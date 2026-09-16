"""Train and validate a CPU-only TF-IDF + Ridge essay scoring baseline."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import cohen_kappa_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline


def make_model(alpha):
    return make_pipeline(
        TfidfVectorizer(
            ngram_range=(1, 2), min_df=3, max_df=0.98,
            max_features=100_000, sublinear_tf=True, dtype=np.float32,
        ),
        Ridge(alpha=alpha, solver="lsqr"),
    )


def integer_scores(predictions):
    return np.clip(np.floor(predictions + 0.5), 1, 6).astype(int)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=1.0)
    args = parser.parse_args()
    train = pd.read_csv(args.data_dir / "train.csv", dtype={"essay_id": str})
    test = pd.read_csv(args.data_dir / "test.csv", dtype={"essay_id": str})
    sample = pd.read_csv(args.data_dir / "sample_submission.csv", dtype={"essay_id": str})
    for name, frame in [("train", train), ("test", test)]:
        if frame[["essay_id", "full_text"]].isna().any().any() or not frame.essay_id.is_unique:
            raise ValueError(f"{name}: missing text/IDs or duplicate IDs")
    if not train.score.isin(range(1, 7)).all():
        raise ValueError("Training scores must be integers from 1 to 6")
    if set(train.essay_id) & set(test.essay_id):
        raise ValueError("Training and test IDs overlap")
    if not sample.essay_id.is_unique or set(sample.essay_id) != set(test.essay_id):
        raise ValueError("Sample submission IDs do not match test IDs")

    fit, valid = train_test_split(
        train, test_size=0.2, random_state=args.seed, stratify=train.score,
    )
    print(f"Training: {len(fit)}; validation: {len(valid)}; test: {len(test)}", flush=True)
    model = make_model(args.alpha)
    # Fit the vocabulary and IDF only on the training split to avoid leakage.
    model.fit(fit.full_text, fit.score)
    raw = model.predict(valid.full_text)
    predicted = integer_scores(raw)
    metrics = {
        "validation_qwk": float(cohen_kappa_score(valid.score, predicted, weights="quadratic")),
        "validation_mae": float(mean_absolute_error(valid.score, predicted)),
        "constant_baseline_qwk": float(cohen_kappa_score(
            valid.score, np.full(len(valid), int(fit.score.mode().iloc[0])), weights="quadratic",
        )),
        "train_rows": len(train), "fit_rows": len(fit), "validation_rows": len(valid),
        "test_rows": len(test), "seed": args.seed, "alpha": args.alpha,
        "sklearn_version": sklearn.__version__,
    }
    print(json.dumps(metrics, indent=2), flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    valid[["essay_id", "score"]].assign(raw_prediction=raw, prediction=predicted).to_csv(
        args.output_dir / "validation_predictions.csv", index=False,
    )
    print("Refitting on all training essays...", flush=True)
    model = make_model(args.alpha)
    model.fit(train.full_text, train.score)
    predictions = pd.Series(integer_scores(model.predict(test.full_text)), index=test.essay_id)
    submission = sample[["essay_id"]].copy()
    submission["score"] = submission.essay_id.map(predictions)
    if submission.score.isna().any() or not submission.score.isin(range(1, 7)).all():
        raise ValueError("Invalid submission scores")
    submission.to_csv(args.output_dir / "submission.csv", index=False)
    joblib.dump(model, args.output_dir / "model.joblib")
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"Saved model, metrics and submission to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
