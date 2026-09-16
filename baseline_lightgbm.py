"""Evaluate handcrafted essay features + LightGBM on the baseline split."""

import argparse
import json
from pathlib import Path

import joblib
import lightgbm
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import cohen_kappa_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

from baseline import integer_scores
from essay_features import EssayFeatures


def make_model(seed):
    # Fixed configuration; the holdout is used only for evaluation.
    return make_pipeline(EssayFeatures(), LGBMRegressor(
        objective="regression", n_estimators=600, learning_rate=0.03,
        num_leaves=15, min_child_samples=30, colsample_bytree=0.9,
        reg_lambda=5.0, random_state=seed, n_jobs=4, verbosity=-1,
        deterministic=True, force_col_wise=True,
    ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent
    parser.add_argument("--data-dir", type=Path, default=root)
    parser.add_argument("--output-dir", type=Path, default=root / "outputs_lightgbm")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train = pd.read_csv(args.data_dir / "train.csv", dtype={"essay_id": str})
    test = pd.read_csv(args.data_dir / "test.csv", dtype={"essay_id": str})
    sample = pd.read_csv(args.data_dir / "sample_submission.csv", dtype={"essay_id": str})
    for name, frame in [("train", train), ("test", test)]:
        if frame[["essay_id", "full_text"]].isna().any().any() or not frame.essay_id.is_unique:
            raise ValueError(f"{name}: missing text/IDs or duplicate IDs")
    if not train.score.isin(range(1, 7)).all() or set(train.essay_id) & set(test.essay_id):
        raise ValueError("Invalid training scores or overlapping train/test IDs")
    if not sample.essay_id.is_unique or set(sample.essay_id) != set(test.essay_id):
        raise ValueError("Sample submission IDs do not match test IDs")
    fit, valid = train_test_split(train, test_size=0.2, random_state=args.seed, stratify=train.score)
    print(f"Training: {len(fit)}; validation: {len(valid)}; test: {len(test)}", flush=True)
    model = make_model(args.seed)
    model.fit(fit.full_text, fit.score)
    raw = model.predict(valid.full_text)
    predicted = integer_scores(raw)
    metrics = {
        "validation_qwk": float(cohen_kappa_score(valid.score, predicted, weights="quadratic")),
        "validation_mae": float(mean_absolute_error(valid.score, predicted)),
        "train_rows": len(train), "fit_rows": len(fit), "validation_rows": len(valid),
        "test_rows": len(test), "seed": args.seed, "lightgbm_version": lightgbm.__version__,
        "feature_count": model[-1].n_features_in_, "model_parameters": model[-1].get_params(),
    }
    print(json.dumps(metrics, indent=2), flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    valid[["essay_id", "score"]].assign(raw_prediction=raw, prediction=predicted).to_csv(
        args.output_dir / "validation_predictions.csv", index=False,
    )
    print("Refitting on all training essays...", flush=True)
    model = make_model(args.seed)
    model.fit(train.full_text, train.score)
    predictions = pd.Series(integer_scores(model.predict(test.full_text)), index=test.essay_id)
    submission = sample[["essay_id"]].copy()
    submission["score"] = submission.essay_id.map(predictions)
    if submission.score.isna().any() or not submission.score.isin(range(1, 7)).all():
        raise ValueError("Invalid submission scores")
    submission.to_csv(args.output_dir / "submission.csv", index=False)
    joblib.dump(model, args.output_dir / "model.joblib")
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    pd.DataFrame({"feature": model[-1].feature_name_, "split_importance": model[-1].feature_importances_}).sort_values(
        "split_importance", ascending=False,
    ).to_csv(args.output_dir / "feature_importance.csv", index=False)
    print(f"Saved model, metrics and submission to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
