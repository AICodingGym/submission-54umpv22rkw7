"""Tune on inner cross-validation, then evaluate once on the original holdout."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import cohen_kappa_score, mean_absolute_error
from sklearn.model_selection import ParameterSampler, StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline

from baseline import integer_scores
from baseline_lightgbm import make_model
from essay_features import EssayFeatures


def qwk(y, predictions):
    return float(cohen_kappa_score(y, predictions, weights="quadratic"))


def calibrate(raw, center, scale, offset):
    return integer_scores(center + scale * (raw - center) + offset)


def main():
    root = Path(__file__).resolve().parent
    output = root / "outputs_tuned"
    output.mkdir(exist_ok=True)
    train = pd.read_csv(root / "train.csv", dtype={"essay_id": str})
    test = pd.read_csv(root / "test.csv", dtype={"essay_id": str})
    sample = pd.read_csv(root / "sample_submission.csv", dtype={"essay_id": str})
    if train.isna().any().any() or test.isna().any().any():
        raise ValueError("Missing input values")
    if not train.essay_id.is_unique or not test.essay_id.is_unique or set(train.essay_id) & set(test.essay_id):
        raise ValueError("Duplicate or overlapping essay IDs")
    if not train.score.isin(range(1, 7)).all():
        raise ValueError("Invalid labels")
    if not sample.essay_id.is_unique or set(sample.essay_id) != set(test.essay_id):
        raise ValueError("Invalid sample submission IDs")
    fit, valid = train_test_split(train, test_size=0.2, random_state=42, stratify=train.score)
    # Extraction uses only each essay's own text, with no learned corpus statistics.
    features = EssayFeatures()
    X = features.transform(fit.full_text)
    y = fit.score.to_numpy()
    center = float(y.mean())
    folds = list(StratifiedKFold(n_splits=3, shuffle=True, random_state=43).split(X, y))
    base = make_model(42)[-1].get_params()
    candidates = [{}] + list(ParameterSampler({
        "num_leaves": [7, 15, 31], "min_child_samples": [20, 50, 100],
        "n_estimators": [400, 800, 1200], "learning_rate": [0.02, 0.04],
        "reg_lambda": [1.0, 5.0, 15.0], "colsample_bytree": [0.8, 1.0],
    }, n_iter=11, random_state=42))
    results = []
    best = None
    best_oof = None
    for trial, overrides in enumerate(candidates):
        params = {**base, **overrides}
        oof = np.zeros(len(y))
        for a, b in folds:
            model = LGBMRegressor(**params)
            model.fit(X.iloc[a], y[a])
            oof[b] = model.predict(X.iloc[b])
        # Regression shrinks extreme predictions; tune a modest affine correction.
        calibration_options = [
            (qwk(y, calibrate(oof, center, scale, offset)), scale, offset)
            for scale in [1.0, 1.1, 1.2, 1.3, 1.4]
            for offset in [0.0, -0.1, 0.1]
        ]
        score, scale, offset = max(calibration_options, key=lambda x: x[0])
        result = {
            "trial": trial, "cv_qwk_raw": qwk(y, integer_scores(oof)),
            "cv_qwk_calibrated": score, "scale": scale, "offset": offset,
            "center": center, "parameters": params,
        }
        results.append(result)
        if best is None or score > best["cv_qwk_calibrated"]:
            best, best_oof = result, oof.copy()
        (output / "search_results.json").write_text(json.dumps(results, indent=2) + "\n")
        print(f"Trial {trial + 1}/{len(candidates)}: raw CV={result['cv_qwk_raw']:.5f}, calibrated CV={score:.5f}", flush=True)

    # Model and calibration are now frozen before accessing holdout outcomes.
    model = LGBMRegressor(**best["parameters"])
    model.fit(X, y)
    raw = model.predict(features.transform(valid.full_text))
    predicted = calibrate(raw, center, best["scale"], best["offset"])
    metrics = {
        "validation_qwk": qwk(valid.score, predicted),
        "validation_qwk_without_calibration": qwk(valid.score, integer_scores(raw)),
        "validation_mae": float(mean_absolute_error(valid.score, predicted)),
        "selected_trial": best, "seed": 42, "cv_seed": 43, "cv_folds": 3,
        "fit_rows": len(fit), "validation_rows": len(valid), "test_rows": len(test),
        "note": "CV scores are selection scores; outer holdout is not used for tuning.",
    }
    print(json.dumps(metrics, indent=2), flush=True)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    fit[["essay_id", "score"]].assign(raw_prediction=best_oof).to_csv(output / "selected_oof.csv", index=False)
    valid[["essay_id", "score"]].assign(raw_prediction=raw, prediction=predicted).to_csv(output / "validation_predictions.csv", index=False)
    pipeline = make_pipeline(EssayFeatures(), LGBMRegressor(**best["parameters"]))
    pipeline.fit(train.full_text, train.score)
    raw_test = pipeline.predict(test.full_text)
    predictions = pd.Series(calibrate(raw_test, center, best["scale"], best["offset"]), index=test.essay_id)
    submission = sample[["essay_id"]].copy()
    submission["score"] = submission.essay_id.map(predictions)
    if submission.score.isna().any() or not submission.score.isin(range(1, 7)).all():
        raise ValueError("Invalid submission")
    submission.to_csv(output / "submission.csv", index=False)
    joblib.dump({"pipeline": pipeline, "center": center, "scale": best["scale"], "offset": best["offset"]}, output / "model.joblib")
    print(f"Saved tuned artifacts to {output}", flush=True)


if __name__ == "__main__":
    main()
