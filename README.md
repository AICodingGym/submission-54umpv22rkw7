# Essay scoring baseline

CPU baseline: word unigram/bigram TF-IDF (up to 100,000 features) followed
by Ridge regression (`alpha=1`, LSQR solver). Predictions are rounded to
the nearest integer and clipped to the range 1–6.

## Run

Tested with Python 3.11.15. From this directory:

```bash
python -m pip install -r requirements.txt
# If CSV files have not yet been extracted:
unzip -n data/learning-agency-lab-automated-essay-scoring-2.zip
python baseline.py
```

In the provided workspace, use `../.venv/bin/python baseline.py`.
Optional arguments: `--data-dir`, `--output-dir`, `--seed`, `--alpha`.

## Validation

An 80/20 stratified split with seed 42 uses 12,460 essays for training and
3,116 for validation. TF-IDF is fitted only on the training split. No
hyperparameter or rounding-threshold tuning is performed on validation data.

| Metric | Result |
| --- | ---: |
| Validation quadratic weighted kappa | 0.735663 |
| Validation MAE (integer predictions) | 0.436136 |
| Constant majority-score baseline QWK | 0.000000 |

These are local holdout results, not competition test-set scores. A single
split is a starting point; cross-validation would give a more robust estimate.

## Artifacts

After validation, the pipeline is refitted on all 15,576 labeled essays.
The ignored `outputs/` directory contains:

- `submission.csv`: 1,731 predictions in sample-submission ID order.
- `metrics.json`: validation results and configuration.
- `validation_predictions.csv`: true labels, raw and rounded predictions.
- `model.joblib`: fitted TF-IDF and Ridge pipeline.

To submit the generated predictions to AI Coding Gym:

```bash
aicodinggym mle submit learning-agency-lab-automated-essay-scoring-2 -F outputs/submission.csv -m "TF-IDF + Ridge baseline"
```

## Alternative: handcrafted features + LightGBM

```bash
../.venv/bin/python baseline_lightgbm.py
```

`essay_features.py` extracts 51 per-essay features: text length, word length,
sentence/paragraph length statistics, lexical diversity, punctuation and
selected pronoun/connective/modal frequencies. Tokenization is a simple English
regex; these features do not directly measure grammar or argument quality.
No external dictionaries, models or corpus-level feature fitting are needed.

LightGBM uses 600 trees, learning rate 0.03 and 15 leaves with fixed parameters,
without holdout tuning or early stopping. It uses exactly the same 80/20 split
and rounding rule as Ridge:

| Model | Validation QWK ↑ | Validation MAE ↓ |
| --- | ---: | ---: |
| TF-IDF + Ridge | 0.735663 | 0.436136 |
| Handcrafted features + LightGBM | 0.743975 | 0.423620 |

The QWK improvement is 0.008312 on this single split; generalization of this
gain has not been established with cross-validation or platform evaluation.
After evaluation the model is refitted on all labeled essays. Artifacts are
saved separately in `outputs_lightgbm/`, including `submission.csv`,
`model.joblib`, `metrics.json`, `validation_predictions.csv` and
`feature_importance.csv` (split counts from the final model).
Load the pipeline from the project directory so Python can import
`essay_features.EssayFeatures`, then call `model.predict(texts)` for raw scores
and `baseline.integer_scores(...)` for submission scores.

## Parameter search and score calibration

Run `../.venv/bin/python tune_lightgbm.py` to reproduce the search. The original
20% holdout is excluded from tuning. On the remaining 12,460 essays, a fixed
three-fold stratified CV (seed 43) evaluates the baseline and 11 seeded random
configurations varying tree count, learning rate, leaves, minimum leaf samples,
column sampling and L2 regularization. Each configuration's out-of-fold
predictions are used to select an affine score correction from 15 combinations
of scale (1.0–1.4) and offset (0, -0.1, 0.1). CV selection scores are optimistic;
the untouched holdout is evaluated only after choosing all parameters.

The selected model retained the original LightGBM parameters. Its calibration
is `round_and_clip(2.9501605136436595 + 1.3 * (prediction - 2.9501605136436595))`.
This expands predictions around the training mean before rounding to 1–6.

| Model | Holdout QWK ↑ | Holdout MAE ↓ |
| --- | ---: | ---: |
| Original LightGBM | 0.743975 | 0.423620 |
| Selected LightGBM + calibration | 0.765028 | 0.475610 |

QWK increases by 0.021053, while MAE gets worse: calibration is selected for
the competition metric, not absolute error. This remains a local comparison,
not a platform score or proof of a statistically significant improvement.

`outputs_tuned/` contains the submission, validation predictions, selected
out-of-fold predictions, complete search results and metrics. Its `model.joblib`
is a dictionary, unlike the earlier baseline artifacts. To predict:

```python
import joblib
from tune_lightgbm import calibrate

bundle = joblib.load("outputs_tuned/model.joblib")
raw = bundle["pipeline"].predict(texts)
scores = calibrate(raw, bundle["center"], bundle["scale"], bundle["offset"])
```

The model is refitted on all labeled essays, retaining the calibration chosen
without using the holdout labels. Earlier submissions remain in their own
output directories.
