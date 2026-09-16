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
