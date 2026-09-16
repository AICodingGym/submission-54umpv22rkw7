# Essay scoring baseline

新增 DeBERTa-v3-base 10 轮训练结果：同一 DeBERTa 选择集上，最佳第 5 轮
B0/B1 QWK 为 0.8143 / 0.8146；原 small 为 0.7910 / 0.8010。
第 6–10 轮未进一步改善。模型使用 `score_deberta.py --model-dir outputs_deberta_base`
加载，详见 [base 结果、逐轮指标和使用方式](DEBERTA_BASE_RESULTS.md)。
这些是开发集比较；下述 LightGBM 使用不同划分，不能直接横向比较数字。

## 使用评分系统

现有 `outputs_tuned/model.joblib` 已在全部 15,576 篇标注作文上重新训练。
`score_essay.py` 提供单篇和批量评分，输出 1–6 的整数总分，无需 GPU。
在本目录运行：

```bash
# 单篇：读取 UTF-8 英文作文文件，输出 JSON
../.venv/bin/python score_essay.py --file essay.txt
# 或直接输入作文全文
../.venv/bin/python score_essay.py --text "Your complete English essay here."
# 批量：输入含 essay_id、full_text 的 CSV，输出 essay_id、score
../.venv/bin/python score_essay.py --csv test.csv --output artifacts/scored.csv
```

Python 应用可复用加载后的模型：

```python
from score_essay import EssayScorer

scorer = EssayScorer()
scores = scorer.predict(["Your complete English essay here."])
```

空白作文会被拒绝；批量输入需要非空且唯一的作文 ID。模型路径默认为
脚本旁的 `outputs_tuned/model.joblib`，也可通过 `--model` 指定相同格式的
可信本地模型。模型文件不随 Git 提交；缺失时运行 `tune_lightgbm.py` 重建。

现有独立留出集共 3,116 篇，QWK 为 0.7650，平均绝对误差为 0.4756 分，
完全同分率为 56.13%，误差不超过 1 分的比例为 96.57%。这些是训练流程
留出模型的评估结果，不是最终全量重训模型的新测试结果，也不是置信度。
评分入口已核验：1,731 篇测试作文的输出与原有预测完全一致。

当前版本用于与训练数据相似的英文议论文整体评分。数据只有整体分数，
不能据此验证中文评分、语法/内容等分项分数或评语的准确性。
Qwen 微调脚本已存在，但 `outputs_lora/` 尚无完整评估结果；本入口使用
已经训练完成并经过留出评估的 LightGBM 模型。

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

## Qwen3-1.7B zero-shot pilot

`download_qwen.py` downloads the official model to ignored `models/`.
`evaluate_qwen.py` evaluates 96 essays sampled with stratification (seed 44)
from the existing holdout. It uses experimental scoring guidance, not the
official rubric text, disables thinking and normalizes the next-token
probabilities over the six score digits. Expected scores are rounded to 1–6;
argmax scores are also reported. No fine-tuning or score calibration is used.

Install the baseline requirements and `requirements-qwen.txt`. For CPU-only
execution, install the CPU PyTorch wheel from its official CPU index first.
The recorded environment used torch 2.14.0+cpu and Transformers 4.57.6.

```bash
../.venv/bin/python download_qwen.py
HF_HUB_OFFLINE=1 ../.venv/bin/python evaluate_qwen.py
```

The recorded model revision is `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`.
The downloader resolves the current upstream revision; the evaluator records
the exact downloaded revision. Artifacts are stored in `outputs_qwen/`.

| Method | QWK on the same 96 essays |
| --- | ---: |
| Qwen expected score, rounded | 0.117277 |
| Qwen argmax score | 0.124447 |
| TF-IDF + Ridge | 0.734106 |
| LightGBM | 0.748231 |
| Calibrated LightGBM | 0.744633 |

CPU BF16 inference took 43.4 seconds (excluding model loading). Qwen assigned
4 essays a score of 3, 56 a score of 4 and 36 a score of 5, indicating a strong
upward scoring bias with this prompt. This small zero-shot pilot does not
measure fine-tuned performance and is not comparable to full-holdout scores.
It produces evaluation artifacts, not a test-set submission.

Hardware inspection outside the sandbox confirmed an RTX 3080 with 20 GiB
VRAM and a working NVIDIA driver. The initial GPU detection failed because
the sandbox hides GPU devices. GPU experiments require approved device access
and a CUDA-enabled PyTorch installation; the pilot used CPU PyTorch.

## Qwen3-1.7B LoRA fine-tuning

Install CUDA PyTorch first, then the LoRA requirements:

```bash
uv pip install --python ../.venv/bin/python torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python ../.venv/bin/python -r requirements-lora.txt
HF_HUB_OFFLINE=1 ../.venv/bin/python finetune_qwen.py --smoke --epochs 1
HF_HUB_OFFLINE=1 ../.venv/bin/python finetune_qwen.py --batch-size 8 --accumulation 2
```

Run with GPU access (the workspace sandbox hides NVIDIA devices). Training
uses BF16, gradient checkpointing, rank-8 LoRA on query/value projections,
learning rate 1e-4 and effective batch size 16. Only 1,605,632 parameters are
trainable. The objective is six-class cross entropy on the score-digit logits
at the answer position; it does not train free-form explanations. Predictions
are rounded expectations over the six classes without additional calibration.

The original 12,460-row training split is divided into 11,214 fitting examples
and 1,246 inner validation examples (stratified, seed 45). Two epochs are run;
the adapter with higher inner QWK is reloaded for evaluation on the original
3,116-row outer holdout and test prediction. The final adapter remains trained
on 11,214 examples; it is not refitted on the full labeled dataset.

Inputs are capped at 1,024 tokens with head/tail retention, preserving the
instructions and answer position. The configured length truncates 365 fitting,
36 inner-validation, 110 outer-validation and 59 test essays. Longer context
is a potential later experiment. The prompt and experimental rubric match
the zero-shot pilot; the model learns the task's scoring scale from labels.

`outputs_lora/` stores configuration, split IDs, progress, per-epoch adapters,
the selected `adapter/`, training history, validation predictions, metrics and
`submission.csv`. `outputs_lora_smoke/` is separate and contains only the small
end-to-end smoke check. Adapters require the original base-model revision and
the same prompt/tokenization/scoring procedure used by `finetune_qwen.py`.
