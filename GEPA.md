# Qwen3-4B + GEPA 本地提示词优化

这是冻结语言模型权重的提示词搜索，不是梯度微调。使用 gepa==0.1.4 官方
优化引擎及自定义 adapter，Qwen/Qwen3-4B 同时承担评分和反思，GPU 串行运行，
不使用外部 API。与 DeBERTa 复用固定分组划分，最终验证集不评估。

## 固定协议

- 原始模型 revision：1cfa9a7208912126459214e8b04321603b3df60c。
- BF16，SDPA，评分 batch 2，thinking 关闭，六个分数字符的条件概率。
- G0/G1：选择条件概率最高的分数。保存六类概率及条件期望。
- G1_expected_fixed：条件期望按固定阈值取整，用于隔离校准本身的增益。
- G2：冻结 G1，在 calibration 上对条件期望拟合五个递增阈值。
- 输入保留大小写与标点，作文最多前 2,048 tokens；总评分上下文最多 4,096。
- 分数输出受限为 1–6，因此不产生格式解析失败；这不表示开放式生成格式正确率为 100%。
  `score_token_mass` 记录未条件化时六个数字的总概率，可诊断受限输出的合理性。

## 搜索与数据隔离

从 train 按分数分层抽取 300 篇（seed 46），分为十组；每轮使用一组完整
预测的 QWK 进行候选接受比较。反思只取该训练组误差最大的三篇及人工分数，
作文反馈限制为前 2,200 字符。此限制可能丢失后文依据，属于本轮反思能力的限制。
不向反思模型提供 selection 作文、标签或逐篇错误。

从 selection 分层固定抽取 300 篇（seed 47）作为一个整体评估单元，因此
GEPA 的候选目标就是这 300 篇的整组 QWK，而不是平均逐篇误差冒充 QWK。
使用 current_best 候选选择、单组件 instructions、无 merge，最多 20 次
反思提案。反思 greedy 解码最多 1,100 新 tokens，要求返回完整提示词；
解析失败或超限时保留父提示词，并保存原始反思输出。每次反馈及候选均留档。

仅搜索子集排名前三的已接受候选与基础提示词进入完整 1,557 篇 selection
比较，按 QWK 选择最终提示词。然后使用 1,557 篇 calibration 拟合阈值。
最终 1,557 篇 final_validation 不进行预测、评估或调参。选择集成绩是开发
成绩，经过多次候选选择，不能作为无偏测试成绩。原始数据曾用于更早实验，
本轮保留不代表历史从未见过。

## 执行

```bash
uv pip install --python ../.venv/bin/python -r requirements-gepa.txt
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ../.venv/bin/python gepa_qwen.py --smoke
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ../.venv/bin/python gepa_qwen.py
```

权重位置读取 `models/qwen4b_snapshot.txt`。初次可用 huggingface_hub 的
snapshot_download 下载上述固定 revision，并将返回的绝对路径写入此文件。
已有运行配置时脚本拒绝覆盖，请先归档旧运行。试跑使用 60 篇训练作文、
30 篇选择作文、30 篇校准作文，只尝试一次提案，不代表正式结果。

产物位于 `outputs_gepa/`，试跑位于 `outputs_gepa_smoke/`：

- `config.json`、两个 `search_*_ids.csv`：完整配置与固定搜索样本。
- `search/`、`search_result.json`、`proposal_*.json`：GEPA 轨迹和反思记录。
- `prompt_G0.txt`、`prompt_G1.txt`：基础及最终提示词。
- `thresholds.json`：仅由 calibration 拟合的 G2 阈值。
- `candidate_*_selection.csv`、`selection_predictions.csv`、`calibration_predictions.csv`：逐篇结果。
- `report.json`：指标、混淆矩阵、评分耗时、总运行时间、显存、截断和调用次数。
- `cache/`：按候选提示词及有序作文 ID 缓存预测，仅供同一固定运行复用。

G2 与 G1 的差异同时包含 argmax → 条件期望的改变，因此应另外比较
G1_expected_fixed 与 G2 才能衡量阈值校准本身的影响。条件期望不是置信度。

冻结产物生成后，可对新作文评分（需要 CUDA）：

```bash
HF_HUB_OFFLINE=1 ../.venv/bin/python score_gepa.py --file essay.txt --version G1
HF_HUB_OFFLINE=1 ../.venv/bin/python score_gepa.py --csv new_essays.csv --version G2 > predictions_gepa.csv
```

CLI 的默认版本为 G2，不表示校准必然优于 G1；根据结果报告选用。
