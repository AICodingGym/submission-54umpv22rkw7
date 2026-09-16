# DeBERTa-v3-small 监督评分 baseline

## DeBERTa-v3-base 长训练对照

训练脚本支持 `--model-size base`、任意正整数 `--epochs` 和独立的
`--output-dir`。small 的默认配置保留。base 使用相同的固定划分、
512-token 输入、平均池化回归头、有效 batch 32 和最佳 B0 QWK 选择规则，
已完成 10 轮训练，最佳为第 5 轮，完整产物保存到 `outputs_deberta_base/`。
同一选择集 B0/B1 QWK 为 0.814285 / 0.814573，详见
[base 结果与逐轮比较](DEBERTA_BASE_RESULTS.md)。
学习率衰减按 10 轮总步数计算，因此其第 4 轮并不等价于一个独立的 4 轮实验。

```bash
../.venv/bin/python download_deberta.py --model-size base --revision 8ccc9b6f36199bec6961081d44eb72fb3f7353f3
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ../.venv/bin/python deberta_baseline.py --model-size base --epochs 10 --smoke
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ../.venv/bin/python deberta_baseline.py --model-size base --epochs 10
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ../.venv/bin/python verify_deberta_run.py --run-dir outputs_deberta_base
../.venv/bin/python score_deberta.py --model-dir outputs_deberta_base --device cuda --file essay.txt
```

重跑时通过 `--output-dir` 指定新目录；已有实验不会被覆盖。核验程序重算
三个开发集合的指标，检查划分与标签，独立重载全部选择集预测，并保存
`verification.json`、环境版本和权重哈希。最终验证集继续保留。

## 原 small 基线

作文原文 → DeBERTa-v3-small → attention mask 平均池化 → 线性回归头。
编码器和回归头一起训练，目标是 FP32 MSE；B0 使用固定阈值
`1.5, 2.5, 3.5, 4.5, 5.5` 转为 1–6 分，阈值相等时进入较高分。
B1 使用同一个冻结的检查点，只改变五个评分阈值。

## 数据与选择规则

固定划分保存在 `splits/deberta_seed42.csv`，元数据包含原始 CSV 的 SHA256，
复用时检查数据是否变化。按分数分层、重复组分组，种子 42：

| 集合 | 篇数 | 用途 |
| --- | ---: | --- |
| train | 10,905 | 更新模型权重 |
| selection | 1,557 | 按每轮 B0 QWK 选择检查点；比较 B0/B1 |
| calibration | 1,557 | 模型冻结后拟合 B1 阈值 |
| final_validation | 1,557 | 本轮不预测、不计算指标 |

重复检测规则是小写化、非单词字符归一化后完全相同，或者二值词三元组
余弦相似度 ≥ 0.95；连通分量整体分到同一集合。这个规则发现 15,574 个组，
不保证识别所有语义改写。归一化仅用于分组，模型输入保留原始文本。
这些数据已用于仓库此前的实验，因此 final_validation 只是在本轮及后续
候选比较中预留，并非历史上从未使用过的数据。

检查点选择只看 selection 的 B0 QWK，同分保留较早轮次。B1 在 calibration
上从固定阈值出发做确定性的坐标搜索，保持五个阈值严格递增。
校准集指标属于拟合指标；selection 也参与了检查点选择，不能当作最终
无偏测试成绩。与旧 LightGBM 的划分不同，分数不应直接解释为公平比较。

## 训练配置与复现

| 项目 | 配置 |
| --- | --- |
| 原始权重 | microsoft/deberta-v3-small |
| 固定 revision | a36c739020e01763fe789b4b85e2df55d6180012 |
| 序列长度 | 512 tokens，右侧截断 |
| batch / 累积 | 4 / 8，有效 batch 32 |
| 显存不足时 | batch 2 / 累积 16 |
| 验证 batch | 4 |
| 精度 | CUDA 支持时 BF16 autocast，否则 FP32 |
| 内存优化 | 梯度检查点、batch 内动态 padding |
| 优化器 | AdamW，weight decay 0.01 |
| 学习率 | 编码器 2e-5，回归头 1e-4 |
| 调度 | 总更新步数 10% warmup，再线性衰减 |
| 最大梯度范数 | 1.0 |
| 轮数 / 随机种子 | 4 / 42 |

最后一个梯度累积窗口按实际样本数归一化。保存的模型只在 train 上训练，
不会重新拟合 selection、calibration 或 final_validation。
固定随机种子便于复现，不承诺跨硬件、CUDA 版本的逐位一致。

在项目目录运行（先安装对应硬件的 CUDA PyTorch）：

```bash
../.venv/bin/python -m pip install -r requirements-deberta.txt
# 仅当原始权重缺失时联网下载；其余步骤均可离线运行
../.venv/bin/python download_deberta.py
../.venv/bin/python deberta_baseline.py --prepare-only
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ../.venv/bin/python deberta_baseline.py --smoke
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ../.venv/bin/python deberta_baseline.py
```

训练需要 GPU 设备访问。已有运行目录时脚本拒绝覆盖；复现前先将旧目录
重命名归档。短试跑使用 256 篇长训练作文、每个开发集合 32 篇长作文，
只做 64 个微批次、8 次优化器更新，用于验证执行链路，不代表正式成绩。
本轮使用 Transformers 的 `DebertaV2TokenizerFast`，训练和推理复用
保存的 tokenizer。其 SentencePiece byte fallback 转换限制可能影响
罕见字符，因此不将结果扩展到未验证的语言或输入分布。

## 产物与评分入口

正式结果位于 `outputs_deberta/`，试跑结果位于 `outputs_deberta_smoke/`：

- `model/`：编码器和回归头完整权重、模型配置、tokenizer。
- `config.json`：训练设置、权重 revision、划分哈希、各集合截断比例。
- `history.json`：每轮 B0 指标、训练和验证耗时。
- `thresholds.json`：B1 的五个阈值及所选轮次。
- `report.json`：B0/B1 QWK、MAE、混淆矩阵、截断分组结果与推理耗时。
- `train_predictions.csv`、`selection_predictions.csv`、`calibration_predictions.csv`：
  逐篇标签、连续预测、B0/B1 分数、截断前 token 数和截断标记。

混淆矩阵的行是真实分数，列是预测分数，顺序均为 1–6。推理耗时包括
batch padding、GPU 计算与结果传回 CPU，不包括加载权重和文本分词。
连续预测没有裁剪，也不是置信度；整数输出保持在 1–6。

```bash
# 默认 CPU、B1；GPU 请指定 --device cuda
../.venv/bin/python score_deberta.py --file essay.txt
../.venv/bin/python score_deberta.py --file essay.txt --version B0
../.venv/bin/python score_deberta.py --csv new_essays.csv > predictions.csv
```

Python 中可加载一次并复用：

```python
from score_deberta import DebertaScorer

scorer = DebertaScorer(device='cpu', version='B1')
raw, scores = scorer.predict(['Your complete English essay here.'])
```

CPU 推理使用 FP32，与 GPU BF16 的连续输出可能略有差异，靠近阈值时
整数分也可能改变。比较实验时应使用相同设备、精度与 batch 配置。
大权重和运行产物不纳入 Git；固定划分、代码和结果摘要纳入版本管理。

本轮实测结果见 [DEBERTA_RESULTS.md](DEBERTA_RESULTS.md)。
当前 Transformers 4.57.6 在读取本地 DeBERTa 配置时也会触发 Mistral
regex 警告；已检查本地库的检测分支，该警告未修改 tokenizer。
训练与模型重载的分词及全部选择集预测已核验一致。
