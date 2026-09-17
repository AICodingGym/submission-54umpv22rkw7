# DeBERTa 基线与评分头对照起点

本页只整理已经完成的纯 DeBERTa 基线，以及下一阶段评分头实验的比较条件。
RLT 与融合成绩另见 [最终结果](RLT_FINAL_RESULTS.md)。

## 已完成的纯 DeBERTa 基线

| 模型 | token 上限 | 训练轮数 / 最佳轮次 | 选择集 B0 QWK | 选择集 B1 QWK | 已提交平台 QWK |
| --- | ---: | ---: | ---: | ---: | ---: |
| DeBERTa-v3-small | 512 | 4 / 4 | 0.791044 | 0.801024 | 未提交 |
| DeBERTa-v3-base | 512 | 10 / 5 | 0.814285 | 0.814573 | 0.82349（B1） |
| DeBERTa-v3-base | 2048 | 10 / 5 | 0.834117 | 0.831143 | 0.82030（B0） |

三组都使用动态 padding。512 与 2048 两组的主要输入差异是最大长度，不能将其
称为“固定 padding 对动态 padding”的实验。512 组选择集有 442 / 1,557 篇截断；
2048 组开发集和测试集均未截断。两组 base 在第 6–10 轮均没有刷新最佳检查点。

B0：连续回归输出按固定半整数阈值映射为 1–6 分。
B1：同一个回归模型，只在独立 calibration 集拟合五个分档阈值。
**B1 是回归后的校准，不是 ordinal regression 模型。**

## 当前架构与训练条件

`原始作文 → DeBERTa → attention-mask mean pooling → Linear(H, 1) → 连续分数`

- 训练目标：连续分数与人工 1–6 分之间的 FP32 MSE。
- 编码器和线性头一起微调；编码器学习率 2e-5，评分头 1e-4。
- micro batch 4，梯度累积 8，有效 batch 32；BF16 与梯度检查点。
- AdamW，weight decay 0.01，10% warmup 后线性衰减；seed 42。
- 同一固定分组划分：train 10,905；selection、calibration、final_validation 各 1,557。
- 按 selection 的 B0 QWK 选择检查点，随后冻结模型并拟合 B1。

实现位置：`deberta_baseline.py` 的 `EssayRegressor`、训练中的 `mse_loss`、
`integer_scores` 和 `fit_thresholds`。推理入口为 `score_deberta.py`。

## 后续比较保留两个基准

- 全文与本地效果基准：base-2048 B0，选择集 QWK 0.834117。
- 平台效果基准：base-512 B1，平台 QWK 0.82349。

上一轮冻结方案之后已查看保留集：base-2048 B0 为 0.811768，base-512 B1 为
0.796978。该集合不能在后续迭代中再被称为新的盲测集；后续不据其反复选型。
选择集与平台属于不同集合，不能将两列分数直接相减来判断提升。

## Regression / ordinal regression 的修改边界

此前 `essay_bottleneck/model.py` 已有两种评分方式：

1. RLT latent 表征经过 MLP 输出标量，以 MSE 训练。
2. 同一标量配合五个有序内部阈值，以五项累计二分类 BCE 训练；预测为
   `1 + sum(sigmoid(logits))`。内部阈值由 train 学习，和训练后 B1 校准阈值不同。

后者的选择集 B0 / B1 为 0.831863 / 0.828055，来自冻结编码器＋RLT＋辅助重建。
它不能用于判断“纯 DeBERTa 的 ordinal regression 是否优于 regression”。

建议下一阶段先在纯 DeBERTa 平均池化表征上建立可切换评分头：

| 待做对照 | 输出与损失 | 用途 |
| --- | --- | --- |
| 原回归控制组 | 线性标量＋MSE | 确认新实验流程能复现对照水平 |
| 稳健回归 | 同一线性标量＋Huber | 比较误差惩罚方式 |
| 有序回归 | 共享标量＋有序阈值＋累计 BCE | 利用 1–6 分的次序关系 |

这些是下一阶段建议，尚未在纯 DeBERTa 上实施或训练。
对照组使用相同编码器初始化、平均池化、长度、划分、批次和训练预算。
编码器是否冻结也必须在各组一致；冻结评分头实验和端到端微调应分开报告。
各模型先报告未校准结果，再单独报告 calibration 上拟合的校准结果。

来源：[small](DEBERTA_RESULTS.md)、[base-512](DEBERTA_BASE_RESULTS.md)、
[base-2048](DEBERTA_2048_RESULTS.md)、[机器可读基线清单](reports/deberta_baseline_overview.json)。
