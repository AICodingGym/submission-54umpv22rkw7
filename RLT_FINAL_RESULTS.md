# 全文 DeBERTa 与 RLT 最终结果

最终提交的 **75% 全文 RLT＋25% 512-token DeBERTa 基线融合**，平台 QWK 为
**0.82452**，超过原最佳 **0.82349**，增加 **0.00103**。
RLT 单模型的平台最好成绩为 0.82311，联合微调也未超过原平台最佳；本次提升属于融合结果。

## 固定比较

| 方案 | 选择集 QWK | 平台 QWK | 本轮保留集 QWK |
| --- | ---: | ---: | ---: |
| 512-token DeBERTa 基线，B1 | 0.814573 | 0.82349 | 0.796978 |
| 2048-token DeBERTa 基线，B0 | 0.834117 | 0.82030 | 0.811768 |
| 全文 RLT 单模型，B0 | 0.834201 | 0.82311 | 0.807083 |
| 最终 75% RLT＋25% 512 基线融合，仿射分档 | **0.835118** | **0.82452** | **0.815159** |

选择集与保留集各 1,557 篇，平台提交 1,731 篇。不同集合的数值不能直接相互比较。
本地与平台的固定目标分别为 0.8341167069970928、0.82349，未随候选变化而移动。
保留集在最终模型及分档参数冻结、平台达标之后做一次固定比较，本轮不用于调参。
已经进行多次开发比较，这些分数是当前样本上的结果，不宣称统计显著优势。

## 2048 上限与动态 padding

DeBERTa-v3-base 完成 10 轮，最佳第 5 轮。输入上限为 2048，每个批次仅补齐到
批内最长文本。14,019 篇开发样本和 1,731 篇测试样本均未被截断；最长训练样本
为 1,778 tokens，最长测试样本为 1,605 tokens。
2048 基线已单独提交，平台为 0.82030，因此保留原 512 基线作为平台对照。

## 最终方案与复现

- RLT 组件：`runs/rlt_cached_norm_v1` 第 1 轮；冻结已监督训练的 2048 基线编码器，
  K=4、D=256、归一化查询、重建权重 0.1，全文动态 padding。
- 基线组件：`outputs_deberta_base` 第 5 轮，512-token DeBERTa-v3-base。
- 连续分数：`0.75 × RLT + 0.25 × baseline`，融合权重预先固定。
- 分档：仅在 calibration 拟合整体斜率 1.04、偏移 -0.19，再使用半整数分界；
  实现保存等价的原始分数阈值。
- 提交文件：`runs/rlt_baseline512_blend075_v1/submission.csv`。
  SHA-256：`df8dfb9fef3694356a4e92ee92eaf59334a37bdeeea9e983e46e5bd8e5b64fc3`。

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ../.venv/bin/python score_rlt_blend.py \
  --model-dir runs/rlt_baseline512_blend075_v1 --device cuda --version affine --csv test.csv
```

两个组件的完整选择集重载预测差均为 0，融合预测与指标复算一致。
新测试预测与此前已验证组件的加权结果逐条相同，提交的 ID 顺序、行数和 1–6 整数范围
均通过检查。保留集四套固定预测的整数分数和指标也已复算。

有序评分试验另外发现并修正了 CSV 的 float32 小数精度问题：改为保留精确数值的
float64 十进制导出。修复没有改变该模型的权重、阈值、整数预测或评分指标。

[最终候选记录](reports/rlt_candidate.json) ·
[平台提交记录](reports/rlt_baseline512_blend075_v1/submission.json) ·
[保留集比较与核验](reports/rlt_final_comparison/) ·
[完整实验过程](RLT_EXPERIMENTS.md) ·
[2048 基线记录](DEBERTA_2048_RESULTS.md)
