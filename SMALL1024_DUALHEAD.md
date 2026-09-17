# small-1024 回归与 ordinal 辅助任务对照

## 目的与架构

验证 ordinal 辅助损失能否改善回归头，而不是比较双头融合。

```text
原始预训练 DeBERTa-v3-small（参与训练）
                  ↓
          masked mean pooling
             ├── Linear(H, 1) → 连续回归分数 → MSE
             └── 独立 Linear(H, 1, bias=False) → z
                       logits[k] = z - cutpoint[k] → mean BCE
```

ordinal targets 为 `y > 1, y > 2, …, y > 5`。阈值从 `[-2, -1, 0, 1, 2]`
开始：第一个阈值自由学习，随后四个间距为 `softplus(raw_gap) + 1e-4`，
保证严格递增；ordinal 标量是内部潜在尺度，不要求等于作文的 1–6 分。
其诊断分数为 `1 + sum(sigmoid(logits))`。

## 固定首轮方案

| 目录 | 损失 | 主预测 |
| --- | --- | --- |
| regression | MSE | 回归头 |
| dual_01 | MSE + 0.1 × mean BCE | 回归头 |
| dual_03 | MSE + 0.3 × mean BCE | 回归头 |

三组都从原始预训练 small 开始，Encoder 全量微调，不从上轮监督检查点继续。
保持 1024 上限、mean pooling、10 轮、seed 42、micro batch 4、累积 8、
BF16、梯度检查点、AdamW、weight decay 0.01、10% warmup 后线性衰减。
编码器学习率 2e-5，回归头与 ordinal 头学习率 1e-4。

ordinal 模块初始化隔离随机数流，因此公共编码器、回归头和后续 dropout 的随机数
起点一致。每组保存 `initialization.json`，流水线逐组核对公共参数哈希与 RNG 哈希；
两组双头额外核对 ordinal 初始参数哈希。

沿用 `splits/deberta_seed42.csv`，selection 的回归 **B0 QWK** 选择最佳轮次。
训练完成后只在 calibration 拟合回归 B1 阈值。ordinal 阈值来自训练，和 B1
后处理阈值是两套参数。保存 ordinal 指标用于诊断，不据其选检查点，不做两头融合。
本轮不预测 final_validation，不自动提交平台。

## 运行和核验

```bash
../.venv/bin/python -u run_small1024_dualhead.py
```

输出在 `runs/small1024_dualhead_v1/`：

- `pipeline_status.json`：运行阶段及异常；同目录拒绝重复启动。
- 各组 `progress.json` / `history.json`：MSE、BCE、总损失和选择集结果。
- 各组 `report.json`：最佳检查点的回归和 ordinal 指标。
- 各组预测 CSV：回归输出、ordinal 五项 logits 和期望分数。
- `comparison_partial.json`：已完成并核验组的结果；`comparison.json`：完整比较。

每组先运行既有回归审计，再对双头运行 `audit_deberta_ordinal.py`：复算 ordinal
指标、核对阈值顺序、核对两头与 Encoder 均更新、逐项重载选择集预测。
流水线记录源文件哈希，若运行期间相关实现变更则停止后续阶段。

先完成 CPU 梯度/初始化/兼容性检查和 GPU 长文本压力检查，再启动正式训练。
CPU 检查结果保存在 `reports/small1024_dualhead_verification/`。

启动前检查已通过：ordinal 损失单独反传可到达 Encoder，公共参数及随机数起点一致，
两头 GPU 保存重载差为 0，双头压力测试峰值分配显存约 3.20 GiB。
旧版与新版纯回归的 CPU 参数和梯度完全一致；两次 GPU 压力测试的 B0/B1 整数
预测完全一致，连续输出最大差 0.015625。本流程未强制 GPU 确定性，不把跨运行
逐位一致当作要求；正式回归控制组仍重新训练，并在后续多 seed 验证稳定性。

本轮是单 seed 筛选。若出现收益，下一步多 seed 复核，稳定后再扩大 backbone。
历史冻结 mean+B1 的平台 0.83049 是已有方案成绩；本轮辅助任务的因果对照是新跑的
端到端 regression 组，不能把训练阶段差异当成 ordinal 收益。
