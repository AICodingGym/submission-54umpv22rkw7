# small-1024 pooling 对照

2026-09-17 启动。目的：先在 DeBERTa-v3-small、1024 token 上筛选输出汇总方式，
再用联合微调和多随机种子复核，确认后才扩大 backbone。

## 首轮方案

| 阶段 | 编码器 | Pooling | 评分头 / 损失 | 轮数 |
| --- | --- | --- | --- | --- |
| baseline | 原始预训练 small，全量微调 | masked mean | Linear / MSE | 10 |
| frozen_mean | baseline 最佳编码器，冻结 | masked mean | 重新初始化 Linear / MSE | 10 |
| frozen_attention | 同一个 baseline 最佳编码器，冻结 | token attention | 相同初始化 Linear / MSE | 10 |

- 沿用 `splits/deberta_seed42.csv`，seed 42，动态 padding，micro batch 4，累积 8。
- baseline 编码器学习率 2e-5；评分头和 attention 学习率 1e-4。
- AdamW、weight decay 0.01、10% warmup 后线性衰减；BF16。
- attention 为 `softmax(Linear(H, 1, bias=False))`，屏蔽 padding；权重零初始化，
  起点为均匀汇总。新增模块不改变公共评分头的初始化或随机数流。
- 冻结时编码器禁用梯度及 dropout。两组重新训练评分头，避免 mean 组独占原有评分头。
- selection B0 QWK 选最佳轮次；calibration 仅在训练结束后拟合 B1；不评估 final_validation。
- `attention − mean` 的冻结组选择集 B0 QWK 差值是首轮筛选指标。baseline 单列，
  不将重新训练评分头带来的差异解释为 pooling 收益。
- 单 seed 收益不是有效性的最终结论；后续需匹配预算的联合微调和多 seed 复核。

## 执行和检查

```bash
../.venv/bin/python -u run_small1024_pooling.py
```

输出：`runs/small1024_pooling_v1/`。每组完成后自动运行已有的
`verify_deberta_run.py`，核对预测、指标、长度和检查点重载。
全部结束后逐张量检查冻结编码器与来源一致，并生成 `comparison.json`。
任一训练或核验失败，流水线停止并在 `pipeline_status.json` 记录失败阶段。
同一输出目录拒绝重复启动，避免覆盖已有实验。

状态：`pipeline_status.json`；逐轮结果：各组 `history.json`；当前批次：各组
`progress.json`；日志：顶层 `baseline_train.log`、`frozen_mean_train.log`、
`frozen_attention_train.log`。

已通过 CPU 小模型功能检查：相同初始化、初始输出一致、padding 不变性、
评分头/attention 梯度、冻结权重不变、保存重载一致、解冻后编码器梯度。
small-1024 mean 全量训练长文本压力检查通过，峰值分配显存约 3.20 GiB。
冻结编码器 attention 的 GPU 压力检查也通过，峰值分配显存约 1.24 GiB，
已验证 BF16 训练、保存及重载推理。正式流水线已启动，初始阶段为 baseline_train；
实时进度以 `pipeline_status.json` 为准。
压力检查只用于验证执行，不作为效果结果。
