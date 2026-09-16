# 强基线与 RLT 表征瓶颈实验

用户目标：完成 2048-token 强基线训练并提交；再评估 RLT 表征瓶颈，
若冻结 backbone 不足则联合微调，持续迭代直至超过当前最佳模型。

## 评价约定

- 使用既有固定 train / selection / calibration / final_validation 划分。
- 以 QWK 为主，报告 MAE 和原先超过 512 tokens 的同一作文组。
- 每次实验按 selection 的 B0 QWK 选择检查点，同分保留较早轮次；
  冻结模型后才使用 calibration 拟合 B1 阈值。
- 开发集比较与平台分数分开报告；不把一个集合的数值当作另一个集合的目标。
- 固定目标：选择集超过 2048-token B0 的 **0.834117**；平台超过此前
  512-token B1 的 **0.82349**。2048 的平台提交为 0.82030，没有替代旧平台最佳。
- 强基线完成后冻结并提交一次。RLT 候选主要用本地开发集选择，
  最终方案冻结后再按既有协议对保留验证集统一复核，避免反复用平台调参。
- 所有尝试均记录，不把失败候选删掉或只报告最优一次。

## 当前阶段

1. `outputs_deberta_base_2048/`：DeBERTa-v3-base、2048 上限、动态 padding、
   10 轮完整监督训练已完成。最佳第 5 轮 B0 QWK 0.834117、MAE 0.340398；
   B1 校准未改善，固定 B0 为强基线。14,019 篇开发样本长度与指标核验、
   1,557 篇选择集重载核验均通过，平台提交完成，得分 0.82030。
   本版本保留为本地对照，旧 512-token B1 保留为平台最佳。
2. RLT 首轮：已有 `essay_bottleneck` 实现，从原始 base 权重出发，
   冻结编码器，K=4、D=256、重建权重 0.1，2048 上限、有效 batch 32、
   10 轮已完成；最佳第 10 轮 B0 QWK 0.808142、B1 QWK 0.823761，
   均未超过强基线，进入联合微调。
3. 联合微调首轮：保持上述架构、原始 base 初始化、数据划分和 10 轮调度，
   放开学生编码器，编码器学习率 2e-5，新模块 1e-4。重建教师为独立冻结的
   初始编码器。10 轮已完成，最佳第 5 轮；正式输出为 `runs/rlt_joint_v1/`。
4. 下一轮：冻结已经监督训练的 2048-token 强基线编码器，训练带查询归一化的
   RLT 瓶颈和重建模块。此方案改变了初始化和查询处理，不能把收益单独归因于
   其中一项。仅在每个有效 batch 内按长度排列小批次以减少 padding，显式记录。

现有实现的 `essay_bottleneck.verify` 和 `essay_bottleneck.verify_lengths`
均已通过。它们验证实现正确性，不衡量正式评分质量。日志保存在
`artifacts/bottleneck_verify_before_training.log` 和
`artifacts/bottleneck_lengths_before_training.log`。

## 已完成结果：同一选择集 1,557 篇

| 方案 | 最佳轮次 | B0 QWK | B0 MAE | B1 QWK | B1 MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2048-token 强基线 | 5 | 0.834117 | 0.340398 | 0.831143 | 0.368658 |
| RLT 冻结编码器＋重建 | 10 | 0.808142 | 0.355812 | 0.823761 | 0.373154 |
| RLT 联合微调＋重建 | 5 | 0.815773 | 0.356455 | 0.817984 | 0.375723 |

冻结实验训练与逐轮选择集评估共 1,537.68 秒（25.63 分钟）。
正式核验通过：3,114 条开发集预测指标与整数分数全部一致，重新加载的
1,557 篇选择集预测差为 0；14,019 篇开发作文长度已重新分词核对且均无截断；
保存的编码器权重与原始模型逐项完全相同。
[冻结实验配置、指标、曲线与核验](reports/rlt_frozen_v1/)。
该方案尚未超过强基线，没有提交平台，也没有使用 final_validation。

联合微调实验共 7,476.51 秒（124.61 分钟，训练与逐轮选择集评估）。
完整核验通过：3,114 条开发预测指标、14,019 篇开发作文长度均复核一致，
1,557 篇选择集重载连续预测最大差为 0。最佳检查点仍未超过本地基线，
没有提交平台，也没有打开保留验证集。
[联合微调配置、指标与核验](reports/rlt_joint_v1/)。

## 冻结模型的表征诊断

在固定 train 中每档取前 3 篇，共 18 篇，以 CPU FP32 检查：
平均归一化注意力熵为 0.999628（接近 1 表示接近均匀），
平均 latent 两两余弦相似度为 0.994363。
查询参数本身没有重合，但其输出表征很相似。此小样本诊断不证明表征没有
互补预测信息，也不证明性能差距由此造成；它为后续查询归一化或初始化
对照提供假设。当前联合微调实验保持原配置，不在运行中改动模型。

复现：`../.venv/bin/python diagnose_rlt_latents.py --run-dir runs/rlt_frozen_v1`。
[逐篇诊断](reports/rlt_frozen_v1/latent_diagnostic.json)、
[查询参数统计](reports/rlt_frozen_v1/query_diagnostic.json)。

同样的 18 篇训练作文上，联合微调第 5 轮检查点的归一化注意力熵为
0.999978，latent 两两余弦为 0.997647。这一中间检查点仍表现为近似均匀
注意力和相似表征，尚不能据此解释性能因果。检查点哈希与逐篇统计见
[第 5 轮诊断](reports/rlt_joint_v1/epoch5_latent_diagnostic.json)；它不是最终
选定模型的诊断，也不改变当前训练配置。

计算量估算（未应用于当前训练）：在每个已经打乱的 32 篇有效批次内部，
按 token 长度排序后再切成 4 篇小批次，可以保持每次参数更新使用的样本集合
及权重不变。对 10 个 epoch 的实际训练长度估算，padding 后 token 总数约为
原来的 75.60%，attention 的 `batch × max_length²` 代理量约为 63.65%。
这不是实测运行时间；重排会改变 dropout 随机数对应和浮点累加次序，
若用于后续实验会显式记录。[逐轮估算](reports/rlt_batching_estimate.json)。

已为可能的后续 warm-start 实验导出 `models/deberta_base_2048_supervised/`，
只包含本地强基线第 5 轮编码器和 tokenizer，不含评分头。198 个编码器张量
重新加载后逐项相同；初始化来自已监督训练的编码器，不能称为原始预训练权重。
当前 `rlt_joint_v1` 仍从原始 base 出发，没有使用此导出。
[来源与校验记录](reports/deberta_base_2048/supervised_encoder_export.json)。
导出工具：`export_deberta_encoder.py --source-run outputs_deberta_base_2048
--output-dir models/deberta_base_2048_supervised`（已有目标目录会拒绝覆盖）。

## 最终候选复核工具（尚未执行评估）

候选通过完整审计、选择集超过固定目标且最终模型方案确定后，使用：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ../.venv/bin/python compare_final_candidate.py \
  --candidate-run runs/<最终候选> --version <B0或B1> --device cuda
```

该命令会打开保留验证集。脚本在读取其标签前，将模型权重哈希、模型配置、
阈值、数据哈希和两个既定基线固定到 `runs/rlt_final_comparison/frozen_manifest.json`。
同一目录只允许重试完全相同的比较，且不拟合阈值。冻结 RLT 首轮未达到选择集
目标，已验证脚本会在读取保留验证集前拒绝该候选；没有运行实际保留集评估。
