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
   10 轮；完整评分采用可学习 latent readout 和 MLP。
3. 若首轮不足，开展联合微调并依据训练和开发集误差做有记录的调整。
   相关结果尚未产生，不能预先宣称超过强基线。

现有实现的 `essay_bottleneck.verify` 和 `essay_bottleneck.verify_lengths`
均已通过。它们验证实现正确性，不衡量正式评分质量。日志保存在
`artifacts/bottleneck_verify_before_training.log` 和
`artifacts/bottleneck_lengths_before_training.log`。
