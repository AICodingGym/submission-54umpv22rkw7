# 可学习文章压缩表征＋评分器

借鉴 RLT 的可学习表征瓶颈，使用人工作文总分进行监督学习；不包含 actor、
在线强化学习或 RECAP。独立于原有 DeBERTa 训练入口，复用其固定划分和指标函数。
这是待训练验证的候选架构，没有提升 QWK 的实证结论。

## 模型

```text
作文 → 预训练 DeBERTa → token 表示＋位置
                           ↓
                K 个可学习查询做 cross-attention
                           ↓
                  K × D 个连续表征数值
                    ↙               ↘
             展平＋MLP 评分头      位置查询＋重建 decoder
                    ↓               ↓
                 连续分数      固定教师的 token 表示
                    ↓
              校准阈值 → 1–6 分
```

默认 K=4、D=256、4 个注意力头，重建权重 0.1。这些是初始实验配置，未调优。
这里的 latent token 是连续向量，不是文本摘要或离散词元，也没有预设分项含义。
查询将输入序列汇总为固定大小表征，但完整推理仍须执行编码器，不代表省掉编码成本。

总损失为 `MSE(预测分数, 人工分数) + λ × 重建 MSE`。重建目标在每个 token 的
hidden 维做 LayerNorm；重建 MSE 先平均 hidden 维，再对每篇有效 token 求均值，
最后对 batch 求均值。池化和重建损失均屏蔽 padding。
decoder 只接收 latents 和正弦位置查询，不接收原始 token ID 或编码器特征。

默认冻结编码器并保持 eval 模式，其表示直接作为固定重建目标。
`--finetune-encoder` 可联合训练编码器；启用重建时会额外复制一份初始编码器，
作为冻结、eval、无梯度的教师，不随学生更新。这会增加显存占用。
λ=0 时不创建 decoder 和教师。推理仅执行学生编码器、聚合模块和评分头。

默认**保留全文、不截断**，每个 microbatch 只补齐到该 batch 最长作文的 token 数；
训练和评分共用分词与动态 padding 实现。例如两批最长分别为 730、280 tokens，
其输入长度分别为 730、280。batch 中的篇数仍由 `--batch-size` 控制。
不按整份数据的最长作文补齐，也不受 tokenizer 的默认 512-token 建议长度截断。

如需显式限制资源，可指定 `--max-length 512` 等长度上限，截断情况会写入报告。
保存的长度配置在评分时复用；旧的带上限检查点仍保留其原始截断行为。
本地 DeBERTa-v3 small/base 使用相对位置且不添加绝对位置嵌入，可执行超过 512
tokens 的输入；其他带固定绝对位置表的 DeBERTa 配置会在超限时明确报错。
本版没有全文分块或段落层次模型；超长 batch 的显存需求仍随注意力序列长度显著增加，
需要时减小 batch size，不会因显存不足而悄悄截断。执行长输入不等于已验证评分质量。

## 离线验证

在仓库根目录（包含 `deberta_baseline.py` 的目录）运行，依赖复用 `requirements-deberta.txt`：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ../.venv/bin/python -m essay_bottleneck.verify
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ../.venv/bin/python -m essay_bottleneck.verify_lengths
```

验证使用随机初始化的微型 DeBERTa 和临时 tokenizer，在 CPU 上检查：
梯度边界、教师不变性、padding 不变性、重建梯度、保存重载、推理跳过 decoder，
默认全文保留、各 batch 动态长度、超过 512-token 输入及显式截断兼容性，
以及三个配置的训练/评分入口和目录覆盖保护。CLI 短跑使用固定 train、selection、
calibration 各分档至多两篇作文；临时产物在 `artifacts/` 下，结束后自动清理。
这些检查不衡量评分质量，不改动已有测试文件。

## 后续训练

所有命令均从原始预训练模型重新初始化，使用本地 `models/deberta_base_snapshot.txt`。
本模块不下载权重。有效 batch 默认 32，编码器学习率 2e-5，新模块 1e-4。
输出放在已经被 Git 忽略的 `runs/` 中，现存目录会被拒绝覆盖。

```bash
# A：平均池化对照，冻结编码器
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ../.venv/bin/python -m essay_bottleneck.train \
  --pooling mean --reconstruction-weight 0 --output-dir runs/bottleneck_mean

# B：可学习压缩，仅评分损失，冻结编码器
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ../.venv/bin/python -m essay_bottleneck.train \
  --reconstruction-weight 0 --output-dir runs/bottleneck_score

# C：可学习压缩＋辅助重建，冻结编码器
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ../.venv/bin/python -m essay_bottleneck.train \
  --reconstruction-weight 0.1 --output-dir runs/bottleneck_reconstruct
```

每组都可加 `--finetune-encoder` 做第二轮对照，并使用新输出目录。
轮数默认 10，可用 `--epochs` 修改；`--model-size small` 可选择 small 权重。
`--smoke` 只跑一个极小 epoch 并跳过阈值拟合；CPU 验证可加 `--device cpu`。
平均池化对照使用相同形式的 MLP 评分头，但输入维度随表征改变，因此参数量不同；
也不同于旧基线的线性评分头，报告时不能把收益全部归因于池化方式。

`--normalize-queries` 在 cross-attention 前归一化学习到的查询，并将归一化查询
用于残差；默认关闭，以兼容旧检查点。`--group-microbatches` 在每次更新的有效
batch 内按长度排序，再切小批次，减少 padding；每次更新的样本集合和样本权重
不变，但 dropout 随机数对应及浮点累加次序会变化。它也默认关闭。

`--source` 可以使用 `export_deberta_encoder.py` 导出的监督训练编码器。
若存在 `supervised_source.json`，训练会校验权重哈希、数据和划分来源，并将其
纳入配置记录。这属于显式 warm start，不是从原始预训练模型开始的新实验。

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ../.venv/bin/python -m essay_bottleneck.train \
  --source models/deberta_base_2048_supervised --model-size base --max-length 2048 \
  --epochs 10 --normalize-queries --group-microbatches --reconstruction-weight 0.1 \
  --output-dir runs/rlt_warm_norm_v1
```

## 评估与产物

- 固定 train 更新权重；selection 按固定半整数阈值的 QWK 选择最佳 epoch，同分取较早轮。
- 最佳模型重载后，只在 calibration 上拟合五个严格递增阈值。
- 输出 selection/calibration 的连续分数、B0/B1 整数分、原始 token 长度、截断标记。
- 保存配置、代码哈希、数据/划分指纹、每轮分项损失、指标、最佳权重和 tokenizer。
- final_validation 不分词、不预测、不计算指标；不自动提交平台。
- selection 参与选模，calibration 参与拟合，二者指标均不代表独立测试表现。

`model/` 是部署检查点，保存学生、压缩模块、评分头及可选 decoder；不保存教师、
优化器和随机状态，**本版不支持中断续训**。校准阈值在运行目录的 `thresholds.json`。

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ../.venv/bin/python -m essay_bottleneck.score \
  --model-dir runs/bottleneck_reconstruct --file essay.txt

# GPU 上使用与训练相同的精度；批量 CSV 需要 essay_id、full_text
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ../.venv/bin/python -m essay_bottleneck.score \
  --model-dir runs/bottleneck_reconstruct --device cuda --csv essays.csv
```

批量输出含 `essay_id,raw_prediction,score`。如需比赛提交文件，应按要求仅保留
`essay_id,score`，并核对 ID。CPU FP32 与 GPU BF16 可能在阈值附近产生不同整数分。

参考：[RLT 官方介绍](https://www.pi.website/research/rlt)。辅助重建是否有益、
几枚 latent 是否足够，均需通过以上对照验证，不预先宣称优于平均池化。
