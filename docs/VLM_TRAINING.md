# VLM 伪标签微调实验记录

记录日期：2026-11-03。已完成模型权重校验、数据准备、独立 CUDA 环境验证、完整模型的两步训练测试、**64 步正式训练及完整 24 图块的基座／adapter 对照**。主要结果以全部 24 图的严格统计为准：3 px 教师一致性 F1 从 0.097826 提高到 0.572308，有效输出从 16/24 提高到 19/24；密集图块仍有明显截断和漏点问题。

可直接查看 [24 图前后并排比较](../runs/vlm_train_20261103/comparison_test24/index.html)，复用 [best_adapter](../runs/vlm_train_20261103/qlora_2b_r4_64steps/best_adapter)，或核对 [训练环境记录](VLM_TRAINING_ENV.md)。

本轮问题是：在本机 6 GB 显存条件下，能否让 VLM 学会输出离子亮斑中心，并提高与 classical 检测结果的一致性。监督来自 classical 伪标签，没有独立人工真值，因此本轮属于教师蒸馏探索。

## 资源、环境和模型

| 项目 | 已知条件或本轮配置 |
| --- | --- |
| GPU | NVIDIA RTX 3060 Laptop GPU，6144 MiB 显存 |
| 已有 `face` 环境 | Python 3.10.13，PyTorch 1.13.1+cu116，可使用 CUDA |
| 新训练环境 | 项目内 `.venv-vlm-train`，Python 3.12.14、PyTorch 2.5.1+cu121；CUDA NF4 BF16 与高效 SDPA 反传均已实测通过 |
| 基座 | `Qwen/Qwen3-VL-2B-Instruct`，原始 BF16 safetensors |
| 固定 revision | `89644892e4d85e24eaac8bacfd4f463576704203` |
| 本地模型目录 | `checkpoints/Qwen3-VL-2B-Instruct-bf16` |
| 权重文件 | `model.safetensors`，4,255,140,312 bytes |
| 下载审计 | 模型目录的 `source_provenance.json`，记录来源、版本、文件字节数及 SHA256；下载阶段已完成校验 |

新环境用于隔离 Qwen3-VL 所需的新版本 Transformers、PEFT、bitsandbytes 等依赖。已有 CUDA 环境并非不可用；它的旧版 PyTorch 不作为本次新模型训练依赖环境。原 `.venv`、`face` 环境和已有 classical／masked 代码继续保留。原先用于零样本推理的 4B GGUF 及 llama.cpp 服务也单独保留。

本轮从 2B 原始权重加载，运行时使用 NF4 4 位量化及 double quantization，计算类型为 BF16。只在语言模型 self-attention 的 `q_proj`、`v_proj` 上训练 LoRA：`r=4`、`alpha=8`、`dropout=0`。视觉编码器、基座权重及其他模块冻结；视觉模块和 `lm_head` 不做 NF4 量化。保存的是小型 adapter，不导出完整合并权重。

代码启用梯度检查点和 SDPA。Windows 下若没有 FlashAttention，则使用高效 SDPA 后端，并在本进程内展开 GQA 的 KV，避免默默退回显存开销更大的 math attention。两步完整模型训练已通过；64 步正式实验的 PyTorch 分配显存峰值为 2.546 GiB，缓存预留峰值为 4.082 GiB。这是当前进程的分配器统计，不是整机 GPU 总占用。完整环境记录见 [VLM_TRAINING_ENV.md](VLM_TRAINING_ENV.md)。

## 数据 v2

正式输入是 `runs/vlm_train_20261103/dataset_v2`；`dataset` 为保留的 v1 审计版本。原始采集图像未改动。

已有 `runs/crystal_region_v1/classical` 的 250 帧完整结果来自原始 test 组，不能直接转作训练集。数据准备脚本在原 `data/train`、`data/val` 和保留的 test 帧上重新运行相同教师流程，各 split 的采集组、原图 SHA256 均不交叉。

教师使用 `configs/existing_crystal.yaml`，先在原图上 `segment_crystal`，再以原图作为 `noise_reference`，在晶体掩膜内 `detect_ions`。只有检测到晶体且未触及搜索区域边界的原帧才产生候选图块。train 每采集组均匀选择至多 3 帧，val/test 各均匀选 8 帧；不同相机尺寸的旧 NPY 图像不混入此次 512×512 相机实验。

从原帧随机裁取 64×64 图块，保留 1–24 个亮斑中心。教师分别使用 `threshold_sigma=3.5/4.0/4.5`：三者必须在整个图块内点数一致，完整一对一匹配距离不超过 1 px。边缘内外 1.5 px 保护范围内出现候选中心时舍弃整块；不逐点删除不稳定候选，以免产生假负标签。`no_crystal` 和零检测图块不作为可信空图标签。

v1 使用 3 px 边缘保护，抽查发现它过度偏向晶格外缘的 1–2 点图块。因此 v2 在训练之前固定为 1.5 px 整块保护，并按 1–4／5–12／13–24 点三个桶均衡抽样，同时均衡采集组和原帧。val/test 只按这些预定教师稳定性规则选择，不根据模型预测或训练结果挑选。

| Split | 图块数 | 原帧数 | 采集组数 | 1–4 点 | 5–12 点 | 13–24 点 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 256 | 92 | 32 | 86 | 85 | 85 |
| val | 32 | 8 | 1 | 11 | 11 | 10 |
| test | 24 | 8 | 1 | 8 | 8 | 8 |

保存图像是 64×64 RGB PNG。同一原帧的所有图块固定使用原帧的 0.5／99.9 百分位灰度范围，避免逐块改变显示对比度。`contact_sheet.png` 已目视抽检，包含晶格内部、中等密度和密集图块。v2 总大小约 3 MB。

每个 split 的 JSONL 记录图像相对路径及 SHA256、原图绝对路径及 SHA256、采集组、裁剪框、局部浮点坐标、统一 prompt 和 JSON 文本 `response`。所有行均标为 `label_source=classical_pseudo`、`label_is_ground_truth=false`。响应格式为 `{"points":[[x,y],...],"uncertain":false}`，点按 y、x 顺序排列；归一化 0 和 1000 对应第一与最后像素中心，即 64 px 图块以 63 为分母。

`manifest.json` 保存教师配置、代码哈希、选择规则、计数、独立性检查和局限；`frame_audit.json` 保存原帧检测／筛选记录；`generator_snapshot.py` 固定生成代码。数据模块的 9 项测试已通过，覆盖边缘排除、教师一致性、坐标、组／哈希隔离、均衡抽样和输出／磁盘保护。

如需重新生成，使用新目录，不能覆盖已经冻结的 v2：

```powershell
Set-Location 'F:\CQI\Bell Camera\Net Recog'
& '.\.venv\Scripts\python.exe' scripts/prepare_vlm_training.py --output runs/vlm_train_20261103/dataset_v2_rebuild --edge-margin 1.5 --reserve-gib 8
```

## 训练运行与复现

先检查 GPU 占用和磁盘空间；避免本次 llama.cpp 服务与训练同时占用显存。若需停止服务，只操作本次服务的已确认 PID。训练程序启动时和每个 optimizer step 检查磁盘，要求保留至少 8 GiB；输出目录必须为空，历史结果不会覆盖。训练从本地加载模型，并关闭 Hugging Face 在线请求。

在项目根目录运行。以下 smoke 执行 2 次 optimizer step，核验模型加载、梯度、adapter 保存和验证损失，不生成前后预测，不能据此声称识别效果提升。该命令已成功完成；复现时必须使用新输出目录：

```powershell
Set-Location 'F:\CQI\Bell Camera\Net Recog'
& '.\.venv-vlm-train\Scripts\python.exe' -m ionrecog.vision_language.finetune `
  --dataset runs/vlm_train_20261103/dataset_v2 `
  --model checkpoints/Qwen3-VL-2B-Instruct-bf16 `
  --output runs/vlm_train_20261103/smoke_2b_r4 `
  --steps 2 --accumulate 1 --rank 4 --lr 0.0001 `
  --image-tokens 1024 --eval-every 1 --eval-samples 1 `
  --reserve-gib 8 --skip-generation
```

smoke 成功后，已从基座重新开始并完成以下 64 step 正式试验，没有延续 smoke adapter。复现时须将输出目录改为新的空目录：

```powershell
& '.\.venv-vlm-train\Scripts\python.exe' -m ionrecog.vision_language.finetune `
  --dataset runs/vlm_train_20261103/dataset_v2 `
  --model checkpoints/Qwen3-VL-2B-Instruct-bf16 `
  --output runs/vlm_train_20261103/qlora_2b_r4_64steps `
  --steps 64 --accumulate 4 --rank 4 --lr 0.0001 `
  --image-tokens 1024 --eval-every 16 --eval-samples 8 `
  --max-new-tokens 512 --seed 20261103 --reserve-gib 8
```

microbatch 为 1，梯度累积 4 次，总计呈现 256 个训练样本，相当于本轮训练集一次打乱遍历。优化器为 AdamW，weight decay 0.01，梯度范数裁剪到 1；监督仅计算 assistant 答案 token 的交叉熵。64×64 原图块经 processor 缩放成 1024 个视觉 token，缩放不会产生新的真实图像细节。

`--eval-samples 8` 表示从 val 按三档密度固定选取 3/3/2 个图块，计算初始及每 16 step 的验证损失，以最低验证损失保存 `best_adapter`；不表示用全部 32 个 val 图块选模型。使用密度分层是为了避免均匀下标与数据轮转排序发生周期性重合而漏掉密集图块。初始未训练 adapter 也保留为候选，若验证损失没有降低，best step 可以是 0。`last_adapter` 独立保存最后一步。

正式脚本已在同一组 8 个 test 图块上生成基座和所选 adapter 的前后结果，采用确定性解码。test 结果不参与 best adapter 选择。完整 24 图块对照使用下面的独立评估命令。

## adapter 复用与完整对照

以下命令已完成，复用已保存 adapter，不执行训练；再次运行须更换输出目录。两个命令使用同一基座、同一数据、同一 token 设置和解码上限。省略 `--adapter` 时是同样 NF4 加载方式的原始 2B 基座；它是本轮适当的微调前对照。以前的 4B GGUF 零样本结果具有模型规模、运行时和图块差异，不能直接用作唯一的微调前对照。

```powershell
& '.\.venv-vlm-train\Scripts\python.exe' -m ionrecog.vision_language.evaluate_adapter `
  --model checkpoints/Qwen3-VL-2B-Instruct-bf16 `
  --dataset runs/vlm_train_20261103/dataset_v2 --split test --limit 0 `
  --output runs/vlm_train_20261103/base_test24 `
  --image-tokens 1024 --max-new-tokens 512 --reserve-gib 8

& '.\.venv-vlm-train\Scripts\python.exe' -m ionrecog.vision_language.evaluate_adapter `
  --model checkpoints/Qwen3-VL-2B-Instruct-bf16 `
  --adapter runs/vlm_train_20261103/qlora_2b_r4_64steps/best_adapter `
  --dataset runs/vlm_train_20261103/dataset_v2 --split test --limit 0 `
  --output runs/vlm_train_20261103/adapter_test24 `
  --image-tokens 1024 --max-new-tokens 512 --reserve-gib 8
```

`--limit 0` 评估所选 split 的全部图块。可将 `--split test` 改为 `--split val`，并指定新的输出目录查看全部验证图块。评估目录必须尚不存在。

训练目录包含 `run_config.json`、`progress.json`、`trainable_parameters.json`、`training_history.json`、`generation_evaluation_ids.json`、`training_summary.json`、`best_adapter`、`last_adapter` 和 processor。异常时记录 `interrupted.json`；单有文件或 adapter 不代表完整训练成功。

独立评估目录保存 `evaluation_config.json`、逐图原始回复及 `report/index.html`、`report/summary.json`、CSV 和叠图。报告区分全部样本严格统计与仅有效输出统计；应主要报告严格统计，避免忽略 JSON 错误、截断、拒答等失败样本。指标是给定像素容差下的坐标匹配、点数及整块一致性；叠图青色圆为教师中心，橙色叉为模型输出。

## 64 步训练记录

训练记录位于 `runs/vlm_train_20261103/qlora_2b_r4_64steps`。`training_summary.json` 的状态为 `complete`；以下数值来自该文件、`training_history.json`、`trainable_parameters.json` 及实际 adapter 文件。

| 项目 | 实测结果 |
| --- | --- |
| optimizer steps／呈现样本数 | 64／256 |
| 可训练参数量 | 802,816，仅语言 attention 的 q/v LoRA |
| 初始 → 最优验证答案损失 | 1.997779 → 0.762078，固定 8 个 val 图块 |
| best step | 64，所选 adapter 与最后一步相同 |
| adapter 相对初始化的参数变化 L2 | 2.531051，确认参数发生更新 |
| PyTorch 显存峰值 allocated／reserved | 2.546／4.082 GiB |
| 优化主体计时 | 约 371.3 s，取 history 的 step 64 `elapsed_seconds` |
| 含训练后评估的计时 | 约 444.7 s，取 summary 的 `training_seconds`；不包含模型加载、训练前生成和初始验证 |
| 磁盘空余（正式训练前 → 后） | 25.591 → 25.579 GiB，均高于 8 GiB 保留线 |
| `best_adapter/adapter_model.safetensors` | 3,227,648 bytes，约 3.1 MiB |
| 代码回归验证 | 225 项测试通过；这是代码验证，不代表识别准确率 |

完整评估重新加载了磁盘上的 `best_adapter`。与训练后初评重叠的 8 个图块，其生成 `text` 和 `finish_reason` 均完全一致，检查记录见 [adapter_reload_verification.json](../runs/vlm_train_20261103/adapter_reload_verification.json)。这验证了此次保存／重载结果的一致性，不代表其他硬件或依赖版本也能逐字复现。

训练与评估进程已结束，随后 `nvidia-smi` 显示 GPU 总占用回到 152 MiB；F 盘可用空间约 25.57 GiB，仍高于 8 GiB 保留线。该读数是结束后的机器快照，训练峰值以上表的进程分配器记录为准。

## 完整 24 图对照：本轮主要结果

数据来自 `base_test24/report/summary.json`、`adapter_test24/report/summary.json` 和 [comparison_test24/comparison.json](../runs/vlm_train_20261103/comparison_test24/comparison.json)。比较脚本已核对图块 ID、实际图像 SHA256、裁剪框、参考点数和完整参考坐标一致。两次评估均覆盖全部 24 个 test 图块、226 个教师参考中心，没有按模型表现移除图块。下表使用包含失败输出的严格统计。

| 指标 | 原始 2B（同样 NF4 加载） | step 64 adapter |
| --- | ---: | ---: |
| 有效输出图块 | 16/24 | 19/24 |
| malformed／截断／拒答 | 6／2／0 | 0／5／0 |
| 1 px 教师一致性 micro F1 | 0.027174 | 0.449231 |
| 1 px 匹配 TP／FP／FN | 5／137／221 | 73／26／153 |
| 1 px 整块全对 | 0/24 | 5/24 |
| 3 px 教师一致性 micro F1 | 0.097826 | 0.572308 |
| 3 px 匹配 TP／FP／FN | 18／124／208 | 93／6／133 |
| 3 px 整块全对 | 0/24 | 9/24（37.5%） |

按教师点数分层，每档 8 个图块；桶内同样保留全部格式错误、截断和拒答，F1 按桶内 TP／FP／FN 汇总计算：

| 教师点数 | 1 px F1 前 → 后 | 3 px F1 前 → 后 | 有效输出 前 → 后 | 训练后 3 px 整块全对 |
| --- | ---: | ---: | ---: | ---: |
| 1–4 | 0.0000 → 0.6857 | 0.0000 → 0.8571 | 5/8 → 8/8 | 5/8 |
| 5–12 | 0.0621 → 0.7312 | 0.1366 → 0.9462 | 7/8 → 8/8 | 4/8 |
| 13–24 | 0.0000 → 0.2741 | 0.0741 → 0.3452 | 4/8 → 3/8 | 0/8 |

小规模微调改善了当前保留集的格式遵循和教师坐标一致性，稀疏及中等密度图块表现较好；密集图块仍是主要问题。训练后的 5 个截断输出全部来自 13–24 点档，且该档 3 px 严格统计仍有 128 个未匹配教师中心，没有一个图块全部匹配。总体 3 px 的 133 个 FN 中，110 个来自截断图块被严格计为无预测，另外 23 个来自有效输出中的未匹配参考中心。不能只保留 19 个有效输出后报告较高的条件 F1，也不能将 9/24 整块全对解释为系统已经能够可靠识别完整离子晶格。

左右叠图、逐图状态及原始回复入口集中在 [完整 24 图比较页](../runs/vlm_train_20261103/comparison_test24/index.html)。这些结果仍然是对 classical 伪标签的教师一致性，不是真实识别准确率。

## 8 图初评：保留的过程记录

以下仅保留初评记录；本轮主要结论使用上面的完整 24 图对照。初评来源为本次训练目录的 `before/summary.json`、`after/summary.json`。两者使用同一组 8 个保留 test 图块，共 61 个教师参考中心。下表使用 **全部 8 图的严格统计**，包含输出失败样本，不能与“仅有效输出”的条件统计混用。

| 指标 | 原始 2B（同样 NF4 加载） | step 64 adapter |
| --- | ---: | ---: |
| 有效输出图块 | 3/8 | 7/8 |
| malformed／截断／拒答 | 4／1／0 | 0／1／0 |
| 1 px 教师一致性 micro F1 | 0.0331 | 0.5625 |
| 1 px 匹配 TP／FP／FN | 2／58／59 | 27／8／34 |
| 1 px 整块全对 | 0/8 | 3/8 |
| 3 px 教师一致性 micro F1 | 0.0661 | 0.7083 |
| 3 px 匹配 TP／FP／FN | 4／56／57 | 34／1／27 |
| 3 px 整块全对 | 0/8 | 5/8 |

这组初评支持“小规模微调改善了指定 8 图的格式遵循和教师坐标一致性”。训练后仍有 1 图输出截断，严格 3 px 统计仍有 27 个未匹配教师中心，不能表述为已经可靠识别所有离子。8 图结果属于初评，不能替代全部 24 图块的最终对照。

## 解释边界与后续方向

教师稳定性只能减少阈值敏感样本，不能证明标签正确。弱亮斑漏检、相邻点分离错误、背景误检仍可能被模型学到。1.5 px 边缘保护避免中心贴边，但仍可能保留被截断的 PSF 尾部；后续需要人工核查边缘案例。

train/val/test 的采集组不重叠，但数据主要来自同日相邻采集，不能据此证明跨天、跨晶体或跨相机泛化。test 图像过去已在诊断中查看，因此只是未用于优化的保留集，不是论文意义上的盲测。多个图块共享原帧，也不能视为完全独立的实验样本。清晰、教师稳定、非空的筛选会偏向容易样本；本轮不验证空图识别、完整原帧覆盖、暗态标签或量子态读出保真度。

本轮训练、保存／重载检查及完整 test24 对照均已完成。后续研究应优先核查密集图块的截断、漏点和边缘案例，并引入独立标注；不根据本次 test 表现重新筛选 test 图块。若用这些结果调整 prompt、解码上限、训练或数据方案，需要另设新的独立评估，不能继续把当前 test 当作未见数据。

即使教师一致性提高，结论仍应表述为“模型更好地拟合了当前 classical 伪标签”。确认真实识别改进需要独立人工标注、覆盖困难与空图的测试集，以及不同采集条件的验证。
