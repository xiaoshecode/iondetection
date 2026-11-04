# VLM 扩容与长训练记录

本轮已完成 **4096 个训练图块、6 轮、6144 个 optimizer steps** 的训练及完整 test24 对照。任务于 **2026-11-03 23:45:03 启动，2026-11-04 09:24:41 完成（香港时间，UTC+8）**，总历时 **9 小时 39 分 38 秒**。按验证集选中的权重来自第 5 轮（step 5120）。

同一组 24 个测试图块上，与 classical 伪标签的严格一致性 F1 从上一轮的 **0.449231 提高到 0.970917（1 px）**、从 **0.572308 提高到 0.988814（3 px）**；全部 24 个输出均正常结束，上一轮的 5 个截断输出消失。3 px 容差下仍有 3 个密集图块合计漏掉 5 个教师参考中心。这里报告的是**教师一致性，不是独立真值下的识别准确率**。

可直接查看 [上一轮 → 本轮逐图对照](../runs/vlm_expand_20261103/experiment_4096_6epochs/comparison_previous/index.html)，复用 [best_adapter](../runs/vlm_expand_20261103/experiment_4096_6epochs/training/best_adapter)，或查看[所选权重的完整测试报告](../runs/vlm_expand_20261103/experiment_4096_6epochs/training/after/index.html)。

运行目录是 `runs/vlm_expand_20261103/experiment_4096_6epochs`。初始阶段先在完整 32 个验证图块上生成预测并计算基线，再开始更新权重；模型选择只使用验证集，选定后才生成本轮 test24 结果。

- [运行状态与结果入口](../runs/vlm_expand_20261103/experiment_4096_6epochs/index.html)：运行时每 30 秒刷新，本轮已记录为 `complete`。
- [启动器状态与实际命令](../runs/vlm_expand_20261103/experiment_4096_6epochs/launcher_status.json)：包含进程号、阶段、参数和时间；进程号可能因恢复而变化。
- [训练进度 JSON](../runs/vlm_expand_20261103/experiment_4096_6epochs/training/progress.json)。
- [训练输出日志](../runs/vlm_expand_20261103/experiment_4096_6epochs/logs/training.stdout.log)与[错误／警告日志](../runs/vlm_expand_20261103/experiment_4096_6epochs/logs/training.stderr.log)。

## 数据扩容与固定对照

输入数据为 [dataset_4096/manifest.json](../runs/vlm_expand_20261103/dataset_4096/manifest.json)。扩容扫描了原 `data/train` 的 426 张原图，423 张能够提供通过现有教师规则的图块；其余 3 张未找到可靠晶体区域，没有将它们误当作可信空图标签。

| 项目 | 本轮数据 |
| --- | ---: |
| train／val／test 图块 | 4096／32／24 |
| train 原帧／采集组 | 423／32 |
| 相比上一轮新增覆盖的训练原帧 | 331 |
| train 每原帧图块数量 | 7–10 |
| 1–4／5–12／13–24 点训练图块 | 1366／1365／1365 |
| 数据目录大小 | 约 34.16 MiB |

上一轮 256 个 train 图块和 PNG 原样保留，原训练 JSONL 是新文件的精确前缀；本轮增加 3840 个唯一图块。val32 和 test24 的 JSONL 与 PNG **逐字节保持不变**，文件哈希记录在 `manifest.json` 的 `expansion.frozen_splits` 中。采集组、原图及图块哈希跨集合隔离检查均通过，训练数据不从 val/test 获取。

标签仍由相同 classical 教师生成：64×64 原像素图块、原帧固定灰度显示范围、3.5／4.0／4.5 三门限计数与中心稳定性、1.5 px 整块边界保护、相同 prompt 和归一化坐标格式。新增图块拒绝与同原帧已选图块 IoU 大于 0.9 的情况；旧 256 中已有的 5 对高重叠图块仍保持原样。普通重叠和相似帧仍然存在，4096 个图块不能解释成 4096 次独立实验；完整重叠统计在 manifest 中。

候选扫描缓存位于 `runs/vlm_expand_20261103/candidate_pool_48`，按原帧哈希和固定采样契约核对，可用于避免重复计算。[扩容脚本](../scripts/expand_vlm_training.py)与数据集内的 `expansion_generator_snapshot.py` 记录生成方法。原始图像和上一轮数据、模型、报告均未覆盖。

## 训练参数与选模规则

| 参数 | 本轮设置 |
| --- | --- |
| 基座 | 原始 `Qwen3-VL-2B-Instruct`，同一固定本地 revision，NF4 4-bit 加载 |
| 起点 | 从同一原始基座开始；上一轮 64-step adapter 保留作结果对照 |
| 可训练参数 | 语言侧 q/v attention LoRA，rank 4、alpha 8；视觉编码器与基座冻结 |
| 输入 | 64×64 RGB 图块，1024 个视觉 tokens |
| 优化 | AdamW，学习率 `1e-4`，microbatch 1、梯度累积 4 |
| 总训练量 | 6144 optimizer steps × 4 = 24576 个样本呈现，即 4096 图块的 6 轮 |
| 验证 | 初始及每 1024 steps，用完整 val32 计算答案 loss 和自由生成坐标 |
| 生成设置 | 确定性解码，`max_new_tokens=512`；不对重复／截断前缀补算成功 |
| 磁盘保留线 | 至少 8 GiB；启动时 F 盘可用约 25.05 GiB |

本轮 `--selection-metric f1_3px` 以 **完整 val32 的严格 3 px 教师一致性 F1** 选择 `best_adapter`，格式错误、截断和弃答全部计入。F1 相同时先选失败图块更少者，再选答案 loss 更低者；初始未训练 adapter 也参与比较。选模不使用 test24，不能将 loss 持续下降等同于定位和终止行为都已改善。

原始 2B 的历史 test24 结果已存在，因此启动器使用 `--skip-baseline-generation` 避免重复生成；验证已正常执行。训练结束后已生成所选 adapter 的完整 test24 结果，并自动产生两份比较：

- [comparison_base/index.html](../runs/vlm_expand_20261103/experiment_4096_6epochs/comparison_base/index.html)：原始 2B → 本轮所选 adapter。
- [comparison_previous/index.html](../runs/vlm_expand_20261103/experiment_4096_6epochs/comparison_previous/index.html)：上一轮 256 图块、64-step adapter → 本轮所选 adapter。

两份比较均已完成，退出码为 0。当前 test24 已多次查看，来自同一测试采集组的 8 张原始帧，应视为**固定历史回归对照**，不是新的盲测。最终指标仍是对 classical 伪标签的教师一致性；即使提升，也不能据此宣称独立真实准确率或优于 classical。

## 完整 24 图块结果

以下来自[本轮 summary.json](../runs/vlm_expand_20261103/experiment_4096_6epochs/training/after/summary.json)与[前后 comparison.json](../runs/vlm_expand_20261103/experiment_4096_6epochs/comparison_previous/comparison.json)。比较覆盖同样的全部 24 个图块、226 个教师参考中心；图块 ID、图像哈希、裁剪框和完整参考坐标已在比较阶段核对一致。主表采用包含格式错误、截断和弃答的严格统计，没有剔除失败样本。

| 指标 | 上一轮：256 图块、1 轮 | 本轮：4096 图块、完成 6 轮后选第 5 轮 |
| --- | ---: | ---: |
| 有效输出 | 19/24 | 24/24 |
| malformed／截断／弃答 | 0／5／0 | 0／0／0 |
| 1 px 教师一致性 micro F1 | 0.449231 | 0.970917 |
| 1 px 匹配 TP／FP／FN | 73／26／153 | 217／4／9 |
| 1 px 整块全对 | 5/24 | 20/24 |
| 3 px 教师一致性 micro F1 | 0.572308 | 0.988814 |
| 3 px 匹配 TP／FP／FN | 93／6／133 | 221／0／5 |
| 3 px 整块全对 | 9/24 | 21/24 |

按教师点数分层，每档均包含 8 个图块：

| 教师点数 | 1 px F1 上一轮 → 本轮 | 3 px F1 上一轮 → 本轮 | 本轮有效输出 | 本轮 3 px 整块全对 |
| --- | ---: | ---: | ---: | ---: |
| 1–4 | 0.685714 → 1.000000 | 0.857143 → 1.000000 | 8/8 | 8/8 |
| 5–12 | 0.731183 → 0.978261 | 0.946237 → 1.000000 | 8/8 | 8/8 |
| 13–24 | 0.274112 → 0.965517 | 0.345178 → 0.984326 | 8/8 | 5/8 |

原先密集图块的重复续写和截断在这组回归样本中已消失，密集组的 3 px 教师一致性 F1 从 0.345178 提高到 0.984326。残余 3 px 漏点全部来自下列 3 个密集图块；其余 21 个图块的全部预测和教师中心均完成一对一匹配：

| 图块 ID | 教师点数 | 预测点数 | 3 px TP／FP／FN |
| --- | ---: | ---: | ---: |
| `test_4dbaaccf74f007b9` | 19 | 18 | 18／0／1 |
| `test_f0605adb0d42dff6` | 24 | 21 | 21／0／3 |
| `test_fa846571814bb864` | 21 | 20 | 20／0／1 |

1 px 容差下另外有中心偏差超过阈值的点，因此 1 px 的 FP/FN 与 3 px 不同。较宽容差下的无多余预测，不代表所有中心都具有同样的亚像素精度，也不证明教师没有漏标。

## 逐轮验证与所选权重

以下读取自 [validation_history.json](../runs/vlm_expand_20261103/experiment_4096_6epochs/training/validation_history.json)。所有轮次使用相同的完整 val32，共 253 个教师参考中心；没有用 test24 选择轮次。

| 轮次 | Step | 验证答案 loss | 1 px 严格 F1 | 3 px 严格 F1 | 失败图块 | 3 px 整块全对 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0，初始基座 | 0 | 1.987768 | 0.004535 | 0.068027 | 10/32 | 0/32 |
| 1 | 1024 | 0.498775 | 0.976096 | 0.984064 | 0/32 | 28/32 |
| 2 | 2048 | 0.480575 | 0.988142 | 0.996047 | 0/32 | 30/32 |
| 3 | 3072 | 0.440587 | 0.998028 | 0.998028 | 0/32 | 31/32 |
| 4 | 4096 | 0.448164 | 0.998028 | 0.998028 | 0/32 | 31/32 |
| 5，所选权重 | 5120 | 0.439271 | 1.000000 | 1.000000 | 0/32 | 32/32 |
| 6 | 6144 | 0.447335 | 1.000000 | 1.000000 | 0/32 | 32/32 |

第 5、6 轮的验证集严格 F1 与失败数相同；按预先设定的第三排序项，第 5 轮的答案 loss 更低，因此 `best_adapter` 保存 step 5120，`last_adapter` 保存 step 6144。本轮测试报告使用 `best_adapter`，不是最后一步权重。验证集上的满分只说明匹配了这 32 个图块的教师中心，不能当作新采集条件下的性能保证。上一轮仅用 val 的 8 个图块计算 loss，本轮使用全部 32 个，两个实验的验证 loss 不宜直接作同口径比较。

## 实际运行与解释边界

训练摘要见 [training_summary.json](../runs/vlm_expand_20261103/experiment_4096_6epochs/training/training_summary.json)，完整任务时间见 [launcher_status.json](../runs/vlm_expand_20261103/experiment_4096_6epochs/launcher_status.json)。

| 项目 | 实测记录 |
| --- | --- |
| 总任务时长 | 9 小时 39 分 38 秒，含初始验证与自动比较 |
| 训练摘要计时 | 34402.985 秒，约 9 小时 33 分钟；计时起点在初始验证之后，含训练后评估 |
| 训练总量 | 6144 次 optimizer 更新，24576 个样本呈现，6 轮 |
| 所选权重 | 第 5 轮，step 5120 |
| 显存分配／缓存预留峰值 | 2.546／4.082 GiB |
| 结束后 GPU 占用 | 2026-11-04 09:35 香港时间检查为 0 MiB，训练进程已退出 |
| F 盘余量 | 约 24.61 GiB，高于 8 GiB 保留线 |

本轮结果支持“扩大原图覆盖并延长训练后，在固定历史回归图块上，教师坐标一致性和正常终止行为明显改善”。同时改变了训练图块数、原帧覆盖、轮数和验证选模方式，不能把提升单独归因于其中一项。

数据主要来自同日相邻采集，test24 仅包含 1 个采集组的 8 张原帧，多个图块和原帧之间仍有关联。classical 标签仍可能漏掉弱点或误分相邻点，且筛选偏向教师稳定的非空图块。本轮尚不能验证跨天、跨晶体、空图及完整大晶格的独立识别准确率；这些需要新的采集组和独立复核标签。当前测试集已用于历史比较，后续调整不能继续把它称作未见盲测集。

## 保存与中断恢复

本轮已每 128 steps 保存一次完整恢复状态，保留最新 3 份，指针为 `training/recovery/latest.json`；恢复状态包含 adapter、优化器状态、数据遍历位置、随机状态及验证／选模历史。每 1024 steps 另存一份 epoch adapter 到 `training/checkpoints/step_XXXXXX`，独立保留各轮权重。普通 adapter 用于推理，**完整恢复应使用 recovery 指针或对应恢复检查点**。本轮未中断，已正常完成；下列模板保留作后续中断任务的操作参考。

恢复必须使用新的输出目录，保留原始失败或中断记录；`--steps 6144` 表示训练总目标步数，不是再追加 6144 步。启动器不会无限自动重试。以下是恢复模板，**仅在确认原任务已停止后使用**，不要与现有 GPU 任务同时启动：

```powershell
Set-Location 'F:\CQI\Bell Camera\Net Recog'
& '.\.venv-vlm-train\Scripts\python.exe' scripts/run_vlm_long_training.py `
  --dataset runs/vlm_expand_20261103/dataset_4096 `
  --output runs/vlm_expand_20261103/experiment_4096_6epochs_resume_01 `
  --resume-from runs/vlm_expand_20261103/experiment_4096_6epochs/training/recovery/latest.json `
  --steps 6144 --accumulate 4 --rank 4 --lr 0.0001 `
  --image-tokens 1024 --max-new-tokens 512 --seed 20261103 `
  --eval-every 1024 --eval-samples 32 --selection-metric f1_3px `
  --save-every 1024 --recovery-every 128 --reserve-gib 8
```

使用同一数据集、基座和兼容的训练设置；恢复时会核对数据、配置和检查点完整性。上例启动器也会在续训完成后执行历史对照比较。

## 已完成的运行前验证

- 代码回归共 **245 项测试通过**；这些是实现验证，不是识别准确率。
- CPU helper 的中断恢复等效性已验证。
- 完整 GPU 模型完成了“训练 2 steps → 保存 → 新进程恢复到总 step 4”的实际检查，恢复后 optimizer 步数为 4；记录在 `runs/vlm_expand_20261103/smoke_resume_source` 与 [smoke_resume_continued/training_summary.json](../runs/vlm_expand_20261103/smoke_resume_continued/training_summary.json)。
- 该检查证明恢复链路能够继续更新，不声称 GPU 跨进程与不中断运行逐位一致。

本轮启动器在运行期间临时阻止系统因空闲而自动睡眠，结束时已释放，没有修改永久电源设置。后续长训练期间仍需接电、保持开机，并避免合盖触发系统睡眠或手动关机；临时防空闲睡眠不能阻止这些操作。原 `face` 环境、显卡驱动及上一轮权重继续保留，依赖与硬件记录见 [VLM_TRAINING_ENV.md](VLM_TRAINING_ENV.md)。

## 2026-11-04：3 张密集原图推理与代码打包

已用第 5 轮的 `best_adapter` 完成 test、val、train 各 1 张密集原图，共 12 个图块；香港时间 **09:53:29–09:58:29，历时 5 分钟**，其中生成耗时 260.915 秒。12 次生成均以 `stop` 正常结束，没有截断、弃答或失败。状态及设置见 [status.json](../runs/vlm_dense_originals_20261104/status.json)、[plan.json](../runs/vlm_dense_originals_20261104/plan.json)，交互结果见[原图报告](../runs/vlm_dense_originals_20261104/index.html)，逐图指标见 [report_summary.json](../runs/vlm_dense_originals_20261104/report_summary.json)。

| 原图 | classical 点数 | VLM 点数 | 1 px F1 | 3 px F1 | 3 px TP／FP／FN | 原图输出 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| test `1ef622ec4cdaa22a` | 105 | 95 | 0.890000 | 0.950000 | 95／0／10 | [坐标 CSV](../runs/vlm_dense_originals_20261104/images/test_1ef622ec4cdaa22a/vlm_coordinates.csv) · [整帧叠图](../runs/vlm_dense_originals_20261104/images/test_1ef622ec4cdaa22a/vlm_overlay.png) |
| val `42bd8e322ed12078` | 110 | 91 | 0.865672 | 0.895522 | 90／1／20 | [坐标 CSV](../runs/vlm_dense_originals_20261104/images/val_42bd8e322ed12078/vlm_coordinates.csv) · [整帧叠图](../runs/vlm_dense_originals_20261104/images/val_42bd8e322ed12078/vlm_overlay.png) |
| train `182e332072c65732` | 140 | 115 | 0.768627 | 0.862745 | 110／5／30 | [坐标 CSV](../runs/vlm_dense_originals_20261104/images/train_182e332072c65732/vlm_coordinates.csv) · [整帧叠图](../runs/vlm_dense_originals_20261104/images/train_182e332072c65732/vlm_overlay.png) |

这仍是 **classical 教师一致性**；train 原图属于训练分布内，不能与 test 合并宣称独立测试准确率。训练图块最多 24 个教师点，本次 test 原图的单个图块可达 61 个（val、train 分别可达 63、72 个），存在明显的密度变化。漏点仍然较多，但本次不是因生成截断造成；当前证据不能把全部漏点归因于密度。

推理保持 **64×64 原始像素图块、16 px 重叠**，已知 512×512 相机先由原图信号定位晶体包络，再向外扩 8 px；这 3 张图只覆盖声明的 ROI，整帧叠图不表示 ROI 外也做了推理。不同尺寸的旧相机原图按整帧平铺处理。重叠预测以 2 px 距离合并，最终坐标使用原图的零起始像素中心，`x` 为列、`y` 为行；classical 坐标仅用于选密集样本和事后比较，不输入 VLM，也不用于替换或筛掉 VLM 点。

以下命令复现已完成的选择集；使用新输出目录，原结果继续保留：

```powershell
Set-Location 'F:\CQI\Bell Camera\Net Recog'
& '.\.venv-vlm-train\Scripts\python.exe' -m ionrecog.vision_language.fullframe `
  --data data --config configs/existing_crystal.yaml `
  --model checkpoints/Qwen3-VL-2B-Instruct-bf16 `
  --adapter runs/vlm_expand_20261103/experiment_4096_6epochs/training/best_adapter `
  --input-manifest runs/vlm_dense_originals_20261104_selection.json `
  --output runs/vlm_dense_originals_20261104_replay `
  --max-new-tokens 1024 --reserve-gib 8
```

[选择清单](../runs/vlm_dense_originals_20261104_selection.json) 的必需格式是 `{"paths":["F:/绝对路径/原图.png", "F:/绝对路径/另一张原图.npy"]}`，可附加选图原因等说明；路径需指向 `--data` 下的原图。

**全部 944 张原图尚未启动推理。** 去掉选择清单即可遍历当前 `data` 中 test、val、train、diagnostic 的全部原图；下列仅为后续操作命令，不代表已有全量结果：

```powershell
& '.\.venv-vlm-train\Scripts\python.exe' -m ionrecog.vision_language.fullframe `
  --data data --config configs/existing_crystal.yaml `
  --model checkpoints/Qwen3-VL-2B-Instruct-bf16 `
  --adapter runs/vlm_expand_20261103/experiment_4096_6epochs/training/best_adapter `
  --output runs/vlm_all_originals_20261104 `
  --max-new-tokens 1024 --reserve-gib 8
```

启动、准备计划及逐图／逐块处理均检查磁盘，保留 8 GiB；本次结束余量为 24.606 GiB。中断后用同一输出目录、原命令参数加 `--resume`；代码会核对保存的计划、模型、配置、选择清单及代码指纹，并复用已保存图块响应。恢复时必须保留原 `--max-new-tokens 1024` 等设置，不能退回默认值或把选择集改成全量。代码或设置变化应另开输出目录。

本次打包前完整代码回归为 **335 passed，11.86 秒**，有 10 条已有的 `torch.load` FutureWarning；前文的 245 项是长训练启动前的历史检查。Git 仅打包代码、测试与文档；原图、模型权重以及 `runs/` 下的响应、坐标 CSV、叠图和报告均保留在本机，不纳入 Git。
