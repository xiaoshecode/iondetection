# 离子图像 Diffusion：服务器训练、推理、评测与 DiT 特征实验

本目录提供可执行的两条路线：

1. **像素空间条件 diffusion**：物理近似合成配对数据或真实稳定重复帧 → DDPM 噪声预测训练 → DDIM 恢复 → 现有点源定位器 → 独立真值评测。
2. **冻结 FLUX.1-Kontext-dev 特征探测**：提取编辑 Transformer 中间空间 token → 保存特征与原图坐标映射 → 离线余弦匹配/一对一匹配。

这是前述[可行性评估](<F:/CQI/Bell Camera/Net Recog/docs/2026-11-04-diffusion-ion-localization-feasibility.md>)中的先行实验实现。主线训练的是专用小 U-Net；FLUX 路线冻结全部权重，不含大模型 LoRA/VAE 联合重训。定位器复用传统多峰与质心方法；`fused` 是原图与恢复图的强度加权融合，**不是已训练的联合定位头**。后续可与项目 `point_localization`、`psf_fitting` 分支比较。

CPU 端已验证数据泄漏检查、真实优化步骤、断点续训一致性、两进程 gloo、DDIM 推理和评测。FLUX 接口依据官方 diffusers 0.35.1 编写，完成 CPU 模拟模型测试；**尚未运行真实 FLUX 权重，也未验证服务器 CUDA/NCCL/AMP 吞吐或真实离子准确率**。微运动去模糊是否有效，需要校准模拟器和真实参考数据检验。

## 1. 目录与运行位置

```text
ionrecog/diffusion/
  __main__.py                 命令行入口
  config.py                   严格配置与固定强度映射
  data.py                     JSONL 数据清单、分组/内容泄漏检查、配对采样
  synthetic.py                运动积分、PSF、Poisson/读噪/可选电子倍增模拟
  model.py                    条件 SmallUNet 与扩散计划
  engine.py                   单卡/DDP训练、EMA、AMP、checkpoint、DDIM恢复
  pipeline.py                 批量标圈、CSV坐标、恢复误差与点匹配评测
  features.py                 冻结 FLUX 特征与离线相似度
  configs/smoke.yaml          CPU 流程检查
  configs/server.yaml         服务器起步配置
  requirements-server.txt    基础依赖版本
  requirements-features.txt  可选 FLUX 依赖版本
  scripts/smoke.sh            一键合成→训练→评测
  scripts/train.slurm         单节点 Slurm 示例
```

**将整个 `Net Recog` 项目复制/克隆到服务器，在项目根目录执行命令。**本目录复用了 `ionrecog/data.py`、`classical/` 和 `self_supervised/models.py`，不能只上传 diffusion 文件夹。数据和权重不在 Git 中，需要另行传输；建议上传包含相对路径清单的数据目录。

入口是 `python -m ionrecog.diffusion`。原来的 `python -m ionrecog train --method diffusion` 仍是旧版基线，两个入口及 checkpoint 格式不同，不能互换。

## 2. 服务器环境

建议 Linux、Python 3.11。以下是一个明确的 PyTorch 2.5.1/CUDA 12.4 环境示例，具体 GPU/驱动应选择对应的[官方 PyTorch 安装组合](https://pytorch.org/get-started/previous-versions/)；较新架构可能需要更高版本的 CUDA/PyTorch。

```bash
cd /path/to/NetRecog
python3.11 -m venv .venv-diffusion
source .venv-diffusion/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r ionrecog/diffusion/requirements-server.txt
python -m pip install -e . --no-deps
python -m pip check
python -c "import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
python -m ionrecog.diffusion --help
```

所有脚本都可以直接调用，无需另开服务。基础训练无需 diffusers、无需联网下载预训练模型。需要 FLUX 实验时，在单独环境安装同样的基础依赖，再安装 `requirements-features.txt`；固定到 diffusers 0.35.1，因为特征代码依赖该版本的空间 token 布局和 hook API。

## 3. 首先跑完整小样本流程

```bash
bash ionrecog/diffusion/scripts/smoke.sh runs/diffusion_smoke
```

等价命令：

```bash
python -m ionrecog.diffusion synthetic --output runs/diffusion_smoke/data --train-groups 3 --val-groups 1 --test-groups 1 --size 32
python -m ionrecog.diffusion validate --manifest runs/diffusion_smoke/data/manifest.jsonl
python -m ionrecog.diffusion train --manifest runs/diffusion_smoke/data/manifest.jsonl --config ionrecog/diffusion/configs/smoke.yaml --output runs/diffusion_smoke/train --device cpu
python -m ionrecog.diffusion evaluate --manifest runs/diffusion_smoke/data/manifest.jsonl --checkpoint runs/diffusion_smoke/train/best.pt --output runs/diffusion_smoke/evaluate --device cpu --samples 2 --save-images
```

smoke 只有 4 个训练步骤，模型不会收敛，误检/漏检很大也正常；它检查数据、训练、恢复、保存和评测能否连通。输出目录须为新目录或空目录，重跑请换名字。

## 4. 数据格式与分组规则

输入支持单幅二维灰度 PNG、TIFF、NPY，保留原始数值，不用带圈截图或伪彩色预览图训练。强度映射统一为 `clip((ADU-offset)/scale, 0, 1)`，不能对每张图分别自动拉伸。先用训练集标定选定 offset/scale，再固定到验证、测试和微调；日志中的 `clipped_fraction` 应纳入检查。

JSONL 每行一个记录，路径相对于清单文件所在目录。`id` 全局唯一；`group_id` 标识同一晶格构型谱系，所有条件、重复帧、裁剪必须在同一 split。程序能检查显式分组和重复内容，不能自动判断两个不同文件夹是否拍了同一晶格，需采集者正确填写。

**A. 有可靠配对参考：**

```json
{"id":"g001_exposure1","split":"train","group_id":"crystal001","input":"images/g001_noisy.npy","target":"targets/g001_reference.npy","target_kind":"clean","points":[[24.2,15.7],[33.1,16.0]]}
{"id":"g101_exposure1","split":"val","group_id":"crystal101","input":"images/g101_noisy.npy","target":"targets/g101_reference.npy","target_kind":"low_noise"}
```

`target_kind` 为 `clean`、`synthetic_clean` 或 `low_noise`。`clean` 只是你提供的目标声明，代码不会验证其物理真实性。input 与 target 必须已经配准、尺寸相同。`points` 为原图零基坐标 `[x,y]`，x 是列、y 是行；点标签可省略。**省略意味着没有定位真值，`[]` 表示已确认的空场**。伪标签不能冒充真实测试点，建议只把独立审核或合成精确中心写入测试 points。

**B. 无手工标签，只有稳定重复帧：**

```json
{"id":"crystal001_condition1","split":"train","group_id":"crystal001","stable":true,"frames":["train/crystal001/001.png","train/crystal001/002.png","train/crystal001/003.png"]}
```

每组至少 3 张**独立曝光**，必须保持位置、亮暗状态、曝光和增益稳定。每次取一帧作输入，其余最多 `max_target_frames` 帧平均作目标，输入自身绝不进入目标。没有自动漂移配准；条件不满足时应重新分组/配准，不能仅设置 `stable:true`。

重复均值仍有残余噪声和运动展宽：它支持去噪训练，不等于无 micromotion 真值。训练不需要手工点标签，但如果测试也无 points，报告只能给恢复代理误差，定位 precision/recall 为 null。

现有文件夹可按以下结构准备：

```text
data/diffusion_real/
  train/crystal001/*.png
  train/crystal002/*.png
  val/crystal101/*.png
  test/crystal201/*.png
```

```bash
python -m ionrecog.diffusion prepare-repeats --root data/diffusion_real --output data/diffusion_real/manifest.jsonl --confirm-static
python -m ionrecog.diffusion validate --manifest data/diffusion_real/manifest.jsonl --output runs/diffusion_data_audit.json
```

命令不做随机帧拆分。若同一晶格有多个子条件文件夹，自动生成后应把它们的 `group_id` 统一为该晶格谱系，并再次 validate；不能让每个曝光条件被误当成独立晶格。清单中 train/val 必须都非空才能训练；evaluate 默认只读 test。

审计检查分组交叉、相同文件路径、原文件 SHA256 和解码像素 SHA256 的跨 split 重复，并拒绝重复帧组里复制出来的同一张图。完全相同的公共参考若跨 split 复用也会被拒绝；训练数据应重新组织成物理上独立的组。

## 5. 合成预训练与真实微调

```bash
python -m ionrecog.diffusion synthetic --output data/diffusion_synthetic --train-groups 2000 --val-groups 200 --test-groups 200 --size 128 --seed 42
```

每个组生成一对 `observed.npy`/`target.npy`（具体路径见生成清单）和精确中心。目标是无运动、无探测噪声的光学 PSF 图；观测含正弦运动时间积分、Poisson 散粒噪声与读噪。Python API `make_synthetic_dataset(..., em_gain=300)` 可打开 Gamma 电子倍增近似；它使用输入等效强度，不能不校准就等同于相机 ADU。模拟还包含不同构型、弱点、缺陷、空场和背景，完整物理假设见生成的 `metadata.json`。

这是校验/研究模拟器，不是实验 EMCCD 标定：没有完整 RF 动力学、CIC、饱和、全部像差和荧光率随相位变化。默认强度适合约 `scale=1000`，正式预训练前需要用标定结果调整模拟器或统一物理单位，不能把默认模拟参数当成测量值。

复制服务器配置到新文件并设置：

- `data.patch_size` 不得大于最小训练图尺寸；从 64/128 起步。
- `data.offset/scale` 为训练集固定标定映射。示例 server.yaml 的 65535 只是 uint16 范围占位，不一定适合你的实际信号。
- `model.base_channels=64` 为约 2.18M 参数的起步模型；设 128 可提高容量，同时增加显存。
- `train.precision=bf16` 需要支持 BF16 的卡；不支持时用 fp32，或按实际情况用 fp16。
- `region.mode: "off"` 必须带引号，避免 YAML 把 off 当作布尔值；自动晶格区域应先在验证集校准。

单卡训练：

```bash
python -m ionrecog.diffusion train --manifest data/diffusion_synthetic/manifest.jsonl --config ionrecog/diffusion/configs/server.yaml --output runs/diffusion_pretrain --device cuda
```

真实数据重新开始训练：

```bash
python -m ionrecog.diffusion train --manifest data/diffusion_real/manifest.jsonl --config ionrecog/diffusion/configs/server.yaml --output runs/diffusion_real --device cuda
```

从合成权重初始化一个真实微调 run：

```bash
python -m ionrecog.diffusion train --manifest data/diffusion_real/manifest.jsonl --config ionrecog/diffusion/configs/server.yaml --init-checkpoint runs/diffusion_pretrain/best.pt --output runs/diffusion_finetune --device cuda
```

`--init-checkpoint` 仅载入 EMA 权重，优化器从头开始；模型、扩散设置、offset/scale 必须兼容，微调可降低学习率。若预训练用无运动真值、微调用有运动的重复均值，目标语义发生改变，可能重新学回模糊；应把这种训练单独作为实验，不把它称为持续去运动监督。最清楚的第一轮对照是分别训练“重复帧去噪”和“物理合成去模糊”，并在独立真实参考上评价。

## 6. 多卡与续训

```bash
torchrun --standalone --nproc_per_node=2 -m ionrecog.diffusion train --manifest data/diffusion_real/manifest.jsonl --config ionrecog/diffusion/configs/server.yaml --output runs/diffusion_ddp --device cuda
```

这是小 U-Net 的 DDP 数据并行，每卡保存一份模型；不是给 20B 编辑模型用的 FSDP。有效 batch 为 `卡数 × batch_size × grad_accum`，不要改变卡数后忽略 batch 变化。所有 rank 必须看到相同数据与共享输出目录；Linux CUDA 使用 NCCL，CPU 使用 gloo。

输出包括 `last.pt`、`best.pt`、`history.json`、`config.json`、`data_audit.json`、`provenance.json`。checkpoint 含网络、EMA、优化器、GradScaler、步数和数据指纹；best 按固定验证样本/噪声上的 EMA 噪声预测损失选择，**不是按定位准确率选择**。

中断后保留原输出路径和配置，执行：

```bash
python -m ionrecog.diffusion train --manifest data/diffusion_real/manifest.jsonl --config ionrecog/diffusion/configs/server.yaml --output runs/diffusion_real --resume runs/diffusion_real/last.pt --device cuda
```

原计划完成后继续训练，可在同配置文件中增大 `train.steps`。续训只允许 steps 增加，其余设置、world size、数据文件内容必须相同；不能用 resume 偷换数据/归一化。best.pt 必须保留。随机 crop/扩散噪声由 step、rank、microbatch 决定，在测试的 CPU 环境中恢复与连续训练逐张量一致；不同 CUDA 环境仍可能有数值差异。

训练前与保存时检查磁盘，保留约 2 GiB 空间；实际项目建议预留更多。日志有训练/验证损失和裁剪比例。fp16 遇到梯度溢出会按 GradScaler 降低 scale 并跳过更新，不更新 EMA，日志记录 optimizer_updates；steps 表示尝试步数。BF16/FP32 非有限值直接报错。

Slurm 单卡示例见 `scripts/train.slurm`，先按集群修改分区/显卡资源声明，然后在项目根目录调用：

```bash
sbatch --export=ALL,MANIFEST="$PWD/data/diffusion_real/manifest.jsonl",CONFIG="$PWD/ionrecog/diffusion/configs/server.yaml",OUTPUT="$PWD/runs/diffusion_slurm" ionrecog/diffusion/scripts/train.slurm
```

## 7. 批量推理与测试

```bash
python -m ionrecog.diffusion infer --input data/diffusion_real/test --checkpoint runs/diffusion_real/best.pt --output runs/diffusion_predictions --device cuda --samples 4
python -m ionrecog.diffusion evaluate --manifest data/diffusion_real/manifest.jsonl --checkpoint runs/diffusion_real/best.pt --output runs/diffusion_test --device cuda --split test --tolerance 2.0 --samples 4 --save-images
```

省略 `--config` 时使用 checkpoint 的完整设置。如果想修改检测/区域阈值，提供完整的同结构 YAML，并保留 checkpoint 的 offset/scale；采样步数及 tile/overlap 可改变。检测阈值从原始图估计噪声，晶格区域也由原图决定，避免恢复后低噪声导致门限失效。region 默认关闭，真实大晶格可在验证集校准后启用 `auto`。

推理每个输入保存：

- `sample_000_restored.npy`：float32 原强度尺度的恢复数组。
- `sample_000_overlay.png`：在原图上标圈，只供观察。
- `sample_000_points.csv`：零基 x/y 坐标与启发式 SNR、flux。
- `prediction.json`：每次采样各自的候选、随机种子、区域诊断和耗时。

`--samples 4` 保留四套独立结果，不挑最好一张，也不把不一致的点直接平均；这些采样不是四份独立实验。`snr` 是启发式局部对比度，不是校准的概率。圆圈不是量子态标签。

评测 `report.json` 和 `per_image.json` 给出原图与恢复结果的 MSE/MAE/PSNR、匹配 TP/FP/FN、precision/recall/F1、成功匹配点的平均误差、整帧全正确比例、每晶格分组统计。PSNR 使用固定训练 scale；完全相同的图用 `identical=true, psnr_db=null` 表示，避免 JSON Infinity。定位指标仅在有 points 的记录上计算，并报告有标签图数；无标签不当成空场。`low_noise`/`repeat_mean` 误差不能叫无噪真值误差。

匹配容差在测试前固定，不能挑到最有利的值。汇总统计保留每组结果，但没有把重复帧当独立样本计算置信区间；论文分析需按构型簇做统计。最小间距太大仍会漏检重叠离子，传统定位并不保证解决一颗双峰/两颗重叠歧义。

同一配置的原图基线及简单融合对照：

```bash
python -m ionrecog.diffusion evaluate --manifest data/diffusion_real/manifest.jsonl --checkpoint runs/diffusion_real/best.pt --mode raw --output runs/diffusion_test_raw --device cpu
python -m ionrecog.diffusion evaluate --manifest data/diffusion_real/manifest.jsonl --checkpoint runs/diffusion_real/best.pt --mode fused --raw-weight 0.5 --output runs/diffusion_test_fused --device cuda
```

多步采样与 tile 会增加耗时，恢复使用全图共享随机状态、重叠 tile 的 epsilon 合成，避免各 tile 独立生成接缝；GroupNorm 等局部统计仍可能使分块与整图结果不同，要在验证集比较。

## 8. 冻结编辑式 DiT 的特征实验

本版本实际支持 **FLUX.1-Kontext-dev**，不是任意 DiT 通用加载器，也不是 Qwen 后端。获取并放置已授权的官方 diffusers 格式权重后运行。默认 `local_files_only=True`，不会自动下载几十 GB；需要联网时显式加 `--allow-download`，并自行完成模型仓库访问条件。

```bash
python -m pip install -r ionrecog/diffusion/requirements-features.txt
python -m ionrecog.diffusion extract-features --input data/probe_a.npy --model /models/FLUX.1-Kontext-dev --output runs/probe_a --layers 0 5 10 --width 512 --height 512 --steps 1 --capture-step 0 --precision bf16 --intensity-scale 10000
python -m ionrecog.diffusion extract-features --input data/probe_b.npy --model /models/FLUX.1-Kontext-dev --output runs/probe_b --layers 0 5 10 --width 512 --height 512 --steps 1 --capture-step 0 --precision bf16 --intensity-scale 10000
python -m ionrecog.diffusion compare-features --source runs/probe_a/features.npz --target runs/probe_b/features.npz --output runs/probe_matches --layer 5 --stream reference --normalization l2 --one-to-one --mutual
```

输入科学灰度以明确的 offset/scale 转 float32 RGB，保留相同强度映射；512×512 只是输入尺寸示例，非默认保证。选择尺寸需满足模型要求，注意裁剪/插值改变点源尺度。两张比较图保持提示词、层、capture step、总步数、精度和强度映射一致。

输出 `features.npz` 中 `reference_layer_5`/`target_layer_5` 是二维格点×通道特征，`reference_xy`/`target_xy` 保存格点中心到原图的坐标映射；`metadata.json` 记录尺寸、时间步与坐标约定。流的区分来自 transformer 实际 `img_ids`，不把文本 token 当空间点。

**reference 是条件图像空间 token，target 是生成轨迹 token。**当 `steps=1, capture_step=0` 时，target 从噪声状态参与一次前向；这不是对原始离子图做标准 DDPM 加噪得到的 DIFT，也不能将 target 当成已恢复图。先检查 reference 的稳定性，再改变层/步数做消融。

`compare-features` 输出 `cosine.npy` 全矩阵和 `matches.json`；默认逐 token 最近邻，可选 Hungarian 最大总相似度匹配和互相最近邻过滤。输出是**token 区域对应，不是逐离子亚像素坐标或持续身份**。`--normalization channel_standardize` 使用两幅图的联合通道标准化再算余弦，作为探索对照；不是 DiTF 论文的 AdaLN 调制/通道剔除复现。未处理前景与背景混合问题，不应把全部 token 匹配直接当检测结果。

显存不足可尝试 `--cpu-offload`，这需要足够主存，且可能显著变慢。当前没有量化训练后端。特征缓存和全相似度矩阵都可能很大，代码限制矩阵条目数；先用 2–10 张图验证原始空间信息，再扩展。

官方接口依据：[FLUX 模型卡](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev)、[diffusers v0.35.1 流水线](https://github.com/huggingface/diffusers/blob/v0.35.1/src/diffusers/pipelines/flux/pipeline_flux_kontext.py)。

## 9. 验证与交付边界

```bash
python -m pytest tests/test_diffusion_data.py tests/test_diffusion_synthetic.py tests/test_diffusion_engine.py tests/test_diffusion_pipeline.py tests/test_diffusion_features.py -q
```

这些测试包括训练/推理集成、断点续训、数据变更拒绝、空场与无标签区分、泄漏、EM 噪声采样、空间 token 拆分及匹配。服务器先跑 smoke，再跑 CUDA 小数据任务，并测显存/每步时间；不要直接用默认50,000步启动所有实验。

后续主结果应来自独立大晶格真实测试：当前没有附带有效训练权重，没有“去雾后准确率提高”的实验结论。随机采样一致不等于物理正确；需要可信参考来评价离子数与中心误差。
