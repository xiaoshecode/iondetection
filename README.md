# 大晶格离子图像识别：PC 自监督实验框架

**源码导航：** 三类方案已分别整理到 `ionrecog/classical/`、`ionrecog/self_supervised/`、`ionrecog/vision_language/`。masked 与 diffusion 是第二类中的两个不同方法。文件职责、旧导入兼容及版本说明见 [ionrecog/README.md](ionrecog/README.md)。

**新增 VLM 实验分支：** 支持本地 Qwen3-VL 分块候选定位、原图质心细化与信号检查、相同区域/分块的 classical 对照、原始回复审计及中断续跑。使用独立 `python -m ionrecog vlm` 命令，说明见 [VLM 实验指南](docs/VLM_PILOT.md) 和 [本机模型运行环境](docs/VLM_RUNTIME.md)。模型输出是候选，真实定位准确率仍需独立真值。

**VLM 微调实验：** 新增 classical 稳定伪标签筛选、Qwen3-VL-2B 的 NF4 LoRA 训练及保存 adapter 的独立评估入口。使用隔离环境 `.venv-vlm-train`，数据、运行状态和复现命令见 [VLM 训练记录](docs/VLM_TRAINING.md)。前后对照使用同一个 2B 基座；指标表示与伪标签的一致性。

本项目用于二维离子晶格的**候选离子中心定位、标圈与原图亮度测量**。面向每张几百至约一千个离子、边缘存在拖尾和微运动模糊的实验，先在 PC 上建立可复现实验流程，不涉及 FPGA 部署。

已实现可运行的训练、推理、合成数据、评估和测试代码。**这是研究起点，代码不附带通用预训练权重，真实大晶格准确率尚未验证。** 自监督可以减少去噪训练所需标注，但不会自动提供可信的离子状态标签。论文中的定位准确率仍需要独立验证依据。

**本机已有数据已导入，并完成第一次真实数据训练。** 426 张用于 masked 训练、250 张验证、263 张测试，另 5 张不同相机设置的图片单独保留。训练和测试记录见 [已有数据运行说明](docs/EXISTING_DATA_RUN.md)。本地权重位于 `runs/existing_20261103/masked_training/best.pt`，这是本次数据训练的初步模型，不是通用预训练模型。

**晶格外误检修复：** 现有 512×512 相机图请使用 [configs/existing_crystal.yaml](configs/existing_crystal.yaml)。它在一次设定的粗搜索框内，从每张原图估计实际晶格边界，classical/masked 共用区域及原图噪声阈值；找不到可靠区域时不输出候选。已重跑 250 张图，见 [修复说明与命令](docs/CRYSTAL_REGION.md) 和本地 [新版对照报告](runs/crystal_region_v1/comparison/index.html)。通用配置仍保留 `region.mode: off` 的历史基线行为，不能把该相机的像素框直接套到其他尺寸或新光路。推理会估算磁盘需求并默认保留 2 GiB 空间。

## 1. 两个想法如何落到代码

| 模式 | 流程 | 数据要求 | 是否训练 |
| --- | --- | --- | --- |
| `classical` | 背景估计 → 前景区域 → 区域内多峰 → 亚像素中心 → 画圈 | 单张原始图即可 | 否，首先运行的基线 |
| `masked` | 遮挡像素自监督 U-Net 去噪 → 同一定位器 | 无标注单帧，可以是不同构型 | 是，建议先尝试 |
| `diffusion` | 条件扩散恢复 → 同一定位器 | 每组至少 3 张稳定晶格的独立重复帧 | 是，计算成本较高 |

第一条对应“先分区域，再找亮点”。这里一个连通区域允许有多个离子，不能只取整个区域最亮的一个点，否则密集晶格会大量漏检。检测器使用多个局部峰、最小间距抑制和局部强度质心细化中心。画圈半径是显示及测量参数，不是已估计的离子物理尺寸。

当前区域分割是图像算法，**尚未实现学习式实例分割、Fast/Faster R-CNN 或端到端检测头**。两条学习路线负责恢复强度图，随后共用定位器，方便做公平消融。程序不会按理想晶格补出不存在的离子。

## 2. 目录结构

```text
Net Recog/
├── README.md
├── requirements.txt / requirements-dev.txt / pyproject.toml
├── configs/
│   ├── default.yaml             # 初始实验参数
│   └── smoke.yaml               # 极小参数，仅检查流程
├── data/
│   ├── README.md                # 采集、格式、分组说明
│   ├── train/session_001/       # 放训练图；可创建更多组
│   ├── val/session_101/         # 放独立验证组
│   ├── test/session_201/        # 放待识别图
│   ├── reference/session_201/   # 可选：对应构型的全亮参考
│   └── calibration/dark/        # 暗场预留，当前不自动应用
├── ionrecog/
│   ├── README.md               # 三类方案、四个运行入口与兼容说明
│   ├── classical/              # 区域分割、多峰定位、原图测光
│   ├── self_supervised/        # masked 与 diffusion 的学习实现
│   ├── vision_language/        # VLM 提议、分块和原图检查
│   ├── cli.py                  # 共用命令行
│   ├── data.py / validation.py # 共用原始图读取和集合检查
│   ├── synthetic.py            # 有位置真值的合成演示
│   ├── evaluation.py           # 一对一匹配定位评估
│   └── detection.py 等         # 旧导入路径的轻量兼容入口
├── scripts/run_pipeline.ps1     # 一键检查、训练和推理
├── tests/                      # 有意义的数据/算法/训练测试
├── docs/                       # 讨论记录和验证记录
├── checkpoints/                # 可选权重归档位置
└── runs/                       # 默认结果位置；不会上传 Git
```

图片放置细节见 [data/README.md](data/README.md)。单帧二维灰度 PNG/TIFF/NPY 可直接用，推荐保留原始 16 位图。无需先制作框、圆、分割掩码或 COCO 标签。

## 3. 环境与安装（Windows PowerShell）

所有示例在项目根目录执行，路径有空格，保留引号：

```powershell
Set-Location 'F:\CQI\Bell Camera\Net Recog'
.\.venv\Scripts\python.exe -m ionrecog --help
```

**这台电脑已准备项目 `.venv`**：使用已有 Anaconda Python 3.9 的 PyTorch CPU 安装，在 `.venv` 内安装兼容的 SciPy/Pillow/tifffile，未更改原来的 Anaconda 包。该环境依赖 `E:\Anaconda`，不要直接搬到另一台电脑。实际检查记录见 [docs/VALIDATION.md](docs/VALIDATION.md)。

新电脑建议安装 Python 3.11–3.12，重新创建独立环境，不需要激活 PowerShell 环境脚本。本项目兼容范围为 3.9–3.12，NumPy 固定小于 2 以兼容已有 PyTorch 1.13：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
# 无 NVIDIA CUDA 需求时，先安装 CPU 版，避免无谓下载 GPU 运行库。
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q
```

从项目根目录使用 `python -m ionrecog` 无需安装本项目。也可执行 `python -m pip install -e .` 注册 `ionrecog` 命令。GPU 环境按 [PyTorch 官方安装页](https://pytorch.org/get-started/locally/)选择适配驱动的版本；程序的 `--device auto` 在 CUDA 可用时使用 CUDA，否则 CPU。CPU 可以运行两种模型，完整大数据训练尤其扩散模型可能很慢。代码不下载预训练模型。

如果 pip 联网失败，先检查该终端的代理，不要关闭 SSL 验证。项目不修改系统或 Git 代理。当前电脑的 `.venv` 已能使用，不需要每次运行重新安装。

## 4. 最快开始：先放测试图直接画圈

把几张原始图放进 `data/test/session_201/`，运行：

```powershell
.\.venv\Scripts\python.exe -m ionrecog infer --method classical --input data/test --output runs/first_classical --config configs/default.yaml
```

打开 `runs/first_classical/images/session_201/` 里的 `*_overlay.png` 看圈，读取 `*_ions.csv` 看中心。重复运行时改一个输出目录名；默认拒绝覆盖已有结果。只有明确希望替换同名产物时才加 `--overwrite`，它不清理旧目录中的其他产物。

也可一条命令启动，脚本自动产生时间戳目录：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_pipeline.ps1 -Method classical
```

这里 `-ExecutionPolicy Bypass` 仅作用于这一 PowerShell 进程，不修改系统执行策略。原始数据不会被脚本改写。

## 5. 不手工标注：先运行 masked 自监督

把训练图放 `data/train/不同采集组/`，验证图放 `data/val/另外的采集组/`，测试图放 `data/test/`。

**先按实验组划分数据，再放文件。** 例如用不同日期/构型的一部分组做验证。不要把同一个晶格的相邻连拍帧随机分到 train 和 val；即使噪声不同，也会造成严重的数据泄漏。train 和 val 的组名应唯一。

```powershell
.\.venv\Scripts\python.exe -m ionrecog inspect --method masked --train data/train --val data/val --output runs/masked_dataset.json
.\.venv\Scripts\python.exe -m ionrecog train --method masked --train data/train --val data/val --config configs/default.yaml --output runs/masked_v1 --device auto
.\.venv\Scripts\python.exe -m ionrecog infer --method masked --input data/test --checkpoint runs/masked_v1/best.pt --config configs/default.yaml --output runs/masked_v1_predictions --device auto
```

或一键完成同一流程：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_pipeline.ps1 -Method masked
```

训练过程随机抽图和图像块，遮挡少量像素，只在被遮挡位置计算恢复误差。原始被遮挡值不会通过替换操作复制到另一个输入位置，防止直接泄露目标。这是受 Noise2Void 启发的基线，未声称完整复现其论文或官方实现。

该方法依赖像素噪声在给定信号下近似独立且均值适当。条纹、固定图样、串扰等相关噪声可能被学成信号；过强模糊本身也不是随机噪声。非常暗或欠采样的孤立点可能被过度平滑。请同时保留 classical 基线，检验去噪是否真的改善定位。

## 6. 扩散模型：多样构型 + 每组短重复采集

推荐采集组织方式是“**许多不同构型，每个构型短连拍一组**”，不需要只拍同一个晶格。示例：

```text
data/train/session_001/frame_000.png
data/train/session_001/frame_001.png
data/train/session_001/frame_002.png
data/train/session_002/frame_000.png
...
data/val/session_101/frame_000.png
data/val/session_101/frame_001.png
data/val/session_101/frame_002.png
```

同一最末级目录必须是同一稳定场景。每组至少 3 帧，条件允许时可采 8–16 帧。晶格位置、占据、亮暗状态、曝光和增益不能在组内改变；每帧必须是独立曝光。代码没有自动漂移配准，重复帧均值会把漂移、跃迁或晶格重排变成模糊错误目标。

```powershell
.\.venv\Scripts\python.exe -m ionrecog inspect --method diffusion --train data/train --val data/val --output runs/diffusion_dataset.json
.\.venv\Scripts\python.exe -m ionrecog train --method diffusion --train data/train --val data/val --config configs/default.yaml --output runs/diffusion_v1 --device auto --confirm-static
.\.venv\Scripts\python.exe -m ionrecog infer --method diffusion --input data/test --checkpoint runs/diffusion_v1/best.pt --config configs/default.yaml --output runs/diffusion_v1_predictions --device auto --sample-steps 20
```

一键版本：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_pipeline.ps1 -Method diffusion -ConfirmStaticRepeats
```

`--confirm-static` 是数据条件声明，不能让不稳定的数据变稳定。若完全没有重复帧，先用 masked；当前 diffusion 分支不会把任意不同晶格拼成训练对。

训练时从一组选择输入 `y_i`，以不含它的其他帧均值为目标：

```text
c = y_i
z = mean(y_j, j != i)           # 最多 data.max_target_frames 帧
x_t = sqrt(alpha_bar_t) z + sqrt(1-alpha_bar_t) epsilon
loss = MSE(epsilon_theta(x_t, c, t), epsilon)
```

实际 `z`、`c` 先经过同一个固定强度映射再映射到 [-1, 1]。网络是带时间嵌入的小 U-Net，训练做条件 DDPM 噪声预测，推理做 DDIM，默认 cosine schedule，100 个训练时间步、20 个采样步。训练和采样时间步是两个参数。

**这里 epsilon 是人为加入的扩散高斯噪声，不等于已识别出的相机物理噪声。** 模型学习的是重复帧低噪均值的条件分布。均值仍有残余噪声，不是精确干净真值；它也不消除稳定存在的 PSF 拖尾和微运动模糊。独立重复观测的思想与 Noise2Noise 有关，但条件扩散目标不等同于其 MSE 理论，不能直接引用该理论宣称无偏恢复。

可用不同种子采样检查结果稳定性：

```powershell
.\.venv\Scripts\python.exe -m ionrecog infer --method diffusion --input data/test --checkpoint runs/diffusion_v1/best.pt --output runs/diffusion_samples --samples 3 --sample-steps 20
```

不同采样的圈数或位置变化仅是诊断，**不是校准置信度或准确率**。扩散先验可能补出点或抹掉弱点。因此本项目保留原图上的测量和标圈，不能只凭恢复图更好看来认定性能提高。

## 7. 训练参数、归一化与恢复训练

默认配置是初始实验设置，不是验证过的最优参数：

| 参数 | 默认 | 含义及修改时机 |
| --- | --- | --- |
| `data.patch_size` | 64 | 训练块边长；不应小于需要解释的拖尾/相邻结构；须不大于图像短边 |
| `data.offset` | 0 | 相机黑电平；支持手动设定，当前不自动暗场标定 |
| `data.scale` | auto | 使用训练集最大值减 offset 得到一次固定尺度，保存到模型 |
| `data.max_target_frames` | 8 | diffusion 每个平均目标最多用几张其他帧 |
| `model.base_channels` | 16 | U-Net 宽度，越大计算/显存需求越高 |
| `train.epochs` | 10 | 每个 epoch 是固定次数随机抽块，不是遍历全部图片 |
| `train.steps_per_epoch` | 100 | 每轮优化步数；一万张图需自行增大覆盖量 |
| `train.batch_size` | 4 | 每步图像块数；显存不足时减小 |
| `train.mask_fraction` | 0.02 | masked 监督像素比例 |
| `inference.tile_size / overlap` | 256 / 32 | 恢复网络的整图分块及重叠；定位仍在拼接后的整图上进行 |

默认训练总共仅抽取约 10×100×4=4,000 个随机块，**对于一万张实验图远不足以保证覆盖和收敛**。先用少量图调通，随后根据验证曲线、位置稳定性及独立评测增大步数。训练读取采用最多 16 张图缓存，不会将一万张图全部放入内存。

所有训练/验证/推理使用模型保存的同一尺度，不按每帧单独拉伸。训练和推理会将归一化值裁剪到 [0, 1]，所以验证或测试图高于训练最大值时可能被截断。auto 的最大值也可能被热像素主导；相机量程、黑电平明确时，建议显式设数值 scale，并检查饱和。不同曝光/增益混合需要专门设计，目前没有自动曝光校正或条件建模。显示用的对比度拉伸仅用于 overlay。

训练可使用参数覆盖，例如：

```powershell
.\.venv\Scripts\python.exe -m ionrecog train --method masked --config configs/default.yaml --output runs/masked_long --epochs 50 --steps 500 --batch-size 4 --patch-size 96
```

恢复训练必须使用原输出目录和同样的数据/配置，仅允许增加总 epoch 数。下面的 50 是总轮数：

```powershell
.\.venv\Scripts\python.exe -m ionrecog train --method masked --config configs/default.yaml --output runs/masked_v1 --resume runs/masked_v1/last.pt --epochs 50
```

如果首次训练用过 `--steps` 等覆盖项，恢复时也须传同样的值。权重是本地 PyTorch checkpoint，只加载自己生成或可信来源的文件。模型与归一化以 checkpoint 为准；推理时 YAML 的 detection/reference 设置及显式写出的 inference 分块设置生效，不会改写原权重。summary.json 同时记录 CLI 参数配置和生效的模型恢复配置。

## 8. 如何调圈的位置与数量

编辑 `configs/default.yaml` 的 `detection`，先看 classical 的原图叠加结果：

| 参数 | 默认 | 作用 |
| --- | --- | --- |
| `sigma` | 1.0 px | 初始平滑；过大会合并相邻点 |
| `background_sigma` | 10 px | 慢变背景估计范围；应明显大于单个离子亮斑 |
| `threshold_sigma` | 4.0 | 前景/峰检测阈值；降低通常提高召回也增加假点 |
| `min_area` | 3 px | 删除极小连通区域 |
| `min_distance` | 4 px | 峰间最小距离；要明显小于最小离子间距，过大会漏相邻离子 |
| `radius` | 3 px | 局部中心细化、画圈与原图测量半径；不是统一物理离子大小 |
| `max_ions` | 5000 | 最多保留的候选数，不是已知离子总数约束 |

也支持 `threshold_abs`（默认 0，ADU 背景扣除后的阈值下限）和 `saturation_value`（默认 null）。饱和标记只有设定实际阈值才生效；学习分支的该标记来自恢复图，不可当作原始像素饱和检测。相邻峰分不开、环形或严重拖尾产生双峰时，单调阈值可能无法解决，需要后续 PSF/微运动成像模型与更强定位器。

## 9. 输出文件与坐标约定

一次 infer 输出：

```text
runs/某次运行/
├── detections.csv                  # 逐张追加的所有候选
├── summary.csv / summary.json      # 每张/每个采样的圈数、用时、参数
└── images/原目录/原文件名.png_...
    ├── *_ions.json / *_ions.csv    # 候选中心、峰值、原图测量
    ├── *_overlay.png              # 原图显示 + 圆圈，8 位 RGB 预览
    ├── *_foreground.png           # 前景二值区域，不是实例真值
    └── *_restored.npy             # 仅学习模式；float32，恢复回原始数值单位
```

`x` 是从左到右的列坐标，`y` 是从上到下的行坐标，均从 0 开始，可以是小数。`id` 仅为当前图的排序编号，**不是跨帧追踪的离子身份**。默认 `status=observed` 只表示检测到候选点。

`peak/flux/snr` 来自定位器输入图，学习模式下即恢复图。`raw_flux/raw_peak/raw_background/raw_snr` 在原始采集图的对应位置测量，不能混用两类量。SNR 是局部噪声的启发式量，没有按完整 EMCCD 噪声模型标定。拥挤晶格中背景环可能受到邻居污染，光度值也需要验证。

单张图用 `--input '完整文件路径'`，批量递归读取目录。先试几张可用 `--max-images 5`。多次扩散采样分别保留结果，不会自动平均坐标或认定某个采样正确。10,000 张大图的恢复数组和预览会占用明显磁盘空间，先小批量估算后再扩展。

## 10. 状态识别与全亮参考：可选，不能等同于真实暗态

单张荧光图中，一个完全不发光位置可能对应暗离子、离子丢失或背景。没有先验位置或独立信号时，自监督不能凭空消除这种歧义。

如果能采到**同一构型的全亮参考**，可以从参考平均图定位各格点，再在测试原图测量这些位置，即使测试时某些位置很暗：

```powershell
.\.venv\Scripts\python.exe -m ionrecog reference --input data/reference/session_201 --output runs/reference_201.json --config configs/default.yaml --confirm-static
.\.venv\Scripts\python.exe -m ionrecog infer --method classical --input data/test/session_201 --reference runs/reference_201.json --output runs/reference_query_201 --config configs/default.yaml
```

reference 输入只能是一个稳定构型，不要指向包含多个不同晶格的上层目录。程序拒绝多组输入，并要求用 `--confirm-static` 声明全亮且稳定。当前没有配准，参考与测试必须保持相同坐标、尺寸、曝光和增益。它只记录参考检测出来的格点，不补齐隐藏格点。

额外输出 `*_reference_photometry.csv`：根据原始图局部净通量与参考比值标记 `bright_like`、`dim_candidate` 或 `uncertain`。默认阈值 0.5/0.15 只是可配置示例，**不是量子态分类器**。参考 ROI 截边、无正参考强度等情况保持 uncertain。尚未实现亮/暗光子计数分布标定、状态保真度估计、漂移追踪、离子占据判定和量子态自动标签。

## 11. 没有真实数据也可以先检查整个流程

生成小型合成数据，不会复制或改动实验数据：

```powershell
.\.venv\Scripts\python.exe -m ionrecog demo --output data/demo --groups 4 --frames 6 --size 128
.\.venv\Scripts\python.exe -m ionrecog infer --method classical --input data/demo/val --output runs/demo_classical
.\.venv\Scripts\python.exe -m ionrecog train --method masked --train data/demo/train --val data/demo/val --config configs/smoke.yaml --output runs/demo_masked --device cpu
.\.venv\Scripts\python.exe -m ionrecog infer --method masked --input data/demo/val --checkpoint runs/demo_masked/best.pt --output runs/demo_masked_predictions --device cpu
.\.venv\Scripts\python.exe -m ionrecog train --method diffusion --train data/demo/train --val data/demo/val --config configs/smoke.yaml --output runs/demo_diffusion --device cpu --confirm-static
.\.venv\Scripts\python.exe -m ionrecog infer --method diffusion --input data/demo/val --checkpoint runs/demo_diffusion/best.pt --output runs/demo_diffusion_predictions --device cpu
.\.venv\Scripts\python.exe -m pytest -q
```

`configs/smoke.yaml` 仅一轮、极少步数，输出通常没有可用识别质量，只用于排查安装、维度、保存和加载。合成器包含固定晶格、简单拖尾、Poisson+Gaussian 噪声；不是完整 EMCCD 或离子阱动力学模拟。它的 `truth.json` 只为合成验证提供位置。

## 12. 怎样评估与推进论文

有独立中心真值时，把单张图真值存成 JSON：

```json
[[30.25, 45.5], [42.75, 46.0]]
```

再进行阈值内一对一匹配：

```powershell
.\.venv\Scripts\python.exe -m ionrecog evaluate --prediction runs/first_classical/images/session_201/frame_000.png_ions.json --truth path/to/independent_truth.json --tolerance 3 --output runs/position_evaluation.json
```

输出 TP/FP/FN、precision、recall、F1、匹配中心误差和该图是否全部正确。容差单位为像素，应在报告中注明。合成 `truth.json` 是总清单，需先提取对应图片的 `ions` 中的 `[x,y]`，不能直接把整个清单传给 evaluate。

无人工标签时可用高信噪参考、独立物理定位或少量抽样复核建立评测集。参考也有误差，需注明来源与不确定性；**不得把同一模型的伪标签当真值计算识别准确率**。训练日志中的 `train_proxy_loss/val_proxy_loss` 只是对应恢复任务损失；masked 和 diffusion 的损失定义不同，数值不可直接横比。

建议后续先固定一个小而可靠的评测集，比较原图直接定位、masked、重复帧平均和 diffusion，然后按晶格中心/边缘、离子间距、信号强度、拖尾长度与离子数分层统计。大量弱标签用于训练、小量可信真值用于评估，比完全不验证更适合支撑论文结论。

值得进一步研究但**尚未实现**的方向包括：位置相关 PSF/微运动前向模型、与原图光子证据一致的扩散约束、去噪与中心定位联合优化、真实像素噪声标定，以及避免补出虚假离子的可校准不确定性。当前代码组合本身不构成已经证实的 CVPR 创新；需要方法、对照实验和跨条件泛化证据。

## 13. 常见问题

- `No input images`：确认图片已放到对应目录且后缀支持，`.gitkeep` 不是图像。
- `Expected ... grayscale`：读到了彩色预览或截图。导出原始单通道帧，不要无说明地转灰度冒充原始数据。
- `smaller than patch_size`：修改 patch_size 或使用 `--patch-size`，训练最小为 8；本版不通过反射填充伪造独立噪声。
- `Every diffusion group ... at least 3`：每个最末级目录至少三张稳定重复帧，包括验证集；做不到就先用 masked。
- `duplicate/shared group`：检查训练验证是否重复，按完整采集组重新划分；不要只改文件名掩盖泄漏。
- 圈太多或没有圈：先检查原图、像素间距、阈值和强度范围，再看恢复效果；不要把未收敛的 smoke 模型当正式模型。
- 输出目录非空：换一个运行名；训练恢复用 `--resume`。原始图目录及其父子目录均不应用作运行输出。
- GPU 不可用：先 `--device cpu` 跑通，再检查 PyTorch/CUDA 安装；不会自动修改驱动。
- 内存/显存不足：减小 batch_size、patch_size 或 checkpoint 的 inference.tile_size。单个超大 TIFF 解码仍需整帧内存；训练缓存最多 16 帧。
- NumPy/PyTorch 旧版本兼容：本机验证使用 NumPy 1.26 和 PyTorch 1.13；若坚持旧 PyTorch，请使用 `numpy<2`。全新环境建议安装相互兼容的当前稳定版。

## 14. 方法来源

本项目是针对离子图像编写的轻量实验实现，并非以下论文的官方复现：

- Krull et al., [Noise2Void — Learning Denoising From Single Noisy Images, CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Krull_Noise2Void_-_Learning_Denoising_From_Single_Noisy_Images_CVPR_2019_paper.html)：单张噪声图的遮挡像素自监督思想。
- Lehtinen et al., [Noise2Noise: Learning Image Restoration without Clean Data, ICML 2018](https://proceedings.mlr.press/v80/lehtinen18a.html)：独立带噪观测用于学习恢复的思想；不直接等价于本项目扩散目标。
- Ho et al., [Denoising Diffusion Probabilistic Models, 2020](https://arxiv.org/abs/2006.11239)：扩散噪声预测基础。
- Song et al., [Denoising Diffusion Implicit Models, 2020/ICLR 2021](https://arxiv.org/abs/2010.02502)：DDIM 采样基础。

早期研究讨论保留在 [docs/2026-11-03-ion-detection-discussion.md](docs/2026-11-03-ion-detection-discussion.md)。实现与研究设想有区别，以本 README 的“已实现/尚未实现”说明及代码为准。
