# 直接点源定位：仿真监督实验分支

本目录直接从原始灰度图预测离子中心概率热图和亚像素偏移，再输出坐标。它是自建的小型卷积基线，**不是 DECODE、LiteLoc 或 SHOT 官方实现，也没有附带可用的真实离子权重**。实现参考这类方法的任务定义，不声称复现其论文性能。

与 `self_supervised` 中先恢复图像的流程不同，本分支监督中心位置；已有 masked/diffusion/VLM 权重不能加载到这里。没有调用大模型服务、自动下载或后台训练。

## 文件与接口

| 文件 | 用途 |
|---|---|
| `simulation.py` | 在线生成点源标签、椭圆 PSF、对称运动展宽、随机/晶格布局与空场景 |
| `models.py` | 小型全卷积网络；中心热图 focal loss＋中心位置处的偏移损失 |
| `training.py` | 仿真监督训练、独立固定验证流、阈值选择、最佳/最后权重与续训 |
| `inference.py` | 带上下文的分块、核心像素拼接、全局点提取、可选原图信号检查 |
| `cli.py` / `__main__.py` | `python -m ionrecog points train/infer` 与独立模块入口 |
| `simulation.example.json` | **示例参数，尚未按相机实测标定** |

## 明天开始时的命令

在项目根目录运行。默认使用 CPU 和 2 个线程，不自动占用正在做 VLM 训练的 GPU。

```powershell
Set-Location 'F:\CQI\Bell Camera\Net Recog'
.\.venv\Scripts\python.exe -m ionrecog points --help

# 极短流程检查；生成的权重不能作为已经收敛的识别器。
.\.venv\Scripts\python.exe -m ionrecog points train --output runs/points_smoke --epochs 1 --steps 2 --batch-size 1 --val-samples 2 --channels 4 --device cpu

# 初始训练示例；训练规模和模拟参数需要根据验证结果调整。
.\.venv\Scripts\python.exe -m ionrecog points train --simulation ionrecog/point_localization/simulation.example.json --output runs/points_v1 --epochs 10 --steps 100 --batch-size 4 --val-samples 32 --device cpu

# 先取少量真实图检查；现有相机使用已有晶格区域配置。
.\.venv\Scripts\python.exe -m ionrecog points infer --checkpoint runs/points_v1/best.pt --input data/test --output runs/points_v1_test --config configs/existing_crystal.yaml --max-images 10 --device cpu
```

也可将命令开头的 `python -m ionrecog points` 换成 `python -m ionrecog.point_localization`。仅显式选择 `--device cuda` 或 `auto` 才可能占用 GPU；当前 `.venv` 是 CPU PyTorch，CUDA 需另一个兼容的本地环境。

代码创建阶段只做语法/命令帮助检查，以上训练、推理及算法测试**尚未执行**。测试已写在 `tests/test_point_localization.py`，后续可以运行：

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_point_localization.py
```

测试包含短训练和推理，运行它们会实际执行算法。

## 模拟与标签的定义

训练图按需生成，不会在磁盘保存成千上万张模拟图片。每个场景随机选择点数、坐标、亮度、背景梯度、椭圆宽度和方向，并对均匀运动相位积分模拟对称正弦展宽。标签是这一对称模型的运动中心坐标，采用零基 `x=列、y=行`。

- `sigma_min/max`：两个主轴的光学高斯标准差范围，单位像素。
- `motion_max`：对称正弦位移振幅上限，单位像素；不代表已经测得真实 micromotion。
- `amplitude_min/max`：未运动时高斯峰值的期望计数尺度，不是积分总光子数。
- `read_noise`、`camera_bias`：高斯读出噪声标准差和固定偏置。
- `input_offset/scale`：固定线性归一化，在所有训练、验证和推理图中一致，保存到 checkpoint；不会对每张图重估尺度。
- `empty_fraction`：训练中加入无离子的纯背景，减少网络在空图上输出候选。
- `lattice_fraction`：部分场景采用有抖动和缺点的三角网格，其余采用随机位置。标签始终来自实际生成的点，程序不补齐理想晶格。

这是 **Poisson＋Gaussian** 简化噪声模型；尚无完整 EMCCD 倍增统计、实测 PSF 数组导入或空间变化像差标定。`simulation.example.json` 只提供明确可编辑的起点。应先用真实数据测量光斑宽度、峰值/背景范围和展宽，再修改配置。当前不读取人工点标签或 classical 伪标签进行训练，因此不能把本分支叫作真实数据自监督。

每像素最多表示一个中心，模拟器要求点间距至少 1.5 px 并避免同像素标签碰撞。越密集的场景越需要另行验证。点数过大或晶格候选不足时实际生成数可能低于请求上限，标签记录实际生成的全部点。

## 训练记录、验证和续训

`configuration.json` 保留全部模拟/训练参数；`progress.json`、`history.json` 保留进度、损失和验证结果。`best.pt` 按 1 px 容差下的验证 F1 选择；`last.pt` 每个完整 epoch 保存模型和优化器。采用临时文件写完后替换，默认保留至少 2 GiB 磁盘空间。

训练和验证使用分域的随机种子流，验证场景固定；阈值在该验证集的 0.1–0.7 网格上选择。报告同时提供 1 px/3 px 的 TP/FP/FN、precision/recall/F1、整图正确比例。**这些是用于选模/选阈值的合成验证结果，既不是未触碰测试集成绩，也不是真实离子准确率。**应保留独立真实参考或独立模拟测试流做最终评测。

默认训练不做零样本或实验准确率承诺。极短训练可能没有检测能力；先看 `history.json` 里的漏检与误检，不能仅因生成了 checkpoint 就称为成功识别。

续训使用新输出目录并保持原模拟参数、网络大小、batch、steps、seed、lr、验证样本数，将 `--epochs` 设置为目标**总轮数**：

```powershell
.\.venv\Scripts\python.exe -m ionrecog points train --simulation ionrecog/point_localization/simulation.example.json --output runs/points_v2 --epochs 20 --steps 100 --batch-size 4 --val-samples 32 --resume runs/points_v1/last.pt --device cpu
```

续训目录中的 `best.pt` 是此次继续训练阶段的最佳，原目录的历史最佳权重仍保留。epoch 内中断后重新开始该 epoch；不是逐步断点恢复。只加载自己生成、可信的 checkpoint（兼容的 PyTorch 1.13 使用 pickle）。

## 推理与结果解释

按核心像素拼接各图块的预测热图和偏移，再统一提点，避免独立裁剪输出的重复坐标。默认 `--tile-size 256 --halo 32`；halo 提供邻居上下文，强制至少 16 px，以覆盖当前网络的局部感受野。图块起点对齐原图的二像素池化网格，奇数尺寸仅在下侧/右侧重复边缘补齐，保留原坐标，防止各块缩放比例不同。`--threshold` 可覆盖合成验证选出的阈值，真实实验应在独立验证集上确定它。

`--support-snr 0` 默认不额外筛选；大于零时使用原图的局部对比度检查候选，但不会把网络中心替换成 classical 质心。该选项可能删掉弱离子，必须单独记录并重新验证，不能声称一定更好。

输出遵循公共格式：逐图 JSON、CSV、原图标圈、总坐标表和 summary。神经网络 score 不是校准概率，也不是量子态保真度。达到 `--max-ions` 上限会标记结果覆盖不完整，不能当作全图已经正确检测。

目前仍使用 3×3 局部峰抑制和最小间距合并，特别相近的真实离子可能互相抑制。这是本版明确的局限，并未实现 SHOT 式无 NMS 点集预测。后续可在相同评估协议下对比这些不同的解码方法。
