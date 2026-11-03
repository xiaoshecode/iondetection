# 新定位方法：代码交付与后续运行

日期：2026-11-04（香港时间）。按用户要求，本次完成代码、目录及说明，**只做语法与命令接口检查，不执行新方法的训练、识别或算法测试**。以下命令供后续手动运行；当前没有新训练权重、识别报告或实验准确率。

## 1. 文件位置与实现范围

所有方法代码均在 `F:\CQI\Bell Camera\Net Recog\ionrecog` 内：

| 目录 | 入口 | 已写入的功能 | 第一版边界 |
| --- | --- | --- | --- |
| [psf_fitting](../ionrecog/psf_fitting/README.md) | `psf` | 固定椭圆/已知运动光斑的局部多离子联合拟合、背景和数量比较、残差补点、回退诊断 | PSF 参数需自行设置；大连通组、未建模邻居等会显式保留 classical 结果 |
| [point_localization](../ionrecog/point_localization/README.md) | `points train/infer` | 在线模拟、中心热图＋亚像素偏移网络、独立模拟验证、阈值选择、分块推理、续训 | 模拟监督基线，非 DECODE/LiteLoc/SHOT 官方复现；没有真实数据标定或预训练权重 |
| [equivariant_localization](../ionrecog/equivariant_localization/README.md) | `equivariant train/infer` | 单离子无坐标裁剪上的旋转/翻转/平移一致性训练，带信号支持的神经加权质心 | LodeSTAR-inspired 候选细化；classical 决定数量，不能找回漏检或保证拆分重叠 |

共用 [localization_common.py](../ionrecog/localization_common.py) 负责原始灰度图读取、晶格区域限制、原图测光、CSV/JSON/标圈导出、输出路径保护、磁盘余量及原子权重写入。现有 `classical`、`masked`、`diffusion`、VLM 入口保持各自实现。

本次没有实现稀疏反演/深度展开、LUNAR 式物理自监督、时序/HMM 或新的相机采集系统；这些保留为后续研究方向。代码不下载第三方模型，不自动启动后台任务。

## 2. 建议的运行顺序

所有命令从项目根目录执行。先用 CPU 检查流程，选择 CUDA 时须使用装有 CUDA 版 PyTorch 的环境；默认 `.venv` 是现有 CPU 环境。原有 VLM 后台训练没有被本次操作停止、重启或占用新训练任务。

```powershell
Set-Location 'F:\CQI\Bell Camera\Net Recog'
.\.venv\Scripts\python.exe -m ionrecog --help
```

### A. 先执行准备好的测试

下面的测试会执行数值计算及微型 CPU 训练，**本次没有运行**：

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_localization_common.py tests/test_psf_fitting.py tests/test_point_localization.py tests/test_equivariant_localization.py
```

通过后可运行完整回归：

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

这些用例涵盖输出保护、坐标变换、重叠光斑、模拟标签、分块坐标、checkpoint 和训练/推理接口。通过合成测试也不能证明真实图像定位更准确；尚未执行的测试可能暴露需要修正的问题。

### B. 联合 PSF 拟合：无需训练，先看一张实图

```powershell
.\.venv\Scripts\python.exe -m ionrecog psf --input data/test --output runs/psf_first --config configs/existing_crystal.yaml --max-images 1 --sigma-x 1.4 --sigma-y 1.4
```

`1.4 px` 仅为默认示例，需与实际光斑匹配。`existing_crystal.yaml` 只针对现有 512×512 相机几何；其他尺寸或光路需对应区域配置。重点检查逐图 JSON 中 `fallback_ions`、每组优化状态和数量变化：输出有圆圈不等于每个圆圈都经过联合拟合。

### C. 直接点源网络：模拟数据自动提供中心标签

先小规模检查训练管线：

```powershell
.\.venv\Scripts\python.exe -m ionrecog points train --output runs/points_smoke --epochs 1 --steps 2 --batch-size 1 --val-samples 2 --channels 4 --device cpu
```

随后按实际光斑、背景、亮度和噪声修改 [simulation.example.json](../ionrecog/point_localization/simulation.example.json)，再启动较完整的训练：

```powershell
.\.venv\Scripts\python.exe -m ionrecog points train --simulation ionrecog/point_localization/simulation.example.json --output runs/points_v1 --epochs 10 --steps 100 --batch-size 4 --val-samples 32 --device cpu
.\.venv\Scripts\python.exe -m ionrecog points infer --input data/test --checkpoint runs/points_v1/best.pt --output runs/points_v1_predictions --config configs/existing_crystal.yaml --max-images 5 --device cpu
```

短训练只检查能否运行，不能当可用权重。模型的固定灰度归一化和模拟条件保存在 checkpoint 中，推理沿用这些条件。阈值由模拟验证集选择，验证分数不是独立测试分数，更不是真实离子准确率。模拟器目前没有真实 EMCCD 电子倍增过程，也不会自动学出实际 PSF。

### D. 几何自监督：先准备单离子裁剪

已预留空目录：

```text
data/isolated_ions/train/session_001/
data/isolated_ions/val/session_101/
```

每张裁剪只含一颗完整孤立离子，不需要中心坐标。默认设置要求每边 5–13 像素；更长的完整光斑需要调整 patch 大小，详见分支 README。同一原图或同构型连拍的裁剪留在同一个划分，不能把多离子整图直接放进去。

数据准备好并确认条件后：

```powershell
.\.venv\Scripts\python.exe -m ionrecog equivariant train --train data/isolated_ions/train --val data/isolated_ions/val --output runs/equivariant_smoke --confirm-single-ion --epochs 1 --steps 2 --val-steps 1 --batch-size 2 --device cpu
.\.venv\Scripts\python.exe -m ionrecog equivariant train --train data/isolated_ions/train --val data/isolated_ions/val --output runs/equivariant_v1 --confirm-single-ion --device cpu
.\.venv\Scripts\python.exe -m ionrecog equivariant infer --input data/test --checkpoint runs/equivariant_v1/best.pt --output runs/equivariant_v1_predictions --config configs/existing_crystal.yaml --max-images 5 --device cpu
```

`--confirm-single-ion` 是对输入数据条件的声明，不会自动验证或隔离光斑。该分支只细化孤立候选；密集邻居、边缘及异常位移会保留 classical 位置并记录原因。初始质心参与验证选模，训练没有改善时保留基线。几何一致性降低不能解释为定位误差降低。

## 3. 输出、比较与资源

推理结果都写到各自独立的 `runs/...` 目录：

- `images/..._ions.json`：浮点中心、方法诊断、晶格区域及覆盖/弃权标记。
- `images/..._ions.csv` 和 `detections.csv`：坐标及原始图像光度测量。
- `images/..._overlay.png`：原图标圈，用于人工检查。
- `summary.json` / `summary.csv`：完成进度、候选数和耗时。逐图详细算法诊断在各自 JSON 中。

坐标为从零开始的 `x=列、y=行`，单位像素。圈半径是显示/测光范围，不是物理离子半径或置信区间。模型 score、拟合分数、光度 SNR 均未校准为正确概率或量子态保真度。

输出默认要求空目录，推理仅在显式 `--overwrite` 时替换同名结果；不会清空其他旧文件。训练没有自动覆盖已有实验。默认保留 2 GiB 磁盘余量，并在处理/训练过程中复查；模型训练应使用独立输出目录，原始图像、VLM 权重及现有结果不会被改写。

方法比较应使用同一独立采集划分、同一晶格区域，报告漏检、多检、中心误差及耗时。对于 PSF/几何分支，还应报告实际拟合/细化比例，避免把大量回退的 classical 结果误认为新算法的效果。没有可靠晶格区域时会弃权，不把空输出解释成物理上没有离子。

## 4. 本次检查记录

已完成：22 个新增/相关 Python 文件按 Python 3.9 语法解析通过；11 个主命令帮助入口及 3 个独立模块帮助入口通过；`git diff --check` 无空白错误。上述检查均不运行模型或图像算法。

测试代码已准备，数值算法、checkpoint 往返、训练收敛、真实数据效果和完整回归均留待后续执行。历史 README 中已有的测试通过数量与 VLM/恢复实验成绩不适用于这三个新分支。
