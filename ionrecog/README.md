# ionrecog：方案与代码位置

更新日期：2026-11-04。项目包版本仍为 `0.1.0`，见 [__init__.py](__init__.py)。原有三类方案包含 **classical、masked、diffusion、vlm** 四种方法；本次新增三个独立定位目录及 **psf、points、equivariant** 命令。新增代码仅完成语法与接口检查，训练、推理和算法测试留待用户后续执行，尚无新增实验成绩。

**masked 和 diffusion 是两种不同的学习方案。** 把它们放在 `self_supervised` 中，是因为二者共用图像读取、网络骨架、训练管理和后续定位器，并不表示二者的训练目标、数据要求或推理过程相同。

## 1. 方案目录

| 方案 | 源码文件夹 | 运行入口 | 作用与当前状态 |
| --- | --- | --- | --- |
| 方案一：传统图像定位 | [classical/](classical/) | `infer --method classical` | 原图晶格区域分割、多峰检测、亚像素中心和亮度测量；不需要训练 |
| 方案二 A：遮挡像素自监督 | [self_supervised/](self_supervised/) | `train/infer --method masked` | 单张噪声图遮挡部分像素，学习恢复后再定位；已完成现有真实数据的初步训练与推理 |
| 方案二 B：条件扩散恢复 | [self_supervised/](self_supervised/) | `train/infer --method diffusion` | 用独立重复帧均值作为低噪代理目标，条件 DDPM 训练、DDIM 推理；已跑通合成数据流程，尚未在当前真实数据上完成有效性验证 |
| 方案三：视觉语言模型 | [vision_language/](vision_language/) | `vlm` | VLM 根据图块提出中心坐标，再做原图支持检查与细化；支持本地服务、离线 replay、审计和续跑 |
| 方案四：联合光斑拟合 | [psf_fitting/](psf_fitting/README.md) | `psf` | 用局部多离子 PSF 联合拟合细化位置及检查数量；无需训练，需选择符合成像的 PSF 参数 |
| 方案五：直接点源定位 | [point_localization/](point_localization/README.md) | `points train/infer` | 在线模拟图及坐标监督，直接学习中心检测和亚像素偏移；不附带训练权重 |
| 方案六：几何自监督细化 | [equivariant_localization/](equivariant_localization/README.md) | `equivariant train/infer` | 从单离子无坐标裁剪学习几何一致性，细化传统候选；受 LodeSTAR 启发，非官方复现 |

这些目录是研究路线的分类，不是相互独立的历史快照。历史变更由 Git 记录；实验配置、训练权重和运行产物分别位于项目根目录的 `configs/`、`runs/` 等目录。新分支的使用顺序和待执行检查见 [新定位方法指南](../docs/NEW_LOCALIZATION_METHODS.md)。

## 2. 目录结构及文件职责

```text
ionrecog/
├── README.md
├── classical/                    # 方案一的实际实现
│   ├── __init__.py
│   ├── detection.py              # 多峰、最小间距、中心细化、原图噪声门限、画圈
│   ├── region.py                 # 从原图估计晶格包络，排除晶格外候选
│   └── reference.py              # 原图局部测光、可选全亮参考强度比较
├── self_supervised/              # 方案二：两个不同学习分支
│   ├── __init__.py
│   ├── learning.py               # masked/diffusion 训练、采样、checkpoint、恢复
│   └── models.py                 # 共用小 U-Net；扩散时间嵌入、噪声计划、分块推理
├── vision_language/              # 方案三的实际实现
│   ├── __init__.py
│   ├── pipeline.py               # VLM 命令、模型请求、replay、审计、报告、续跑
│   └── geometry.py               # 图块坐标、响应解析、跨块去重、原图中心细化
├── psf_fitting/                  # 新：多离子联合光斑拟合
├── point_localization/           # 新：模拟监督直接点预测
├── equivariant_localization/     # 新：几何自监督候选中心细化
├── localization_common.py        # 新：定位分支输出、原图测光、磁盘检查
├── cli.py                        # 共用命令行，负责调用各类方案
├── data.py                       # 共用原始图读取、分组与强度映射
├── validation.py                 # 共用数据集检查，防止训练验证泄漏
├── evaluation.py                 # 共用独立真值点匹配评估
├── synthetic.py                  # 共用合成演示数据
├── __init__.py / __main__.py     # 包版本与 python -m ionrecog 入口
├── detection.py / region.py / reference.py  # 旧导入路径的兼容入口
├── learning.py / models.py                 # 旧导入路径的兼容入口
└── vlm.py / vlm_geometry.py                 # 旧导入路径的兼容入口
```

**原有三类方案的实际源码位于对应目录。** 根目录保留的同名 `.py` 只有少量转发代码，不是另一份算法副本。修改算法时请编辑对应目录中的实现文件，避免误改兼容入口。

公共数据、校验、评估和命令行仍在 `ionrecog` 根目录，三种路线共用。自监督恢复分支和 VLM 分支也会复用 `classical` 中的定位或测光功能，不重复维护同一段代码。数据图片、模型文件和运行结果没有因本次整理而复制。

## 3. masked 与 diffusion 的具体区别

| 项目 | masked | diffusion |
| --- | --- | --- |
| 训练图片 | 可以是不同晶格的无标注单帧 | 每组至少 3 张同一稳定场景的独立重复帧 |
| 目标 | 预测被遮挡位置的原始噪声像素 | 学习其他重复帧均值的条件分布，训练预测加入的扩散噪声 |
| 输入目标隔离 | 不把被遮挡的原始值复制回模型输入 | 输入帧不能参与自身平均目标 |
| 模型 | 单通道小 U-Net | 输入含噪扩散状态与观测图，加入时间嵌入的条件 U-Net |
| 推理 | 一次网络恢复，整图可分块 | 从随机噪声开始，多步 DDIM 恢复 |
| 主要代码 | `learning.py: mask_pixels / _batch_loss / Restorer` | `learning.py: choose_repeat_paths / make_repeat_pair / _batch_loss / Restorer`；`models.py: diffusion_schedule` |

共同入口是 [self_supervised/learning.py](self_supervised/learning.py) 的 `train_model(method, ...)`、`load_restorer(...)` 和 `Restorer.restore(...)`。`method` 明确区分两种方案，checkpoint 中也保存了该字段，不能把 masked 权重作为 diffusion 权重使用。

二者的损失定义不同，训练损失数值不能直接横比。都不能仅凭恢复图更清晰就认定定位更准确。扩散均值目标仍有残余噪声，也不能自动消除微运动模糊或标定量子态。

## 4. 传统定位中的两个配置状态

方案一当前包含历史整图基线和后来加入的晶格区域约束，它们共用同一套代码，通过配置选择：

- `region.mode: off`：历史整图基线，便于对照。
- `region.mode: auto`：在搜索范围内，从每张原图提取自适应晶格边界，再限定峰选择与最终中心。
- `inference.noise_reference: raw`：恢复图也用原图噪声来确定定位门限，防止去噪后的低残余噪声放大背景误检。

现有 512×512 相机推荐 [configs/existing_crystal.yaml](../configs/existing_crystal.yaml)。粗搜索框是一次设备/视场先验，不是逐离子标签；改变相机尺寸、晶格大小或光路后需重新设置。`no_crystal` 只表示没有可靠区域，不是“物理上没有离子”。详见 [晶格区域修复说明](../docs/CRYSTAL_REGION.md)。

## 5. 运行命令保持不变

从项目根目录执行，不要切换到方案子文件夹中直接运行单个 `.py`：

```powershell
Set-Location 'F:\CQI\Bell Camera\Net Recog'
.\.venv\Scripts\python.exe -m ionrecog --help
.\.venv\Scripts\python.exe -m ionrecog train --help
.\.venv\Scripts\python.exe -m ionrecog infer --help
.\.venv\Scripts\python.exe -m ionrecog vlm --help
```

方案一，当前相机数据：

```powershell
.\.venv\Scripts\python.exe -m ionrecog infer --method classical --input data/test/ion5_20261027_172623_633512 --config configs/existing_crystal.yaml --output runs/layout_example/classical
```

方案二 A，直接使用已有 masked 权重：

```powershell
.\.venv\Scripts\python.exe -m ionrecog infer --method masked --input data/test/ion5_20261027_172623_633512 --config configs/existing_crystal.yaml --checkpoint runs/existing_20261103/masked_training/best.pt --output runs/layout_example/masked --device cpu
```

方案二 B，只有确认 train/val 各组为稳定重复帧后再运行：

```powershell
.\.venv\Scripts\python.exe -m ionrecog train --method diffusion --train data/train --val data/val --config configs/default.yaml --output runs/layout_example/diffusion_training --confirm-static
```

现有数据的目录分组本身不保证组内位置/状态不变；不能仅为让命令运行而加 `--confirm-static`。完整要求见 [项目 README](../README.md)。

方案三的本地模型准备、manifest、实际 API 推理与 replay 使用方法见 [VLM 指南](../docs/VLM_PILOT.md)、[VLM 运行环境](../docs/VLM_RUNTIME.md) 与 [已有 VLM 结果](../docs/VLM_RESULTS.md)。整理源码不会启动模型服务、下载模型或调用 API。

## 6. 导入兼容、旧权重与 VLM 续跑

推荐的新导入路径：

```python
from ionrecog.classical.detection import detect_ions
from ionrecog.classical.region import segment_crystal
from ionrecog.self_supervised.learning import train_model, load_restorer
from ionrecog.vision_language.pipeline import run_vlm
```

旧路径继续有效，例如 `from ionrecog.detection import detect_ions` 和 `from ionrecog.learning import train_model`。兼容入口指向同一个实现模块，保留原有私有函数及 monkeypatch 行为；不是复制一份代码后分别维护。

现有恢复 checkpoint 保存的是模型参数 `state_dict` 和配置，本次没有改变网络层结构或参数名，因此已训练的 masked/diffusion 权重可以继续加载。

VLM 运行指纹会记录实际实现源码的 SHA256。目录调整同时修改了导入和运行时查找代码，因此**整理前已保存的 VLM 结果仍可查看，但不能直接对其使用 `--resume`**；新任务使用新输出目录。整理后创建的任务仍支持相同参数下的续跑。不会为了兼容目录移动而跳过原图、模型和源码的一致性校验。

## 7. 历史记录：2026-11-03 目录整理范围与验证

2026-11-03 的目录整理仅涉及源码组织、导入转发、VLM 文件查找与指纹路径、文档索引；当时没有修改定位、恢复或 VLM 提示的算法逻辑。以下测试数量属于该历史整理，不是新增定位分支的验证结果。

验证结果：完整测试 **188 passed（6.15 秒）**；7 个旧模块路径与新模块为同一对象；主命令和 VLM 帮助正常；已有 masked 权重处理同一张 512×512 原图的结果与整理前逐像素一致，最大差值为 0。磁盘空间主要只增加这个 README、少量入口文件及 Python 缓存，图片、旧报告、权重及模型文件均使用原位置。

## 8. 2026-11-04 新定位分支交付

新增三个目录、共用结果导出和测试源代码。按用户要求，只进行语法编译及命令帮助入口检查；未运行新增模型的训练、识别或数值算法测试。模型默认使用 CPU，GPU 需显式选择。原有 VLM 后台训练不因新分支创建而停止或重启。

使用命令、数据需求、尚未执行的测试及方法局限见 [新定位方法指南](../docs/NEW_LOCALIZATION_METHODS.md)。
