# 多离子联合 PSF 拟合

本目录实现 CPU 上的**固定光斑模型联合定位基线**。无需训练或下载权重。代码已准备；按本次要求，尚未执行识别、数值测试或实图验证。下面的命令留待下一次测试。

入口：`python -m ionrecog psf`；等价独立入口：`python -m ionrecog.psf_fitting`。

## 如何工作

1. 复用 classical 的晶格区域和候选点。
2. 按距离将相互影响的候选组成局部组。
3. 在局部原始 ADU 图像上联合拟合中心、非负强度和常数背景。
4. 在预算内比较背景、合并／删点以及正残差补点解释。
5. 写出中心、原图测光、叠图和逐组拟合诊断。

光学模型为旋转椭圆高斯，可叠加**已知振幅和方向**的正弦运动时间平均。`sigma_x/y`、光学角度和运动参数在本次运行中固定；代码不会从单帧自动分离像差和微运动，也不会自动标定相机。

本版本采用像素中心采样，尚未做精确像素面积积分。高斯 sigma 必须至少 0.5 px；强欠采样、高饱和、非对称光斑以及空间变化 PSF 需要后续模型适配。输出中心是指定 PSF 的中心；对对称运动，是其时间平均位置。

## 明天从一张图开始

在项目根目录执行。把 `--input` 换成实际原始图路径；已有相机的 512×512 图使用现有晶格配置。

```powershell
Set-Location 'F:\CQI\Bell Camera\Net Recog'
.\.venv\Scripts\python.exe -m ionrecog psf `
  --input 'data/test' `
  --output 'runs/psf_first' `
  --config 'configs/existing_crystal.yaml' `
  --max-images 1 `
  --sigma-x 1.4 --sigma-y 1.4 `
  --reserve-gib 2
```

`1.4 px` 是演示初值，不是本机已测 PSF。应先从清晰、孤立光斑估计合适参数，再判断拟合是否有效。不同相机尺寸／光路不能直接沿用 `existing_crystal.yaml` 中的搜索框。

若已通过实验得到运动参数，可尝试：

```powershell
.\.venv\Scripts\python.exe -m ionrecog psf `
  --input 'data/test' --output 'runs/psf_known_motion' `
  --config 'configs/existing_crystal.yaml' --max-images 1 `
  --sigma-x 1.4 --sigma-y 1.4 `
  --motion-amplitude 2.0 --motion-angle 0 --motion-samples 16
```

此处 `2.0 px` 仅示例。`--psf-angle` 描述光学椭圆方向；`--motion-angle` 描述运动方向，两者以图像 x 轴为零、向图像 y 轴旋转。运动振幅是相对中心的单侧振幅，而非总拖尾长度。

## 边界、预算和回退

| 参数 | 默认 | 含义 |
| --- | --- | --- |
| `--group-distance` | 6 px | 将距离较近的初始候选连成一组 |
| `--max-emitters` | 5 | 一组允许的最大离子数；超出则回退 |
| `--max-roi-pixels` | 4096 | 单组最大拟合区域像素数 |
| `--max-fit-groups` | 256 | 一帧最多尝试多少组 |
| `--max-nfev` | 120 | 单次最小二乘优化的迭代预算 |
| `--max-models` | 12 | 每组最多评估多少个模型 |
| `--selection-rounds` | 3 | 最多几轮局部模型搜索 |
| `--min-delta-score` | 6 | 改换模型需要的分数改善量 |
| `--noise-std` | 自动估计 | 原始 ADU 中的常数噪声标准差 |

`--max-nfev` 是 SciPy 的函数评估预算；数值雅可比可能调用额外的残差计算，不能把它解读为严格的 CPU 时间上限。光斑外侧使用至少“运动振幅＋4 倍最大光学 sigma”的 halo。若其他候选落入本组 halo 而未被联合建模，该组也回退，以避免把邻居的光错误归给本组。

密集晶格可能连成大组，本版本会保留该组的 classical 候选，并显式标记 `status=classical_fallback` 和原因，不进行千离子无界联合优化。较大的 `group-distance` 并不一定更好；需检查输出中的 `fallback_ions` 和每组原因。大组分块联合建模、未知 PSF 学习不在此版本范围。

相邻组的 halo 可能分别从残差补出同一个点。最终会对**不同来源组**的坐标作一对一去重，距离严格小于 `--min-separation`（默认 0.75 px）才匹配；优先保留成功拟合、离 ROI 边缘更远、标准化残差更低的结果，不平均坐标。每个保留点最多接收同一来源组的一个候选，避免一个居中假设吞掉另一组已经分开的两个点；不做传递式聚类。距离达到或超过阈值的点不会直接合并。记录 `cross_group_duplicates_removed` 及逐点详情，这也是启发式规则，阈值应结合预期最小物理间距验证。

模型比较使用 `RSS / noise_std² + 参数数 × log(像素数)` 的 Gaussian BIC 风格分数。真实 EMCCD 的噪声通常不是常方差高斯，因此该分数只作启发式选择，**不是严格 EMCCD 似然、贝叶斯证据或正确概率**。允许空模型拒绝无信号候选，但不能推断暗态与离子丢失。

## 输出与 Python 接口

- `images/..._ions.json`：坐标、背景、拟合状态、模型比较及回退信息。
- `images/..._ions.csv`、`detections.csv`：坐标与原图测光。
- `images/..._overlay.png`：绿色圆圈与估计晶格边界。
- `summary.json`：运行设置、完成进度、每帧统计。

`amplitude_adu` 是模型中未运动展宽高斯的峰值幅度；它不是实际像素峰值、光子数或量子态概率。量化测光以公共输出的 `raw_flux/raw_peak` 等字段为准，但这些也尚未作态读出标定。

```python
from ionrecog.psf_fitting import PSFConfig, fit_patch, locate_ions

cfg = PSFConfig(sigma_x=1.3, sigma_y=1.6, noise_std=8.0)
# 小块内所有坐标相对该图块；seeds 可以是 [[x,y], ...] 或坐标字典。
result = fit_patch(raw_patch, [[12.2, 14.1], [15.0, 14.4]], cfg)
# 全图流程内部生成 classical 候选，并保留各类回退诊断。
result = locate_ions(raw_image, cfg, detection_config=None, region_mask=None)
```

## 待执行测试

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_psf_fitting.py
```

准备的测试覆盖亚像素重叠点、已知运动双峰合并、背景拒绝、无晶格区域、大组回退、优化失败、交叠 halo 的残差重复点去重和参数校验。这些用例尚未运行；即使通过也只是合成条件验证，不代表真实离子识别准确率。

方法背景：[多发射体联合拟合研究](https://pmc.ncbi.nlm.nih.gov/articles/PMC3087594/)。本目录为本项目的简化实现，不宣称复现该论文的算法或结果。
