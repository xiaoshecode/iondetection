# 晶格区域约束：去除散射光带和晶格外候选

日期：2026-11-03。本次针对用户指出的“只有晶格内的候选才有效”问题，修改定位流程，并对现有 250 张 512×512 测试图重新运行 classical 和 masked。原始图、此前的输出和训练权重均保留，未重新复制数据或训练网络。

直接查看 [新版完整对照报告](../runs/crystal_region_v1/comparison/index.html) 或 [首张图修复前后](../runs/crystal_region_v1/before_after_first_frame.png)。蓝色边界表示从原图估计出的有效晶格范围。

## 修改了什么

原流程在整张图中找局部峰，无法排除两侧亮散射光带；masked 还会因为恢复后的高频噪声变小，把检测门限降得过低。

当前流程：

```text
原始采集图 ──→ 粗搜索范围内的自适应晶格区域 ───┐
          └──→ 原图噪声估计 → 同一 ADU 门限 ─┤
原图 / masked 恢复图 ──→ 区域内多峰与质心定位 ─→ 中心范围检查 → 输出
```

1. **搜索范围只是一项光路先验。** 现有相机配置使用 `[x0,y0,x1,y1)=[180,220,300,430)`，排除已知散射光带。不是给每个离子标注，也不会把整个矩形当作有效区域。
2. **每张图独立从原图估计边界。** Gaussian 包络与慢变背景相减，使用低阈值支持、高阈值种子、连通分量、闭运算和填孔，选择一个有足够原图信号的晶格范围，并保留 3 像素边缘余量。学习模型不能决定晶格在哪里。
3. **两种方法使用完全相同的区域。** 掩码在连通区域和峰选择之前生效；最终亚像素中心也校验是否在范围内。没有找到可靠晶格时返回空掩码，状态为 `no_crystal`，不会退回整图或粗矩形画圈。
4. **恢复图定位仍使用原图噪声门限。** `inference.noise_reference: raw` 防止去噪后很小的起伏被自动当成显著峰。首张图两种方法的门限均为约 92.80 ADU，旧 masked 门限只有约 11.80 ADU。

实现文件：`ionrecog/region.py`、`ionrecog/detection.py` 与 `ionrecog/cli.py`。这是无需逐离子标注的传统区域分割约束，**不是新训练的分割网络或经过真值验证的实例掩码**。

## 本次验证结果

| 检查 | 结果 |
| --- | --- |
| 重跑范围 | 250 张现有 512×512 测试 PNG，两个方法各处理 250 张 |
| classical / masked 区域 | 逐图像素完全一致 |
| 输出中心位于估计晶格范围外 | 两种方法均为 0；报告生成时逐点校验 |
| 找到可靠区域 | 237 张 |
| 未找到可靠区域 | 13 张；两种方法均输出 0 个候选 |
| 首张 classical 候选 | 1363 → 105 |
| 首张 masked 候选 | 1858 → 105 |
| 第 66 帧中央低信号失败样例 | 旧 masked 在显示 ROI 输出 138 个候选；现在区域为 `no_crystal`，两种方法均输出 0 |
| 新 classical 全图平均候选 | 87.340 |
| 新 masked 全图平均候选 | 85.048 |

这些数量是算法输出，不是真实离子总数或识别准确率。区域外为 0 是明确实现并校验的范围约束；它不能证明区域内所有点都对，也不能证明边缘弱离子没有遗漏。`no_crystal` 表示证据不足，**不是“离子数为零”或“离子处于暗态”的物理结论**。

区域参数先在训练/验证样例与背景诊断图上检查，再运行上述 250 张图。当前修复使用已知光路的粗搜索范围，测试集之前也已被查看，因此这轮结果应视为工程诊断与修复验证，不能作为未见测试集上的最终论文结果。

合成回归测试覆盖：强散射光带排除、外围离子中心保留、无晶格返回空区域、相机尺寸不匹配、掩码边界内定位、原图门限抑制弱恢复伪峰、磁盘不足时不创建输出。执行时当前整个工作区测试为 **185 passed**；该总数包含工作区的其他实验分支测试。

## 如何运行

当前相机配置：[configs/existing_crystal.yaml](../configs/existing_crystal.yaml)。输入需直接指向 512×512 数据组，避免与旧 300×800 NPY 混用像素坐标。

```powershell
Set-Location 'F:\CQI\Bell Camera\Net Recog'

.\.venv\Scripts\python.exe -m ionrecog infer --method classical --input data/test/ion5_20261027_172623_633512 --config configs/existing_crystal.yaml --output runs/crystal_region_v2/classical

.\.venv\Scripts\python.exe -m ionrecog infer --method masked --input data/test/ion5_20261027_172623_633512 --config configs/existing_crystal.yaml --checkpoint runs/existing_20261103/masked_training/best.pt --output runs/crystal_region_v2/masked --device cpu

.\.venv\Scripts\python.exe scripts/compare_runs.py --input data/test/ion5_20261027_172623_633512 --classical runs/crystal_region_v2/classical --masked runs/crystal_region_v2/masked --output runs/crystal_region_v2/comparison --examples 6
```

`v1` 是已经生成的本次结果；示例使用 `v2`，防止覆盖。无需重新训练 masked。新掩码、原图门限也可用于 diffusion 模式，但本次未重跑 diffusion。

新增输出：

- `*_crystal_mask.png`：原图估计的有效晶格掩码，区别于局部峰的 `*_foreground.png`。
- `*_ions.json` 中 `region`：搜索框、实际边界框、面积、状态、阈值和失败原因。
- `noise_reference` 与 `detection_threshold_adu`：定位所依据的噪声图和实际门限。
- `summary.csv` 中区域状态与面积。新版对照报告校验两种方法掩码一致，且不允许任何导出的中心落在掩码外。

## 新相机或更大的晶格

需要按新光路设置一个宽松的 `search_roi`，并更新 `expected_shape: [高度, 宽度]`。这是一次设备/视场设置，不是逐帧人工标注。相机尺寸不匹配会报错，避免悄悄套用错误坐标。

`search_roi: null` 可启用全图自动搜索，但陌生背景中的紧凑亮结构可能被误认为晶格，不能保证完全自动区分所有光学杂散结构。默认只选取一个最可信的连通晶格；多晶格、断裂晶格、信号极弱或靠近搜索框边缘时需要另行验证。若 `touches_search_boundary=true`，应检查并适当扩大搜索范围。

`padding` 是边缘留量，`min_contrast`/`seed_sigma` 控制最小原图证据，`support_fraction`/`support_sigma` 控制范围。降低门限可能保留弱边缘，也可能纳入背景；不能仅凭圈数更多来选参数。通用配置的 `region.mode: off` 保留历史整图基线。要复现旧恢复图门限，另设 `inference.noise_reference: processed`；不推荐作为当前相机的实际定位配置。

## 磁盘空间

批量推理开始前检查目标磁盘，按源文件大小估算图像、JSON、掩码及可选恢复数组的输出需求，默认额外预留 **2 GiB**。处理中每 25 张再次检查余量，不足则停止并保留已完成的结果。可用 `--reserve-gib` 调整；预估不是严格文件大小上限，也无法控制其他程序同时占用磁盘。

本次预估 classical 约 0.50 GiB、masked 约 0.78 GiB；实际整套新运行和对照报告约 **483 MiB**，小型开发检查另约 **1.6 MiB**。只生成 6 张报告示例，没有复制原始数据、重新训练或删除已有结果。详细空间记录见 [space_and_verification.json](../runs/crystal_region_v1/space_and_verification.json)。
