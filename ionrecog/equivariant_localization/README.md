# 几何自监督候选中心细化

这是受 [LodeSTAR](https://www.nature.com/articles/s41467-022-35004-y) 几何自蒸馏思想启发的小型实验分支，**不是官方 LodeSTAR 复现，也不是独立的密集离子检测器**。当前方案用已有 classical 提供候选，再用单离子裁剪训练的网络细化孤立候选的位置。它不会找回 classical 漏掉的离子，不会改变候选数量；相邻候选过近、图像边缘或细化位移过大时保留原 classical 坐标，并输出原因计数。

当前交付代码、测试及运行说明。尚未执行本分支训练、推理或模型测试；不附带训练好的权重，不宣称实验准确率改善。

## 原理与边界

每个没有坐标标签的单离子裁剪生成多个旋转、翻转、整数平移视图，网络预测各视图中心，再按精确的坐标变换映回原图，最小化这些预测的差异。训练加入少量独立噪声扰动，鼓励对扰动稳定。网络只学习重新加权前景像素；非负信号支持及权重下限限制固定中心退化，初始模型等价于前景质心。

这里使用离散的 90° 旋转/翻转和平移，避免任意角度插值把坐标变换误差混入第一版训练。它没有实现 LodeSTAR 的局部向量投票、整图多目标检测或任意角度旋转训练。

**几何一致性低不等于位置准确。**均匀重加权的质心也可能很一致；非对称拖尾中的光学中心未必是离子平衡位置。候选间距检查依赖 classical 已发现的邻居，无法识别未检出的干扰离子。需要独立参考评价，不能用验证一致性或 classical 一致性当真实准确率。

## 数据

在项目根目录准备：

```text
data/isolated_ions/train/session_001/*.png
data/isolated_ions/val/session_101/*.png
```

每个文件必须是原始尺度的二维灰度 PNG/TIFF/NPY，含**一颗完整、孤立的离子**，不用标中心。不要放整张晶格、截图或画过圈的图。

默认 `--patch-size 17 --max-shift 2` 要求输入每边 5–13 像素；程序只居中填充背景，不缩放、不自动截断目标。如果完整拖尾放不下，可同时加大 patch，例如 `--patch-size 33 --max-shift 3` 允许每边 5–27 像素。较大窗口在密集晶格中会导致更多候选因邻居过近而跳过细化，这是预期限制。

同一原图或同一晶格连拍得到的裁剪全部留在一个划分。程序检查训练/验证输入路径不重叠，并用像素 SHA-256 拒绝跨集合复制；它无法自动知道不同裁剪是否来自同一原图，因此仍须按采集来源组织数据。

## 明天执行的命令

所有命令在 `F:\CQI\Bell Camera\Net Recog` 执行。默认 CPU，不抢占 VLM 使用的 GPU。训练前需检查单离子条件，`--confirm-single-ion` 只是你的数据条件声明，不能把密集图变成合法输入。

```powershell
Set-Location 'F:\CQI\Bell Camera\Net Recog'
.\.venv\Scripts\python.exe -m ionrecog equivariant train --train data/isolated_ions/train --val data/isolated_ions/val --output runs/equivariant_v1 --confirm-single-ion --device cpu
```

建议先做小规模管线检查，再正式训练：

```powershell
.\.venv\Scripts\python.exe -m ionrecog equivariant train --train data/isolated_ions/train --val data/isolated_ions/val --output runs/equivariant_smoke --confirm-single-ion --epochs 1 --steps 2 --val-steps 1 --batch-size 2 --device cpu
```

小规模检查只验证能否执行，不能说明训练收敛。每轮按固定验证裁剪和固定验证变换评估，输出 `best.pt`、`last.pt`、`history.json`、`progress.json`、`summary.json`、`config.json` 和源裁剪哈希清单 `dataset.json`。初始前景质心作为 epoch 0 对照，也参与最佳权重选择；如果训练没有改善一致性，`best.pt` 保留初始基线，汇总中的 `improved_over_initial_centroid` 为 false。没有断点续训功能；`last.pt` 是推理模型快照，不包含优化器或采样器状态。

推理使用原始图，模型的强度尺度固定来自训练集，候选配置和晶格边界复用已有 YAML：

```powershell
.\.venv\Scripts\python.exe -m ionrecog equivariant infer --input data/test --checkpoint runs/equivariant_v1/best.pt --config configs/existing_crystal.yaml --output runs/equivariant_v1_predictions --device cpu --max-images 5
```

`configs/existing_crystal.yaml` 只适用现有 512×512 相机几何，其他图像需对应配置。输出为 `detections.csv`、各图 `*_ions.json/CSV`、原图叠圈、汇总与诊断。每个候选包含 `proposal_x/y` 和 `refined`，诊断包含实际细化数量及邻居、边缘、位移、区域检查失败数量。原图测光由公共模块另算，不能把旧 proposal 的峰值等当作网络预测。

`--max-refine-shift` 默认 2 px；`--isolation-distance` 默认按训练裁剪大小保守设置。降低间距门限可能让邻居污染单目标网络，必须作为独立实验记录。

只有显式执行命令才训练或推理；没有自动下载、后台启动或导入时执行。模型载入使用 PyTorch 本地序列化，只使用可信 checkpoint。训练与推理均保留默认 2 GiB 磁盘余量；输出不可覆盖输入数据。

## 待执行测试

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_equivariant_localization.py
```

测试包括坐标变换、固定中心退化检查、训练验证复制检查、checkpoint 类型/尺寸校验、微型 CPU 训练、空图/区域限制和候选细化。**这些测试包含模型运行，按本次要求留待明天执行。**
