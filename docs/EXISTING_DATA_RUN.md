# 已有真实数据：masked 与 classical 首轮运行

后续已针对晶格外误检加入原图区域约束与原图噪声门限，见 [晶格区域修复记录](CRYSTAL_REGION.md)。本页保留修复前的原始实验结果，便于比较。

日期：2026-11-03。已完成原图复制校验、20 轮 masked 自监督训练，以及两种方法在相同 263 张测试图上的推理。**当前 masked 恢复图更平滑，但后续检测产生了更多背景候选，不能认定它提高了识别准确率。**

直接打开本地 [完整对照报告](../runs/existing_20261103/comparison/index.html)，可查看同一图片的原图、classical 标圈、masked 标圈和恢复图，另有逐帧 CSV/JSON 链接。所有同帧面板共用原图的显示范围。页面按预定规则均衡、等间隔选取 12 张示例，并列出全部 263 张结果。

## 数据导入

原始数据来自：

- `F:\CQI\Bell Camera\Camera Link\PCIE-Update\captures`：931 张原始 512×512、16 位灰度 PNG，35 个采集目录。
- `F:\CQI\Bell Camera\Recog\CCD-Recognition\single_scan_images`：13 张名称符合原始采集格式的 300×800 uint16 NPY。

未导入 `_noise`、`attenuate_output`、`try`、旧算法输出、带圈截图或像素映射尚未验证的 Camera Delay RAW。原文件未修改，新目录中的每个复制文件都核对了 SHA256，另检查了解码像素哈希，944 张图未发现像素完全重复。采集 JSON/JSONL 同时保留。

| 目录 | 图片数 | 用途 |
| --- | ---: | --- |
| `data/train/ion5_20261027_*/` | 426 | 10 月 27 日前 32 个完整采集组 |
| `data/val/ion5_20261027_172551_412370/` | 250 | 验证和选择 checkpoint |
| `data/test/ion5_20261027_172623_633512/` | 250 | 同次实验后续批次测试 |
| `data/test/legacy_20260116/` | 13 | 旧数据额外测试，未用于训练 |
| `data/diagnostic/ion5_20261023_134104_873492/` | 5 | 不同曝光/放大器设置，单独保留 |

10 月 27 日各组相机快照记录为曝光 0.1 s、EM gain 300、Gain 3、17 MHz。10 月 23 日五帧为约 0.001 s、常规放大器、EM 关闭、3 MHz，未混入训练，也不能在缺少闭快门证据时当作标定暗场。相机设置是采集开始时的 UI 快照，不是逐帧读回。

采用完整采集组时间顺序划分，没有把同一文件夹的相邻帧随机拆开。但是验证末帧和测试首帧仍只间隔 9 个相机 frame ID，属于同次实验的时间留出，**不是跨晶格、跨日期的独立泛化验证**。旧 NPY 的尺寸、背景和信号分布也与训练 PNG 不同，应单独报告。

完整来源和哈希清单在 [data/existing_import_manifest.json](../data/existing_import_manifest.json)。导入脚本可重复运行，只接受已存在且哈希相同的目标文件，不覆盖不同内容。

## 实际运行的配置

使用 [configs/existing_masked.yaml](../configs/existing_masked.yaml)，未根据测试图调参：

- masked 小 U-Net，base_channels=16，64×64 随机块，batch_size=4。
- CPU，4 个线程，seed=42，学习率 0.001，遮挡比例 0.02。
- 20 epoch，每轮 100 个优化步，共 2,000 步、8,000 个抽样训练块；不表示完整遍历每张图的所有像素。
- 每轮 20 个验证步，固定验证随机序列；按最低 `val_proxy_loss` 选择权重。
- offset=0，统一 scale=8426，仅从训练集计算，验证和测试不重新归一化。
- 整图推理使用 256 像素 tile、32 像素重叠。
- 两种方法定位参数相同：sigma=1、background_sigma=10、threshold_sigma=4、min_distance=4、radius=3、min_area=3，没有固定晶格 ROI 限制检测。

Python 3.9.12 / PyTorch 1.13.0+cpu，使用项目已有 `.venv`。没有手工坐标、亮暗状态或分割掩码标签。

## 训练结果与权重

最佳 checkpoint 来自 **第 17 轮**：

| 指标 | 数值 |
| --- | ---: |
| 第 1 轮验证 proxy loss | 0.0009099222 |
| 最佳验证 proxy loss | 0.0004467309 |
| 第 20 轮验证 proxy loss | 0.0006913000 |

第 17 轮优于最后一轮，因此实际推理使用 `best.pt`。这些值衡量遮挡像素预测任务，**不是定位准确率或状态保真度**；随机块损失也可能主要受背景像素支配。

权重：[best.pt](../runs/existing_20261103/masked_training/best.pt)，约 1.47 MB。完整训练记录、固定归一化、训练验证校验和最后一轮权重位于 `runs/existing_20261103/masked_training/`。另有 [training_summary.json](../runs/existing_20261103/training_summary.json)。

## 两种方法的测试输出

classical 与 masked 均完整处理 263/263 张图片并正常退出。保留原图标圈、候选坐标 CSV/JSON、前景掩码；masked 另外保存原始 ADU 尺度的 float32 恢复数组。

下面全部是**候选点数量，不是真实离子数，也不是准确率**：

| 测试组 | 图数 | classical 全图平均候选 | masked 全图平均候选 | classical 显示 ROI 平均候选 | masked 显示 ROI 平均候选 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 10 月 27 日 PNG | 250 | 1349.684 | 1861.440 | 87.572 | 138.368 |
| 1 月 16 日旧 NPY | 13 | 182.615 | 2431.462 | 不适用 | 不适用 |

显示 ROI 为 `[x0,y0,x1,y1)=[180,220,300,430)`，仅对 512×512 图用于放大查看和统计，**不是检测约束，也不是晶格真值边界**。两侧散射光带使整图计数包含大量背景候选。ROI 中也包含暗背景，不能用其计数代替真实离子数。

结果目录：

- [classical](../runs/existing_20261103/classical/)
- [masked](../runs/existing_20261103/masked/)
- [逐帧对照 CSV](../runs/existing_20261103/comparison/comparison.csv)
- [分组统计 CSV](../runs/existing_20261103/comparison/group_summary.csv)
- [示例总览](../runs/existing_20261103/comparison/contact_sheet.png)

## 本次发现的问题

1. **恢复后的检测阈值变得过低。** 两种方法虽然使用同一个 `threshold_sigma=4`，但各自从检测输入估计噪声。首张测试图原图的实际阈值为 92.80 ADU，masked 恢复图为 11.80 ADU。恢复图的高通 MAD 变小，使小幅起伏更容易通过门限。数值见 [threshold_diagnostic.json](../runs/existing_20261103/threshold_diagnostic.json)。恢复后的相关噪声不满足原来白噪声修正的假设。
2. **平滑不等于减少误检。** 首张图中央亮点仍可辨认，masked 却在周围暗背景增加了圈。尚无独立标签，因此不计算召回率或误检率。
3. **极端失败样例需要保留。** 按 classical 显示 ROI 候选数最小这一规则选择的第 66 帧，中央区域看不到清晰离子亮点；classical 在 ROI 输出 0，masked 输出 138。该图是诊断性极端样例，不是代表性抽样，不能据它计算整体性能。见 [失败样例对照](../runs/existing_20261103/comparison/additional_failure_case.png) 和 [选择记录](../runs/existing_20261103/additional_failure_case.json)。
4. **旧 NPY 上存在明显泛化问题。** 可见恢复亮度分块不一致、背景大量画圈。训练和测试场景分布不同，以及分块推理中的归一化，都是需要分别检验的因素；当前不能把原因全部归于某一项。旧图最大值 9416 超过训练归一化上限 8426，一些输入还会被截断。

本次没有为了让报告更好看而删除失败帧、增加手工标签或在测试集上调阈值。原始输出完整保留。后续应优先在训练/验证数据上建立原图噪声标定与候选信号支持约束、验证晶格区域选择，并检查分块强度一致性；随后用新独立测试集评价。现有结果不支持“加去噪网络即可获得更高识别准确率”的结论。

## 可重复执行的命令

```powershell
Set-Location 'F:\CQI\Bell Camera\Net Recog'
# 已经完成；重复导入只校验并复用相同文件。
.\.venv\Scripts\python.exe scripts/import_existing_data.py

# 新一轮运行用新目录，避免覆盖这次记录。
.\.venv\Scripts\python.exe -m ionrecog train --method masked --train data/train --val data/val --config configs/existing_masked.yaml --output runs/existing_v2/masked_training --device cpu
.\.venv\Scripts\python.exe -m ionrecog infer --method classical --input data/test --config configs/existing_masked.yaml --output runs/existing_v2/classical
.\.venv\Scripts\python.exe -m ionrecog infer --method masked --input data/test --checkpoint runs/existing_v2/masked_training/best.pt --config configs/existing_masked.yaml --output runs/existing_v2/masked --device cpu
.\.venv\Scripts\python.exe scripts/compare_runs.py --input data/test --classical runs/existing_v2/classical --masked runs/existing_v2/masked --output runs/existing_v2/comparison
```

如果只是识别新图，不必重新训练：把图放入单独输入文件夹，使用 `runs/existing_20261103/masked_training/best.pt`，另选新的结果目录即可。鉴于上面的失败现象，这个权重目前只适合继续研究和诊断，不能作为已校准的测量工具。

真实图像、相机元数据、训练权重和全部运行产物均保留在本机，并被 `.gitignore` 排除。项目下的导入/比较脚本、配置和这份说明可随代码版本管理。
