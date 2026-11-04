# CCD 原始图查看与识别工具

本目录包含二维 NPY 图像查看器，以及调用项目现有 classical / masked 算法的适配与结果对照脚本。算法仍位于 `../ionrecog/`。

Git 跟踪脚本、说明及选定的 [267 张图像识别总结](recognition_20261104_20260403/comparison/README.md)，包括逐帧统计、7 组示例图和运行记录。原始采集目录、`previews/`、全量逐帧识别产物及缓存保留在本地，不随仓库克隆。模型权重同样不在 Git 中。

## 本地数据与依赖

本机采集文件位于 `20260403_2000pure/`（267 张）和 `20260404_2000pure/`（1277 张）。新电脑需另行复制原始 NPY 数据到对应目录。

- 查看器需要 NumPy 和 Matplotlib（Matplotlib 是查看器的额外依赖，未列入项目 `requirements.txt`；需支持 `RangeSlider`）。
- 识别与汇总使用项目 Python 环境及 `../requirements.txt` 中的依赖。
- `run_recognition.py` 同时执行 classical 和 masked，必须先准备 `../runs/existing_20261103/masked_training/best.pt`，并至少留出 5 GiB 可用空间；脚本不会自动下载权重或训练。

## 查看原始图

以下命令从 `Net Recog` 项目根目录执行；Python 环境需已安装上述依赖：

```powershell
.\.venv\Scripts\python.exe -B CCD_2000ion\view_ion.py CCD_2000ion\20260403_2000pure
```

也可双击 `查看离子图.bat`，它使用 PATH 中的 `python`。查看器支持逐帧浏览、缩放、显示范围调节和 PNG 预览导出；显示调整不会改写原始 NPY。详细操作见 [查看说明.txt](查看说明.txt)。

## 运行现有识别与对照

```powershell
.\.venv\Scripts\python.exe -B -u CCD_2000ion\script\run_recognition.py --input CCD_2000ion\20260403_2000pure --output CCD_2000ion\recognition_new_run
.\.venv\Scripts\python.exe -B CCD_2000ion\script\summarize_recognition.py --run-dir CCD_2000ion\recognition_new_run --input CCD_2000ion\20260403_2000pure --offset 1161 --expected-images 267
```

输出目录必须尚不存在。识别脚本对整批有符号 NPY 使用共同偏移，再四舍五入为 16 位 PNG；不缩放、不裁零、不逐帧归一化，并记录转换清单与文件完整性核对。汇总命令中的 `--offset` 必须与该次 `conversion_manifest.json` 一致；1161 仅为本机 20260403 批次的值。

汇总脚本针对当前 201×1024 数据编写，默认预期 267 张，报告文字包含该批次和配置说明；不应直接当作适用于任意相机、尺寸或配置的通用报告器。对照结果表示两方法的候选一致性，不是有独立真值的准确率，文件夹名中的 2000 也不是逐帧真值。

参数、强度尺度和操作细节见 [script/使用说明.txt](script/使用说明.txt)。
