# VLM 离子候选定位：实现、运行与结果解释

本分支把视觉语言模型作为候选中心生成器。输入原始灰度图的显示副本，输出坐标，再在原始数值图上做局部质心细化与信号检查。它独立于 masked/diffusion 恢复流程，不需要原有恢复 checkpoint。

**目前实现的是零样本坐标定位实验，不是经过离子数据微调的模型。原图支持也不是准确率、量子态标签或已标定概率。** 本机实际结果另见 `VLM_RESULTS.md`（实验完成后生成）。

## 环境

本机使用 Qwen3-VL-4B-Instruct Q4_K_M，经独立 llama.cpp Vulkan 服务推理。详细下载来源、固定版本、SHA256、硬件与服务命令见 [VLM_RUNTIME.md](VLM_RUNTIME.md)。图像不上传外部服务；已有 `.venv` 不需安装 transformers 或更换 PyTorch。

```powershell
Set-Location 'F:\CQI\Bell Camera\Net Recog'
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\start_vlm_server.ps1 -Background
Invoke-RestMethod 'http://127.0.0.1:8087/health'
```

服务就绪后使用原项目 `.venv\Scripts\python.exe`。启动脚本打印 PID 和日志位置；结束后只停止该 PID，释放显存。新的电脑需按运行环境文档准备模型与便携运行时。

## 一、准备固定样本

```powershell
.\.venv\Scripts\python.exe scripts/prepare_vlm_pilot.py --output runs/my_vlm_preparation
```

默认选择32张：3张验证集校准、16张测试 PNG、8张旧 NPY、4张其他设置诊断、1张此前失败诊断。每张记录原图绝对路径、SHA256、采集组、尺寸、选择原因和用途。原始图不复制、不修改，也不把同一模型的预测冒充真值。

完整图像按128×128、重叠32分块需792次请求。为了先控制时间，本轮使用 `--study-regions` 生成固定区域实验：512图共享历史显示 ROI `[180,220,300,430)`，旧300×800图使用中心128像素横带。32张共112块。这些范围是显式先验，不是 VLM 自动寻找的结果，也不能把区域结果称为完整晶格召回率。

```powershell
.\.venv\Scripts\python.exe scripts/prepare_vlm_pilot.py --study-regions --output runs/my_vlm_regions
```

仅准备图块与经典算法对照，不调用模型：

```powershell
.\.venv\Scripts\python.exe -m ionrecog vlm --manifest runs/my_vlm_regions/manifest.json --output runs/my_vlm_preview --backend dry-run
```

`prepared_not_inferred` 表示尚未推理。对应的0个 VLM 候选只是缺少结果，不是模型判断无离子。查看 `plan.json` 的 `planned_requests`，先估算模型耗时再扩大规模。

## 二、实际 VLM 推理

```powershell
.\.venv\Scripts\python.exe -m ionrecog vlm --manifest runs/my_vlm_regions/manifest.json --output runs/my_vlm_run --backend api --model ion-qwen3-vl-4b
```

先仅检查校准组可加 `--roles calibration`，输出用独立目录。可用 `--max-requests 6` 限制本次新增请求；其余图块保持 pending。继续运行时使用相同参数和 `--resume`，可提高或去掉 `--max-requests`。更换模型、提示、图块大小、筛选阈值、原图或代码后必须新建运行目录。

```powershell
.\.venv\Scripts\python.exe -m ionrecog vlm --manifest runs/my_vlm_regions/manifest.json --output runs/my_vlm_run --backend api --model ion-qwen3-vl-4b --resume
```

成功图块的响应会复用，失败图块可以重试。原始响应、实际服务返回的模型名、token用量和耗时均保存。非JSON、越界坐标、截断输出视为失败，不能当作空预测；`uncertain=true` 的空列表会记录为模型弃权。

## 三、输出与公平对照

打开输出目录的 `index.html`。每张图六个面板：

1. 原图；
2. 相同研究区域的 classical；
3. 相同分块与去重策略的 classical；
4. VLM 直接输出的去重坐标；
5. 原图支持且通过位移/边界检查的细化候选；
6. 所有未通过检查或模型不确定的候选。

面板使用相同原图显示范围，质心细化仍读取原始16位数据。相同分块的经典对照用于区分“裁剪本身的收益”和模型收益。VLM默认半径3px、最大细化位移2px、合并距离2px、支持度阈值5，均是可配置初始参数，未在真实测试真值上校准。

```text
run/
  plan.json                  原图hash、图块坐标、模型/提示/代码hash
  summary.json / summary.csv 状态、候选数与耗时，不是准确率
  disk_space.json            运行前后剩余空间及输出体积
  index.html                 本地对照报告
  images/<image_id>/
    tiles/<tile_id>.png       实际送给模型的图块
    tiles/<tile_id>.json      完整原始回复、解析结果与完成状态
    classical_ions.json/csv
    classical_tiled_ions.json/csv
    vlm_direct_ions.json/csv
    vlm_checked_ions.json/csv
    uncertain.json            未接受候选及具体原因
    comparison.png
```

`x`为列、`y`为行，从0起计。模型使用0–1000端点坐标：`x=x0+u*(width-1)/1000`。放大采用最近邻，只帮助观看，不增加光学分辨率。每个保留点都有去重来源和原始提议；局部细化使用最近候选的像素归属，减少强邻点吸附。它不是联合PSF拟合，不能解决所有拖尾/重叠。

`supported_candidate`仅表示启发式原图支持。局部背景环可能被邻居污染，弱点可能被过滤，固定距离也可能合并真实近邻。因此同时保留direct与checked；不能只展示看起来更整齐的一组。

## 四、带已知坐标的合成检查

```powershell
.\.venv\Scripts\python.exe scripts/prepare_vlm_synthetic.py --output runs/my_vlm_synthetic_preparation
.\.venv\Scripts\python.exe -m ionrecog vlm --manifest runs/my_vlm_synthetic_preparation/manifest.json --output runs/my_vlm_synthetic --model ion-qwen3-vl-4b --display-low 100 --display-high 1100
.\.venv\Scripts\python.exe scripts/evaluate_vlm_synthetic.py --run runs/my_vlm_synthetic
```

四张图包含清晰9点、近邻2点、包含弱点的4点、纯背景0点。真值直接来自生成中心，独立于识别算法。固定跨帧显示范围避免把纯背景噪声拉满。

评估使用1px与3px容差的一对一匹配，报告TP/FP/FN、F1及定位误差。未完成或弃权结果单独标记，不静默删除困难样本后宣称整体准确率。**简单合成结果不能外推成真实离子图准确率。** 真实数据的独立中心真值仍需另行建立；不清楚的位置允许不可判定。

## 五、资源控制与外部结果

默认 `--min-free-gib 5`，在写图块前检查磁盘，空间不足就停止，保留已完成响应。没有原图副本、恢复数组或全局模型缓存。正常运行不覆盖其他结果，不自动清理原始数据或已有权重。

接口支持显式设置 `--base-url` 和 `--model`；本机默认仅访问loopback。远端服务需要用户自行选择端点，密钥通过 `--api-key-env 环境变量名` 读取，不写入配置/报告。`pixels`坐标仅允许离线replay，防止API的归一化提示与解析方式不一致。

`--backend replay --responses path/to/responses.json` 可导入外部坐标实验，格式是以`image_id/tile_id`为键的字典，值含`text`和`finish_reason: "stop"`，例如：

```json
{"image_abc/0": {"text": "{\"points\":[[250,500]],\"uncertain\":false}", "finish_reason":"stop", "response_model":"external-method-name"}}
```

导入必须使用已知图块和明确坐标制，记录外部模型与来源；这不是调用模型、也不是训练标签验证。CountGD++的实际对照需要其真实权重和运行环境，不能用classical伪装成CountGD++。当前主分支未把VLM输出自动当真值训练下一代模型。
