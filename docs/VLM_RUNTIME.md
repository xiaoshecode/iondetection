# 本机 VLM 运行环境

本项目的 VLM 服务采用独立的 llama.cpp 便携运行时，原来的 Python `.venv` 和 masked/classical 训练环境保持不变。服务只监听 `127.0.0.1:8087`，实验图像只在本机推理。

## 固定版本和文件

| 组件 | 版本／文件 | 文件大小 |
| --- | --- | ---: |
| llama.cpp | b12069，commit `e6192de0a`，Windows Vulkan x64 | 压缩包 37,554,286 bytes |
| 语言权重 | Qwen3-VL-4B-Instruct，Q4_K_M GGUF | 2,497,282,336 bytes |
| 视觉投影 | 同一仓库的 `mmproj-F16.gguf` | 836,180,640 bytes |

总下载量 3,371,017,262 bytes（约 3.14 GiB），另有运行时解压空间和少量日志。没有复制已有实验数据，不使用全局 Hugging Face 模型缓存。下载的 `.part` 文件完成后原地重命名，不能把半成品当模型启动。

- [llama.cpp 官方发布 b12069](https://github.com/ggml-org/llama.cpp/releases/tag/b12069)
- [Unsloth GGUF 来源](https://huggingface.co/unsloth/Qwen3-VL-4B-Instruct-GGUF/tree/00c00da0690c4b14b5539b02c4ea5d7c9102b35e)
- [原始 Qwen3-VL 模型项目](https://github.com/QwenLM/Qwen3-VL)
- [llama.cpp 多模态服务说明](https://github.com/ggml-org/llama.cpp/blob/b12069/docs/multimodal.md)

GGUF 固定 revision 为 `00c00da0690c4b14b5539b02c4ea5d7c9102b35e`。4 位量化是本机 6 GB 显存下的资源折中，不能将其表现等同于未量化、更大规模模型的表现。

来源公布的 SHA256：

```text
llama-b12069-bin-win-vulkan-x64.zip
e1b1c4e5481fe88e3f49f97908b53bb7699ad8aa3f6332a58f944e7a8f068b33

Qwen3-VL-4B-Instruct-Q4_K_M.gguf
d4dcd426bfba75752a312b266b80fec8136fbaca13c62d93b7ac41fa67f0492b

mmproj-F16.gguf
1b9f4e92f0fbda14d7d7b58baed86039b8a980fe503d9d6a9393f25c0028f1fc
```

下载来源、字节数和实际校验记录保存在本地 `.vlm-runtime/manifest.json`。二进制和权重不提交 Git。

## 本机硬件与启动

本机是 RTX 3060 Laptop GPU，显存 6144 MiB，NVIDIA driver 546.80。采用 Vulkan 后端，不安装或升级 CUDA、显卡驱动或 PyTorch。该机器 `--list-devices` 中 `Vulkan0` 是 Intel UHD、`Vulkan1` 才是 RTX 3060，启动脚本明确选择后者。换电脑后先重新检查设备编号。

在项目目录执行：

```powershell
Set-Location 'F:\CQI\Bell Camera\Net Recog'
& '.\.vlm-runtime\llama-b12069\llama-server.exe' --list-devices
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\start_vlm_server.ps1
```

服务地址为 `http://127.0.0.1:8087/v1`，模型名为 `ion-qwen3-vl-4b`。默认单并行、8192 上下文、图像最少／最多均为 1024 tokens、GPU 卸载、q8_0 KV cache。模型视觉编码器仍可能缩放输入；保存的图块并不代表编码器逐像素读取。

llama.cpp 在加载此模型时提示：Qwen-VL 的 grounding 任务建议至少 1024 image tokens，参见 [上游问题说明](https://github.com/ggml-org/llama.cpp/issues/16842)。因此脚本显式传入 `--image-min-tokens 1024 --image-max-tokens 1024`，避免小图块使用很少的视觉 tokens。首次运行的服务 PID 29388 只设置了 max=1024；该首次服务的 smoke 结果与以后显式设置 min=1024 的实验应区别记录，正式对照以实际运行参数为准。

后台启动使用隐藏窗口并保存日志：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\start_vlm_server.ps1 -Background
Invoke-RestMethod 'http://127.0.0.1:8087/health'
Invoke-RestMethod 'http://127.0.0.1:8087/v1/models'
```

后台模式打印 PID，停止服务时只停止这个 PID：`Stop-Process -Id <打印的PID>`。前台模式按 Ctrl+C。不要停止其他 Python 或 llama 服务。显存不足时先关闭本次服务，用 `-ContextSize 4096` 重启；不要同时加载多个模型实例。

支持 `/v1/chat/completions`：图片使用 `image_url` 的 PNG data URL，`stream=false`，模型参数通过请求传入。temperature=0 用于可重复的初步实验；仍应保留原始回复并检查格式、坐标范围、截断及重复点。

## 文件布局与恢复

```text
.vlm-runtime/llama-b12069/llama-server.exe
.vlm-runtime/manifest.json
.vlm-runtime/active_server.json
.vlm-runtime/server_*.stderr.log
checkpoints/vlm/Qwen3-VL-4B-Instruct-Q4_K_M.gguf
checkpoints/vlm/mmproj-F16.gguf
```

在新电脑上根据固定来源下载同名文件并核对 SHA256，将运行时 ZIP 解压到上述目录。启动脚本不会自动下载，也不会静默调用远端模型。模型包和 projector 必须配套。

每次启动，脚本重新计算本地权重和 projector 的 SHA256，并记录模型别名、运行时版本、GPU 后端、上下文、图像 token 范围、KV cache 和完整启动参数到 `.vlm-runtime/active_server.json`。该文件不包含时间或 PID，相同配置的内容保持稳定，供实验流程记录和计算运行指纹。端口已有服务时脚本拒绝启动并保留旧 metadata；应先停止旧服务再重启。metadata 是本机启动记录，调用前仍须确认 `/health` 正常。

可用以下命令自行核对文件：

```powershell
Get-FileHash -Algorithm SHA256 -LiteralPath '.\checkpoints\vlm\Qwen3-VL-4B-Instruct-Q4_K_M.gguf'
Get-FileHash -Algorithm SHA256 -LiteralPath '.\checkpoints\vlm\mmproj-F16.gguf'
```

本机下载曾通过已运行的本地代理 `127.0.0.1:7897` 完成；这不修改系统代理，也不是模型推理所需依赖。

## 部署验证记录

2026-11-03，本机三个下载文件均完成整文件 SHA256 校验。服务加载约 13 秒后，`GET /health` 返回 HTTP 200、`{"status":"ok"}`。主流程已开始提交图像请求，服务器日志记录了正常生成且未截断的回复；实验识别效果由对应的运行报告说明。

初始服务运行中的一次 `nvidia-smi` 快照为总显存 6144 MiB、已用 4281 MiB、空闲 1728 MiB，GPU 利用率 27%。这是整个 GPU 的瞬时读数，不是服务独占显存或峰值。机器当时还有足够余量，本次未安装其他模型。具体部署检查和磁盘字节数见 `.vlm-runtime/runtime_verified.json`。

随后已按上述 grounding 设置重启服务，显式 min=max=1024，加载约 8.5 秒后 `/health` 返回正常。正式实验应记录并使用这次配置，不复用早期低 token smoke 的缓存。
