# VLM 训练环境记录

2026-11-03 已在本机建立并验证独立环境 `.venv-vlm-train`。原项目 `.venv`、已有 Conda 环境、显卡驱动均未更改。

## 使用方式

从项目目录 `F:\CQI\Bell Camera\Net Recog` 调用：

```powershell
.\.venv-vlm-train\Scripts\python.exe -m pip check
.\.venv-vlm-train\Scripts\python.exe <训练脚本及参数>
```

无需激活环境。训练入口、数据和结果说明见 [VLM_TRAINING.md](VLM_TRAINING.md)。

## 固定版本

| 组件 | 版本 |
| --- | --- |
| Python | 3.12.14，Windows x64 |
| PyTorch | 2.5.1+cu121 |
| torchvision | 0.20.1+cu121 |
| transformers | 4.57.1 |
| peft | 0.17.1 |
| accelerate | 1.10.1 |
| bitsandbytes | 0.46.1 |
| numpy | 1.26.4 |
| Pillow | 11.3.0 |
| PyYAML | 6.0.3 |
| safetensors | 0.6.2 |
| scipy | 1.14.1 |
| tifffile | 2025.3.30 |
| pytest | 8.4.2 |

完整锁定版本保存在 `.vlm-runtime/train-wheels/requirements-lock.txt`。环境由本机已有的 Python 3.12 创建；未替换系统 Python 或 CUDA。已存在的 `face` 环境为 Python 3.10 / PyTorch 1.13.1+cu116，CUDA 可用，但版本不足以直接运行当前 Qwen3-VL 训练代码。

torchvision 是实际必需依赖：即使只使用图片且 `use_fast=False`，Qwen3-VL 的 `AutoProcessor` 仍会构造 `AutoVideoProcessor`；没有 torchvision 时本地 processor 加载失败。

## 已通过的验证

- `pip check`：无依赖冲突；离线安装前后的版本均固定。
- `torch.cuda.is_available()` 为真，识别到 NVIDIA GeForce RTX 3060 Laptop GPU（6 GiB），驱动 546.80。
- bitsandbytes `Linear4bit`：NF4、double quantization、BF16 计算，在 CUDA 上完成前向和输入梯度反传，梯度有限且非零。小测试 loss 为 0.2230108678，输入梯度绝对值之和为 0.6640625。
- 强制 `SDPBackend.EFFICIENT_ATTENTION`：BF16 的 Q/K/V，形状 `[1,16,64,128]`，CUDA 前向和反向均成功且梯度有限。
- `AutoProcessor.from_pretrained('checkpoints/Qwen3-VL-2B-Instruct-bf16', local_files_only=True, use_fast=False)` 成功构造 `Qwen3VLProcessor`。
- `Qwen3VLForConditionalGeneration` 可导入。

这些验证证明本机依赖、CUDA 和量化反向通路可用；完整模型是否适配显存以及训练质量须以真实训练结果为准。上述小测试均已退出，不保留 GPU 进程。

本 Windows PyTorch 版本使用 memory-efficient SDPA。训练时应避免不支持的 GQA 分支退回会创建完整注意力矩阵的 math 实现；训练器的具体处理与整模验证见训练文档。

机器可读记录位于 `.vlm-runtime/training_environment_verified.json`。

## 下载来源与磁盘

所有 wheel 位于项目内 `.vlm-runtime/train-wheels/`，按官方发布的 SHA256 校验后离线安装。PyTorch 与 torchvision 来源为 [PyTorch CUDA 12.1 wheel 索引](https://download.pytorch.org/whl/cu121/)，其余依赖来自 [PyPI](https://pypi.org/) 的对应版本元数据和 `files.pythonhosted.org`。未关闭 TLS 校验。依赖的版本、文件名、大小、SHA256 与声明依赖保存在 `packages_provenance.json`。

PyTorch wheel 大小为 2,449,331,371 字节，SHA256 为 `473d76257636c66b22cbfac6f616d6b522ef3d3473c13decb1afda22a7b059eb`。torchvision wheel 大小为 6,063,634 字节，SHA256 为 `12b764936fc3fde29d7165adc7c721e491fa75c7a498c9fbaa08757b5fe59fe7`。

网络下载最初经本机代理很慢，改用获准的官方源直连后完成；最终依赖清单重新查询官方 PyPI 并复核所有缓存哈希，结果无未满足依赖。

安装后的实测磁盘记录（会随训练输出变化）：

| 项目 | 大小 |
| --- | ---: |
| `.venv-vlm-train` | 5,357,954,969 字节，约 4.99 GiB |
| `.vlm-runtime/train-wheels` | 2,627,518,114 字节，约 2.45 GiB |
| `.vlm-runtime/train-temp` | 0 字节 |
| F 盘可用空间 | 25.60 GiB |

下载阶段按至少保留 8 GiB 检查空间，安装临时目录固定在工作区。wheel 缓存暂时保留，便于离线重建和审计；本任务没有删除用户已有数据或缓存。
