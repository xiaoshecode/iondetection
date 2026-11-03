# 本次构建验证记录

日期：2026-11-03。平台：本机 Windows，CPU。范围：代码可执行性、输入隔离、合成数据定位与输出流程。**没有完成真实大晶格模型训练或准确率评估。**

## 环境

项目 `.venv` 由 `E:\Anaconda\python.exe -m venv --system-site-packages .venv` 创建，复用已有 CPU PyTorch；仅在项目环境安装兼容依赖，未更新 Anaconda 的包。

| 项目 | 实际版本 |
| --- | --- |
| Python | 3.9.12 |
| NumPy | 1.26.3 |
| SciPy | 1.10.1 |
| Pillow | 11.3.0 |
| tifffile | 2024.8.30 |
| PyTorch | 1.13.0+cpu |
| PyYAML | 6.0 |
| pytest | 7.1.1 |
| CUDA 可用 | 否 |

旧 Anaconda SciPy 1.7.3 与现有 NumPy 不兼容，`scipy.optimize` 曾出现 DLL 导入失败；项目 `.venv` 的 SciPy 1.10.1 已解决。本地 `.wheelhouse` 保留了本次下载的三个安装包，已排除在 Git 之外。此虚拟环境依赖原 Anaconda，不是可搬运的独立安装包。

## 自动测试

执行：

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

结果：**112 passed，2.81 s，无警告**。包含：

- uint16 数值保留、单帧/单通道限制、固定归一化。
- 空白图、相连双峰、亚像素中心、叠加图不改原始数组。
- masked 隐藏值隔离、重复帧目标排除输入、重复像素泄漏拒绝。
- 两种模型各完成真实 CPU 优化步、验证、保存/加载和奇数尺寸推理。
- 非有限验证损失不写 checkpoint。
- 训练验证目录隔离、跨格式相同像素检查及同组复制帧拒绝。
- 一对一匹配、空集合、非法坐标/半径/阈值、空测光 ROI。
- 结果目录保护、评估真值文件防覆盖、不同参考晶格组禁止平均。

## CLI 与 PowerShell 端到端

生成 `data/smoke_demo`：2 组、每组 3 帧、96×96、seed=42。每组一个固定场景，帧间噪声独立；一组训练、一组验证。

| 流程 | 实际结果 | 本地结果目录 |
| --- | --- | --- |
| PowerShell classical | 3 张图均输出 51 个中心、CSV/JSON、原图标圈及区域掩码 | `runs/script_smoke/predictions/` |
| masked 训练 | smoke 配置 1 epoch × 2 steps，输出 best.pt/last.pt 与日志 | `runs/smoke_masked/` |
| masked 推理 | 3 张完整图读取权重并写出恢复数组与标圈 | `runs/smoke_masked_predictions/` |
| diffusion 训练 | 同样极短配置，条件 DDPM 优化与验证完成 | `runs/smoke_diffusion/` |
| diffusion 推理 | 每张 2 个种子样本，DDIM 推理及空检测输出正常 | `runs/smoke_diffusion_predictions/` |
| PowerShell diffusion 全链路 | inspect → train → best.pt → infer 均成功 | `runs/pipeline_diffusion_smoke/` |
| 全亮参考测光 | 参考提取 51 格点，逐帧写出参考位置光度表 | `runs/smoke_reference_query/` |
| evaluate CLI | 成功读取对应单图独立合成真值，完成一对一匹配 | `runs/smoke_position_evaluation.json` |

短训练的 masked 输出 84/85/86 个候选；diffusion 输出 0 个候选。这些是**未收敛模型的流程测试结果，不是可用权重**。不应将该输出作为方法有效性的证据。没有把这些权重放入待提交代码。

## 合成数据上的定位检查

使用原图 classical、3 px 容差、一对一最优匹配，与生成器的位置真值比较：

| 验证图 | TP | FP | FN | 平均中心误差 / px |
| --- | ---: | ---: | ---: | ---: |
| group_001/frame_000.png | 51 | 0 | 0 | 0.0406 |
| group_001/frame_001.png | 51 | 0 | 0 | 0.0406 |
| group_001/frame_002.png | 51 | 0 | 0 | 0.0428 |

这是简单、高信噪合成样例，不覆盖真实边缘微运动、相机相关噪声、离子重叠或暗态。三个帧来自同一验证构型，不能当作三个独立实验重复报告。完整本机数值保存在 `runs/verification_report.json`。

## 已有实验图的读取检查及发现的限制

对已有项目 `Recog/CCD-Recognition/single_scan_images/` 中的 `single_scan_frame000006_20260116_211228_885.npy` 做了一次只读推理，尺寸 300×800；原文件未修改。

默认 classical 参数输出 199 个候选。查看原图叠加后，**噪声条带中存在明显多余候选，默认阈值不适配这张实际相机图**。提高 `threshold_sigma` 至 6/8/10/12，候选数变为 50/20/11/7；这仅说明阈值敏感，不证明剩余候选全对，也不证明弱离子没有被删除。没有据此改写默认参数以迎合这一帧。

结果位于 `runs/real_image_classical_check/`。该帧没有独立真值，因此不报告真实准确率。后续应使用用户的新大晶格数据与可靠评测依据，处理空间非均匀噪声和拖尾造成的误检。

## 尚未验证

CUDA、10,000 张图的全量训练时间及磁盘使用、真实大晶格泛化、状态保真度、微运动去模糊、学习式实例分割均未验证。当前交付的是可运行的 PC 研究框架，不是已校准的实验测量系统。
