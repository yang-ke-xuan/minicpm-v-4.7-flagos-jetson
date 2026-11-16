# 此文档用于记录将Triton3.6.0安装于Jetson orin nano super 8GB中遇到的问题
## 【Triton官方文档】 （https://github.com/triton-lang/triton/blob/main/RELEASE.md#release-compatibility-matrix）

## 🖥️ 针对 Triton 安装的环境摸底 (Targeted Env for Triton)

| 关键维度 | 本机现状 | Triton 官方要求/限制 | 
| :--- | :--- | :--- | :--- |
| **CPU 架构** | `aarch64` (ARM64) | 官方 3.6.0 提供 `manylinux_2_28_aarch64` wheel | 
| **GPU 架构** | Tegra, `sm_87` (Ampere) | 依赖底层 `ptxas` 支持该计算能力 | 
| **系统 C 库** | `glibc 2.39`  | 需 `glibc >= 2.27` (Manylinux 要求) | 
| **Python 环境**| `Python 3.12.3`  | 需 `>=3.10` 且 `<=3.14` | 
| **底层驱动** | CUDA 12.9 | 需 CUDA Toolkit 支持 | 

## 发现新问题：FlagGems和FlagTree目前没有针对ARM+UMA+GPU的适配
FlagOS 的 NVIDIA 后端从构建层面就没有 aarch64。torch、torchvision、torchaudio、flagtree、flag-gems-cpp-cuda 五个包，全部零 aarch64 wheel
