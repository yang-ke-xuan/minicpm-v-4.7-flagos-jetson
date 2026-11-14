# Jetson Orin Nano Super 8GB — 软件环境基线

目标设备的只读实测记录。

| | |
|---|---|
| 设备 | NVIDIA Jetson Orin Nano Super 8GB(开发套件 `3767-300-0005`) |
| 采集时间 | 2026-11-14 |

## 1. 平台

| 项 | 值 |
|---|---|
| L4T | R39.2.1（`nvidia-l4t-core 39.2.1-20260806224157`） |
| JetPack | 7.2.1 |
| 操作系统 | Ubuntu 24.04.4 LTS |
| 架构 | aarch64 / sbsa |
| 内核 | 6.8.12-1021-tegra |
| GPU 驱动 | 595.78（上报 CUDA 13.2） |
| 统一内存 | 7.4 GiB |
| 根分区 | 185 GB，可用 62 GB |

## 2. CUDA

三套并存，属刻意配置。

| 路径 | 版本 | 内容 |
|---|---|---|
| `/usr/local/cuda-13.2` | 13.2.86 | 完整工具链，含 `nvcc`；`/usr/local/cuda` 指向它 |
| `/usr/local/cuda-12.9` | cudart 12.9.79 / nvrtc 12.9.86 | 仅运行库，PyTorch 依赖 |
| `/usr/local/cuda-13.0` | 仅运行库 | 未使用的残留 |

| 软件包 | 版本 |
|---|---|
| `cuda-nvcc-13-2` | 13.2.86-1 |
| `cuda-cccl-13-2` | 13.2.86-1 |
| `cuda-cudart-13-2` / `-dev` | 13.2.86-1 |
| `cuda-cupti-13-2` | 13.2.86-1 |
| `cuda-cudart-12-9` | 12.9.79-1 |
| `cuda-cupti-12-9` | 12.9.79-1 |
| `cuda-nvrtc-12-9` | 12.9.86-1 |

**CUDA 12.9 不带 `nvcc`** —— 若要针对 PyTorch 的 cu129 ABI 编译 CUDA C++，需另装 `cuda-nvcc-12-9`。`nvcc` 也不在 `PATH` 中，须用绝对路径 `/usr/local/cuda-13.2/bin/nvcc`。

## 3. 推理与视觉

| 组件 | 版本 |
|---|---|
| cuDNN | 9.25.0.15-1 |
| TensorRT（系统 / Python） | 10.16.2.10-1+cuda13.2 / 10.16.2.10 |
| DeepStream | 9.1.0-1 |
| Triton Inference Server | 已装于 `/opt/tritonserver` |
| OpenCV | 4.6.0（系统包） |

Triton Inference Server 是 NVIDIA 的模型服务运行时，与 Triton Python 算子 DSL 是两回事。

## 4. 工具链

| 工具 | 版本 |
|---|---|
| GCC / G++ | 13.3.0 |
| GNU Make | 4.3 |
| CMake | 3.28.3 |
| glibc | 2.39 |
| binutils | 2.42 |
| Python / pip | 3.12.3 / 24.0 |

## 5. Python 环境

| 包 | 版本 |
|---|---|
| torch | 2.8.0 |
| torchvision | 0.23.0 |
| numpy | 1.26.4 |
| transformers | 5.16.1 |
| tokenizers | 0.23.3 |
| safetensors | 0.8.0 |
| huggingface_hub | 1.33.0 |
| onnx / onnxruntime / onnxruntime-gpu | 1.16.2 / 1.29.0 / 1.29.0 |
| pillow | 12.1.1 |
| scipy | 1.15.3 |
| pytest | 7.4.4 |
| packaging | 26.0 |
| PyYAML | 6.0.3 |

PyTorch 构建信息：`torch.version.cuda = 12.9`，`arch_list = ['sm_87']`，算力 `(8, 7)`，cuDNN 92500。wheel 来自 `pypi.jetson-ai-lab.io`（`jp6/cu129`）。

## 6. 服务

| 组件 | 版本 |
|---|---|
| Docker | 29.1.3 |
| containerd | 2.2.1 |
| ROS 2 | Jazzy |
| Node.js / npm | 26.7.0 / 11.19.0 |
| Ollama | 0.33.3 |

## 7. 缺失组件

| 包 | 用途 |
|---|---|
| `triton` | FlagGems 的算子内核 |
| `flag_gems` | 算子库本体 |
| `accelerate` | 低内存模型加载（CPU 可用内存仅约 4 GB） |
| `cuda-nvcc-12-9` | 针对 cu129 ABI 编译 CUDA C++ |

## 8. FlagGems 兼容性

官方要求出处：`src/flag_gems/backends.yaml` → `pyproject.toml [project.optional-dependencies]`。

| 项 | FlagGems 要求 | 本机 | 状态 |
|---|---|---|---|
| torch | 2.10.0+cu128 或 2.11.0+cu130 | 2.8.0+cu129 | 唯一偏离项 |
| triton | 3.6.0 | PyPI 有 cp312 aarch64 | 可对齐 |
| flagtree | 0.6.1 / 0.7.0 | PyPI 未发布 | 改用 `--compiler triton` |
| python | 3.12 | 3.12.3 | 符合 |
| packaging | >=26.0 | 26.0 | 符合 |
| PyYAML | ==6.0.1 | 6.0.3 | 安装时用 `--no-deps` |

TensorRT 与 PyTorch 无耦合；aarch64 Jetson 生态每个 JetPack 版本只发布一个 PyTorch wheel。因此真正的耦合点只有 Triton。

## 9. 复现

```bash
cat /etc/nv_tegra_release
dpkg -l | awk '/^ii/ && $2 ~ /^cuda-/ {print $2, $3}'
dpkg-query -W -f='${Version}\n' libcudnn9-cuda-12 libnvinfer10 deepstream-9.1
gcc -dumpfullversion; cmake --version | head -1
python3 -V; python3 -m pip list
python3 -c "import torch; print(torch.__version__, torch.version.cuda,
             torch.cuda.get_arch_list(), torch.cuda.get_device_capability(0))"
docker --version; node -v; free -h; df -h /
```

数值校验（只看 `is_available()` 不够）：

```python
import torch
torch.manual_seed(0)
a, b = torch.randn(512, 512), torch.randn(512, 512)
print(((a @ b) - (a.cuda() @ b.cuda()).cpu()).abs().max().item())
# fp32 ≈ 5e-5, fp16 ≈ 5e-2
```