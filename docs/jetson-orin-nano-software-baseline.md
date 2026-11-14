# Jetson Orin Nano Super 8GB — Software Baseline

Read-only capture of the target device. Everything below was measured, not inferred.

| | |
|---|---|
| Device | NVIDIA Jetson Orin Nano Super 8GB (devkit `3767-300-0005`) |
| Captured | 2026-11-14 |

## 1. Platform

| Item | Value |
|---|---|
| L4T | R39.2.1 (`nvidia-l4t-core 39.2.1-20260806224157`) |
| JetPack | 7.2.1 |
| OS | Ubuntu 24.04.4 LTS |
| Architecture | aarch64 / sbsa |
| Kernel | 6.8.12-1021-tegra |
| GPU driver | 595.78 (reports CUDA 13.2) |
| Unified memory | 7.4 GiB |
| Root filesystem | 185 GB, 62 GB free |

## 2. CUDA

Three toolkits coexist. This is intentional.

| Path | Version | Contents |
|---|---|---|
| `/usr/local/cuda-13.2` | 13.2.86 | Full toolkit with `nvcc`; `/usr/local/cuda` points here |
| `/usr/local/cuda-12.9` | cudart 12.9.79 / nvrtc 12.9.86 | Runtime only — required by PyTorch |
| `/usr/local/cuda-13.0` | runtime only | Unused leftover |

| Package | Version |
|---|---|
| `cuda-nvcc-13-2` | 13.2.86-1 |
| `cuda-cccl-13-2` | 13.2.86-1 |
| `cuda-cudart-13-2` / `-dev` | 13.2.86-1 |
| `cuda-cupti-13-2` | 13.2.86-1 |
| `cuda-cudart-12-9` | 12.9.79-1 |
| `cuda-cupti-12-9` | 12.9.79-1 |
| `cuda-nvrtc-12-9` | 12.9.86-1 |

**CUDA 12.9 ships no `nvcc`** — compiling against the PyTorch cu129 ABI needs `cuda-nvcc-12-9`. `nvcc` is also not on `PATH`; use `/usr/local/cuda-13.2/bin/nvcc`.

## 3. Inference & Vision

| Component | Version |
|---|---|
| cuDNN | 9.25.0.15-1 |
| TensorRT (system / Python) | 10.16.2.10-1+cuda13.2 / 10.16.2.10 |
| DeepStream | 9.1.0-1 |
| Triton Inference Server | installed at `/opt/tritonserver` |
| OpenCV | 4.6.0 (system package) |

Triton Inference Server is NVIDIA's model-serving runtime; it is unrelated to the Triton Python kernel DSL.

## 4. Toolchain

| Tool | Version |
|---|---|
| GCC / G++ | 13.3.0 |
| GNU Make | 4.3 |
| CMake | 3.28.3 |
| glibc | 2.39 |
| binutils | 2.42 |
| Python / pip | 3.12.3 / 24.0 |

## 5. Python Stack

| Package | Version |
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

PyTorch build: `torch.version.cuda = 12.9`, `arch_list = ['sm_87']`, compute capability `(8, 7)`, cuDNN 92500. Wheel comes from `pypi.jetson-ai-lab.io` (`jp6/cu129`).

## 6. Services

| Component | Version |
|---|---|
| Docker | 29.1.3 |
| containerd | 2.2.1 |
| ROS 2 | Jazzy |
| Node.js / npm | 26.7.0 / 11.19.0 |
| Ollama | 0.33.3 |

## 7. Missing

| Package | Needed for |
|---|---|
| `triton` | FlagGems kernels |
| `flag_gems` | Operator library |
| `accelerate` | Low-memory model loading (only ~4 GB CPU RAM free) |
| `cuda-nvcc-12-9` | Compiling CUDA C++ against the cu129 ABI |

## 8. FlagGems Compatibility

Official requirements: `src/flag_gems/backends.yaml` → `pyproject.toml [project.optional-dependencies]`.

| Item | FlagGems requires | This device | Status |
|---|---|---|---|
| torch | 2.10.0+cu128 or 2.11.0+cu130 | 2.8.0+cu129 | Only deviation |
| triton | 3.6.0 | available on PyPI (cp312 aarch64) | Alignable |
| flagtree | 0.6.1 / 0.7.0 | not published on PyPI | Use `--compiler triton` |
| python | 3.12 | 3.12.3 | OK |
| packaging | >=26.0 | 26.0 | OK |
| PyYAML | ==6.0.1 | 6.0.3 | Install with `--no-deps` |

TensorRT is independent of PyTorch, and the aarch64 Jetson ecosystem publishes one PyTorch wheel per JetPack line — so Triton is the only real coupling point.

## 9. Reproduce

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

Numerical sanity check (a working GPU should not just report `is_available()`):

```python
import torch
torch.manual_seed(0)
a, b = torch.randn(512, 512), torch.randn(512, 512)
print(((a @ b) - (a.cuda() @ b.cuda()).cpu()).abs().max().item())
# fp32 ≈ 5e-5, fp16 ≈ 5e-2
```