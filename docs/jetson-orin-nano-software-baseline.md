# Jetson Orin Nano Super 8GB — Software Baseline

Complete, reproducible record of every low-level component on the target device.
Captured **read-only** (no installs, no upgrades, no configuration changes).

| | |
|---|---|
| **Device** | NVIDIA Jetson Orin Nano Super 8GB (official devkit) |
| **Captured** | 2026-11-14 14:32 (Asia/Shanghai) |
| **Captured by** | read-only probe over SSH |
| **Purpose** | Pin the exact environment before operator adaptation / benchmarking |
| **Previous capture** | 2026-11-13 — **no changes detected** |

---

## 1. Hardware & Platform

| Item | Value |
|---|---|
| Board spec (TNSPEC) | `3767-300-0005-` (Orin Nano Super devkit) |
| Tegra CHIPID | `0x23` (Tegra234) |
| Boot storage | NVMe (`nvme0n1`) |
| L4T release | **R39.2.1** |
| `nvidia-l4t-core` | `39.2.1-20260806224157` |
| JetPack | **7.2.1** |
| OS | Ubuntu 24.04.4 LTS (Noble) |
| Architecture | `aarch64` / **sbsa** |
| Kernel | `6.8.12-1021-tegra` |
| GPU driver | **595.78** |
| Driver-reported CUDA | **13.2** |
| CPU / SM count | 6-core ARM Cortex-A78AE / 8 SMs |
| Unified memory | 7.4 GiB |
| Root filesystem | 185 GB total, **62 GB available (66%)** |

---

## 2. CUDA Family (three toolkits coexist)

| Path | Version | Contents |
|---|---|---|
| `/usr/local/cuda-13.2` | **13.2.86** | Full toolchain, includes `nvcc`. `/usr/local/cuda` points here. |
| `/usr/local/cuda-12.9` | cudart **12.9.79** / nvrtc **12.9.86** | **Runtime libraries only**, required by the PyTorch wheel |
| `/usr/local/cuda-13.0` | runtime only | Leftover, safe to remove |
| `/usr/local/cuda-12` | symlink → `cuda-12.9` | Resolves `libcublas.so.12` for PyTorch |

### Installed CUDA apt packages

| Package | Version |
|---|---|
| `cuda-nvcc-13-2` | 13.2.86-1 |
| `cuda-cccl-13-2` | 13.2.86-1 |
| `cuda-cudart-13-2` | 13.2.86-1 |
| `cuda-cudart-dev-13-2` | 13.2.86-1 |
| `cuda-cupti-13-2` | 13.2.86-1 |
| `cuda-cudart-12-9` | 12.9.79-1 |
| `cuda-cupti-12-9` | 12.9.79-1 |
| `cuda-nvrtc-12-9` | 12.9.86-1 |

> ⚠️ **There is no `nvcc` for CUDA 12.9** — only the runtime. Compiling CUDA C++
> extensions against the PyTorch cu129 ABI requires installing `cuda-nvcc-12-9`.
>
> ⚠️ `nvcc` is **not on `PATH`**. Invoke it as `/usr/local/cuda-13.2/bin/nvcc`.

---

## 3. Inference & Vision

| Component | Version | Notes |
|---|---|---|
| cuDNN | **9.25.0.15-1** | `libcudnn9-cuda-12`, resolved via `/lib/aarch64-linux-gnu` |
| TensorRT (system) | **10.16.2.10-1+cuda13.2** | Pinned; do **not** upgrade to 11.x |
| TensorRT (Python) | **10.16.2.10** | Imports without loading PyTorch (zero coupling) |
| DeepStream | **9.1.0-1** | |
| Triton Inference Server | `/opt/tritonserver` present | Version file absent, version could not be read |
| OpenCV | **4.6.0** | System apt package (not the pip wheel) |

> ℹ️ **Triton Inference Server ≠ Triton Language.** `/opt/tritonserver` is NVIDIA's
> C++ model-serving runtime. It is unrelated to the `triton` Python kernel DSL
> required by FlagGems.

---

## 4. Compiler & Build Toolchain

| Tool | Version |
|---|---|
| GCC | 13.3.0 |
| G++ | 13.3.0 |
| GNU Make | 4.3 |
| CMake | 3.28.3 |
| glibc | 2.39 |
| binutils (`ld`) | 2.42 |
| Python headers (`Python.h`) | present |

---

## 5. Python Stack

Python **3.12.3**, pip **24.0** (system interpreter, PEP 668 externally-managed).

### Installed

| Package | Version |
|---|---|
| torch | **2.8.0** |
| torchvision | 0.23.0 |
| torch-complex | 0.4.4 |
| numpy | **1.26.4** |
| transformers | **5.16.1** |
| tokenizers | 0.23.3 |
| safetensors | 0.8.0 |
| huggingface_hub | 1.33.0 |
| onnx | 1.16.2 |
| onnxruntime | 1.29.0 |
| onnxruntime-gpu | 1.29.0 |
| onnxslim | 0.1.96 |
| pillow | 12.1.1 |
| scipy | 1.15.3 |
| matplotlib | 3.6.3 |
| matplotlib-inline | 0.2.2 |
| pytest | 7.4.4 |
| packaging | **26.0** |
| PyYAML | **6.0.3** |
| psutil | 7.2.2 |
| decorator | 5.1.1 |

### Not installed

| Package | Why it matters |
|---|---|
| **accelerate** | Required for `low_cpu_mem_usage` / `device_map` model loading (only ~4 GB CPU RAM free) |
| **triton** | Prerequisite for FlagGems |
| **flag_gems** | The operator library itself |
| einops, timm | Not referenced by `MiniCPMV4_6` (verified by source grep) |

### PyTorch build identity

| Property | Value |
|---|---|
| `torch.__version__` | 2.8.0 |
| `torch.version.cuda` | 12.9 |
| `torch.backends.cudnn.version()` | 92500 (9.25.0) |
| `torch.cuda.get_arch_list()` | `['sm_87']` |
| `torch.cuda.get_device_capability()` | (8, 7) |
| `torch.cuda.get_device_name()` | `Orin` |
| Wheel origin | `pypi.jetson-ai-lab.io` → `jp6/cu129` build, installed from a local `.whl` |

Runtime linkage of `libtorch_cuda.so`:

```
libcudart.so.12   → /usr/local/cuda-12/targets/sbsa-linux/lib
libcublas.so.12   → /usr/local/cuda-12/targets/sbsa-linux/lib
libcublasLt.so.12 → /usr/local/cuda-12/targets/sbsa-linux/lib
libcudnn.so.9     → /lib/aarch64-linux-gnu
```

---

## 6. Services & Runtimes

| Component | Version |
|---|---|
| Docker | 29.1.3 |
| containerd | 2.2.1 |
| ROS 2 | Jazzy |
| Node.js | 26.7.0 |
| npm | 11.19.0 |
| Ollama | 0.33.3 |

---

## 7. Known Gaps

| Gap | Impact | Fix |
|---|---|---|
| `triton` missing | Blocks the entire FlagGems path | `pip install triton==3.6.0` (PyPI, cp312 aarch64) |
| `flag_gems` missing | No operator library | `pip install flag_gems --no-deps` |
| `accelerate` missing | Model loading may thrash swap | `pip install accelerate` |
| No `nvcc` for CUDA 12.9 | Cannot compile CUDA C++ extensions against the cu129 ABI | `apt install cuda-nvcc-12-9` |
| `PyYAML` mismatch | FlagGems pins `==6.0.1`, system has 6.0.3 | Install with `--no-deps` |
| `huggingface.co` unreachable | Download hangs | `export HF_ENDPOINT=https://hf-mirror.com` |

---

## 8. FlagGems Compatibility Matrix

Official requirement source:
`src/flag_gems/backends.yaml` (lines 56–75) → generated into
`pyproject.toml [project.optional-dependencies]`.

| Item | FlagGems official | This device | Verdict |
|---|---|---|---|
| torch | `2.10.0+cu128` or `2.11.0+cu130` | `2.8.0+cu129` | ⚠️ Only deviation |
| triton | `3.6.0` | available on PyPI (cp312 aarch64) | ✅ Alignable |
| flagtree | `0.6.1` / `0.7.0` | **not published on PyPI (404)** | 🟢 use `--compiler triton` |
| python | `3.12` | 3.12.3 | ✅ |
| packaging | `>=26.0` | 26.0 | ✅ (at the lower bound) |
| PyYAML | `==6.0.1` | 6.0.3 | ⚠️ minor |

**Why only triton matters:** TensorRT is fully decoupled from PyTorch
(`import tensorrt` does not load `torch`), and the aarch64 Jetson ecosystem
publishes exactly one PyTorch wheel per JetPack line. FlagGems is pure Python
and couples only to Triton.

---

## 9. How to Reproduce This Record

```bash
# Platform
cat /etc/nv_tegra_release
dpkg-query -W -f='${Version}\n' nvidia-l4t-core
. /etc/os-release && echo "$PRETTY_NAME"
uname -r; uname -m

# CUDA packages
dpkg -l | awk '/^ii/ && $2 ~ /^cuda-/ {print $2, $3}'
/usr/local/cuda-13.2/bin/nvcc --version

# Inference stack
dpkg-query -W -f='${Version}\n' libcudnn9-cuda-12 libnvinfer10 deepstream-9.1
python3 -c "import tensorrt; print(tensorrt.__version__)"

# Toolchain
gcc -dumpfullversion; cmake --version | head -1; ldd --version | head -1

# Python stack
python3 -V; python3 -m pip --version
python3 -m pip list | grep -iE '^(torch|torchvision|numpy|transformers|tokenizers|safetensors|huggingface|accelerate|triton|flag|onnx|pillow|scipy|pytest|packaging|pyyaml)'

# PyTorch identity
python3 -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_arch_list(), torch.cuda.get_device_capability(0))"

# Services
docker --version; node -v; ollama --version

# Resources
free -h; df -h /
```

### Correctness check (not just `is_available()`)

```python
import torch
torch.manual_seed(0)
a, b = torch.randn(512, 512), torch.randn(512, 512)
cpu = a @ b
gpu = (a.cuda() @ b.cuda()).cpu()
print((cpu - gpu).abs().max().item())   # expected ~5e-5 for fp32
```

Reference values on this device: fp32 `5.34e-05`, fp16 `4.83e-02`.

---

## 10. Change Log

| Date | Change |
|---|---|
| 2026-11-12 | Initial environment audit (read-only) |
| 2026-11-13 | Fixed broken `transformers` pin: `tokenizers` 0.22.2 → 0.23.3, `huggingface_hub` 1.4.1 → 1.33.0 |
| 2026-11-13 | Reclaimed ~15 GB disk (removed `DeepStream.tar.gz` + HF/uv/pip caches) |
| 2026-11-14 | Full re-capture — **no version changes** |