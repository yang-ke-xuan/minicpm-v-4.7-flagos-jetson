# Devlog

## 2026-11-14 · 定位卡点：FlagGems 在 ARM64 + 统一内存 + NVIDIA GPU 上没有官方路径

### 问题

目标是把 MiniCPM-V 通过 FlagOS（FlagGems + Triton）跑到 Jetson Orin Nano Super 8GB。
Triton 层面大概率可行，卡点在 FlagGems / FlagTree。

### 已确认的事实

| 项 | 结论 | 依据 |
|---|---|---|
| Triton 3.6.0 可装 | ✅ Python 3.10–3.14、glibc 2.27+、AArch64 均在支持范围 | triton-lang/triton `RELEASE.md` |
| Triton 3.6.0 wheel | ✅ PyPI 有 `cp312 + manylinux_2_28 aarch64` | PyPI 索引 |
| FlagOS NVIDIA 索引 | ❌ torch / torchvision / torchaudio / flagtree / flag-gems-cpp-cuda **全部只有 x86_64**，零 aarch64 | `resource.flagos.net/repository/flagos-pypi-nvidia/simple/` |
| FlagGems 官方依赖 | torch 2.10.0+cu128 或 2.11.0+cu130；triton 3.6.0；flagtree 0.6.1/0.7.0 | `src/flag_gems/backends.yaml` |
| flagtree | ❌ 无 aarch64 构建 | 同上索引 |
| `flag-gems` 本体 | ✅ 纯 Python，装得上 | `py3-none-any` |
| 编译器选择 | `COMPILER=triton` 是官方机制（默认 flagtree） | backends.yaml 头部注释 |

→ **能成立的组合只有一条**：torch 2.8.0+cu129（现成）+ triton 3.6.0（PyPI）+ flag_gems 纯 Python（`--no-deps`），放弃 flagtree 和 `flag-gems-cpp-cuda`。

### 社区已有的 ARM 工作 —— 全都是 CPU

`src/flag_gems/runtime/backend/_arm/` 已存在（ops / fused / int8 / quantized_linear / tune_configs.yaml），`Vendor.ARM = 12`。

| PR | 内容 | 状态 |
|---|---|---|
| #3454 | [arm64] ARM64 CPU backend 骨架 + ~30 算子 | closed，**未合并** |
| #3455 | [Arm64] 融合算子（RMSNorm / RoPE / SiLU+mul） | closed |
| #3456 | [Arm64] int8 算子 | closed |
| #5904 | [ARM] Triton W4A8-G128 linear（基于 FlagTree-cpu） | **已合并** |
| #8752 | [ARM] O6N aarch64 CI 测试套件 | open |

`flagtree-cpu` 仓库自述：*An experimental CPU backend for Triton*。

**关键区别：社区所有 ARM 支持都是 CPU 后端。** 本项目要的是 ARM SoC + 统一内存 + NVIDIA GPU（sm_87），仓库内搜索 `Jetson` 结果为 **0**。

另外：`backends.yaml` 里没有 `arm` / `cpu` 条目，所以 `flaggems-setup` 装不了 ARM 后端。

### CIX P1 案例是怎么做的

对象是 **MiniCPM5-1B on ARMv9（CIX P1 / CD8180，8×A720 + 4×A520）**，不是 V-4.7-1B。

- **纯 CPU 推理**：容器标签 `gpu_cpu` / `driver_none`，无 GPU、无驱动
- 软件基线：Triton 3.3 / LLVM a66376b0 / **PyTorch 2.10 CPU** / Python 3.11
- **交付形式是 Docker 镜像**，不是 pip 包：`harbor.baai.ac.cn/flagrelease-public/flagrelease-minicpm5-armv9-tree_0.5.0-gems_5.0.2-...-torch_2.10.0_cpu-...-arc_arm64-driver_none`
- 优化手段两层：
  - **编译器层**：ARM 指令选择、OMP 线程调优、让 `tl.dot` 在 ARM 上正确高效
  - **算子层**：6 个 TLE 扩展算子（`triton.language.extra.cpu.tle_ops`）
    1. `sdot_pack_weights` — INT8 权重打包成 SDOT 布局
    2. `sdot_gemv` — M=1 INT8 GEMV
    3. `sdot_gemv_fused_bf16` — BF16 动态量化 → SDOT → 反量化，decode 主路径
    4. `rms_norm` — 单 kernel 替代 5 个 ATen op
    5. `swiglu` — 融合激活
    6. `flash_attn_decode` — M=1 Flash Attention，online softmax
- 效果：INT4 后权重 0.5 GB，可在 90% 终端运行

**可借鉴的方法论**：把 decode 热点收敛成少量融合算子、按 M=1 形状专门优化、用统一内存带宽而非算力做设计约束。**不可借鉴的部分**：NEON / SDOT / i8mm / SVE2 全部是 ARM CPU 指令，与 sm_87 无关。

### 结论

1. FlagOS 的 ARM 叙事目前是 **ARM CPU**；ARM + GPU + UMA 在官方和社区都没有实现。
2. 官方组合依赖的是 FlagOS 私有索引里的 **aarch64 wheel**，而这些 wheel 不存在——不是版本对不上，是包不存在。
3. 因此本项目的真实课题应是：**把 FlagOS 的 NVIDIA 后端首次带到 ARM64 / 统一内存设备上**。
4. 代价：没有 `flag-gems-cpp-cuda`，每次算子调用的 Python 分发开销无法消除。这本身是可量化的研究点。

### 下一步（待确认）

1. 先在 Jetson 上做 Triton 冒烟测试（venv + `triton==3.6.0` + 向量加法），确认 sm_87 能出 kernel
2. 同步向社区提问，问题聚焦：
   - 是否有 ARM64 + NVIDIA GPU（Tegra / sm_87）的支持计划？
   - `_arm` 后端是否设计为 CPU-only，是否规划 `_arm_gpu`？
   - `flagtree-cpu` 能否扩展到 aarch64 + CUDA？
   - 是否有 Jetson 上的实验数据？
   - 若无计划，社区是否接受外部 PR？
3. 拿到两边结果后决定：提 PR / 自研 / 换路线