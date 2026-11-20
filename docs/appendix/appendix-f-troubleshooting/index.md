---
title: "附录 F 常见失败模式与排查清单"
description: "Hello MLSys · 附录 F 常见失败模式与排查清单 · 含 rocprofv3 PMC 修复记录"
---

# 附录 F 常见失败模式与排查清单

::: danger ⚠️ Alpha内测版本警告
此为早期内部构建版本，尚不完整且可能存在错误，欢迎大家提Issue反馈问题或建议。
:::

## 本章导读

本附录记录课程真机实验中已复现、已定位的失败模式。每条包含现象、根因、可执行修复和验证边界，避免把“工具没配好”误判成“硬件不支持”。

当前收录：

- `rocprofv3 --pmc` 报 `aqlprofile API table load failed`；
- `--pmc` 能启动，但 VALU、L2、GRBM、TA 等计数器全部为 0；
- RDNA3/RDNA4 在 `AUTO` 性能档下采集深计数器的注意事项。

## F.1 `rocprofv3 --pmc` 加载失败或计数器全零

### 现象

在 AMD Radeon RX 7900 XTX（`gfx1100`）、ROCm 10.0、`rocprofv3 1.3.5` 上出现过两类不同现象：

| 现象 | 典型输出 | 实际根因 |
|---|---|---|
| 一启动就失败 | `aqlprofile API table load failed: HSA_STATUS_ERROR` | conda/TheRock 运行时 wheel 缺少无版本 aqlprofile symlink |
| 能采集，但目标计数器全为 0 | `SQ_BUSY_CYCLES`、`SQ_WAVES_sum` 非零；`SQ_INSTS_VALU`、`SQ_WAVE_CYCLES`、`GL2C_HIT/MISS`、`GRBM_*`、`TA_TA_BUSY` 为 0 | GPU 处于 `AUTO` 动态调频，部分 PMC 事件不累计 |

两个问题相互独立：只修 aqlprofile 路径不会修复全零；只用 root 也不会修复全零。

### 根因一：缺少无版本 aqlprofile

HSA runtime 通过无版本名 `libhsa-amd-aqlprofile64.so` 执行 `dlopen`，但运行时 wheel 只带 SONAME：

```text
_rocm_sdk_core/lib/libhsa-amd-aqlprofile64.so.1
```

这是上游打包问题，见 [ROCm/TheRock#7085](https://github.com/ROCm/TheRock/issues/7085)。临时修复：

```bash
PY=$(command -v python)
CONDA=$("$PY" -c 'import pathlib, _rocm_sdk_core; print(pathlib.Path(_rocm_sdk_core.__file__).resolve().parent / "lib")')
LINK=$(mktemp -d)
ln -s "$CONDA/libhsa-amd-aqlprofile64.so.1" "$LINK/libhsa-amd-aqlprofile64.so"
export LD_LIBRARY_PATH="$LINK:$CONDA"
```

不要在教程中建议把 symlink 写进全局系统目录；临时目录或项目环境内修复更容易回退。后续运行时 wheel 自带该 symlink 后即可删除此 workaround。

### 根因二：`AUTO` dVFS 下事件不累计

补好 aqlprofile 后，`SQ_BUSY_CYCLES` / `SQ_WAVES_sum` 仍有值，但其余计数器为 0。上游 [ROCm/rocm-systems#8523](https://github.com/ROCm/rocm-systems/issues/8523) 已确认：`AUTO` 在 P-state 间切换时，部分事件（如 `SQ_WAVE_CYCLES` 使用的 event 24）不累计；锁定任意稳定性能档后恢复。

先确认当前性能档：

```bash
amd-smi metric -l -g 0 --json
# "perf_level": "AMDSMI_DEV_PERF_LEVEL_AUTO"
```

采集前锁定 GPU0：

```bash
sudo amd-smi set -g 0 -l STABLE_PEAK
```

`STABLE_STD` 与 `STABLE_PEAK` 均验证可用。课程记录使用 `STABLE_PEAK`，以保留较高时钟。采集结束后必须恢复：

```bash
sudo amd-smi set -g 0 -l AUTO
```

::: warning root 不是根因
root 但保持 `AUTO` 时计数器仍为 0；root 只用于执行 `amd-smi set`。`kernel.perf_event_paranoid` 从 4 调到 1、以及不使用 torch 的原生 HIP probe，也都不能修复，因此这不是 perf 权限、容器能力或 torch 库栈问题。
:::

### 完整复现命令

```bash
PY=$(command -v python)
PROF=$(command -v rocprofv3)
CONDA=$("$PY" -c 'import pathlib, _rocm_sdk_core; print(pathlib.Path(_rocm_sdk_core.__file__).resolve().parent / "lib")')
LINK=$(mktemp -d)
ln -s "$CONDA/libhsa-amd-aqlprofile64.so.1" "$LINK/libhsa-amd-aqlprofile64.so"
RAW_OUT=$(mktemp -d /tmp/pmc-raw.XXXXXX)
DERIVED_OUT=$(mktemp -d /tmp/pmc-derived.XXXXXX)

sudo amd-smi set -g 0 -l STABLE_PEAK

LD_LIBRARY_PATH="$LINK:$CONDA" HIP_VISIBLE_DEVICES=0 "$PROF" \
  --pmc "SQ_INSTS_VALU SQ_WAVE_CYCLES SQ_BUSY_CYCLES SQ_WAVES_sum" \
  --pmc "GL2C_HIT_sum GL2C_MISS_sum" \
  --pmc "GRBM_COUNT GRBM_GUI_ACTIVE" \
  --pmc "TA_TA_BUSY" \
  --kernel-trace --marker-trace -f json -d "$RAW_OUT" -- \
  "$PY" runs/course-all-20261120/scripts/prof_gemm_bf16.py

LD_LIBRARY_PATH="$LINK:$CONDA" HIP_VISIBLE_DEVICES=0 "$PROF" \
  --pmc "L2CacheHit OccupancyPercent MemUnitBusy MeanOccupancyPerActiveCU MeanOccupancyPerCU VALUInsts Wavefronts" \
  --kernel-trace --marker-trace -f json -d "$DERIVED_OUT" -- \
  "$PY" runs/course-all-20261120/scripts/prof_gemm_bf16.py

sudo amd-smi set -g 0 -l AUTO
```

### 验证结果

修复后，GEMM 2048³ 的 5 个 `MEASURED` kernel 中位数为：

| 指标 | 结果 |
|---|---:|
| L2 命中率（`L2CacheHit`） | 84.10% |
| `OccupancyPercent` | 18.21% |
| `MeanOccupancyPerActiveCU` | 14.39 |
| `MemUnitBusy` | 55.10% |
| `VALUInsts` | 2374.11 / wave |
| `Wavefronts` | 1936 |

Qwen3 的 prefill/decode 也全部恢复非零：

| 模型/窗口 | L2 命中率 | OccupancyPercent | MeanOcc/ActiveCU | MemUnitBusy |
|---|---:|---:|---:|---:|
| 0.6B prefill | 34.24% | 16.58% | 37.35 | 13.89% |
| 0.6B decode | 19.28% | 0.069% | 7.58 | 2.33% |
| 4B prefill | 33.79% | 23.25% | 38.08 | 25.47% |
| 4B decode | 25.81% | 0.110% | 7.57 | 2.90% |

decode 的 `OccupancyPercent` 会被大量微小 kernel 拉低；判断 active-CU 占用应同时看 `MeanOccupancyPerActiveCU`，不要把单 kernel 的 `OccupancyPercent` 中位数当成整段 decode 的平均占用率。

原始 trace、汇总和复现脚本见 `runs/course-all-20261120/PROFILE.md`、`runs/course-all-20261120/hw-profile-rocprofv3/pmc-stable-peak-*` 与 `runs/course-all-20261120/scripts/summarize_pmc.py`。

### 使用边界

- PMC 在 `STABLE_STD` / `STABLE_PEAK` 下采集；延迟和吞吐在 `AUTO` 下采集。两种时钟状态的绝对耗时不能混用。
- `STABLE_PEAK` 会改变频率和功耗行为；报告必须记录性能档，不应把不同档位的延迟放在同一张表里直接比较。
- gfx1100 没有名为 `VALUBusy` 的派生指标；使用 `VALUInsts` 或原始 `SQ_INSTS_VALU`。
- 多卡机器必须显式指定 `-g 0`，不要对共享 GPU 执行 `-g all`。课程环境中 GPU1 可能承载他人 vLLM 服务。

### 排查顺序

1. `rocprofv3 --list-avail` 确认目标计数器存在。
2. 看到 `aqlprofile API table load failed` 时，先检查无版本 `.so` 是否存在。
3. `--pmc` 能启动但目标值全为 0 时，检查 `amd-smi metric -l` 是否为 `AUTO`。
4. 锁定 `STABLE_STD`/`STABLE_PEAK` 后重跑，并保留一次同 workload 的原始计数与派生计数。
5. 采集结束恢复 `AUTO`，在记录中写明采集档位。

## 延伸阅读

- [ROCm/TheRock#7085](https://github.com/ROCm/TheRock/issues/7085)：运行时 wheel 缺少无版本 aqlprofile symlink。
- [ROCm/rocm-systems#8523](https://github.com/ROCm/rocm-systems/issues/8523)：`AUTO` dVFS 下 `SQ_WAVE_CYCLES` 恒为零；稳定性能档可恢复。
- `runs/course-all-20261120/PROFILE.md`：本课程的完整命令、结果与原始证据索引。
