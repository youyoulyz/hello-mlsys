# rocprofv3 Profile 记录（Ch2 / Ch8）

- 日期：2026-11-20
- 设备：AMD Radeon RX 7900 XTX（GPU0，23.98 GiB）
- 工具：`/home/luyzh/miniconda3/envs/torch/bin/rocprofv3`，版本 1.3.5，ROCm 10.0.0
- Python：`/home/luyzh/miniconda3/envs/torch/bin/python`（torch 2.12.0+rocm10.0.0）
- 采集：`--kernel-trace --marker-trace -f json`；PMC 另见下方
- 产物：`hw-profile-rocprofv3/`

## 为什么必须用 torch 环境自带的 rocprofv3

base/系统 `rocprofv3` 与 torch 的 `_rocm_sdk_core` 会各加载一份 `libamd_comgr.so.3`：

- `/opt/rocm/core-10.0/lib/libamd_comgr.so.3`
- `/home/luyzh/miniconda3/envs/torch/lib/python3.14/site-packages/_rocm_sdk_core/lib/libamd_comgr.so.3`

两者一起加载会报：

```text
LLVM ERROR: inconsistency in registered CommandLine options
CommandLine Error: Option 'spirv-expand-step' registered more than once!
```

改用 torch 环境自带的 `rocprofv3` 后注入正常。

## 命令

```bash
PROF=/home/luyzh/miniconda3/envs/torch/bin/rocprofv3
PY=/home/luyzh/miniconda3/envs/torch/bin/python

# GEMM
HIP_VISIBLE_DEVICES=0 $PROF --kernel-trace --marker-trace -f json \
  -d runs/course-all-20261120/hw-profile-rocprofv3/gemm-2048-markers -- \
  $PY runs/course-all-20261120/scripts/prof_gemm_bf16.py

# Qwen3-0.6B
HIP_VISIBLE_DEVICES=0 $PROF --kernel-trace --marker-trace -f json \
  -d runs/course-all-20261120/hw-profile-rocprofv3/qwen3-0.6b-p512-d8 -- \
  $PY runs/course-all-20261120/scripts/prof_qwen_steps.py \
  --model /mnt/public/models/Qwen3-0.6B --prompt-tokens 512 --decode-steps 8 \
  --out runs/course-all-20261120/hw-profile-rocprofv3/qwen3-0.6b-p512-d8/result.json

# Qwen3-4B
HIP_VISIBLE_DEVICES=0 $PROF --kernel-trace --marker-trace -f json \
  -d runs/course-all-20261120/hw-profile-rocprofv3/qwen3-4b-p512-d4 -- \
  $PY runs/course-all-20261120/scripts/prof_qwen_steps.py \
  --model /mnt/public/models/Qwen3-4B --prompt-tokens 512 --decode-steps 4 \
  --out runs/course-all-20261120/hw-profile-rocprofv3/qwen3-4b-p512-d4/result.json
```

汇总：

```bash
$PY runs/course-all-20261120/scripts/summarize_rocprof.py \
  --results <results.json> --out <summary.json> --labels PREFILL,DECODE
```

## PMC 复测：root、插件路径与 gfx1100 限制

最初的 `aqlprofile API table load failed` **不是 GPU 权限问题**。`strace` 显示 HSA runtime 会查找无版本后缀的 `libhsa-amd-aqlprofile64.so`，而 conda 目录只有 `libhsa-amd-aqlprofile64.so.1`：

```text
_rocm_sdk_core/lib/libhsa-amd-aqlprofile64.so.1
```

用临时目录补一个同版本 symlink 后，PMC 采集可以正常启动：

```bash
CONDA=/home/luyzh/miniconda3/envs/torch/lib/python3.14/site-packages/_rocm_sdk_core/lib
LINK=$(mktemp -d)
ln -s "$CONDA/libhsa-amd-aqlprofile64.so.1" "$LINK/libhsa-amd-aqlprofile64.so"

LD_LIBRARY_PATH="$LINK:$CONDA" HIP_VISIBLE_DEVICES=0 \
  /home/luyzh/miniconda3/envs/torch/bin/rocprofv3 \
  --pmc SQ_BUSY_CYCLES SQ_WAVES_sum SQ_INSTS_VALU GL2C_HIT_sum GL2C_MISS_sum \
  --kernel-trace --marker-trace -f json -d <out-dir> -- \
  /home/luyzh/miniconda3/envs/torch/bin/python \
  runs/course-all-20261120/scripts/prof_gemm_bf16.py
```

另外也验证过 `LD_LIBRARY_PATH=/opt/rocm/lib` 可以绕过加载错误，但它加载的是系统版 aqlprofile，和 conda 的 HSA runtime 不同构建，计数器结果同样为零；这不是推荐路径。

### 结果

| 计数器 | GEMM 2048³ | HIP vector-add | 结论 |
|-|-:|-:|-|
| `SQ_BUSY_CYCLES` | 非零（约 53–54 万 cycles） | 非零 | 可用 |
| `SQ_WAVES_sum` | 1936 | 非零 | 可用 |
| `GRBM_COUNT` | 0 | 0 | 不可用 |
| `GRBM_GUI_ACTIVE` | 0 | 0 | 不可用 |
| `SQ_INSTS_VALU` | 0 | 0 | 不可用 |
| `SQ_INSTS_SMEM` | 0 | 0 | 不可用 |
| `SQ_WAVE_CYCLES` | 0 | 0 | 不可用 |
| `GL2C_HIT_sum` / `GL2C_MISS_sum` | 0 / 0 | 0 / 0 | 不可用 |
| `TA_TA_BUSY` | 0 | 0 | 不可用 |

高层派生指标 `L2CacheHit`、`OccupancyPercent`、`MeanOccupancyPerActiveCU`、`MemUnitBusy` 也全部返回 0。`VALUBusy` 在当前 gfx1100 上甚至没有被 rocprofv3 识别为可用计数器。

### 排除项

- **root 不是解**：用 root + 正确 symlink 重新运行原生 HIP 探针，`rocprof_status=0`，但目标计数器仍为 0。
- **perf 权限不是解**：临时把 `kernel.perf_event_paranoid` 从 4 调到 1，目标计数器仍为 0；测试后已恢复为 4。
- **不是 torch 独有**：不加载 torch 的 HIP vector-add 程序结果相同。
- **不是计数器名写错**：`SQ_BUSY_CYCLES` 和 `SQ_WAVES_sum` 在相同采集里正常返回非零值。

因此当前可用的 PMC 事实是：**gfx1100 + ROCm 10.0 的 rocprofv3 能启动 PMC 并读取 `SQ_BUSY_CYCLES` / `SQ_WAVES_sum`，但 L2、VALU、occupancy 所需计数器块返回 0**。L2 命中率、VALU 利用率和 occupancy 仍然不能作为实测指标写入结论。

## GEMM 2048³ BF16

| 指标 | 结果 |
|-|-:|
| 计入窗口 | `MEASURED`，5 次 GEMM |
| kernel 数 | 5 |
| 每次 kernel 平均 | 0.5019 ms |
| kernel 总时间 | 2.5093 ms |
| 窗口墙钟 | 3.0570 ms |
| kernel 时间 / 墙钟 | 82.08% |
| PyTorch event 中位数 | 0.5753 ms |
| event − kernel | 约 0.0735 ms/次（12.8%） |
| 对应吞吐 | 29.86 TFLOP/s |

归因：GEMM 窗口内 100% 的 kernel 时间是同一个 `Cijk_...` GEMM kernel。event 与 kernel 的差值约 73 µs/次，主要是 launch、同步和事件计时开销。

## Qwen3-0.6B（512 输入，8 decode steps）

| 窗口 | kernel 数 | kernel 时间 | 墙钟 | GPU busy | 主导类别 |
|-|-:|-:|-:|-:|-|
| Prefill | 1619 | 21.227 ms | 51.717 ms | 41.04% | GEMM 44.94%，elementwise 30.12% |
| Decode | 12728 | 93.941 ms | 309.163 ms | 30.39% | elementwise 37.22%，GEMM 32.76%，copy/cat 15.28% |

类别分解：

| 窗口 | GEMM | elementwise | copy/cat | reduce | attention |
|-|-:|-:|-:|-:|-:|
| Prefill | 44.94% | 30.12% | 11.92% | 4.86% | 8.14% |
| Decode | 32.76% | 37.22% | 15.28% | 8.91% | 5.70% |

Top kernels（decode）：

| kernel | 次数 | 总时间 |
|-|-:|-:|
| elementwise manual_unroll | 1344 | 10.332 ms |
| `Cijk_..._MT16x16x128` | 448 | 10.296 ms |
| `Cijk_..._MT16x16x128`（另一变体） | 672 | 8.280 ms |
| reduce_kernel | 904 | 8.252 ms |
| elementwise manual_unroll（另一变体） | 904 | 7.502 ms |

逐 step：1591 kernel/step，11.743 ms kernel/step，墙钟 38.645 ms/step；event 中位数 36.729 ms/step，27.23 tok/s。

## Qwen3-4B（512 输入，4 decode steps）

| 窗口 | kernel 数 | kernel 时间 | 墙钟 | GPU busy | 主导类别 |
|-|-:|-:|-:|-:|-|
| Prefill | 2075 | 61.225 ms | 71.312 ms | 85.86% | GEMM 71.27%，elementwise 14.14% |
| Decode | 8156 | 85.701 ms | 182.582 ms | 46.94% | GEMM 51.74%，elementwise 26.35%，copy/cat 10.96% |

类别分解：

| 窗口 | GEMM | elementwise | copy/cat | reduce | attention |
|-|-:|-:|-:|-:|-:|
| Prefill | 71.27% | 14.14% | 6.16% | 2.26% | 6.15% |
| Decode | 51.74% | 26.35% | 10.96% | 6.42% | 4.49% |

逐 step：2039 kernel/step，21.425 ms kernel/step，墙钟 45.646 ms/step；event 中位数 45.645 ms/step，21.91 tok/s。

显存：peak allocated 7.787 GiB，peak reserved 7.869 GiB。

## 解释

1. **Prefill 是 GEMM 主导**：4B 的 GEMM 占 kernel 时间 71.27%，GPU busy 85.86%，说明 prefill 已经比较接近算力受限。
2. **Decode 不是纯 GEMM**：0.6B 只有 32.76% 的 kernel 时间是 GEMM，4B 为 51.74%；其余是 elementwise、KV cache copy/cat、reduce 和 attention。
3. **Decode 的 GPU busy 明显低于 prefill**：0.6B 30.39%，4B 46.94%。大量时间在 kernel 之间的 gap，说明 eager decode 被小算子 launch 和 KV cache 操作拖住。
4. **框架优化方向**：continuous batching、HIP graph/CUDA graph、算子融合、减少 KV cache 拷贝，比单纯提高 GEMM 峰值更直接。
5. **0.6B vs 4B**：模型越大，GEMM 占比越高，GPU busy 越高；小模型更容易被 launch/小算子开销支配。

## 限制

- **PMC 仅部分可用**：补 conda 的 `libhsa-amd-aqlprofile64.so` symlink 后，`--pmc` 可运行；`SQ_BUSY_CYCLES` / `SQ_WAVES_sum` 有值，但 L2、VALU、occupancy 相关计数器在 gfx1100 上返回 0。root 与 `perf_event_paranoid` 均不改变结果。
- **Profile 开销**：trace 模式会改变墙钟；逐 step 时间只用于同次 profile 内的相对归因，不作为 Ch2/Ch7 的最终延迟。
- **不是 W7900D**：所有数字来自 RX 7900 XTX，不能外推为 W7900D 参数。
- **随机 prompt**：profile 使用随机 token id，不是课程固定数据集；模型结构、层数和 KV 路径与真实请求一致，但 token 内容不影响算子形状。
- **没有 CUDA/HIP graph**：这是 eager PyTorch 路径；vLLM 的 graph/continuous batching 路径未在这里 profile。

## 产物索引

| 路径 | 内容 |
|-|-|
| `hw-profile-rocprofv3/gemm-2048-markers/` | GEMM kernel trace + ROCTx + `summary.json` |
| `hw-profile-rocprofv3/gemm-2048-pmc-matchplugin/` | 匹配 conda aqlprofile 后的 PMC 复测 |
| `hw-profile-rocprofv3/gemm-2048-pmc-full/` | GEMM 原始计数器组合复测 |
| `hw-profile-rocprofv3/hip-vector-add-pmc/` | 不依赖 torch 的原生 HIP vector-add PMC 探针 |
| `hw-profile-rocprofv3/hip-vector-add-pmc-root/` | root + 匹配插件探针 |
| `hw-profile-rocprofv3/hip-vector-add-pmc-paranoid1/` | `perf_event_paranoid=1` 对照 |
| `hw-profile-rocprofv3/qwen3-0.6b-p512-d8/` | 0.6B prefill/decode trace + `result.json` + `summary.json` |
| `hw-profile-rocprofv3/qwen3-4b-p512-d4/` | 4B prefill/decode trace + `result.json` + `summary.json` |
| `scripts/prof_gemm_bf16.py` | GEMM profile 负载 |
| `scripts/prof_qwen_steps.py` | Qwen prefill/decode profile 负载 |
| `scripts/summarize_rocprof.py` | 按 marker 窗口和 kernel 类别汇总 |
| `scripts/probe_hip_pmc.cpp` | 原生 HIP vector-add PMC 对照探针 |

