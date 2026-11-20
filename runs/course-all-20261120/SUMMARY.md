# hello-mlsys Ch1–Ch9.3 实验执行记录

- 执行日期：2026-11-20（Asia/Shanghai）
- 记录根目录：`runs/course-all-20261120/`
- 课程来源：飞书项目安排 `[20260702] hello-mlsys 项目安排`
- 模拟器：`/tmp/mlsys-sim-tutorial`，`mlsys-simulator 1.24.0`，commit `83f03da`
- Book 仓库：`/mnt/luyuzhou/hello-mlsys/hello-mlsys`，commit `fd56681`（Ch7 PR 分支）
- 主机：`z-lab-7900`，Linux 6.14.0-37，Ryzen 9 3900X（12C/24T）
- GPU0：AMD Radeon RX 7900 XTX，23.98 GiB，`gfx1100`，本记录中的单卡实验主要使用此卡
- GPU1：AMD Radeon RX 9070 XT，15.92 GiB，`gfx1201`；Ch7 真机 vLLM 使用该卡上已有的共享实例
- 未获得：W7900D 参数与实机、`rocm-bandwidth-test`、≥4 卡 GPU 集群；`rocprofv3` PMC 可启动，但 gfx1100 仅 `SQ_BUSY_CYCLES` / `SQ_WAVES_sum` 有值，L2/VALU/occupancy 计数器返回 0（root 与 perf 权限已排除）

> 本记录严格区分“模拟/解析预测”“测试夹具（fixture）”“真机测量”“行为语义验证”“合成教学轨迹”。凡 `hardware_validated=false` 的结果都不能表述为实测性能；凡 7900 XTX / 9070 XT 的数字都不能写成 W7900D。

## 总体状态

| 章节 | 实验内容 | 状态 | 证据等级 | 一句话结论 |
|-|-|-|-|-|
| Ch1 | Roofline、宏观算子拆解 | PASS（设计上仅 CPU） | analytical / nominal | 18 点、54 分量全部完成；W7900D 参数仍 pending |
| Ch2 | Profile 校准、留出、在线对账 | PASS（fixture + 部分真机） | fixture + RX 7900 XTX 微基准 + rocprofv3 | 留出 MAPE 1.61%；fixture 对账 19/19；真机 GEMM/带宽/端到端/ kernel profile 已补 |
| Ch3 | 权重与 KV 显存、容量边界 | PARTIAL | analytical + fixture + 现有 vLLM metrics | 4B 静态账本与 4→8 token 容量括号完成；缺受控 rocm-smi 对照 |
| Ch4 | 精度、量化、误差预算 | PARTIAL | analytical + BF16 真机 + FP8 仅加载 | 五方案显存/延迟可行；无 INT4 checkpoint、无 PPL，FP8 推理卡住 |
| Ch5 | 通信代价、PCIe、gloo、native PP | PASS（analytical + 部分真机） | analytical + PCIe probe + gloo semantics | 四拓扑拐点与 48 点对账完成；native PP lookup 留出含负结果 |
| Ch6 | DP/TP/PP 显存与通信拆解 | PASS（analytical） | simulated / declared hardware | 32B 单卡与 DP 失败，TP2/PP2 通过；无真实多卡性能 |
| Ch7 | DES、排队、SLA、vLLM 压测 | PASS（sim + 9070 XT 真机） | simulated + real vLLM（共享实例） | 模拟拐点 16→32 rps；真机 C1–C64 全完成；内置 goodput 有缺陷 |
| Ch8 | 框架取舍、容量上界、Profile | PARTIAL | simulated + vLLM + torch eager + rocprofv3 | 筛选器正常；已补 0.6B/4B kernel profile；缺多框架横评，PMC 仅有限计数器 |
| Ch9.1 | 端侧助手、资源约束 | PASS（sim + BF16 真机） | simulated + RX 7900 XTX | 6/6 完成，真机预热后 TTFT 28–226 ms；INT4 被 422 拒绝 |
| Ch9.2 | 单卡 SLA 容量规划 | PARTIAL | simulated | 12/12 完成，并发平台 16；长 8192 输入无 ≤2 s 点；无真机留出 |
| Ch9.3 | 负载启发的集群设计 | PARTIAL | simulated / synthetic trace | 3 候选 2 通过；52 节点是编写式教学轨迹，不是真实 Claude 轨迹 |

## 证据等级定义

- **analytical / nominal**：只做公式与参数计算，不执行 GPU，不拟合效率；`hardware_validated=false`。
- **simulated**：确定性 CPU 离散事件服务模拟；行为、队列、容量逻辑可用，但绝对性能依赖声明参数。
- **test_fixture**：人工构造或手算夹具，用来验证工具链正确性；不是硬件校准。
- **hardware**：在真实 AMD GPU 上执行并落盘；本记录中仅 GPU0 的 PyTorch/vLLM 测量与 GPU1 的共享 vLLM 压测属于此类。
- **semantics**：验证代码行为、同步、数值一致性，例如 gloo all-reduce；不等于多卡性能。
- **synthetic trace**：作者编写的教学 DAG，不是采集得到的真实生产轨迹。

## 统一口径与限制

- 模拟器结果的时间单位是 ms，吞吐是 token/s 或 req/s；显存同时存在 GB/GiB 口径，本文只在明确写 `GiB` 时按 2^30 换算。
- 真机 TTFT/TPOT/E2E 来自 PyTorch event 或 vLLM benchmark JSON；TTFT = 首 token 时间 − 到达时间，E2E = 完成时间 − 到达时间。
- Ch7 模拟 SLA 使用 plan 级 `p99_ttft_ms=1000`、`p99_tpot_ms=50`、`p99_e2e_ms=5000`；官方 Poisson/closed-loop 教学 plan 使用更严格的 30 ms 口径，二者不可混比。
- Ch7 真机 vLLM 是 GPU1 上已存在的共享容器 `ch7-vllm`，未重启、未改配置；存在外部干扰，C8 首次运行出现异常 TTFT 尖峰，已保留并重跑。
- `ch2-compare` 与 `ch2-holdout` 的 `test_fixture_only=true`，即使误差接近零也只证明工具链自洽。
- Ch1、Ch2、Ch5、Ch6、Ch7、Ch8、Ch9.3 的模拟结果全部为 `hardware_validated=false`，不能作为 W7900D 性能承诺。

## Ch1 模拟器估算、Roofline 与瓶颈

- 产物：`ch1/`（`report.html`、`report.md`、`components.csv`、`prefill.svg`、`batching.svg`、`roofline-*.svg`）
- 命令入口：`mlsys-sim chapter1 --plan examples/course/chapter1.plan.json --out <dir>`
- 状态：`COMPLETED`；18 个实验点、54 个宏观分量；`hardware_validated=false`
- 计算口径：full-square attention；nominal 下界，不含排队、launch、通信、容量判定

### 关键预测

| 模型 | Prefill S128 | Prefill S512 | Prefill S2048 | Decode B1 | Decode B4 | Decode B16 |
|-|-:|-:|-:|-:|-:|-:|
| Qwen3-0.6B | 1.629 ms | 2.713 ms | 14.551 ms | 2.046 ms | 2.597 ms | 4.801 ms |
| Qwen3-1.7B | 4.840 ms | 7.892 ms | 35.270 ms | 5.560 ms | 6.112 ms | 8.318 ms |
| Qwen3-4B | 12.091 ms | 20.167 ms | 90.185 ms | 12.806 ms | 13.516 ms | 16.354 ms |

Qwen3-0.6B Prefill 的全局算术强度：S128 为 111.80，S512 为 334.85，S2048 为 798.51 FLOP/byte。所有点的主瓶颈均为 GEMM，其次 Attention，再次 element-wise。

### 结论与边界

- 本章按课程设计不需要 GPU；结果用于建立后续章节的 nominal 基线。
- W7900D 的峰值算力/带宽未确认，因此 `roofline-w7900d-pending.svg` 只保留占位，不填猜测值。
- 本章时间不能直接与 Ch7 的端到端延迟比较，因为计数口径不同且不含服务开销。

## Ch2 第一次真机 Profile 与性能校准

- 产物：`ch2-fit/`、`ch2-holdout/`、`ch2-compare/`、`ch2-course-calibration/`、`hw-ch2-*`、`hw-ch2-calibration/`、`hw-profile-rocprofv3/`
- 状态：fixture 校准、留出、在线对账完成；GPU0 微基准与端到端测量完成
- 证据：fixture 对账 + RX 7900 XTX 真机微基准

### Fixture 校准与留出

| 项目 | 结果 |
|-|-|
| holdout 点 | `prefill-16` |
| MAE | 0.1 ms |
| MAPE | 1.6129% |
| 最大绝对误差 | 0.2 ms |
| coverage | 1.0 |
| 结论 | 工具链自洽；`test_fixture_only=true`，不是硬件校准 |

### 在线对账

`ch2-compare` 覆盖 19/19 指标，`complete=true`，SLA 判定一致：

| 指标 | 预测 | 观测 |
|-|-:|-:|
| E2E P50 | 8.3 ms | 8.3 ms |
| TTFT | 6.1 ms | 6.1 ms |
| 输出吞吐 | 361.4458 tok/s | 361.4458 tok/s |
| SLA satisfied | true | true |

同样地，这是 fixture 对 fixture 的一致性验证，不应写成真机精度结论。

### 课程校准参数

`ch2-course-calibration` 生成 `compute_efficiency=0.5`、`bandwidth_efficiency=0.8`，来源为 `test_fixture`，`hardware_validated=false`。这些参数可作为“课程历史参考”使用，但必须在后续实验中标注来源，不能在看到测试答案后回填。

### RX 7900 XTX 真机测量

Qwen3 端到端（512 输入 / 32 输出，PyTorch event）：

| 模型 | TTFT P50 | TTFT P99 | TPOT P50 | TPOT P99 | Decode |
|-|-:|-:|-:|-:|-:|
| Qwen3-0.6B | 25.838 ms | 29.908 ms | 23.370 ms | 25.438 ms | 42.79 tok/s |
| Qwen3-1.7B | 34.184 ms | 34.204 ms | 22.833 ms | 23.282 ms | 43.80 tok/s |
| Qwen3-4B | 66.439 ms | 66.558 ms | 28.890 ms | 31.786 ms | 34.61 tok/s |

微基准（`hw-ch2-calibration/result.json`）：

| 项目 | 结果 |
|-|-|
| BF16 GEMM 2048³ | 32.224 TFLOP/s（nominal 195 的 0.165） |
| BF16 GEMM 4096³ | 82.121 TFLOP/s（0.421） |
| BF16 GEMM 5120³ | 84.190 TFLOP/s（0.432） |
| BF16 GEMM 8192³ | 84.832 TFLOP/s（0.435） |
| streaming copy 64 MiB | 741.46 GB/s |
| streaming copy 256 MiB | 790.29 GB/s |
| streaming copy 1 GiB | 774.99 GB/s |
| tiny-launch event 计时 | 18.16 µs |

### rocprofv3 kernel profile（BF16 GEMM 2048³）

使用 torch 环境自带的 `rocprofv3 1.3.5`（ROCm 10.0.0）采集 kernel trace + ROCTx marker：

| 指标 | 结果 |
|-|-:|
| 计入窗口 | `MEASURED`，5 次 GEMM |
| kernel 数 | 5 |
| GEMM kernel 平均 | 0.5019 ms |
| GEMM kernel 总时间 | 2.5093 ms |
| 窗口墙钟 | 3.0570 ms |
| kernel 时间/墙钟 | 82.08% |
| 事件计时中位数 | 0.5753 ms |
| 事件 − kernel 平均 | 约 0.0735 ms/次（12.8%） |
| 对应吞吐 | 29.86 TFLOP/s |

归因：GEMM kernel 本身约 0.502 ms；PyTorch event 计时约 0.575 ms，差值约 73 µs/次，主要是 launch、同步和事件计时开销。GEMM 窗口内 kernel 时间占比 100%，没有其他算子混入。产物：`hw-profile-rocprofv3/gemm-2048-markers/`。

PMC 复测（2026-11-20 19:06）：

- 补 conda `_rocm_sdk_core/lib` 中缺失的无版本 `libhsa-amd-aqlprofile64.so` symlink 后，`--pmc` 不再报 `aqlprofile API table load failed`。
- `SQ_BUSY_CYCLES`、`SQ_WAVES_sum` 有非零值；`SQ_INSTS_VALU`、`SQ_INSTS_SMEM`、`SQ_WAVE_CYCLES`、`GL2C_HIT_sum`、`GL2C_MISS_sum`、`TA_TA_BUSY`、`GRBM_COUNT`、`GRBM_GUI_ACTIVE` 全为 0，派生 `L2CacheHit`、`OccupancyPercent`、`MemUnitBusy` 也为 0。
- root + 匹配插件、`perf_event_paranoid=1`、不依赖 torch 的原生 HIP vector-add 三种对照结果一致，排除了 root 权限、perf 权限和 torch 库栈因素。
- 结论：gfx1100 + ROCm 10.0 的目标计数器块在当前环境不可用/返回零；仍只能使用 kernel 时间线，不能把 L2 命中率、VALU 利用率或 occupancy 写成实测值。详见 `PROFILE.md` 与 `hw-profile-rocprofv3/hip-vector-add-pmc*`。

### 缺项

- `rocprofv3` kernel trace 已补；PMC 仅 `SQ_BUSY_CYCLES` / `SQ_WAVES_sum` 有值，L2 命中率、VALU 利用率、wave occupancy 仍不可用（root 与 perf 限制已排除）。
- 真机在 RX 7900 XTX 上，不是 W7900D；不能把 84.8 TFLOP/s 或 790 GB/s 外推为 W7900D 参数。
- 没有把 BF16 GEMM/带宽微基准反拟合进 Ch2 的 `eta/beta`；当前课程校准仍来自 fixture。

## Ch3 显存占用估算：模型权重与 KV Cache

- 产物：`ch3-predict/`、`ch3-workspace/`、`ch3-capacity/`、`ch3-capacity-validation/`
- 状态：解析账本与容量括号完成；真机只复用了现有 vLLM metrics，缺受控启动/关闭对照

### Qwen3-4B 解析账本

`ch3-predict`（16 GiB 设备声明）：

| 项目 | 数值 |
|-|-:|
| 权重 | 7.492 GiB（8,044,936,192 B） |
| 每 token KV | 147,456 B |
| 物理峰值 | 7.498 GiB |
| 设备容量 | 16 GiB |
| 可行性 | 可行 |
| Prefill 预测 | 1621.9 ms |
| Decode step 预测 | 1610.6 ms |

上述时间为 `analytical_uncalibrated`，没有服务排队与真实 kernel 开销。

`ch3-workspace` fixture：1/1 完成，TTFT 2.022 ms，E2E 3.041 ms；只用于 Workspace 显存路径验证。

### 容量边界

`ch3-capacity-validation`：

- 轴：`input_tokens`，取值 4 / 8
- 每个点重复 2 次，共 4 次 trial
- 4 token：2 次 SUCCESS
- 8 token：2 次 OOM
- 括号：`4 → 8`，resolution 4，状态 `BRACKETED`
- `hardware_validated=false`

### 现有 vLLM metrics

从 GPU1 共享 vLLM 服务读取：

| 项目 | 数值 |
|-|-:|
| block size | 16 |
| num_gpu_blocks | 1529 |
| KV cache tokens | 24464 |
| gpu_memory_utilization | 0.9 |
| prefix caching | enabled |
| KV 最大并发（按 4096 token/请求粗算） | 5.97265625 |

### 缺项

- 没有在模型加载前后用 `rocm-smi --showmeminfo vram` 做受控对照；没有 vLLM 启动日志中的 Total Blocks/block size 原始行。
- 容量边界只验证了 simulator 的 fixture 逻辑，没有在 16 GiB 显卡上真实触发 OOM/拒绝。
- 没有把 vLLM 的 24,464 token KV 池反算到模拟器账本。

## Ch4 精度、量化方法与误差预算

- 产物：`ch4-bf16/`、`ch4-fp8/`、`ch4-int8/`、`ch4-fp4_nvfp4/`、`ch4-int4_gptq128/`、`hw-ch4-qwen4b-fp8*`
- 状态：五种精度的模拟可行域完成；BF16 真机端到端完成；FP8 仅加载/显存完成

### 模拟可行域（Qwen3-4B，16 GiB 声明）

| 变体 | 权重 | 物理峰值 | Decode step | 结论 |
|-|-:|-:|-:|-|
| BF16 | 7.492 GiB | 10.492 GiB | 14.520 ms | 可行 |
| FP8 | 3.746 GiB | 6.746 GiB | 7.578 ms | 可行 |
| INT8 | 3.746 GiB | 6.746 GiB | 7.578 ms | 可行 |
| FP4 NVFP4 | 2.107 GiB | 5.107 GiB | 4.541 ms | 可行 |
| INT4 GPTQ-128 | 1.932 GiB | 4.932 GiB | 4.216 ms | 可行 |

全部未触碰 16 GiB 上限；模拟器只做字节账与反量化开销，不建模 PPL、精度损失、真实内核指令支持。

### 真机

- BF16 Qwen3-4B：完整推理成功（`hw-ch8-torch-eager-4b` 同模型参考：TTFT P50 41.411 ms，TPOT P50 29.304 ms，34.13 tok/s）。
- FP8 Qwen3-4B：`hw-ch4-qwen4b-fp8-loadonly/result.json` 加载成功：
  - load 4.443 s
  - GPU0 显存增量 5.864 GB
  - torch allocated 4.115 GiB
  - torch reserved 5.2207 GiB
- FP8 推理尝试长时间无输出，已 `kill -TERM`；没有 FP8 推理延迟结果。
- 没有匹配的 Qwen3-4B INT4 checkpoint；没有运行 MMLU-mini/PPL。

### 缺项

- 缺 INT4/FP4/INT8 真机权重与同口径 TTFT/TPOT/PPL。
- 缺“INT4 小并发可能劣于 BF16”的反直觉验证。
- FP8 卡住需要最小复现、日志和驱动/库版本定位；当前不能把加载成功写成推理成功。
## Ch5 通信代价模型与 Scale-in / Scale-up

- 产物：`ch5-comm-ledger.json`、`ch5-hardware/`、`ch5-native-pp-*`
- 状态：四拓扑曲线、TP 通信占比、PCIe probe、gloo 语义、native PP 校准/留出全部执行
- 证据：analytical + RX 7900 XTX PCIe probe + gloo semantics + fixture native PP

### 四拓扑 Ring-AllReduce 拐点

| 拓扑 | P=2 拐点 | P=4 拐点 | α | β |
|-|-:|-:|-:|-:|
| PCIe 4.0 x16 | 97.66 KB | 195.31 KB | 2 µs/跳 | 25 GB/s |
| xGMI ring | 125.0 KB | 250.0 KB | 1 µs/跳 | 64 GB/s |
| 100G RoCEv2 | 42.97 KB | 85.94 KB | 2 µs/跳 | 11 GB/s |
| NVLink 300 | 585.94 KB | 1171.88 KB | 1 µs/跳 | 300 GB/s |

与模拟器 `ring_allreduce_ms` 对账 48 个点，最大相对误差 `2.15e-16`。

### TP 单层通信占比

Qwen3-1.7B，hidden 2048，28 层，BF16，64 token：

| 拓扑 | TP=2 通信占比 | TP=4 通信占比 |
|-|-:|-:|
| PCIe 4.0 x16 | 82.12% | 314.38% |
| xGMI ring | 34.56% | 137.69% |
| 100G RoCEv2 | 157.77% | 541.34% |
| NVLink 300 | 16.29% | 82.89% |

> 占比 >100% 表示在该消息尺寸与模型假设下，通信时间已超过计算时间；这是“切分收益被通信吃掉”的直接信号。NVLink 相对 PCIe 在 1 GB 消息上最大加速约 11.994×。

### 真机 PCIe probe（GPU0，pinned host memory）

| 消息 | H2D | D2H |
|-:|-:|-:|
| 64 KB | 2.63 GB/s | 1.38 GB/s |
| 1 MB | 11.56 GB/s | 9.95 GB/s |
| 4 MB | 13.70 GB/s | 12.82 GB/s |
| 16 MB | 14.15 GB/s | 13.79 GB/s |
| 64 MB | 14.27 GB/s | 14.08 GB/s |
| 256 MB | 14.30 GB/s | 14.15 GB/s |
| 1 GB | 14.30 GB/s | 14.17 GB/s |

这是 7900 XTX 的 PCIe 路径，不是 xGMI/NVLink，也不是 W7900D 的 P2P 性能。

### gloo 语义验证

2-rank 与 4-rank 均通过：sum 正确、跨 rank 一致、in-place 支持、bf16 支持、blocking 行为成立。`declaration` 明确写明“语义验证，非多卡性能测量”。不要把 gloo 的耗时当作 GPU 集合通信性能。

### native PP 校准与留出

- 默认 affine 拟合：失败，`INVALID_COMMUNICATION_FIT: needs positive finite time-per-byte slope`。
- 正确路径：`--native-pp-model size-lookup`，20 条训练记录，4 个消息尺寸，校准成功。
- 分段：128–256 B 支持；256–512 B 因 `downward_jump` 不支持插值；512–1024 B 支持。
- 留出：192 B 与 768 B 误差 0；384 B 因落在不支持的向下跳变区间被拒绝，标记 `OUT_OF_DOMAIN`。
- 这是保留的负结果，不是工具失败需要隐藏。

### 缺项

- 没有 `rocm-bandwidth-test`；PCIe probe 是 PyTorch copy 路径。
- 没有 1 KB–1 GB 的统一真机 ring AllReduce 数据集；只有解析曲线和 1 KB–1 MB fixture。
- 没有 Ch5 动画生成器；课程要求的“消息传递 + 主导区过渡”动画未产出。

## Ch6 模型并行策略与显存/算力拆解

- 产物：`ch6-tp2/`、`ch6-pp2/`、`ch6-combined/`、`ch6-32b-*`、`ch6-4b-dp2/`
- 状态：DP/TP/PP 解析与官方示例全部完成；无真实多卡 GPU 性能
- 设备声明：32B 场景使用 48 GiB 设备参数、4 卡；4B 场景使用 16 GiB 设备参数

### 32B 在 48 GiB 声明设备上的决策

| 方案 | 结果 | 每 rank 权重 | 物理峰值 | 关键时间 |
|-|-|-:|-:|-|
| 单卡 | FAIL | — | — | 静态显存 72.2377 GB > 51.5396 GB |
| DP=2 | FAIL | — | — | DP 不减少单模型副本，仍然超限 |
| TP=2 | PASS | 30.514 GiB | 35.514 GiB | Prefill 731.483 ms；Decode 83.354 ms |
| PP=2 | PASS | 30.513 GiB | 35.514–35.552 GiB | Stage0 652.906 / 78.560 ms；Stage1 656.823 / 82.451 ms |

TP=2 通信：Prefill 128 次 AllReduce，共 69.669 ms；Decode 128 次，共 2.626 ms。  
PP=2 通信：Prefill P2P 0.534 ms；Decode P2P 0.0105 ms。

### 4B DP=2

- 每 rank 权重 7.492 GiB，物理峰值 12.492 GiB，PASS
- 无模型并行通信

### 官方示例

`examples/tp2.json`、`examples/pp2_analytical.json`、`examples/v1_7/qwen3-native-r2-tp2-pp2.json` 均 `COMPLETED`。

### 缺项

- 没有现成 DP-only 示例，`4b-dp2` 与 `32b-dp2` 是自行构造的。
- 没有真实 2/4 卡 TP/PP 性能；所有 `communication` 都是 `analytical_uncalibrated`。
- 课程要求的“DP/TP/PP 每卡显存水位 + 通信时序动画”未产出。

## Ch7 服务、排队与 SLA

- 产物：`ch7-workload/`、`ch7-poisson/`、`ch7-clients/`、`ch7-open-loop-1-64/`、`ch7-clients-1-64/`、`ch7-queue-debug/`、`ch7-hardware-vllm/`
- 状态：DES 开环/闭环扫描、官方 Poisson/closed-loop、真机 vLLM 并发阶梯全部执行
- 证据：simulated + 9070 XT 共享 vLLM 实例

### 自定义开环扫描（2 s 窗口，2 replications）

SLO：TTFT P99 < 1000 ms、TPOT P99 < 50 ms、E2E P99 < 5000 ms。

| RPS | 请求数 | SLA 合格 | TTFT P99 | 合格/全部 RPS | 结论 |
|-:|-:|-:|-:|-:|-|
| 1 | 3 | 3 | 59.08 ms | 0.75 / 0.75 | 样本不足 |
| 2 | 7 | 7 | 66.40 ms | 1.75 / 1.75 | 样本不足 |
| 4 | 16 | 16 | 71.11 ms | 4.0 / 4.0 | 样本不足 |
| 8 | 32 | 32 | 81.72 ms | 8.0 / 8.0 | 稳定通过 |
| 16 | 61 | 61 | 114.93 ms | 15.25 / 15.25 | 稳定通过 |
| 32 | 133 | 114 | 1236.06 ms | 28.5 / 33.25 | 过载，14.29% 违约 |
| 64 | 260 | 63 | 4336.26 ms | 15.75 / 65.0 | 过载，75.77% 违约 |

- `sla_max_observed=16`
- 稳定 → 过载分界：16 → 32，resolution 16
- `last_observed_stable_rps=16`

### 自定义 closed-loop 客户端扫描（2 s，2 reps）

| 客户端 | 请求数 | SLA 合格 | TTFT P99 | 平均完成 RPS | 违约率 |
|-:|-:|-:|-:|-:|-:|
| 1 | 83 | 83 | 13.0 ms | 20.75 | 0% |
| 2 | 85 | 85 | 87.56 ms | 21.25 | 0% |
| 4 | 89 | 89 | 226.22 ms | 22.25 | 0% |
| 8 | 97 | 97 | 449.36 ms | 24.25 | 0% |
| 16 | 113 | 113 | 839.14 ms | 28.25 | 0% |
| 32 | 146 | 42 | 1680.36 ms | 10.5 | 71.23% |
| 64 | 209 | 42 | 3286.21 ms | 10.5 | 79.9% |

- `sla_max_observed=16`
- 分界：16 → 32，resolution 16

### 官方 Poisson 扫描（10 s，3 reps，SLO E2E P99 < 30 ms）

| RPS | 请求数 | SLA 合格 | TTFT P99 | 违约率 | 稳定性 |
|-:|-:|-:|-:|-:|-|
| 25 | 736 | 734 | 23.82 ms | 0.27% | stable |
| 50 | 1449 | 1406 | 34.75 ms | 2.97% | stable，但 SLA 失败 |
| 75 | 2156 | 1669 | 87.47 ms | 22.59% | stable，但 SLA 失败 |
| 100 | 2975 | 469 | 436.37 ms | 84.24% | stable，但 SLA 失败 |
| 125 | 3751 | 46 | 2811.67 ms | 98.77% | overload |

- `sla_max_observed=25`
- `last_observed_stable_rps=100`
- 稳定/过载分界：100 → 125
- 注意“稳定”只表示队列没有进入无界增长，不等于 SLA 通过。

### 官方 closed-loop 客户端（5 s，3 reps，SLO E2E < 30 ms + TTFT < 30 ms）

| 客户端 | 请求数 | SLA 合格 | TTFT P99 |
|-:|-:|-:|-:|
| 1 | 1502 | 1502 | 13 ms |
| 2 | 1505 | 1505 | 26 ms |
| 4 | 1511 | 43 | 50 ms |
| 8 | 1523 | 9 | 94 ms |
| 16 | 1547 | 9 | 180 ms |

- `sla_max_observed=2`
- 分界：2 → 4，resolution 2

### 排队时间分解

从 `detailed_events` 的首个 `STEP_START` 重算：

| RPS | 平均排队 | 平均执行 | 排队占比 | TTFT P99 |
|-:|-:|-:|-:|-:|
| 16 | 36.249 ms | 42.144 ms | 46.24% | 118.56 ms |
| 32 | 256.797 ms | 45.652 ms | 84.906% | 564.07 ms |
| 64 | 1090.498 ms | 45.313 ms | 96.011% | 2214.645 ms |

### 真机 vLLM（GPU1 / RX 9070 XT，Qwen3-4B）

共享容器 `ch7-vllm`，host `127.0.0.1:18000`；256 输入 / 64 输出；C1 为 32 prompts，其余为 64 prompts。

| 并发 | 完成/失败 | req/s | 输出 tok/s | TTFT P99 | TPOT P99 | ITL P99 |
|-:|-:|-:|-:|-:|-:|-:|
| 1 | 32/0 | 1.070 | 68.50 | 36.34 ms | 14.33 ms | 14.85 ms |
| 2 | 64/0 | 2.019 | 129.20 | 69.99 ms | 14.98 ms | 15.59 ms |
| 4 | 64/0 | 3.877 | 248.15 | 53.55 ms | 15.75 ms | 16.24 ms |
| 8（首次） | 64/0 | 5.455 | 349.11 | 3187.30 ms | 16.38 ms | 17.21 ms |
| 8（重跑） | 64/0 | 7.432 | 475.66 | 64.95 ms | 16.38 ms | 17.05 ms |
| 16 | 64/0 | 13.297 | 851.02 | 97.71 ms | 18.05 ms | 20.42 ms |
| 32 | 64/0 | 23.293 | 1490.76 | 145.77 ms | 20.54 ms | 24.07 ms |
| 64 | 64/0 | 37.323 | 2388.66 | 268.27 ms | 24.68 ms | 61.10 ms |

结论：真实 vLLM 没有失败请求，吞吐随并发上升，TTFT/TPOT 单调上升；C8 首次的 3.19 s TTFT 尖峰不可复现，必须保留并作为共享实例噪声证据。vLLM 日志中的 `Peak concurrent requests` 与提交的 `max_concurrency` 呈 2× 关系，应同时记录 JSON 的 `max_concurrent_requests`，不能混用两个指标名。

### Ch7 已知工具缺陷

1. `sweep` 的 plan 级 `slo` 只用于 P99 pass/fail，**不会传给** run 级 goodput。
2. run 级 goodput 读取 experiment 级 `evaluation.ttft_sla_ms` / `tpot_sla_ms`；本实验未声明，故 `sla_declared=false`、`sla_satisfied=null`。
3. 32 rps 实际 114/133 合格，但 run 报 `36 req/s`、`1309 tok/s`；64 rps 实际 63/260 合格，但 run 报 `67.5 req/s`。
4. sweep 不直接输出 SLA violation ratio，需要从 `requests` 重算。
5. 普通 request row 没有 queue wait / execution time，必须读 `detailed_events` 的首个 `STEP_START`。
6. 官方教学 plan 是 1-token output，TPOT/ITL 为 null。
7. 没有 DES 驱动的排队动画生成器。
8. `detailed_events` 在 schema 中默认 true，但教程文字暗示需手动打开，存在文档歧义。

## Ch8 推理框架的系统取舍、Profile 与性能分析

- 产物：`ch8-screen-simulation/`、`ch8-screen-sweep/`、`hw-ch8-torch-eager-4b/`、`hw-profile-rocprofv3/`
- 状态：筛选器、扫描、vLLM 复用、PyTorch eager 参考、rocprofv3 kernel profile 完成；缺多框架横评与 PMC

### Screen simulation

4 个候选：1 PASS / 3 FAIL / 0 UNKNOWN。

| 候选 | 结论 | 首次破坏的约束 |
|-|-|-|
| `joint-pass` | PASS，排名 1，59.9 ms | 全约束满足 |
| `static-memory-fail` | FAIL | 固定预算 + base workspace 需 9,373,696 B，容量仅 1,048,576 B |
| `p99-only-fail` | FAIL | P99 E2E 79.307 ms > 70 ms |
| `throughput-only-fail` | FAIL | min req/s 671.14 < 1000 |

### Screen sweep

2 个候选：1 PASS / 1 FAIL。

- `fast-all-points` PASS，最小窗口输出 50.0 tok/s
- `slow-high-load-fails` FAIL：P99 E2E 995.05 ms > 20 ms，`observed_overloaded`，`load.slo=false`

### 真机参考

- Ch7 vLLM C1–C64 已提供框架侧压测证据。
- Qwen3-4B torch eager（256 输入 / 64 输出）：TTFT P50 41.411 ms，TPOT P50 29.304 ms，decode 34.13 tok/s，torch peak allocated 7.684 GiB。

### rocprofv3 算子级 Profile

用 torch 环境自带的 `rocprofv3 1.3.5` 采集 `--kernel-trace --marker-trace`，用 ROCTx 标记切出 `PREFILL` 与 `DECODE` 窗口。下表是 kernel 时间归因；`GPU busy` 定义为窗口内 kernel 时间之和 / 墙钟，不是硬件 PMC。

Qwen3-0.6B，512 输入，8 个 decode step：

| 窗口 | kernel 数 | kernel 时间 | 墙钟 | GPU busy | 主导类别 |
|-|-:|-:|-:|-:|-|
| Prefill | 1619 | 21.227 ms | 51.717 ms | 41.04% | GEMM 44.94%，elementwise 30.12% |
| Decode（8 step） | 12728 | 93.941 ms | 309.163 ms | 30.39% | elementwise 37.22%，GEMM 32.76%，copy/cat 15.28% |

Qwen3-4B，512 输入，4 个 decode step：

| 窗口 | kernel 数 | kernel 时间 | 墙钟 | GPU busy | 主导类别 |
|-|-:|-:|-:|-:|-|
| Prefill | 2075 | 61.225 ms | 71.312 ms | 85.86% | GEMM 71.27%，elementwise 14.14% |
| Decode（4 step） | 8156 | 85.701 ms | 182.582 ms | 46.94% | GEMM 51.74%，elementwise 26.35%，copy/cat 10.96% |

更细的类别分解：

| 模型/窗口 | GEMM | elementwise | copy/cat | reduce | attention |
|-|-:|-:|-:|-:|-:|
| 0.6B prefill | 44.94% | 30.12% | 11.92% | 4.86% | 8.14% |
| 0.6B decode | 32.76% | 37.22% | 15.28% | 8.91% | 5.70% |
| 4B prefill | 71.27% | 14.14% | 6.16% | 2.26% | 6.15% |
| 4B decode | 51.74% | 26.35% | 10.96% | 6.42% | 4.49% |

逐 step：

| 模型 | decode step 中位数 | decode tok/s | 每 step kernel 数 | 每 step kernel 时间 |
|-|-:|-:|-:|-:|
| Qwen3-0.6B | 36.729 ms | 27.23 | 1591 | 11.743 ms |
| Qwen3-4B | 45.645 ms | 21.91 | 2039 | 21.425 ms |

结论：

1. **Prefill 是 GEMM 主导**，尤其 4B 的 GEMM 占 kernel 时间 71.27%，GPU busy 85.86%。
2. **Decode 不是纯 GEMM**：0.6B 只有 32.76% 的 kernel 时间是 GEMM；4B 为 51.74%。其余主要是 elementwise、KV cache copy/cat、reduce 和 attention。
3. **decode 被 launch/小算子开销拖住**：0.6B GPU busy 仅 30.39%，4B 也只有 46.94%；大量时间花在 kernel 之间的 gap，而不是 tensor core 计算。
4. 这为 Ch8 的框架取舍提供了直接证据：continuous batching、CUDA/HIP graph、算子融合、减少 KV cache 拷贝，比单独追求更大 GEMM 更重要。

> Profile 模式会引入 trace 开销；这里的墙钟与逐 step 事件计时只用于同次 profile 内的相对归因，不作为 Ch2/Ch7 的最终延迟数字。

### 缺项

- 没有 llama.cpp / SGLang / Ollama / MIGraphX 横评。
- `rocprofv3` kernel trace 已补；PMC 复测仅 `SQ_BUSY_CYCLES` / `SQ_WAVES_sum` 有值，L2/VALU/occupancy 计数器返回 0（root、perf、torch 因素已排除）。
- 没有多框架失败模式矩阵，无法回答“显存碎片 / 调度开销 / 降频”中哪一个主导。
## Ch9.1 端侧部署助手（资源受限实战）

- 产物：`ch9-agent/task-9.1*.json`、`hw-ch9-1-qwen1.7b/`
- 状态：Agent 模拟 6/6 完成；GPU0 BF16 真机 6/6 完成（含预热）
- 请求：`inputs/agent/task-9.1.protocol.request.json`

### Agent 模拟

| 指标 | 结果 |
|-|-:|
| 完成 | 6/6 |
| makespan | 12021.733 ms |
| TTFT P50 / P99 | 28.386 / 152.581 ms |
| TPOT P50 / P99 | 11.370 / 12.503 ms |
| decode | 87.95 tok/s |
| E2E P50 / P99 | 1661.567 / 2956.700 ms |
| 聚合吞吐 | 1024 tok / 12.022 s = 85.18 tok/s |
| 峰值 VRAM | 12.490 GiB |
| KV 峰值 | 0.451 GiB |

约束（来自 request）：VRAM ≤ 14 GiB、TTFT P50 ≤ 500 ms、decode ≥ 20 tok/s，均预测通过。模型为 Qwen3-1.7B BF16，声明设备为 RX 9070 XT。

### INT4 对比被拒绝

`task-9.1.int4-probe` 返回 HTTP 422：

- `UNSUPPORTED_PRECISION`
- `deployments[] requires weight_precision=BF16 and kv_precision=BF16 in this catalog`

因此当前 Agent catalog 不支持 BF16/INT4 对比，不能产出课程要求的量化收益结论。

### GPU0 真机 BF16

预热后重跑（`hw-ch9-1-qwen1.7b/result.json`）：

| 请求 | 输入/输出 | TTFT | TPOT | Decode |
|-|-:|-:|-:|-:|
| p01 | 48 / 128 | 28.399 ms | 23.754 ms | 42.10 tok/s |
| p02 | 120 / 128 | 23.871 ms | 22.666 ms | 44.12 tok/s |
| p03 | 512 / 256 | 36.062 ms | 23.229 ms | 43.05 tok/s |
| p04 | 1024 / 256 | 57.565 ms | 22.579 ms | 44.29 tok/s |
| p05 | 2048 / 128 | 108.259 ms | 22.969 ms | 43.54 tok/s |
| p06 | 4096 / 128 | 225.812 ms | 22.825 ms | 43.81 tok/s |

- load 3.632 s
- GPU0 load delta 4.335 GiB（4.654 GB）
- torch peak allocated 4.892 GiB
- torch peak reserved 5.895 GiB
- 首次冷启动 p01 TTFT 558.98 ms；预热后消失，说明必须把冷启动与稳态分开记录

### 缺项

- Agent catalog 只支持 BF16，无法做 INT4/BF16 端侧对比。
- 没有真正的端侧模型转换、量化、移动端助手接口；当前是桌面 GPU 上的协议验证。
- 真机是 RX 7900 XTX，不是课程声明的 9070 XT/端侧设备。

## Ch9.2 单机服务满足 SLA（单卡容量规划实战）

- 产物：`ch9-agent/task-9.2.response.json`、`ch9-2-sweep/`
- 状态：12/12 完成；并发扫描、短/长/超长留出完成；无真机留出压测
- 请求：`inputs/agent/task-9.2.protocol.request.json`

### 基础结果（C16 协议点）

| 指标 | 结果 |
|-|-:|
| 完成 | 12/12 |
| makespan | 6697.647 ms |
| TTFT P50 / P99 | 45.070 / 709.700 ms |
| TPOT P50 / P99 | 24.093 / 32.045 ms |
| E2E P50 / P99 | 2000.138 / 5848.922 ms |
| 聚合吞吐 | 1344 tok / 6.698 s = 200.67 tok/s |
| 峰值 VRAM | 12.494 GiB |
| KV 峰值 | 2.675 GiB |

约束：TTFT P99 < 2000 ms、completion_rate ≥ 0.99 通过；TPOT 与 throughput 目标在 plan 中仍为 `tbd`。

### 并发扫描

| 并发 | TTFT P50 | TTFT P99 | TPOT P50 | E2E P99 | makespan |
|-:|-:|-:|-:|-:|-:|
| 1 | 6024.881 ms | 15006.697 ms | 11.152 ms | 15714.429 ms | 17946.690 ms |
| 2 | 2517.412 ms | 7995.570 ms | 13.992 ms | 9140.185 ms | 10957.182 ms |
| 4 | 594.061 ms | 3635.858 ms | 19.141 ms | 6524.591 ms | 7633.577 ms |
| 8 | 308.274 ms | 1075.597 ms | 22.474 ms | 5845.018 ms | 6746.021 ms |
| 16 | 45.070 ms | 709.700 ms | 24.093 ms | 5848.922 ms | 6697.647 ms |
| 32 | 45.070 ms | 709.700 ms | 24.093 ms | 5848.922 ms | 6697.647 ms |
| 64 | 45.070 ms | 709.700 ms | 24.093 ms | 5848.922 ms | 6697.647 ms |

C16/C32/C64 完全相同，说明预测模型已达到批处理/并发平台；C1 慢是因为 12 个请求被串行化。

### 留出验证

短输入 256 @ C32：

| 指标 | 结果 |
|-|-:|
| 完成 | 12/12 |
| TTFT P50 / P99 | 16.339 / 30.588 ms |
| TPOT P50 / P99 | 11.895 / 12.797 ms |
| E2E P99 | 3087.392 ms |
| TTFT < 2 s | PASS |

长输入 8192 @ C32：

| 指标 | 结果 |
|-|-:|
| 完成 | 12/12 |
| TTFT P50 / P99 | 2457.720 / 8574.913 ms |
| TPOT P50 / P99 | 40.588 / 89.850 ms |
| E2E P99 | 9646.466 ms |
| TTFT < 2 s | FAIL |
| TPOT < 50 ms | FAIL |

长输入 8192 并发 1/2/3/4/5/8 的 TTFT P99：19681.6 / 12626.7 / 9450.8 / 7818.3 / 6790.7 / 5335.3 ms。测试范围内没有 ≤2 s 的安全并发点；C5 的 P99 低于 C4，存在非单调性，不能简单按并发排序裁剪。

16K 输入：

- 输入 16384 @ C4/C5/C8 均 HTTP 422
- `context limit cannot hold the full input plus output target`
- 16K context 是明确拒绝边界，不是静默截断

### 缺项

- TPOT 与 throughput 的目标仍是 TBD，无法给出完整 SLA 判定。
- 没有真实 vLLM/真机留出压测；所有数值是 Agent 模拟。
- 16K 输入被上下文限制拒绝，需要更大的 context 或分块/检索方案。

## Ch9.3 负载启发的推理集群设计

- 产物：`ch9-3-screen-dag/`、`ch9-3-dag-candidates/`、`ch9-3-dag-fork-join/`、`ch9-agent/task-9.3.response.json`
- 状态：3 候选筛选、候选排名、fork/join 示例、Agent 10 请求 DAG 全部完成
- 证据：simulated / synthetic teaching trace

### 3 候选屏幕

| 候选 | 结论 | makespan | 备注 |
|-|-|-:|-|
| `bf16-tp2-small-r2` | PASS，排名 1 | 22924.280 ms | 32B BF16 TP=2 + 7B 双副本 |
| `int4-single-small-r3` | PASS，排名 2 | 24380.265 ms | 32B INT4 单卡 + 7B 三副本 |
| `fp8-single-small-r3` | FAIL | 29665.989 ms | 超过 25000 ms 目标 |

三者均 52/52 completed，无 blocked/failed，资源 drain 完成。关键路径：

`planner → search-07 → analyze-07 → reason-07 → verify-07 → summarize-07 → report-07 → join → critic → final`

设备账本：

| 候选 | 32B 设备 | 32B 利用率 | 32B 物理峰值 | 32B KV 峰值 |
|-|-|-:|-:|-:|
| BF16 TP2 | gpu0/gpu1 | 0.896 | 39.803 GiB | 0.625 GiB |
| FP8 单卡 | gpu0 | 0.934 | 36.818 GiB | 1.250 GiB |
| INT4 单卡 | gpu0 | 0.901 | 29.038 GiB | 1.250 GiB |

`dag-candidates.plan.json` 的排名与 `screen-dag` 一致：BF16 22924.280 < INT4 24380.265 < FP8 29665.989。

### fork/join 示例

`dag-fork-join.plan.json`：4/4 completed，elapsed 29 ms，关键路径 `a → c → d`。

### Agent 9.3

- 10/10 completed
- makespan 66664.346 ms
- TTFT P50 / P99：207.271 / 1992.108 ms
- TPOT P50 / P99：8.293 / 33.083 ms
- decode 120.59 tok/s
- E2E P50 / P99：4052.885 / 34201.587 ms
- 聚合吞吐：4864 tok / 66.664 s = 72.96 tok/s
- 4 个声明 9070 XT 设备：峰值 12.490–12.501 GiB；KV 峰值 2.391 / 1.395 / 0.930 / 0.479 GiB

### 重要边界

- 仓库自带 52 节点 DAG 的 `trace_description` 明确写入：**“authored fork/join teaching workload. This is not a captured claudecode trace.”**
- 因此不能声称复现了 50+ subagent 的真实 Claude deep-research 轨迹。
- 设备是 4× 声明 9070 XT 16 GiB，不是 ≥4× W7900D 48 GiB 集群；所有结果都是解析预测。

## 跨章验证结论

1. **Ch1→Ch2 的 nominal 预测与 fixture 对账自洽**，但真机只在 7900 XTX 上做了补充，尚未形成 W7900D 的“预测-实测-校准-留出”闭环。
2. **Ch3/Ch4 的显存账本方向正确**：4B BF16 权重约 7.5 GiB、每 token KV 144 KiB；量化后权重近似线性下降。但缺真实 PPL 与失败边界。
3. **Ch5/Ch6 的通信模型能在解析层自洽**：Ring-AllReduce 48 点对账误差 2e-16；TP 通信占比能解释“什么时候切分不划算”。但缺多卡 GPU 性能与动画。
4. **Ch7 的排队拐点在模拟与真机上方向一致**：模拟开环 16→32 rps 过载，闭环 16→32 clients 违约；真机 9070 XT 在 C64 仍无失败但 TTFT 上升到 268 ms。两边的绝对容量不可直接比较，因为模型、SLA、输入输出长度和硬件不同。
5. **Ch8 的筛选器能正确拒绝静态显存、P99、吞吐三类失败**；rocprofv3 已补 0.6B/4B kernel profile，显示 decode 的 GPU busy 仅 30%/47%，缺多框架横评与实际可用的 PMC。
6. **Ch9 的 Agent 协议与 DAG 调度可运行**；Ch9.3 目前是教学合成轨迹，不足以支撑“真实 deep-research 集群”结论。

## 工具链验证

- `python -m unittest discover -s tests -v`：**1768 tests，OK，3 skipped**，耗时 142.276 s。
- 日志：`logs/simulator-tests.log`。
- 教程基线存档：`tutorial-baseline/`
  - FIFO：3/3，TTFT P50 19.0 ms，输出吞吐 166.6667 tok/s。
  - KV 池 24 KiB：2/2，1 次抢占，32 recomputed tokens，E2E P50 50 ms。
  - KV 池 48 KiB：2/2，0 次抢占，0 recomputed tokens，E2E P50 28 ms。
  - Web 导出回放：Qwen3-4B 20 请求，TTFT P99 54.14 ms，61.98 tok/s。
- 这些基线说明模拟器自身行为可复现；不代表任何硬件性能。
- `rocprofv3` 必须使用 torch 环境自带的版本（`/home/luyzh/miniconda3/envs/torch/bin/rocprofv3`）；base/系统版本会与 torch 的 `libamd_comgr.so.3` 冲突，报 `LLVM ERROR: inconsistency in registered CommandLine options`。
- 采集 PMC 前需要给 conda 的 `_rocm_sdk_core/lib` 补一个无版本 `libhsa-amd-aqlprofile64.so` symlink；否则 HSA 会报 `aqlprofile API table load failed`。这是加载器路径问题，不是 root 权限问题。

## 复现方式

每个运行目录都有 `manifest.json`，记录输入文件路径、SHA-256、simulator 版本和模式。复现时应：

1. 使用 `/tmp/mlsys-sim-tutorial` 的 `83f03da` 或等价 `mlsys-simulator 1.24.0`。
2. 按 `manifest.json` 的 `input_files[].path` 取回输入，并校验 `sha256`。
3. 输出到新目录；模拟器要求输出目录尚不存在，以保留每次预测。
4. 真机脚本在 `scripts/`，硬件结果在 `hw-*`；不要把 `simulated` 与 `hardware_validated` 混写。

常用入口：

```bash
SIM=/tmp/mlsys-sim-venv/bin/python
cd /tmp/mlsys-sim-tutorial

$SIM -m mlsys_sim chapter1 --plan examples/course/chapter1.plan.json --out runs/ch1-replay
$SIM -m mlsys_sim validate --profile <profile.json> --dataset <holdout.json> --out runs/ch2-holdout-replay
$SIM -m mlsys_sim sweep --plan examples/course/load-poisson.plan.json --experiment examples/course/load-queue.experiment.json --out runs/ch7-poisson-replay
$SIM -m mlsys_sim screen --plan <screen.plan.json> --out runs/ch8-screen-replay
$SIM -m mlsys_sim.agent_service --port 8877

# rocprofv3 kernel profile（必须用 torch 环境自带版本）
PROF=/home/luyzh/miniconda3/envs/torch/bin/rocprofv3
HIP_VISIBLE_DEVICES=0 $PROF --kernel-trace --marker-trace -f json -d <out-dir> -- \
  /home/luyzh/miniconda3/envs/torch/bin/python \
  runs/course-all-20261120/scripts/prof_qwen_steps.py \
  --model /mnt/public/models/Qwen3-4B --prompt-tokens 512 --decode-steps 4 \
  --out <out-dir>/result.json

# PMC（gfx1100：先补 aqlprofile 无版本 symlink）
CONDA=/home/luyzh/miniconda3/envs/torch/lib/python3.14/site-packages/_rocm_sdk_core/lib
LINK=$(mktemp -d)
ln -s "$CONDA/libhsa-amd-aqlprofile64.so.1" "$LINK/libhsa-amd-aqlprofile64.so"
LD_LIBRARY_PATH="$LINK:$CONDA" HIP_VISIBLE_DEVICES=0 $PROF \
  --pmc SQ_BUSY_CYCLES SQ_WAVES_sum SQ_INSTS_VALU GL2C_HIT_sum GL2C_MISS_sum \
  --kernel-trace --marker-trace -f json -d <out-dir> -- \
  /home/luyzh/miniconda3/envs/torch/bin/python \
  runs/course-all-20261120/scripts/prof_gemm_bf16.py
```

具体子命令以 `mlsys-sim --help` 和对应 `manifest.json` 为准。

## 产物目录索引

| 路径 | 内容 |
|-|-|
| `logs/` | 30 个 console 日志，含模拟与硬件运行 |
| `inputs/` | 自定义 plan/experiment 与 Agent 协议请求 |
| `scripts/` | 真机测量脚本 |
| `tutorial-baseline/` | 教程 FIFO/KV/TP/PP/DAG/Web replay 的完整小产物；大扫描只保留 compact summary |
| `ch1/` … `ch9-3-*/` | 各章预测、报告、CSV、SVG、HTML |
| `hw-*` | 真机 JSON（BF16/FP8 load/Ch7 vLLM/Ch9.1） |
| `hw-profile-rocprofv3/` | GEMM/Qwen3-0.6B/Qwen3-4B 的 kernel trace、ROCTx 窗口与归因 summary |
| `ch5-comm-ledger.json` | 通信账本与 48 点对账 |
| `ch7-hardware-vllm/` | C1–C64 vLLM benchmark JSON |
| `ch9-agent/` | 9.1/9.2/9.3 Agent 响应与 INT4 拒绝证据 |

