# Ch1–Ch9.3 实验缺口与待确认清单

- 对应记录：`runs/course-all-20261120/SUMMARY.md`
- 日期：2026-11-20
- 用法：请直接按编号反馈“先做哪些 / 哪些不做 / 哪些参数由你提供”。带 **[可自行修]** 的项不需要外部资源，我可以直接补实现或文档；带 **[需资源]** 的项需要硬件、模型、权限或课程口径确认。

## A. P0：阻断“完整闭环”的资源与参数

- [ ] **A01 [需资源] W7900D 准确参数与实机访问**
  - 章节：Ch1、Ch2、Ch3、Ch6、Ch9.3
  - 缺：型号确认（W7900D 还是 W7900）、显存、峰值 BF16/FP16 算力、HBM 带宽、P2P/xGMI 拓扑、单/双卡数量、云端可用时间。
  - 现状：所有 W7900D 结果均为 `pending` 或声明参数下的解析预测；7900 XTX 数字没有冒充 W7900D。
  - 需要反馈：由谁提供参数表？是否有 ≥2 卡/≥4 卡窗口？

- [x] **A02 [已定位] `rocprofv3` kernel trace 已完成；PMC 目标计数器在 gfx1100 返回 0**
  - 章节：Ch2、Ch8
  - 已做：用 torch 环境自带的 `rocprofv3 1.3.5` + `--kernel-trace --marker-trace` 采集 GEMM、Qwen3-0.6B、Qwen3-4B；产物在 `hw-profile-rocprofv3/`。
  - 已得：GEMM kernel 平均 0.502 ms；4B prefill GPU busy 85.86%，decode 46.94%；0.6B decode GPU busy 30.39%。
  - PMC 复测：补 conda `libhsa-amd-aqlprofile64.so` 无版本 symlink 后，`--pmc` 可启动；`SQ_BUSY_CYCLES` / `SQ_WAVES_sum` 有值，但 `SQ_INSTS_VALU`、`SQ_WAVE_CYCLES`、`GL2C_HIT/MISS`、`TA_TA_BUSY`、`GRBM_*` 及派生 `L2CacheHit` / `OccupancyPercent` 均为 0。
  - 排除项：root + 匹配插件、`perf_event_paranoid=1`、不依赖 torch 的原生 HIP 探针结果一致，因此不是 root/容器权限，也不是 torch 库栈问题。
  - 仍缺：可用的 L2 命中率、VALU 利用率和 occupancy 数据。若课程必须要求这些指标，需要换支持该计数器集合的 GPU/ROCm/driver 环境；否则只能以 kernel 时间线归因为最低要求。
  - 需要反馈：课程是否接受“`SQ_BUSY_CYCLES` / `SQ_WAVES_sum` 的有限 PMC + kernel 时间线”，还是必须换环境补齐 L2/VALU/occupancy？

- [ ] **A03 [需资源] `rocm-bandwidth-test`**
  - 章节：Ch5
  - 缺：ROCm 原生 H2D/D2H、Host 延迟、P2P 基线。
  - 现状：只有 PyTorch pinned copy probe，最高约 14.3 GB/s H2D / 14.17 GB/s D2H。
  - 需要反馈：是否安装该工具，或接受 PyTorch probe 作为替代基线？

- [ ] **A04 [需资源] 真实多 GPU 性能环境**
  - 章节：Ch5、Ch6、Ch9.3
  - 缺：2–4 卡 GPU AllReduce/AllGather/P2P 实测，TP/PP 真实通信开销。
  - 现状：gloo 只验证语义；Ch5/Ch6 通信是 `analytical_uncalibrated`；Ch9.3 是 4× 声明 9070 XT。
  - 需要反馈：能否提供 2 卡同机 PCIe、4 卡同机、或跨节点 RoCE 环境？

- [ ] **A05 [需资源] 专用 vLLM 实例**
  - 章节：Ch7、Ch8
  - 缺：可独立启停、固定版本/参数的 vLLM 服务。
  - 现状：GPU1 的 `ch7-vllm` 是他人共享实例，未重启、未改配置；C8 首次出现 3.19 s TTFT 异常尖峰。
  - 需要反馈：能否分配一张空卡或允许我启动独立容器？vLLM 目标版本是否固定 0.29.0？

- [ ] **A06 [需资源] Qwen3-4B INT4/FP4/INT8 权重与精度评测集**
  - 章节：Ch4
  - 缺：匹配 4B 的 AWQ/GPTQ/GGUF/FP4 checkpoint；MMLU-mini 或 PPL 评测脚本与基线。
  - 现状：只有 BF16 完整推理、FP8 仅加载；无 INT4 真机、无 PPL。
  - 需要反馈：用哪个量化格式/仓库？精度评测用 PPL 还是 MMLU-mini 子集？

- [ ] **A07 [需资源/可自行修] FP8 推理卡住定位**
  - 章节：Ch4
  - 缺：最小复现、进程栈、驱动/ROCm/PyTorch/Transformers 版本矩阵、是否 kernel 不支持。
  - 现状：FP8 权重加载与显存测量成功（4.443 s，5.864 GB delta），推理长时间无输出后被 `kill -TERM`。
  - 需要反馈：是否允许我花时间做最小复现并提 issue？还是先以“仅加载成功”结案？

## B. P1：模拟器与工具缺陷（多数可直接修）

- [ ] **B01 [可自行修] Ch7 plan 级 SLA 没有传给 run 级 goodput**
  - 证据：`ch7-open-loop-1-64` 的 plan 有 `p99_ttft_ms=1000` 等，但 run 的 `goodput.sla_declared=false`、`sla_satisfied=null`。
  - 影响：32 rps 实际 114/133 合格，内置 goodput 却报 36 req/s；64 rps 实际 63/260，内置报 67.5 req/s。
  - 建议：sweep 把 plan SLO 显式注入每次 run，或在报告层重算 goodput。

- [ ] **B02 [可自行修] sweep 不输出 SLA violation ratio**
  - 章节：Ch7
  - 现状：只能从 `requests` 逐条重算（16/32/64 rps 分别 0%、14.29%、75.77%）。
  - 建议：在 point/run 输出 `sla_violation_ratio`、`qualified_requests`、`total_evaluated`。

- [ ] **B03 [可自行修] 普通 request row 缺 queue wait / execution time**
  - 章节：Ch7
  - 现状：必须读 `detailed_events` 的首个 `STEP_START` 才能拆排队与执行；16/32/64 rps 的排队占比为 46.24%/84.906%/96.011%。
  - 建议：request 行直接输出 `queue_wait_ms`、`execution_ms`、`prefill_ms`、`decode_ms`。

- [ ] **B04 [可自行修] `detailed_events` 默认值与教程描述不一致**
  - 章节：Ch7
  - 现状：schema 默认 true，教程暗示需手动打开；学生容易拿不到事件。
  - 建议：文档明确“默认开/关 + 如何开 + 产物大小”。

- [ ] **B05 [可自行修] 官方教学 plan 只有 1-token output，TPOT/ITL 为 null**
  - 章节：Ch7
  - 影响：无法用官方 plan 验证 TPOT SLA，只能验证 TTFT/E2E。
  - 建议：补一个 output_tokens ≥ 8 的官方 plan，或在文档注明 TPOT 必须用自定义 plan。

- [ ] **B06 [可自行修] Ch5/Ch6/Ch7 没有动画生成器**
  - 缺：Ring-AllReduce 消息传递、TP/PP 每卡显存水位与通信时序、排队→batching→KV 水位→SLA 突破。
  - 现状：只有静态 SVG/HTML/CSV。
  - 建议：先做 JSON timeline + 最小 HTML/JS 播放器；数据直接读 run 的 events/detailed_events。

- [ ] **B07 [可自行修] Ch5 native PP affine 校准缺文档指引**
  - 现状：默认 affine 拟合失败 `INVALID_COMMUNICATION_FIT`；改用 `--native-pp-model size-lookup` 成功。
  - 负结果：384 B 落在 `downward_jump` 区间，被 `OUT_OF_DOMAIN` 拒绝。
  - 建议：教程补“什么时候用 affine、什么时候必须用 size-lookup”，并把 384 B 负结果作为教学案例。

- [ ] **B08 [可自行修] Ch9 Agent catalog 只支持 BF16**
  - 证据：INT4 请求返回 422 `UNSUPPORTED_PRECISION`，要求 `weight_precision=BF16` 且 `kv_precision=BF16`。
  - 影响：Ch9.1 的“BF16 vs INT4 端侧收益”无法完成。
  - 建议：扩展 catalog 的 INT4/FP8 权重与 KV 支持，或在课程中明确“9.1 只做 BF16”。

- [ ] **B09 [需决策] Ch9.2 的 TPOT/throughput 目标仍是 TBD**
  - 现状：request 中 `tpot_ms.p99=null`、`throughput_tok_s=null`，只判定 TTFT<2 s 与 completion≥0.99。
  - 需要反馈：这两个目标填什么值？是否按 4B/1.7B 分档？

- [ ] **B10 [可自行修/需决策] Ch9.3 轨迹来源与课程文案不一致**
  - 课程文案：给定一个确定 Claude Code deep-research 轨迹，50+ subagent。
  - 仓库现状：52 节点 `authored fork/join teaching workload`，明确“not a captured claudecode trace”。
  - 需要反馈：是否有真实轨迹可脱敏提供？若没有，是否把课程文案改为“合成教学轨迹 + 可替换真实轨迹接口”？

- [ ] **B11 [可自行修] Ch2 课程校准仍是 test_fixture，不是硬件拟合**
  - 现状：`compute_efficiency=0.5`、`bandwidth_efficiency=0.8`，`source_kind=test_fixture`，`hardware_validated=false`。
  - 建议：用 7900 XTX 的 GEMM/带宽数据做一次真实 `calibrate`，并在报告中标注“7900 XTX 校准，不外推 W7900D”。

- [ ] **B12 [可自行修] vLLM 指标语义 `max_concurrency` vs `max_concurrent_requests`**
  - 现状：日志 `Peak concurrent requests` 约为提交 `max_concurrency` 的 2×（C2→4、C4→8、C8→16、C16→32、C32→64）。
  - 建议：报告固定使用 benchmark JSON 的 `max_concurrent_requests`，并解释它是服务端实际并发，不是客户端提交并发。

- [ ] **B13 [可自行修] 缺统一的 run 摘要生成器**
  - 现状：本记录的表格是脚本临时提取；不同章节字段差异大。
  - 建议：增加 `mlsys-sim summarize-run`，输出统一 `summary.json`（status、evidence_level、counts、latency、throughput、SLA、memory、limitations）。

## C. P2：实验覆盖缺口

- [ ] **C01 [需资源] Ch3 缺受控 `rocm-smi` 加载前后对照与 vLLM 启动日志**
  - 缺：冷启动基线、模型加载后显存差、Total Blocks/block size/KV tokens 原始日志。
  - 现状：只有现有 vLLM metrics（block_size 16、num_gpu_blocks 1529、kv tokens 24464、util 0.9）和解析账本。
  - 建议：在一张空卡上做“空载→加载→启动 vLLM→压测”的受控序列。

- [ ] **C02 [需资源] Ch3 容量边界只在 fixture 中 OOM**
  - 现状：4 token SUCCESS、8 token OOM，括号 4→8，resolution 4；`hardware_validated=false`。
  - 缺：16 GiB 真机上的首次 OOM/拒绝点。
  - 建议：用 Qwen3-4B 在 9070 XT 或 7900 XTX 上限制显存，逐步抬高并发/上下文。

- [ ] **C03 [需资源] Ch4 真机多精度矩阵不完整**
  - 缺：INT8、INT4、FP4 的 TTFT/TPOT/PPL；BF16/FP8/INT4 的反直觉对比。
  - 现状：BF16 完整、FP8 仅加载、其余仅模拟。
  - 建议：优先补 INT4 GPTQ-128 与 FP8 推理；PPL 用固定 1–2 个数据集子集。

- [ ] **C04 [需资源] Ch5 缺 1 KB–1 GB 真机 Ring-AllReduce 曲线**
  - 现状：解析曲线完整；真机只有 PCIe copy probe 与 gloo 语义。
  - 缺：2/4 rank GPU 集合通信、xGMI/NVLink/RoCE 实测拐点。
  - 建议：至少补 2 卡同机 PCIe 的 1 KB–1 GB AllReduce；xGMI/RoCE 标注“不可用”。

- [ ] **C05 [需资源] Ch6 缺真实 TP/PP 性能**
  - 现状：32B TP2/PP2 是 48 GiB 声明参数下的解析预测；gloo 不是 GPU 性能。
  - 缺：每 rank 显存、AllReduce 次数/耗时、pipeline 气泡的真机数据。
  - 建议：2 卡先做 7B/4B TP2，4 卡再做 32B TP2/PP2。

- [ ] **C06 [需资源/可自行修] Ch7 缺“模拟拐点 vs 真机拐点”的同口径对齐**
  - 现状：模拟是 Qwen3-4B/自定义负载/1000-50-5000 ms SLO；真机是 9070 XT/共享 vLLM/256-64/不同 SLO。
  - 缺：同一模型、同一输入输出分布、同一 SLO 下的模拟与真机对照。
  - 建议：先固定一套 workload，再分别跑 `sweep` 与 `vllm bench serve`。

- [ ] **C07 [需资源] Ch8 缺多框架横评**
  - 缺：llama.cpp、SGLang、Ollama、MIGraphX 的吞吐/显存/失败模式。
  - 现状：已有 vLLM、torch eager、rocprofv3 kernel profile（0.6B/4B）；仍无多框架横评与 PMC。
  - 建议：至少补 llama.cpp 与 SGLang 的同模型/同 workload 对比；否则课程文案降级为“单框架案例 + 内核归因”。

- [ ] **C08 [需资源] Ch9.1 缺真正端侧部署**
  - 现状：桌面 GPU 上的协议验证 + BF16 HF 推理；无模型转换、量化、移动端/边缘设备接口。
  - 缺：端侧 runtime、内存上限、功耗、温升、离线场景。
  - 建议：明确 9.1 是“资源约束模拟 + 桌面 GPU 代理验证”，或提供端侧设备。

- [ ] **C09 [需资源] Ch9.2 缺真机留出压测**
  - 现状：所有并发/长上下文结果来自 Agent 模拟；16K 输入被 422 拒绝。
  - 缺：真实 vLLM 在短/长输入下的 P99 与 completion。
  - 建议：用独立 vLLM 实例复跑短 256、长 8192、超长 16384 三档。

- [ ] **C10 [需资源] Ch9.3 缺真实轨迹与 ≥4 卡 W7900D 集群**
  - 现状：52 节点合成 DAG、4× 声明 9070 XT；所有墙钟为解析预测。
  - 缺：真实 50+ subagent 轨迹、真实多卡、真实逐请求时间线、功耗/成本。
  - 建议：先提供脱敏轨迹；集群可先用 4× 16 GiB 做缩小版，32B 部分继续模拟并标注外推边界。

## D. P3：记录与流程缺口

- [ ] **D01 [需决策] “预测先于测量”的时间戳没有全部建立**
  - 证据：`ch7-poisson/manifest.json` 中 `historical_prediction_before_measurement_established=false`。
  - 影响：无法严格证明预测未偷看实测。
  - 建议：后续每次先落盘 prediction，再跑 hardware，并在 manifest 记录 `prediction_stored_at` 与 `measurement_started_at`。

- [ ] **D02 [需决策] 课程存在两套 SLO 口径**
  - 现状：自定义 Ch7 plan 用 1000/50/5000 ms；官方 Poisson/closed-loop 用 30 ms E2E/TTFT。
  - 影响：学生容易把“25 rps 通过”和“16 rps 通过”直接比较。
  - 建议：课程明确哪套是主线；两套都保留时，在图表和结论中强制标注 SLO。

- [ ] **D03 [需决策] 大产物留存策略**
  - 现状：`detailed_events`/`requests` 使单个 sweep 达数十 MB；本记录对大扫描只保留 compact summary。
  - 建议：git 只存 `manifest/result/report/points.csv/compact-summary`；原始 events 放对象存储或本地 `runs/` 忽略目录。

- [ ] **D04 [可自行修] 当前 `runs/` 未纳入 book 仓库提交**
  - 现状：`runs/course-all-20261120/` 是未跟踪目录，约 19 MB（归档后略增）。
  - 需要反馈：是否要把 `SUMMARY.md`/`GAPS.md` 放进 `docs/` 或 `experiments/` 提交 PR？原始 artifact 是否只留在本地？

- [ ] **D05 [可自行修] Ch7 共享实例异常值处理规则未定**
  - 现状：C8 首次 TTFT P99 3187.30 ms，重跑 64.95 ms；已同时保留。
  - 建议：固定“至少 3 次、取中位数、异常值保留并标注”的压测协议。

- [ ] **D06 [需决策] 课程章节状态需要统一命名**
  - 现状：本记录用 PASS/PARTIAL；飞书用 🚧；模拟器用 COMPLETED/FAILED。
  - 建议：统一为 `PASS_SIM`、`PASS_HW`、`PARTIAL`、`BLOCKED`、`FAIL`，避免“COMPLETED”被误读为“硬件验证通过”。

## 建议处理顺序

1. **先解决 A01/A03/A04/A05**：没有 W7900D 参数、带宽工具、多卡环境和独立 vLLM，Ch2/5/6/7/9.3 的课程闭环无法真正完成。A02 的 kernel trace 已补；PMC 已确认是 gfx1100 计数器返回 0，不是权限问题。
2. **并行修 B01–B04、B08、B13**：这些是模拟器输出可信度与可教性的核心问题，不需要额外硬件。
3. **再补 C03/C04/C06/C09**：优先做能形成“预测 vs 实测”对照的最小实验，而不是追求全矩阵。
4. **最后处理 D01–D06**：统一记录规范，避免后续再次出现“fixture 被当成实测”“7900 XTX 被写成 W7900D”。

## 需要你直接回复的最小信息

1. W7900D 参数表与可用时间（A01）。
2. 是否接受 A02 的有限 PMC（`SQ_BUSY_CYCLES` / `SQ_WAVES_sum`），还是提供支持 L2/VALU/occupancy 的 GPU/ROCm 环境；是否安装 `rocm-bandwidth-test`（A02/A03）。
3. 是否能提供 2/4 卡 GPU 与独立 vLLM 卡（A04/A05）。
4. Ch9.2 的 TPOT/throughput 目标值（B09）。
5. Ch9.3 是否有真实脱敏轨迹；没有的话是否接受“合成教学轨迹”表述（B10）。
6. 这份记录是否要提 PR；原始 `runs/` 是否入库（D04）。

