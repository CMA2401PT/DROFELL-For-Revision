# DROSIM 审稿级真机补测

> 更完整的研究问题、事件语义、失败判据和数据交接规则见
> `../docs/14-4到8卡性能瓶颈测量流程.md`。本文件保留为命令速查。

目标：辨识 4→8 卡回落中的 GPU compute、D2H/H2D、序列化/TCP、sync wait 和
host/PCIe 共享资源代价。所有正式格固定为**每客户端 256 输出 token**，不再沿用固定总
token 预算。

## 1. 端到端矩阵

在 8 卡机、仓库根目录执行：

```bash
PY=/home/dai/projects/LLM/dflash_related/.venv/bin/python
$PY simulate/tests/test_measurement_tools.py
$PY simulate/hardware/run_hop_breakdown.py \
  --python "$PY" \
  --run-prefix hop-breakdown-v1 \
  --scales 4 8 \
  --clients 1 64 128 \
  --deployments combined-request-agent \
  --repeats 5 \
  --output-tokens-per-client 256 \
  --sample-every 8 \
  --monitor-interval-ms 100
```

这会产生 30 个带 instrumentation 的独立 network process，另加 3 个 L4/c64
instrumentation-off 对照，用于验证观测扰动不超过 5%。每个正式 run 都包含原始 token 时间线、
decode-graph counters、100 ms GPU monitor 和抽样后的：

- `decoder_batch_selected`
- `device_to_host_enqueue` / `host_to_device_enqueue`
- `operate_sync_wait`
- `transport_serialize` / `transport_socket_write` / `transport_deserialize`

如果系统安装了 `pidstat`/`mpstat`，runner 会自动保存进程与 CPU 监控；没有则在
environment 中明确记录 unavailable，不会伪造空数据。

## 2. H2D/D2H 微基准

```bash
$PY simulate/hardware/bench_host_device_transfer.py \
  --devices 0 1 2 3 4 5 6 7 \
  --batches 1 2 4 8 \
  --hidden-size 4096 \
  --warmup 100 \
  --iterations 1000 \
  --include-pinned \
  --out simulate/results/reviewer-grade/hop-breakdown-v1-transfer.json
```

pageable 路径是当前生产路径；pinned 只作对照。脚本分别报告每卡隔离 copy 与 8 卡并发
copy，后者用于识别共享 PCIe/root-complex/host-memory 带宽。

## 3. FasterPickler 微基准

```bash
$PY simulate/hardware/bench_transport_serialization.py \
  --batches 1 2 4 8 \
  --hidden-size 4096 \
  --warmup 100 \
  --iterations 1000 \
  --out simulate/results/reviewer-grade/hop-breakdown-v1-serialization.json
```

## 4. 完整性审计

```bash
$PY simulate/hardware/analyze_hop_breakdown.py \
  --journal simulate/results/reviewer-grade/hop-breakdown-v1-hardware-journal.json \
  --required-repeats 5 \
  --out simulate/results/reviewer-grade/hop-breakdown-v1-analysis.json
```

缺文件、缺事件或不足 5 次独立重复时脚本退出码为 2。通过该测量门仍不等于论文准入；
结果还要进入单机 profile 的 calibration/holdout 与 G1–G4 统一验收。
