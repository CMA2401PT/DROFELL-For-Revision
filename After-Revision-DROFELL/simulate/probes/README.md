# probes/ —— 预研探针

设计文档里的每个关键前提，都对应这里一个可重跑的脚本。
输出与解读见 [`../docs/06-预研验证记录.md`](../docs/06-预研验证记录.md)。

**从仓库根运行**（P7/P8 读 `exps/` 下的相对路径）：

```bash
cd /home/dai/projects/LLM/DROFELL
PY=/home/dai/projects/LLM/dflash_related/.venv/bin/python
$PY simulate/probes/p1_virtual_clock.py
```

| 脚本 | 验证什么 | 结论 |
|---|---|---|
| `p1_virtual_clock.py` | 虚拟时钟能否驱动 DROFELL 自己的 `StdEventLoop` | ✅ 3601 虚拟秒 / 0.2 ms |
| `p2_control_plane_import.py` | 控制平面能否脱离 GPU import | ✅ 未触碰 CUDA |
| `p3_real_shadow_state.py` | **真实** `InferStateData` 能否无 GPU 运行 | ✅ canon 三条语义全复现 |
| `p4_memory_twin_no_gpu.py` | 内存数字孪生能否无 GPU 预测真实模型 | ✅ 1.437 GiB，CUDA 0 字节 |
| `p5_meta_tensor.py` | meta tensor 能否承载激活并 pickle | ✅ |
| `p6_harness_scale.py` | 单进程能否承载 256 逻辑节点 | ✅ 29 MiB RSS，20× 实时 |
| `p7_trace_cost_extract.py` | 代价模型的拟合数据是否已存在 | ✅ 每段 2048–8192 样本 |
| `p8_real_anchors.py` | 3090 校准锚点及**真机自身方差** | ✅ 故障配置方差达 45% |
| `p10a_shim_surface.py` | compute node 实际调用的 shim 方法 | ✅ 12 个（非 57） |
| `p10b_operate_seam.py` | 正确的接缝在哪一层 | ⭐ **operate 队列，共 8 个** |
| `p10c_twin_coverage.py` | 真孪生已覆盖多少 | ⭐ **8 个 handler 全部已存在** |
| `p11_scale_anchors.py` | 新增 8 卡矩阵提供什么 | ⭐ 35 格 × 5 重复；最差散布 89.3% |

P1–P6、P10a–c 无参数。P7/P8/P11 可传入其它 run 的路径：

```bash
$PY simulate/probes/p7_trace_cost_extract.py <traces.json> --top 20
$PY simulate/probes/p8_real_anchors.py <summary.csv> --deployment persistent-agent
$PY simulate/probes/p11_scale_anchors.py --unreliable <summary.csv> --scale <summary.csv>
```

> P9（时钟注入点盘点）是一条 grep，不单独成脚本：
> ```bash
> grep -rn "time\.time_ns()\|time\.monotonic_ns()\|perf_counter_ns()" node/ dealers/ utils/
> ```
> 规则：`perf_counter_ns` 保留真实（测 CPU 开销），
> `time_ns`/`monotonic_ns` 换虚拟（测时间线）。
