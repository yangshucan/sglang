# KVCache Footprint Tracer — 设计文档

**日期**: 2026-06-21  
**状态**: 设计完成，待实现

---

## 1. 概述

为 sglang 实现一个**请求粒度**的 KVCache footprint 追踪器，覆盖 KVCache 的完整生命周期：

- **生成** — prefill 时 KV page 的分配与写入
- **搬运** — PD 分离中 prefill→decode 的 KV 传输，prefill 时 Radix Cache prefix match 复用
- **Offload** — GPU → CPU DRAM (L2)，GPU/CPU → Distributed Storage (L3)
- **Eviction** — Radix Cache 驱逐和请求完成后的释放

核心目标：**零开销热路径**写入，**异步消费**分析，**插件化架构**最小侵入。

---

## 2. 已有基础设施分析

sglang 已经具备实现此追踪器所需的大部分基础设施：

| 组件 | 文件 | 能力 |
|------|------|------|
| `KVCacheEventMixin` | `srt/mem_cache/events.py` | Radix Cache 事件发射 (insert/evict) |
| `KVEventsPublisher` | `srt/managers/scheduler_components/kv_events_publisher.py` | Scheduler 步进收集事件 → publish |
| `EventPublisherFactory` | `srt/disaggregation/kv_events.py` | 可注册自定义 publisher |
| `ZmqEventPublisher` | `srt/disaggregation/kv_events.py` | 可靠 ZMQ 发布 + replay |
| `StorageMedium` 枚举 | `srt/disaggregation/kv_events.py` | GPU/CPU/DISK/EXTERNAL 四级存储 |
| `KVCacheEvent` (msgspec) | `srt/disaggregation/kv_events.py` | tag-based 多态事件，omit_defaults |
| OpenTelemetry trace | `srt/observability/trace.py` | OTLP 导出基础设施 |
| Prometheus metrics | `srt/observability/metrics_collector.py` | 指标收集范式 |

### 现有事件的覆盖缺口

| 生命周期事件 | 现有覆盖 | 缺失 |
|---|---|---|
| KV 分配 (alloc) | ❌ 无事件 | BlockAllocated |
| KV 写入 (prefill) | 部分：BlockStored 在 insert 时 | 无 req_id 关联 |
| Prefix 命中 (复用) | 部分：BlockStored 有 parent_block_hash | 无命中/未命中区分 |
| PD 分离传输 | ❌ 无事件 | BlockTransferred |
| Offload (GPU→CPU/Disk) | 部分：StorageMedium 枚举已定义 | 无 BlockOffloaded/Loaded 事件 |
| Eviction | 部分：BlockRemoved 在 evict 时 | 无 eviction reason |
| 请求完成释放 | ❌ 无事件 | AllBlocksCleared 已有但不完整 |

---

## 3. 架构设计

### 3.1 整体架构

```
  sglang 内部 (3 处最小修改)         插件包 (sglang-kv-footprint)
  ┌──────────────────────┐      ┌──────────────────────────────────┐
  │                      │      │                                  │
  │  KVCacheEventMixin   │      │  FootprintEventPublisher         │
  │  └─ _record_*()      │──→──│  └─ publish(batch)                │
  │     (+req_id透传)     │ ZMQ  │     ├─ RingBufferWriter (无锁)    │
  │                      │ 或   │     ├─ FootprintAggregator        │
  │  KVEventsPublisher   │管道  │     │   └─ req_id → Footprint     │
  │  └─ take_events()    │──→──│     └─ FootprintExporter           │
  │                      │      │        ├─ PrometheusMetrics       │
  │  EventPublisherFactory│     │        ├─ JSONL File              │
  │  └─ register("footprint")│  │        └─ OTLP Span               │
  │                      │      │                                  │
  └──────────────────────┘      │  FootprintQueryAPI (可选)         │
                                │  GET /footprint/{req_id}          │
                                └──────────────────────────────────┘
```

### 3.2 设计原则

1. **插件化**: 通过 `EventPublisherFactory.register_publisher()` 注册，无需修改 sglang 核心逻辑
2. **零开销热路径**: 使用无锁 ring buffer，热路径仅一次 CAS 写入，事件丢失可接受
3. **异步消费**: 独立线程批量处理事件，不影响推理延迟
4. **最小内部修改**: 仅 3 处可选参数透传，不破坏现有消费者兼容性

---

## 4. sglang 内部修改（3 处）

### 4.1 KVCacheEvent 基类增加可选字段

**文件**: `sglang/srt/disaggregation/kv_events.py`

```python
class KVCacheEvent(
    msgspec.Struct, array_like=True, omit_defaults=True, gc=False, tag=True,
):
    req_id: Optional[str] = None       # 新增：关联请求
    sequence_id: Optional[int] = None  # 新增：chunked prefill 序列关联
```

`omit_defaults=True` 确保 `req_id=None` 时不序列化，与现有消费者完全兼容。

### 4.2 KVCacheEventMixin 透传 req_id

**文件**: `sglang/srt/mem_cache/events.py`

- `_record_store_event(self, node, medium=None, req_id=None)` — 新增 req_id 参数，透传到 BlockStored
- `_record_remove_event(self, node, medium=None, req_id=None, reason=None)` — 新增参数，透传到 BlockRemoved

### 4.3 RadixCache 调用处传入 req_id

**文件**: `sglang/srt/mem_cache/radix_cache.py`

- `cache_finished_req()`: 有 `req: Req` 参数，传入 `req.rid`
- `cache_unfinished_req()`: 有 `req: Req` 参数，传入 `req.rid`
- `evict()`: 无请求上下文，传 `reason="eviction"`

---

## 5. 事件数据模型

### 5.1 已有事件（增强）

```python
class BlockStored(KVCacheEvent):
    block_hashes: list[int]
    parent_block_hash: Optional[int]
    token_ids: list[int]
    block_size: int
    lora_id: Optional[int]
    medium: Optional[str] = None

class BlockRemoved(KVCacheEvent):
    block_hashes: list[int]
    medium: Optional[str] = None

class AllBlocksCleared(KVCacheEvent):
    pass
```

### 5.2 新增事件

```python
class BlockAllocated(KVCacheEvent):
    """TokenToKVPoolAllocator 分配新 slot"""
    block_hashes: list[int]
    num_tokens: int
    medium: str = "GPU"

class BlockTransferred(KVCacheEvent):
    """PD 分离 KV 传输完成"""
    block_hashes: list[int]
    src: str                           # prefill 节点
    dst: str                           # decode 节点
    backend: str                       # NVLINK | RDMA | MOONCAKE
    transfer_latency_us: int
    bytes_transferred: int

class BlockOffloaded(KVCacheEvent):
    """KV block GPU → CPU/Disk"""
    block_hashes: list[int]
    src_medium: str                    # GPU
    dst_medium: str                    # CPU | DISK | EXTERNAL
    pool_name: str                     # kv | swa | mamba

class BlockLoaded(KVCacheEvent):
    """KV block CPU/Disk → GPU"""
    block_hashes: list[int]
    src_medium: str
    dst_medium: str                    # GPU
    load_latency_us: int
```

### 5.3 新增埋点位置

| 路径 | 位置 | 事件 |
|------|------|------|
| TokenToKVPoolAllocator.alloc() | `allocator/paged_token_to_kv_pool_allocator.py` | BlockAllocated |
| HiCache L2 transfer | `memory_pool_host.py` / `hicache_storage.py` | BlockOffloaded / BlockLoaded |
| PD 分离 TransferQueue→WaitingQueue | `disaggregation/decode.py` | BlockTransferred |
| DecodeKVCacheOffloadManager | `disaggregation/decode_kvcache_offload_manager.py` | BlockOffloaded |

---

## 6. 插件实现

### 6.1 请求 KV Footprint 状态机

```
                    ┌─────────┐
                    │  INIT   │
                    └────┬────┘
                         │ BlockAllocated
                         ▼
              ┌──────────────────────┐
              │     GPU_RESIDENT     │
              │  gpu_pages: {p1,p2}  │
              └──┬──────┬──────┬─────┘
                 │      │      │
        ┌────────┘      │      └────────┐
        │ BlockStored   │ BlockOffloaded │ BlockRemoved
        │ (radix cached)│ (→CPU/Disk)    │ (evicted/freed)
        ▼                ▼                ▼
   ┌──────────┐  ┌──────────────┐  ┌──────────┐
   │ (仍 GPU, │  │ CPU_RESIDENT │  │  FREED   │
   │  cached) │  │ or DISK_RES  │  └──────────┘
   └──────────┘  └──────┬───────┘
                        │ BlockLoaded/BlockTransferred
                        ▼
                 ┌──────────────┐
                 │ GPU_RESIDENT │
                 └──────────────┘
```

### 6.2 FootprintAggregator

核心数据结构:

```python
@dataclass
class RequestKVFootprint:
    req_id: str
    start_ts: float
    end_ts: Optional[float] = None
    gpu_pages: set[int] = field(default_factory=set)
    cpu_pages: set[int] = field(default_factory=set)
    disk_pages: set[int] = field(default_factory=set)
    total_gpu_bytes: int = 0
    total_cpu_bytes: int = 0
    total_disk_bytes: int = 0
    peak_gpu_bytes: int = 0
    peak_cpu_bytes: int = 0
    events: list[tuple[float, str]] = field(default_factory=list)
```

处理逻辑: 对每个事件类型进行 match-case 分发，更新对应的 page set 和 byte 计数，追踪 peak 值。

### 6.3 FootprintRingBuffer（零开销写入）

无锁环形缓冲区:
- 固定大小预分配（默认 1M 条记录）
- 写入路径: 单个 atomic CAS 操作
- 读取路径: 批量 drain，独立消费线程
- 溢出策略: 覆盖最旧记录（符合"允许丢失"约束）

### 6.4 导出后端

| 后端 | 用途 | 数据粒度 |
|------|------|---------|
| PrometheusExporter | 容量规划、实时水位 | 聚合指标 (histogram/counter/gauge) |
| JSONLFileExporter | 离线分析、trace replay | 逐请求完整记录 |
| OTELSpanExporter | 分布式追踪、关联上游调用链 | per-request Span + Events |

**Prometheus 核心指标**:

- `sglang_kv_footprint_gpu_bytes_per_request` (Histogram) — 请求峰值 GPU footprint
- `sglang_kv_footprint_cpu_bytes_per_request` (Histogram) — 请求峰值 CPU footprint
- `sglang_kv_footprint_events_total` (Counter) — 按 event_type/medium 的事件计数
- `sglang_kv_footprint_active_requests` (Gauge) — 当前活跃请求数
- `sglang_kv_transfer_latency_us` (Histogram) — PD 分离传输延迟
- `sglang_kv_offload_latency_us` (Histogram) — Offload 操作延迟

### 6.5 查询 API（可选）

- `GET /footprint/{req_id}` — 单个请求完整 KV footprint + 事件时间线
- `GET /footprints?status=gpu_resident&limit=100` — 列出活跃 footprint
- `GET /footprint/summary` — 聚合摘要

---

## 7. 分阶段实现计划

### Phase 1 — 基础事件补齐（请求粒度）

- [ ] KVCacheEvent 基类添加 `req_id`, `sequence_id` 可选字段
- [ ] KVCacheEventMixin 透传 req_id 参数
- [ ] RadixCache 调用处传入 req_id
- [ ] 插件包骨架 + FootprintEventPublisher 注册

### Phase 2 — 生成与复用路径

- [ ] TokenToKVPoolAllocator.alloc() 埋点 → BlockAllocated 事件
- [ ] RingBuffer + FootprintAggregator 基础实现
- [ ] JSONLFileExporter

### Phase 3 — 搬运路径（PD 分离）

- [ ] disaggregation/decode.py TransferQueue 埋点 → BlockTransferred
- [ ] PrometheusExporter（传输延迟指标）

### Phase 4 — Offload 路径

- [ ] HiCache L2 (CPU DRAM) 埋点 → BlockOffloaded / BlockLoaded
- [ ] HiCache L3 (Storage) 埋点
- [ ] PrometheusExporter（offload 延迟指标）

### Phase 5 — 查询与可视化

- [ ] FootprintQueryAPI
- [ ] OTELSpanExporter
- [ ] 容量规划 dashboard 推荐配置

---

## 8. 插件安装与使用

```bash
# 安装插件
pip install sglang-kv-footprint

# 启动 sglang 时启用
python -m sglang.launch_server \
    --model meta-llama/Llama-3-8B \
    --enable-kv-cache-events \
    --kv-events-publisher '{"publisher":"footprint"}'

# 可选：通过环境变量配置插件行为
export KV_FOOTPRINT_EXPORTERS=prometheus,jsonl
export KV_FOOTPRINT_JSONL_DIR=/var/log/sglang/footprints
export KV_FOOTPRINT_RING_BUFFER_SIZE=1000000
```

---

## 9. 插件目录结构

```
sglang-kv-footprint/
├── pyproject.toml
├── src/
│   └── sglang_kv_footprint/
│       ├── __init__.py              # register_publisher("footprint", ...)
│       ├── publisher.py             # FootprintEventPublisher
│       ├── ring_buffer.py           # FootprintRingBuffer
│       ├── aggregator.py            # FootprintAggregator
│       ├── exporters/
│       │   ├── __init__.py
│       │   ├── prometheus.py        # PrometheusExporter
│       │   ├── jsonl.py             # JSONLFileExporter
│       │   └── otel.py              # OTELSpanExporter
│       ├── api.py                   # FootprintQueryAPI
│       └── config.py                # FootprintConfig
└── README.md
```

---

## 10. 风险与缓解

| 风险 | 缓解 |
|------|------|
| Ring buffer 溢出丢失事件 | 监控 `sglang_kv_footprint_events_total` 与内部丢弃计数对比；提供 buffer size 配置 |
| req_id 获取不到（如 eviction 无请求上下文） | 允许 req_id=None，事件归类到全局统计 |
| DP attention 下多 rank 事件重复 | 利用现有 `attn_dp_rank` 标签区分，由 consumer 去重 |
| 插件进程崩溃影响主进程 | 插件在独立线程+独立进程运行，ZMQ PUB/SUB 解耦 |
| 向后兼容 | 所有新增字段都是 Optional，omit_defaults=True 确保旧消费者不受影响 |

---

## 11. 未决问题（后续确定）

1. PD 分离传输埋点中，是否需要追踪 NVLINK/RDMA 的逐 page 传输完成还是整批传输完成？
2. HiCache L3 storage backend 的 key 命名规范是否需要标准化以支持跨节点追踪？
3. 是否需要支持 request 粒度的 KV cache 预热/锁定事件的追踪？
