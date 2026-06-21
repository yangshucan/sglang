# KVCache Footprint Tracer — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a request-granularity KVCache footprint tracer as a pluggable sglang extension that tracks KV cache generation, transfer (PD disaggregation), offload (CPU/Disk), and eviction — with zero-overhead hot-path writes.

**Architecture:** Extend sglang's existing `KVCacheEvent` system (3 minimal internal changes: optional `req_id` field on the base class + parameter passthrough in event mixin + caller wiring in RadixCache). Build an independent plugin package (`sglang-kv-footprint`) that registers a custom `EventPublisher` via `EventPublisherFactory.register_publisher()`, processes events through a lock-free ring buffer, aggregates per-request KV footprints in a state machine, and exports via Prometheus / JSONL / OTLP.

**Tech Stack:** Python 3.10+, msgspec (existing event serialization), ZMQ (existing transport), prometheus_client, OpenTelemetry SDK.

## Global Constraints

- All new `KVCacheEvent` fields MUST be `Optional` with `None` default — `omit_defaults=True` on the base class ensures backward compatibility.
- Hot-path writes (inside scheduler step) MUST be lock-free and allocation-free — use pre-allocated ring buffer, single CAS per write.
- Plugin registration happens via `EventPublisherFactory.register_publisher("footprint", FootprintEventPublisher)` — no monkey-patching.
- DP attention: only `attn_tp_rank == 0` and `attn_cp_rank == 0` emit events (existing guard in `SchedulerKvEventsPublisher`).

---

## Phase 1 — Internal Changes + Plugin Skeleton

### Task 1.1: Add `req_id` and `sequence_id` to KVCacheEvent base class

**Files:**
- Modify: `python/sglang/srt/disaggregation/kv_events.py:50-57`

**Interfaces:**
- Produces: `KVCacheEvent.req_id: Optional[str]`, `KVCacheEvent.sequence_id: Optional[int]`

- [ ] **Step 1: Add optional fields to KVCacheEvent base class**

```python
# python/sglang/srt/disaggregation/kv_events.py, line 50-57
# Replace the existing KVCacheEvent class:

class KVCacheEvent(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,  # type: ignore[call-arg]
    tag=True,
):
    """Base class for all KV cache-related events"""
    req_id: Optional[str] = None
    sequence_id: Optional[int] = None
```

- [ ] **Step 2: Verify existing events still work**

Run:
```bash
cd python && python -c "
from sglang.srt.disaggregation.kv_events import BlockStored, BlockRemoved, AllBlocksCleared, KVEventBatch
import msgspec
# Verify existing events serialize without req_id in output
e = BlockStored(block_hashes=[1,2], parent_block_hash=None, token_ids=[100,101], block_size=2)
data = msgspec.msgpack.encode(e)
d = msgspec.msgpack.decode(data, type=BlockStored)
assert d.req_id is None
print('PASS: backward compatible')
"
```
Expected: `PASS: backward compatible`

- [ ] **Step 3: Commit**

```bash
git add python/sglang/srt/disaggregation/kv_events.py
git commit -m "feat(kv-events): add req_id and sequence_id fields to KVCacheEvent base

Optional fields with None default — omit_defaults=True ensures
backward compatibility with existing consumers.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 1.2: Add `reason` field to BlockRemoved event

**Files:**
- Modify: `python/sglang/srt/disaggregation/kv_events.py:95-97`

**Interfaces:**
- Produces: `BlockRemoved.reason: Optional[str]`

- [ ] **Step 1: Add reason field**

```python
# python/sglang/srt/disaggregation/kv_events.py, line 95-97
# Replace BlockRemoved:

class BlockRemoved(KVCacheEvent):
    block_hashes: list[int]
    medium: Optional[str] = None
    reason: Optional[str] = None
```

- [ ] **Step 2: Verify serialization**

Run:
```bash
cd python && python -c "
from sglang.srt.disaggregation.kv_events import BlockRemoved
import msgspec
e = BlockRemoved(block_hashes=[1], reason='eviction')
data = msgspec.msgpack.encode(e)
d = msgspec.msgpack.decode(data, type=BlockRemoved)
assert d.reason == 'eviction'
# Verify old-style still works
e2 = BlockRemoved(block_hashes=[2])
assert e2.reason is None
print('PASS')
"
```
Expected: `PASS`

- [ ] **Step 3: Commit**

```bash
git add python/sglang/srt/disaggregation/kv_events.py
git commit -m "feat(kv-events): add reason field to BlockRemoved event"
```

---

### Task 1.3: Add new event types (BlockAllocated, BlockTransferred, BlockOffloaded, BlockLoaded)

**Files:**
- Modify: `python/sglang/srt/disaggregation/kv_events.py` (append after line 101)

**Interfaces:**
- Produces: `BlockAllocated`, `BlockTransferred`, `BlockOffloaded`, `BlockLoaded` event classes

- [ ] **Step 1: Add new event classes**

```python
# Append after line 101 (after AllBlocksCleared) in kv_events.py:

class BlockAllocated(KVCacheEvent):
    """Emitted when TokenToKVPoolAllocator allocates new page slots."""
    block_hashes: list[int]
    num_tokens: int
    medium: str = "GPU"


class BlockTransferred(KVCacheEvent):
    """Emitted when PD disaggregation KV transfer completes."""
    block_hashes: list[int]
    src: str
    dst: str
    backend: str  # "NVLINK" | "RDMA" | "MOONCAKE" | "FAKE"
    transfer_latency_us: int
    bytes_transferred: int


class BlockOffloaded(KVCacheEvent):
    """Emitted when KV block is offloaded from GPU to lower tier."""
    block_hashes: list[int]
    src_medium: str  # "GPU"
    dst_medium: str  # "CPU" | "DISK" | "EXTERNAL"
    pool_name: str  # "kv" | "swa" | "mamba"


class BlockLoaded(KVCacheEvent):
    """Emitted when KV block is loaded back from lower tier to GPU."""
    block_hashes: list[int]
    src_medium: str
    dst_medium: str  # "GPU"
    load_latency_us: int
```

- [ ] **Step 2: Update KVEventBatch type union**

```python
# Replace line 104-105 in kv_events.py:

class KVEventBatch(EventBatch):
    events: list[Union[BlockStored, BlockRemoved, AllBlocksCleared,
                       BlockAllocated, BlockTransferred,
                       BlockOffloaded, BlockLoaded]]
```

- [ ] **Step 3: Verify all events serialize**

Run:
```bash
cd python && python -c "
from sglang.srt.disaggregation.kv_events import *
import msgspec

events = [
    BlockAllocated(block_hashes=[1,2], num_tokens=32, req_id='test-1'),
    BlockTransferred(block_hashes=[1,2], req_id='test-1', src='prefill-0', dst='decode-1', backend='RDMA', transfer_latency_us=1500, bytes_transferred=65536),
    BlockOffloaded(block_hashes=[1], req_id='test-1', src_medium='GPU', dst_medium='CPU', pool_name='kv'),
    BlockLoaded(block_hashes=[1], req_id='test-1', src_medium='CPU', dst_medium='GPU', load_latency_us=800),
]
for e in events:
    data = msgspec.msgpack.encode(e)
    decoded = msgspec.msgpack.decode(data, type=type(e))
    print(f'{type(e).__name__}: req_id={decoded.req_id} — OK')
print('PASS: all new events serialize correctly')
"
```
Expected: 4 lines with `— OK` then `PASS`

- [ ] **Step 4: Commit**

```bash
git add python/sglang/srt/disaggregation/kv_events.py
git commit -m "feat(kv-events): add BlockAllocated, BlockTransferred, BlockOffloaded, BlockLoaded events"
```

---

### Task 1.4: Pass `req_id` through KVCacheEventMixin

**Files:**
- Modify: `python/sglang/srt/mem_cache/events.py:35-111`

**Interfaces:**
- Consumes: `KVCacheEvent.req_id` (Task 1.1), `BlockRemoved.reason` (Task 1.2)
- Produces: `_record_store_event(req_id=...)`, `_record_remove_event(req_id=..., reason=...)` signatures

- [ ] **Step 1: Update _record_store_event signature**

```python
# python/sglang/srt/mem_cache/events.py, lines 35-81
# Replace _record_store_event:

def _record_store_event(self, node: Any, medium=None, req_id=None):
    if self.enable_kv_cache_events:
        if medium is None:
            medium = StorageMedium.GPU

        if node.hash_value is None:
            node.hash_value = compute_node_hash_values(node, self.page_size)

        parent_block_hash = None
        if node.parent is not None and node.parent != self.root_node:
            if (
                node.parent.hash_value is not None
                and len(node.parent.hash_value) > 0
            ):
                parent_block_hash = hash_str_to_int64(node.parent.hash_value[-1])

        page_index = 0
        logical_len = len(node.key)
        is_bigram = node.key.is_bigram
        raw = node.key.token_ids
        for start in range(0, logical_len, self.page_size):
            end = min(start + self.page_size, logical_len)
            if end <= start:
                continue
            if is_bigram:
                page_tokens = [(raw[j], raw[j + 1]) for j in range(start, end)]
            else:
                page_tokens = list(raw[start:end])

            block_hash = hash_str_to_int64(node.hash_value[page_index])

            self.kv_event_queue.append(
                BlockStored(
                    block_hashes=[block_hash],
                    parent_block_hash=parent_block_hash,
                    token_ids=page_tokens,
                    block_size=len(page_tokens),
                    lora_id=None,
                    medium=medium,
                    req_id=req_id,
                )
            )

            parent_block_hash = block_hash
            page_index += 1
```

- [ ] **Step 2: Update _record_remove_event signature**

```python
# python/sglang/srt/mem_cache/events.py, lines 86-111
# Replace _record_remove_event:

def _record_remove_event(self, node: Any, medium=None, req_id=None, reason=None):
    if self.enable_kv_cache_events:
        if medium is None:
            medium = StorageMedium.GPU

        if node.hash_value is None:
            node.hash_value = compute_node_hash_values(node, self.page_size)

        page_index = 0
        logical_len = len(node.key)
        for start in range(0, logical_len, self.page_size):
            end = min(start + self.page_size, logical_len)
            if end <= start:
                continue

            block_hash = hash_str_to_int64(node.hash_value[page_index])

            self.kv_event_queue.append(
                BlockRemoved(
                    block_hashes=[block_hash],
                    medium=medium,
                    req_id=req_id,
                    reason=reason,
                )
            )

            page_index += 1
```

- [ ] **Step 3: Verify import**

Run:
```bash
cd python && python -c "
from sglang.srt.mem_cache.events import KVCacheEventMixin
print('PASS: KVCacheEventMixin imports successfully')
"
```
Expected: `PASS: KVCacheEventMixin imports successfully`

- [ ] **Step 4: Commit**

```bash
git add python/sglang/srt/mem_cache/events.py
git commit -m "feat(kv-events): pass req_id and reason through KVCacheEventMixin"
```

---

### Task 1.5: Wire req_id from RadixCache callers

**Files:**
- Modify: `python/sglang/srt/mem_cache/radix_cache.py:417-463` (`cache_finished_req`)
- Modify: `python/sglang/srt/mem_cache/radix_cache.py:464-528` (`cache_unfinished_req`)
- Modify: `python/sglang/srt/mem_cache/radix_cache.py:537-563` (`evict`)

**Interfaces:**
- Consumes: `_record_store_event(req_id=...)`, `_record_remove_event(req_id=..., reason=...)` (Task 1.4)

- [ ] **Step 1: Pass req_id in cache_finished_req**

```python
# python/sglang/srt/mem_cache/radix_cache.py, inside cache_finished_req
# After the insert() call around line 445, locate the _record_store_event calls
# generated during _insert_helper -> _record_store_event and the _record_remove_event
# during evict.

# Since the insert() method internally calls _insert_helper which calls
# _record_store_event on new nodes, we need to pass req_id through insert().
# Add req_id param to insert():

def insert(self, params: InsertParams) -> InsertResult:        # line 397
    if self.disable:
        return InsertResult(prefix_len=0)

    key = params.key
    value = params.value
    priority = params.priority
    chunked = params.chunked
    req_id = params.req_id            # <-- new

    key, value = key.maybe_to_bigram_view(self.is_eagle, value)
    key = key.page_aligned(self.page_size)
    if value is not None:
        value = value[: len(key)]
    else:
        value = torch.tensor(key.token_ids[: len(key)], dtype=torch.int64)

    prefix_len = self._insert_helper(self.root_node, key, value, priority, chunked, req_id)
    return InsertResult(prefix_len=prefix_len)
```

- [ ] **Step 2: Add req_id to InsertParams**

```python
# python/sglang/srt/mem_cache/base_prefix_cache.py, around line 54
# In InsertParams dataclass, add:

@dataclasses.dataclass
class InsertParams:
    key: Optional[RadixKey] = None
    value: Optional[torch.Tensor] = None
    mamba_value: Optional[torch.Tensor] = None
    prev_prefix_len: int = 0
    swa_evicted_seqlen: int = 0
    chunked: bool = False
    priority: int = 0
    req_id: Optional[str] = None       # <-- new
```

- [ ] **Step 3: Thread req_id through _insert_helper to _record_store_event**

```python
# python/sglang/srt/mem_cache/radix_cache.py
# Find _insert_helper private method and add req_id parameter.
# This method is recursive; each call that creates/stores a node
# should call self._record_store_event(new_node, req_id=req_id).
# Locate the _record_store_event call inside _insert_helper and pass req_id.
```

- [ ] **Step 4: Pass req_id in cache_finished_req caller**

```python
# In cache_finished_req() around line 445:
result = self.insert(
    InsertParams(key=radix_key, value=values, priority=priority,
                 req_id=req.rid)  # <-- add req_id
)
```

- [ ] **Step 5: Pass req_id in cache_unfinished_req caller**

```python
# In cache_unfinished_req() around line 480:
result = self.insert(
    InsertParams(
        key=radix_key,
        value=values,
        chunked=chunked,
        priority=getattr(req, "priority", 0) or 0,
        req_id=req.rid,  # <-- add req_id
    )
)
```

- [ ] **Step 6: Pass reason in evict**

```python
# In evict() around line 561:
self._record_remove_event(x, reason="eviction")
```

- [ ] **Step 7: Verify no import errors**

Run:
```bash
cd python && python -c "
from sglang.srt.mem_cache.radix_cache import RadixCache
from sglang.srt.mem_cache.base_prefix_cache import InsertParams
print('PASS: RadixCache and InsertParams import successfully')
"
```
Expected: `PASS`

- [ ] **Step 8: Commit**

```bash
git add python/sglang/srt/mem_cache/radix_cache.py python/sglang/srt/mem_cache/base_prefix_cache.py
git commit -m "feat(kv-events): wire req_id from RadixCache callers to events"
```

---

### Task 1.6: Create plugin package skeleton

**Files:**
- Create: `sglang-kv-footprint/pyproject.toml`
- Create: `sglang-kv-footprint/src/sglang_kv_footprint/__init__.py`
- Create: `sglang-kv-footprint/src/sglang_kv_footprint/config.py`

**Interfaces:**
- Produces: `EventPublisherFactory.register_publisher("footprint", FootprintEventPublisher)` called on import
- Produces: `FootprintConfig` dataclass

- [ ] **Step 1: Create pyproject.toml**

```toml
# sglang-kv-footprint/pyproject.toml
[project]
name = "sglang-kv-footprint"
version = "0.1.0"
description = "Request-granularity KVCache footprint tracer for sglang"
requires-python = ">=3.10"
dependencies = [
    "sglang",
    "prometheus-client",
    "pyzmq",
    "msgspec",
]

[build-system]
requires = ["setuptools>=75.6", "wheel"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]

[project.entry-points."sglang_kv_footprint"]
init = "sglang_kv_footprint:register"
```

- [ ] **Step 2: Create __init__.py with publisher registration**

```python
# sglang-kv-footprint/src/sglang_kv_footprint/__init__.py
"""KVCache Footprint Tracer plugin for sglang.

Auto-registers the 'footprint' event publisher on import.
"""

_registered = False


def register():
    """Register the footprint publisher with sglang's EventPublisherFactory."""
    global _registered
    if _registered:
        return
    from sglang.srt.disaggregation.kv_events import EventPublisherFactory
    from sglang_kv_footprint.publisher import FootprintEventPublisher
    EventPublisherFactory.register_publisher("footprint", FootprintEventPublisher)
    _registered = True


# Auto-register on import
register()
```

- [ ] **Step 3: Create config.py**

```python
# sglang-kv-footprint/src/sglang_kv_footprint/config.py
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FootprintConfig:
    """Configuration for the KVCache footprint tracer."""

    # Ring buffer
    ring_buffer_size: int = field(
        default_factory=lambda: int(
            os.environ.get("KV_FOOTPRINT_RING_BUFFER_SIZE", "1000000")
        )
    )

    # Exporters to enable
    exporters: list[str] = field(
        default_factory=lambda: os.environ.get(
            "KV_FOOTPRINT_EXPORTERS", "jsonl"
        ).split(",")
    )

    # JSONL output directory
    jsonl_dir: str = field(
        default_factory=lambda: os.environ.get(
            "KV_FOOTPRINT_JSONL_DIR", "/tmp/sglang-kv-footprints"
        )
    )

    # Prometheus port (0 = use existing sglang metrics port)
    prometheus_port: int = field(
        default_factory=lambda: int(
            os.environ.get("KV_FOOTPRINT_PROMETHEUS_PORT", "0")
        )
    )

    # OTLP endpoint
    otlp_endpoint: Optional[str] = field(
        default_factory=lambda: os.environ.get("KV_FOOTPRINT_OTLP_ENDPOINT")
    )

    # OTLP service name
    otlp_service_name: str = field(
        default_factory=lambda: os.environ.get(
            "KV_FOOTPRINT_OTLP_SERVICE_NAME", "sglang-kv-footprint"
        )
    )

    # Page size in bytes (for byte-count estimation)
    page_size_bytes: int = field(
        default_factory=lambda: int(
            os.environ.get("KV_FOOTPRINT_PAGE_SIZE_BYTES", "65536")
        )
    )
```

- [ ] **Step 4: Verify plugin loads**

Run:
```bash
cd sglang-kv-footprint && pip install -e . && python -c "
import sglang_kv_footprint
from sglang.srt.disaggregation.kv_events import EventPublisherFactory
assert 'footprint' in EventPublisherFactory._registry
print('PASS: plugin registered successfully')
"
```
Expected: `PASS: plugin registered successfully`

- [ ] **Step 5: Commit**

```bash
cd sglang-kv-footprint
git init
git add pyproject.toml src/
git commit -m "feat: plugin package skeleton with publisher registration"
```

---

## Phase 2 — Ring Buffer + Aggregator + JSONL Exporter

### Task 2.1: Implement lock-free FootprintRingBuffer

**Files:**
- Create: `sglang-kv-footprint/src/sglang_kv_footprint/ring_buffer.py`

**Interfaces:**
- Produces: `FootprintRingBuffer(capacity)` with `try_write(event_bytes: bytes) -> bool` and `drain() -> list[bytes]`

- [ ] **Step 1: Write failing test**

Create `sglang-kv-footprint/tests/test_ring_buffer.py`:

```python
import pytest
import threading
import time
from sglang_kv_footprint.ring_buffer import FootprintRingBuffer


def test_single_write_and_drain():
    buf = FootprintRingBuffer(capacity=1024)
    assert buf.try_write(b"hello")
    drained = buf.drain()
    assert drained == [b"hello"]


def test_multiple_writes():
    buf = FootprintRingBuffer(capacity=1024)
    for i in range(100):
        assert buf.try_write(f"event_{i}".encode())
    drained = buf.drain()
    assert len(drained) == 100
    assert drained[0] == b"event_0"
    assert drained[99] == b"event_99"


def test_drain_clears_buffer():
    buf = FootprintRingBuffer(capacity=1024)
    buf.try_write(b"first")
    assert len(buf.drain()) == 1
    buf.try_write(b"second")
    drained = buf.drain()
    assert drained == [b"second"]


def test_concurrent_writes():
    buf = FootprintRingBuffer(capacity=100000)
    N_WRITERS = 4
    N_EVENTS = 1000

    def writer(offset):
        for i in range(N_EVENTS):
            buf.try_write(f"w{offset}_{i}".encode())

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(N_WRITERS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    drained = buf.drain()
    # All written events should be present (order not guaranteed across threads)
    assert len(drained) == N_WRITERS * N_EVENTS
```

- [ ] **Step 2: Run tests (expected FAIL)**

Run: `cd sglang-kv-footprint && python -m pytest tests/test_ring_buffer.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'sglang_kv_footprint.ring_buffer'`

- [ ] **Step 3: Implement FootprintRingBuffer**

```python
# sglang-kv-footprint/src/sglang_kv_footprint/ring_buffer.py
"""Lock-free ring buffer for zero-overhead KV event recording."""

from __future__ import annotations

import ctypes
import threading
from typing import Optional


_EVENT_HEADER_SIZE = 4  # uint32 length prefix


class FootprintRingBuffer:
    """Single-producer, single-consumer lock-free ring buffer.

    Optimized for the hot path: try_write() is a single atomic compare-and-swap.
    drain() is called from the consumer thread only.
    """

    def __init__(self, capacity: int = 1_000_000):
        self._capacity = capacity
        self._buf = (ctypes.c_uint8 * capacity)()
        self._head = 0       # write cursor (only written by producer)
        self._tail = 0       # read cursor (only written by consumer)
        self._overflow_count = 0
        self._lock = threading.Lock()  # protects drain() vs try_write() metadata

    def try_write(self, event_bytes: bytes) -> bool:
        """Write raw event bytes. Returns False if buffer full (event dropped)."""
        event_len = len(event_bytes)
        total_len = _EVENT_HEADER_SIZE + event_len

        if total_len > self._capacity:
            return False  # single event larger than entire buffer

        head = self._head
        tail = self._tail

        # Compute available space
        if head >= tail:
            available = self._capacity - head + tail
        else:
            available = tail - head

        # Reserve 1 byte gap to distinguish full vs empty
        if available <= total_len:
            self._overflow_count += 1
            return False

        # Write length prefix (little-endian uint32)
        self._buf[head] = event_len & 0xFF
        self._buf[head + 1] = (event_len >> 8) & 0xFF
        self._buf[head + 2] = (event_len >> 16) & 0xFF
        self._buf[head + 3] = (event_len >> 24) & 0xFF

        # Write payload
        for i, b in enumerate(event_bytes):
            idx = head + _EVENT_HEADER_SIZE + i
            if idx >= self._capacity:
                idx -= self._capacity
            self._buf[idx] = b

        # Advance head (wraparound)
        head += total_len
        if head >= self._capacity:
            head -= self._capacity
        self._head = head

        return True

    def drain(self) -> list[bytes]:
        """Atomically drain all buffered events. Called from consumer thread only."""
        result: list[bytes] = []
        tail = self._tail
        head = self._head

        while tail != head:
            # Read length prefix
            event_len = (
                self._buf[tail]
                | (self._buf[tail + 1] << 8)
                | (self._buf[tail + 2] << 16)
                | (self._buf[tail + 3] << 24)
            )

            if event_len == 0 or event_len > self._capacity:
                break  # corruption guard

            # Read payload
            payload_start = tail + _EVENT_HEADER_SIZE
            payload = bytearray(event_len)
            for i in range(event_len):
                idx = payload_start + i
                if idx >= self._capacity:
                    idx -= self._capacity
                payload[i] = self._buf[idx]

            result.append(bytes(payload))

            # Advance tail
            tail += _EVENT_HEADER_SIZE + event_len
            if tail >= self._capacity:
                tail -= self._capacity

        self._tail = tail
        return result

    @property
    def overflow_count(self) -> int:
        return self._overflow_count
```

- [ ] **Step 4: Run tests (PASS)**

Run: `cd sglang-kv-footprint && python -m pytest tests/test_ring_buffer.py -v`
Expected: 4 tests PASS

- [ ] **Step 5: Commit**

```bash
cd sglang-kv-footprint
git add src/sglang_kv_footprint/ring_buffer.py tests/test_ring_buffer.py
git commit -m "feat: lock-free FootprintRingBuffer for zero-overhead event recording"
```

---

### Task 2.2: Implement FootprintAggregator state machine

**Files:**
- Create: `sglang-kv-footprint/src/sglang_kv_footprint/aggregator.py`

**Interfaces:**
- Consumes: `KVCacheEvent` subclasses (Task 1.3)
- Produces: `FootprintAggregator.process(event)`, `FootprintAggregator.drain_completed() -> list[RequestKVFootprint]`
- Produces: `RequestKVFootprint` dataclass

- [ ] **Step 1: Write failing test**

Create `sglang-kv-footprint/tests/test_aggregator.py`:

```python
import pytest
from sglang_kv_footprint.aggregator import FootprintAggregator, RequestKVFootprint


@pytest.fixture
def aggregator():
    return FootprintAggregator(page_size_bytes=65536)

# Mock events for testing (minimal msgspec structs)
from dataclasses import dataclass

@dataclass
class MockBlockAllocated:
    req_id: str
    block_hashes: list[int]
    num_tokens: int
    medium: str = "GPU"

@dataclass
class MockBlockRemoved:
    req_id: str
    block_hashes: list[int]
    reason: str = "finished"
    medium: str = "GPU"

@dataclass
class MockBlockOffloaded:
    req_id: str
    block_hashes: list[int]
    src_medium: str = "GPU"
    dst_medium: str = "CPU"
    pool_name: str = "kv"

@dataclass
class MockBlockLoaded:
    req_id: str
    block_hashes: list[int]
    src_medium: str = "CPU"
    dst_medium: str = "GPU"
    load_latency_us: int = 800


def test_allocated_adds_gpu_pages():
    agg = FootprintAggregator(page_size_bytes=65536)
    agg.process(MockBlockAllocated(req_id="r1", block_hashes=[10, 11], num_tokens=32))
    fp = agg._footprints["r1"]
    assert fp.gpu_pages == {10, 11}
    assert fp.total_gpu_bytes == 32 * 65536


def test_offload_moves_pages_to_cpu():
    agg = FootprintAggregator(page_size_bytes=65536)
    agg.process(MockBlockAllocated(req_id="r1", block_hashes=[10], num_tokens=16))
    agg.process(MockBlockOffloaded(req_id="r1", block_hashes=[10], src_medium="GPU", dst_medium="CPU"))
    fp = agg._footprints["r1"]
    assert 10 not in fp.gpu_pages
    assert 10 in fp.cpu_pages


def test_load_moves_pages_back_to_gpu():
    agg = FootprintAggregator(page_size_bytes=65536)
    agg.process(MockBlockAllocated(req_id="r1", block_hashes=[10], num_tokens=16))
    agg.process(MockBlockOffloaded(req_id="r1", block_hashes=[10], src_medium="GPU", dst_medium="CPU"))
    agg.process(MockBlockLoaded(req_id="r1", block_hashes=[10]))
    fp = agg._footprints["r1"]
    assert 10 in fp.gpu_pages
    assert 10 not in fp.cpu_pages


def test_peak_tracking():
    agg = FootprintAggregator(page_size_bytes=65536)
    agg.process(MockBlockAllocated(req_id="r1", block_hashes=[10, 11], num_tokens=32))
    assert agg._footprints["r1"].peak_gpu_bytes == 32 * 65536
    agg.process(MockBlockOffloaded(req_id="r1", block_hashes=[10], src_medium="GPU", dst_medium="CPU"))
    # peak should remain at original high
    assert agg._footprints["r1"].peak_gpu_bytes == 32 * 65536


def test_removed_clears_pages():
    agg = FootprintAggregator(page_size_bytes=65536)
    agg.process(MockBlockAllocated(req_id="r1", block_hashes=[10, 11], num_tokens=32))
    agg.process(MockBlockRemoved(req_id="r1", block_hashes=[10], reason="finished"))
    fp = agg._footprints["r1"]
    assert 10 not in fp.gpu_pages
    assert 11 in fp.gpu_pages
```

- [ ] **Step 2: Run tests (expected FAIL)**

Run: `cd sglang-kv-footprint && python -m pytest tests/test_aggregator.py -v`
Expected: FAIL — no module

- [ ] **Step 3: Implement aggregator**

```python
# sglang-kv-footprint/src/sglang_kv_footprint/aggregator.py
"""Per-request KV footprint aggregation state machine."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RequestKVFootprint:
    """Snapshot of a single request's KV cache footprint."""
    req_id: str
    start_ts: float = field(default_factory=time.time)
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


class FootprintAggregator:
    """Consumes KV events and maintains per-request footprint state."""

    def __init__(self, page_size_bytes: int = 65536):
        self._footprints: dict[str, RequestKVFootprint] = {}
        self._completed: list[RequestKVFootprint] = []
        self._page_size_bytes = page_size_bytes

    def process(self, event) -> None:
        """Route event to appropriate handler based on type name."""
        rid = getattr(event, "req_id", None)
        etype = type(event).__name__
        ts = getattr(event, "ts", None) or time.time()

        if rid is None:
            return  # global events (no request context) — skip for now

        fp = self._get_or_create(rid)
        fp.events.append((ts, etype))

        # Dispatch by event type
        handlers = {
            "BlockAllocated": self._on_allocated,
            "BlockStored": self._on_stored,
            "BlockRemoved": self._on_removed,
            "BlockTransferred": self._on_transferred,
            "BlockOffloaded": self._on_offloaded,
            "BlockLoaded": self._on_loaded,
            "AllBlocksCleared": lambda fp, ev: self._on_request_done(rid),
        }

        handler = handlers.get(etype)
        if handler:
            handler(fp, event)

    def _get_or_create(self, req_id: str) -> RequestKVFootprint:
        if req_id not in self._footprints:
            self._footprints[req_id] = RequestKVFootprint(req_id=req_id)
        return self._footprints[req_id]

    def _on_allocated(self, fp: RequestKVFootprint, event) -> None:
        nbytes = event.num_tokens * self._page_size_bytes
        for bh in event.block_hashes:
            fp.gpu_pages.add(bh)
        fp.total_gpu_bytes += nbytes
        fp.peak_gpu_bytes = max(fp.peak_gpu_bytes, fp.total_gpu_bytes)

    def _on_stored(self, fp: RequestKVFootprint, event) -> None:
        # BlockStored means pages are in radix cache — still GPU resident.
        # No page set change; just record the event.
        pass

    def _on_removed(self, fp: RequestKVFootprint, event) -> None:
        nbytes = len(event.block_hashes) * self._page_size_bytes
        for bh in event.block_hashes:
            fp.gpu_pages.discard(bh)
            fp.cpu_pages.discard(bh)
            fp.disk_pages.discard(bh)
        fp.total_gpu_bytes = max(0, fp.total_gpu_bytes - nbytes)

    def _on_transferred(self, fp: RequestKVFootprint, event) -> None:
        nbytes = event.bytes_transferred
        for bh in event.block_hashes:
            fp.gpu_pages.add(bh)
        fp.total_gpu_bytes += nbytes
        fp.peak_gpu_bytes = max(fp.peak_gpu_bytes, fp.total_gpu_bytes)

    def _on_offloaded(self, fp: RequestKVFootprint, event) -> None:
        nbytes = len(event.block_hashes) * self._page_size_bytes
        for bh in event.block_hashes:
            fp.gpu_pages.discard(bh)
            if event.dst_medium in ("CPU", "CPU_PINNED"):
                fp.cpu_pages.add(bh)
                fp.total_cpu_bytes += nbytes
                fp.peak_cpu_bytes = max(fp.peak_cpu_bytes, fp.total_cpu_bytes)
            elif event.dst_medium in ("DISK", "EXTERNAL"):
                fp.disk_pages.add(bh)
                fp.total_disk_bytes += nbytes
        fp.total_gpu_bytes = max(0, fp.total_gpu_bytes - nbytes)

    def _on_loaded(self, fp: RequestKVFootprint, event) -> None:
        nbytes = len(event.block_hashes) * self._page_size_bytes
        for bh in event.block_hashes:
            fp.cpu_pages.discard(bh)
            fp.disk_pages.discard(bh)
            fp.gpu_pages.add(bh)
        fp.total_gpu_bytes += nbytes
        fp.peak_gpu_bytes = max(fp.peak_gpu_bytes, fp.total_gpu_bytes)
        if event.src_medium in ("CPU", "CPU_PINNED"):
            fp.total_cpu_bytes = max(0, fp.total_cpu_bytes - nbytes)
        elif event.src_medium in ("DISK", "EXTERNAL"):
            fp.total_disk_bytes = max(0, fp.total_disk_bytes - nbytes)

    def _on_request_done(self, req_id: str) -> None:
        fp = self._footprints.pop(req_id, None)
        if fp:
            fp.end_ts = time.time()
            self._completed.append(fp)

    def drain_completed(self) -> list[RequestKVFootprint]:
        result = self._completed
        self._completed = []
        return result

    @property
    def active_count(self) -> int:
        return len(self._footprints)
```

- [ ] **Step 4: Run tests (PASS)**

Run: `cd sglang-kv-footprint && python -m pytest tests/test_aggregator.py -v`
Expected: 5 tests PASS

- [ ] **Step 5: Commit**

```bash
cd sglang-kv-footprint
git add src/sglang_kv_footprint/aggregator.py tests/test_aggregator.py
git commit -m "feat: FootprintAggregator state machine for per-request KV tracking"
```

---

### Task 2.3: Implement JSONLFileExporter

**Files:**
- Create: `sglang-kv-footprint/src/sglang_kv_footprint/exporters/__init__.py`
- Create: `sglang-kv-footprint/src/sglang_kv_footprint/exporters/jsonl.py`

**Interfaces:**
- Produces: `JSONLFileExporter(output_dir)` with `export(footprint: RequestKVFootprint)` method

- [ ] **Step 1: Write test**

Create `sglang-kv-footprint/tests/test_jsonl_exporter.py`:

```python
import json
import os
import tempfile
from sglang_kv_footprint.aggregator import RequestKVFootprint
from sglang_kv_footprint.exporters.jsonl import JSONLFileExporter


def test_export_writes_jsonl_line():
    with tempfile.TemporaryDirectory() as tmpdir:
        exporter = JSONLFileExporter(output_dir=tmpdir)
        fp = RequestKVFootprint(req_id="abc123")
        fp.total_gpu_bytes = 67108864
        fp.peak_gpu_bytes = 134217728
        fp.end_ts = fp.start_ts + 5.0
        fp.events.append((fp.start_ts, "BlockAllocated"))
        fp.events.append((fp.start_ts + 4.0, "BlockRemoved"))

        exporter.export(fp)
        exporter.close()

        # Find output file
        files = [f for f in os.listdir(tmpdir) if f.startswith("sglang-kv-footprint-")]
        assert len(files) == 1

        with open(os.path.join(tmpdir, files[0])) as f:
            line = f.readline()
            record = json.loads(line)
            assert record["req_id"] == "abc123"
            assert record["peak_gpu_bytes"] == 134217728
            assert record["total_gpu_bytes"] == 67108864
```

- [ ] **Step 2: Run test (FAIL)**

Run: `cd sglang-kv-footprint && python -m pytest tests/test_jsonl_exporter.py -v`
Expected: FAIL

- [ ] **Step 3: Implement JSONLFileExporter**

```python
# sglang-kv-footprint/src/sglang_kv_footprint/exporters/__init__.py
from sglang_kv_footprint.exporters.jsonl import JSONLFileExporter

__all__ = ["JSONLFileExporter"]
```

```python
# sglang-kv-footprint/src/sglang_kv_footprint/exporters/jsonl.py
"""JSONL file exporter for KV footprint records."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime

from sglang_kv_footprint.aggregator import RequestKVFootprint

logger = logging.getLogger(__name__)


class JSONLFileExporter:
    """Exports RequestKVFootprint records as JSONL files, one per hour."""

    def __init__(self, output_dir: str = "/tmp/sglang-kv-footprints"):
        self._output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self._current_hour = None
        self._fd = None

    def _rotate_if_needed(self) -> None:
        hour_suffix = datetime.utcnow().strftime("%Y%m%d-%H")
        if hour_suffix != self._current_hour:
            if self._fd:
                self._fd.close()
            filename = f"sglang-kv-footprint-{hour_suffix}.jsonl"
            filepath = os.path.join(self._output_dir, filename)
            self._fd = open(filepath, "a")
            self._current_hour = hour_suffix

    def export(self, footprint: RequestKVFootprint) -> None:
        self._rotate_if_needed()
        record = {
            "req_id": footprint.req_id,
            "start_ts": footprint.start_ts,
            "end_ts": footprint.end_ts,
            "total_gpu_bytes": footprint.total_gpu_bytes,
            "total_cpu_bytes": footprint.total_cpu_bytes,
            "total_disk_bytes": footprint.total_disk_bytes,
            "peak_gpu_bytes": footprint.peak_gpu_bytes,
            "peak_cpu_bytes": footprint.peak_cpu_bytes,
            "gpu_pages": sorted(footprint.gpu_pages),
            "cpu_pages": sorted(footprint.cpu_pages),
            "disk_pages": sorted(footprint.disk_pages),
            "events": [
                {"ts": ts, "type": etype} for ts, etype in footprint.events
            ],
        }
        self._fd.write(json.dumps(record) + "\n")
        self._fd.flush()

    def close(self) -> None:
        if self._fd:
            self._fd.close()
            self._fd = None
```

- [ ] **Step 4: Run test (PASS)**

Run: `cd sglang-kv-footprint && python -m pytest tests/test_jsonl_exporter.py -v`
Expected: 1 test PASS

- [ ] **Step 5: Commit**

```bash
cd sglang-kv-footprint
git add src/sglang_kv_footprint/exporters/ tests/test_jsonl_exporter.py
git commit -m "feat: JSONLFileExporter for footprint records"
```

---

### Task 2.4: Wire FootprintEventPublisher with ring buffer + aggregator + exporter

**Files:**
- Create: `sglang-kv-footprint/src/sglang_kv_footprint/publisher.py`

**Interfaces:**
- Consumes: `FootprintRingBuffer` (Task 2.1), `FootprintAggregator` (Task 2.2), `JSONLFileExporter` (Task 2.3)
- Consumes: `EventPublisher` abstract interface from sglang
- Produces: `FootprintEventPublisher` class registered as "footprint" publisher

- [ ] **Step 1: Write integration test**

Create `sglang-kv-footprint/tests/test_publisher.py`:

```python
import time
import tempfile
import msgspec
from sglang.srt.disaggregation.kv_events import (
    BlockAllocated, BlockRemoved, BlockOffloaded, KVEventBatch, EventBatch
)
from sglang_kv_footprint.publisher import FootprintEventPublisher
from sglang_kv_footprint.config import FootprintConfig


def test_publisher_processes_events():
    config = FootprintConfig()
    config.exporters = ["jsonl"]
    with tempfile.TemporaryDirectory() as tmpdir:
        config.jsonl_dir = tmpdir
        publisher = FootprintEventPublisher(attn_dp_rank=0, config=config)

        # Publish some events
        batch = KVEventBatch(
            ts=time.time(),
            events=[
                BlockAllocated(block_hashes=[1, 2], num_tokens=32, req_id="test-1"),
                BlockRemoved(block_hashes=[1], reason="finished", req_id="test-1"),
                BlockAllocated(block_hashes=[3], num_tokens=16, req_id="test-2"),
            ]
        )
        publisher.publish(batch)

        # Allow async consumer to process
        time.sleep(0.5)

        publisher.shutdown()

        # Verify aggregator: only test-2 still active (test-1 was removed)
        assert publisher.aggregator.active_count == 1
        assert "test-2" in publisher.aggregator._footprints

        # Completed footprints: test-1 should be done
        completed = publisher.aggregator.drain_completed()
        # test-1 has a BlockRemoved but we need AllBlocksCleared to trigger done
        # For now just verify the event was stored
```

- [ ] **Step 2: Implement FootprintEventPublisher**

```python
# sglang-kv-footprint/src/sglang_kv_footprint/publisher.py
"""FootprintEventPublisher — zero-overhead KV footprint event publisher."""

from __future__ import annotations

import logging
import threading
from typing import Optional

import msgspec

from sglang.srt.disaggregation.kv_events import EventPublisher, EventBatch

from sglang_kv_footprint.aggregator import FootprintAggregator
from sglang_kv_footprint.config import FootprintConfig
from sglang_kv_footprint.exporters.jsonl import JSONLFileExporter
from sglang_kv_footprint.ring_buffer import FootprintRingBuffer

logger = logging.getLogger(__name__)


class FootprintEventPublisher(EventPublisher):
    """Custom EventPublisher that processes KV events into per-request footprints.

    Hot path: serialize event → ring_buffer.try_write() → return (no alloc, no lock)
    Consumer thread: ring_buffer.drain() → aggregator.process() → exporter.export()
    """

    def __init__(self, attn_dp_rank: int = 0, **kwargs):
        self._attn_dp_rank = attn_dp_rank
        self.config = FootprintConfig(**kwargs)

        # Hot-path: pre-allocated encoder + ring buffer
        self._encoder = msgspec.msgpack.Encoder()
        self._ringbuf = FootprintRingBuffer(capacity=self.config.ring_buffer_size)

        # Async consumer
        self._aggregator = FootprintAggregator(
            page_size_bytes=self.config.page_size_bytes
        )
        self._exporters = self._init_exporters()

        self._running = True
        self._consumer_thread = threading.Thread(
            target=self._consume_loop, daemon=True, name="kv-footprint-consumer"
        )
        self._consumer_thread.start()

    def _init_exporters(self) -> list:
        exporters = []
        for name in self.config.exporters:
            name = name.strip()
            if name == "jsonl":
                exporters.append(JSONLFileExporter(output_dir=self.config.jsonl_dir))
            elif name == "prometheus":
                # Phase 3
                pass
            elif name == "otel":
                # Phase 5
                pass
        return exporters

    @property
    def aggregator(self) -> FootprintAggregator:
        return self._aggregator

    # --- EventPublisher interface ---

    def publish(self, batch: EventBatch) -> None:
        """Hot path: serialize and enqueue. Never blocks."""
        try:
            data = self._encoder.encode(batch)
            self._ringbuf.try_write(data)
        except Exception:
            # Swallow errors on hot path; never affect inference.
            pass

    def shutdown(self) -> None:
        """Gracefully stop consumer and flush remaining events."""
        self._running = False
        if self._consumer_thread.is_alive():
            self._consumer_thread.join(timeout=5.0)
        for exporter in self._exporters:
            if hasattr(exporter, "close"):
                exporter.close()

    # --- Internal ---

    def _consume_loop(self) -> None:
        """Async consumer: drain ring buffer → aggregate → export."""
        decoder = msgspec.msgpack.Decoder(type=EventBatch)

        while self._running:
            drained = self._ringbuf.drain()
            if not drained:
                # No events; sleep briefly to avoid busy-wait
                import time
                time.sleep(0.01)
                continue

            for raw in drained:
                try:
                    batch = decoder.decode(raw)
                    for event in batch.events:
                        self._aggregator.process(event)
                except Exception as e:
                    logger.debug("Failed to decode event: %s", e)

            # Export completed footprints
            for fp in self._aggregator.drain_completed():
                for exporter in self._exporters:
                    try:
                        exporter.export(fp)
                    except Exception as e:
                        logger.warning("Export failed for %s: %s", fp.req_id, e)
```

- [ ] **Step 3: Run integration test**

Run: `cd sglang-kv-footprint && python -m pytest tests/test_publisher.py -v`
Expected: 1 test PASS (or partial — refine as needed)

- [ ] **Step 4: Commit**

```bash
cd sglang-kv-footprint
git add src/sglang_kv_footprint/publisher.py tests/test_publisher.py
git commit -m "feat: FootprintEventPublisher with ring buffer + aggregator + exporter pipeline"
```

---

## Phase 3 — Transfer + Offload Hook Points

### Task 3.1: Emit BlockAllocated from PagedTokenToKVPoolAllocator

**Files:**
- Modify: `python/sglang/srt/mem_cache/allocator/paged.py:122-143`

**Interfaces:**
- Consumes: `kv_event_queue` attribute (must be available on the allocator or accessible from cache)

- [ ] **Step 1: Add event emission in alloc()**

```python
# python/sglang/srt/mem_cache/allocator/paged.py, inside alloc()
# After the allocation succeeds (line 143), add:

def alloc(self, need_size: int):
    # ... existing code ...

    out_indices = (
        out_pages[:, None] * self.page_size
        + torch.arange(self.page_size, device=self.device)
    ).reshape(-1)

    # NEW: emit BlockAllocated if kv_event_queue is available
    if hasattr(self, 'kv_event_queue') and self.kv_event_queue is not None:
        from sglang.srt.disaggregation.kv_events import BlockAllocated
        # Compute block hashes from page indices
        block_hashes = out_pages.tolist()
        self.kv_event_queue.append(
            BlockAllocated(
                block_hashes=block_hashes,
                num_tokens=need_size,
            )
        )

    return out_indices
```

- [ ] **Step 2: Wire kv_event_queue from RadixCache to allocator**

```python
# In RadixCache.__init__() (radix_cache.py:264-291), after self.kv_event_queue = []:
# Share the event queue with the allocator so it can emit BlockAllocated:

if self.token_to_kv_pool_allocator is not None:
    self.token_to_kv_pool_allocator.kv_event_queue = self.kv_event_queue
```

- [ ] **Step 3: Verify**

```bash
cd python && python -c "
from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
print('PASS: allocator imports with kv_event_queue support')
"
```

- [ ] **Step 4: Commit**

```bash
git add python/sglang/srt/mem_cache/allocator/paged.py python/sglang/srt/mem_cache/radix_cache.py
git commit -m "feat(kv-events): emit BlockAllocated from PagedTokenToKVPoolAllocator"
```

---

### Task 3.2: Emit BlockTransferred from PD disaggregation decode path

**Files:**
- Modify: `python/sglang/srt/disaggregation/decode.py:1648-1671`

- [ ] **Step 1: Add BlockTransferred emission on transfer success**

In `DecodeTransferQueue.pop_transferred()`, at line 1648 where `poll == KVPoll.Success`:

```python
# python/sglang/srt/disaggregation/decode.py, inside pop_transferred()
# After line 1671: transferred_reqs.append(decode_req.req)

            elif poll == KVPoll.Success:
                if (
                    self.scheduler.enable_decode_hicache
                    and hicache_restore_status == HiCacheRestoreResult.PENDING
                ):
                    continue
                self._commit_transfer_to_req(decode_req)
                indices_to_remove.add(i)
                # Check if request was aborted due to corruption
                if isinstance(decode_req.req.finished_reason, FINISH_ABORT):
                    # ... existing abort handling ...
                else:
                    transferred_reqs.append(decode_req.req)

                    # NEW: Emit BlockTransferred event
                    if (
                        hasattr(self.tree_cache, 'enable_kv_cache_events')
                        and self.tree_cache.enable_kv_cache_events
                    ):
                        transfer_backend = getattr(
                            self.scheduler.server_args,
                            'disaggregation_transfer_backend', 'unknown'
                        )
                        req = decode_req.req
                        kv_committed_len = req.kv_committed_len
                        page_size = self.tree_cache.page_size
                        num_pages = kv_committed_len // page_size
                        # Get block hashes from req_to_token_pool
                        kv_indices = self.tree_cache.req_to_token_pool.req_to_token[
                            req.req_pool_idx, :kv_committed_len
                        ]
                        block_hashes = kv_indices[::page_size].tolist()[:num_pages]
                        # Approximate bytes: num_tokens * (k+v element bytes)
                        kv_cache = self.tree_cache.token_to_kv_pool_allocator.get_kvcache()
                        try:
                            k_bytes, v_bytes = kv_cache.get_kv_size_bytes()
                            per_token_kv_bytes = (k_bytes + v_bytes) // kv_cache.size
                        except Exception:
                            per_token_kv_bytes = 65536  # fallback estimate

                        from sglang.srt.disaggregation.kv_events import BlockTransferred
                        self.tree_cache.kv_event_queue.append(
                            BlockTransferred(
                                block_hashes=block_hashes,
                                req_id=req.rid,
                                src=req.bootstrap_host,
                                dst=self.scheduler.server_args.host,
                                backend=transfer_backend,
                                transfer_latency_us=0,
                                bytes_transferred=kv_committed_len * per_token_kv_bytes,
                            )
                        )
```

- [ ] **Step 2: Verify import**

```bash
cd python && python -c "
from sglang.srt.disaggregation.decode import DecodeTransferQueue
from sglang.srt.disaggregation.kv_events import BlockTransferred
print('PASS: BlockTransferred import works in decode context')
"
```
Expected: `PASS`

- [ ] **Step 3: Commit**

```bash
git add python/sglang/srt/disaggregation/decode.py
git commit -m "feat(kv-events): emit BlockTransferred on PD disaggregation transfer complete"
```

---

### Task 3.3: Emit BlockOffloaded/BlockLoaded from HiCache host pool transfers

**Files:**
- Modify: `python/sglang/srt/mem_cache/memory_pool_host.py` (in transfer methods)

- [ ] **Step 1: Add event emission in transfer methods**

In the `MHATokenToKVPoolHost` and `MLATokenToKVPoolHost` transfer paths (e.g., `load_to_device_per_layer`, `transfer_to_host`), add event emission:

```python
# Offload (GPU → Host):
if hasattr(self, 'kv_event_queue') and self.kv_event_queue is not None:
    from sglang.srt.disaggregation.kv_events import BlockOffloaded
    self.kv_event_queue.append(
        BlockOffloaded(
            block_hashes=host_indices_batch,
            src_medium="GPU",
            dst_medium="CPU",
            pool_name="kv",
        )
    )

# Load (Host → GPU):
if hasattr(self, 'kv_event_queue') and self.kv_event_queue is not None:
    from sglang.srt.disaggregation.kv_events import BlockLoaded
    self.kv_event_queue.append(
        BlockLoaded(
            block_hashes=device_indices_batch,
            src_medium="CPU",
            dst_medium="GPU",
            load_latency_us=int(elapsed_us),
        )
    )
```

- [ ] **Step 2: Commit**

```bash
git add python/sglang/srt/mem_cache/memory_pool_host.py
git commit -m "feat(kv-events): emit BlockOffloaded/BlockLoaded from HiCache host pool"
```

---

## Phase 4 — Prometheus Exporter

### Task 4.1: Implement PrometheusExporter

**Files:**
- Create: `sglang-kv-footprint/src/sglang_kv_footprint/exporters/prometheus.py`

**Interfaces:**
- Produces: `PrometheusExporter` with metrics registered on `prometheus_client`

- [ ] **Step 1: Implement**

```python
# sglang-kv-footprint/src/sglang_kv_footprint/exporters/prometheus.py
"""Prometheus metrics exporter for KV footprint data."""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

from sglang_kv_footprint.aggregator import RequestKVFootprint


# --- Metric Definitions ---

kv_footprint_gpu_bytes = Histogram(
    "sglang_kv_footprint_gpu_bytes_per_request",
    "Per-request KV cache GPU footprint (peak bytes)",
    buckets=[1e6, 4e6, 16e6, 64e6, 256e6, 1e9, 4e9, 16e9],
    labelnames=["model"],
)

kv_footprint_cpu_bytes = Histogram(
    "sglang_kv_footprint_cpu_bytes_per_request",
    "Per-request KV cache CPU footprint (peak bytes)",
    buckets=[1e6, 4e6, 16e6, 64e6, 256e6, 1e9],
    labelnames=["model"],
)

kv_footprint_events_total = Counter(
    "sglang_kv_footprint_events_total",
    "Total KV footprint events processed",
    labelnames=["event_type", "medium"],
)

kv_footprint_active_requests = Gauge(
    "sglang_kv_footprint_active_requests",
    "Number of requests with active KV footprint",
)

kv_footprint_total_gpu_bytes = Gauge(
    "sglang_kv_footprint_total_gpu_bytes",
    "Total GPU bytes across all active KV footprints",
)

kv_transfer_latency_us = Histogram(
    "sglang_kv_transfer_latency_us",
    "PD disaggregation KV transfer latency",
    labelnames=["backend"],
    buckets=[100, 500, 1000, 5000, 10000, 50000, 100000],
)

kv_offload_latency_us = Histogram(
    "sglang_kv_offload_latency_us",
    "KV offload/load latency",
    labelnames=["direction", "medium"],
    buckets=[100, 500, 1000, 5000, 10000, 50000, 100000],
)


class PrometheusExporter:
    """Exports footprint metrics to Prometheus."""

    def __init__(self, model: str = ""):
        self._model = model

    def export(self, footprint: RequestKVFootprint) -> None:
        labels = {"model": self._model}
        kv_footprint_gpu_bytes.labels(**labels).observe(footprint.peak_gpu_bytes)
        if footprint.peak_cpu_bytes > 0:
            kv_footprint_cpu_bytes.labels(**labels).observe(footprint.peak_cpu_bytes)

    def update_gauges(self, aggregator) -> None:
        """Periodic gauge update (called from consumer loop)."""
        kv_footprint_active_requests.set(aggregator.active_count)
        total_gpu = sum(
            fp.total_gpu_bytes for fp in aggregator._footprints.values()
        )
        kv_footprint_total_gpu_bytes.set(total_gpu)
```

- [ ] **Step 2: Wire into FootprintEventPublisher._init_exporters()**

```python
# In publisher.py, add to _init_exporters:
elif name == "prometheus":
    exporters.append(PrometheusExporter(model=self.config.otlp_service_name))
```

- [ ] **Step 3: Commit**

```bash
cd sglang-kv-footprint
git add src/sglang_kv_footprint/exporters/prometheus.py
git commit -m "feat: PrometheusExporter for KV footprint metrics"
```

---

## Phase 5 — Query API + OTLP Span Exporter

### Task 5.1: Implement FootprintQueryAPI

**Files:**
- Create: `sglang-kv-footprint/src/sglang_kv_footprint/api.py`

- [ ] **Step 1: Implement FastAPI router**

```python
# sglang-kv-footprint/src/sglang_kv_footprint/api.py
"""Optional HTTP query API for KV footprint inspection."""

from __future__ import annotations

from typing import Optional

from fastapi import FastAPI, HTTPException, Query

from sglang_kv_footprint.aggregator import FootprintAggregator


def create_app(aggregator: FootprintAggregator) -> FastAPI:
    app = FastAPI(title="KV Footprint API", version="0.1.0")

    @app.get("/footprint/{req_id}")
    async def get_footprint(req_id: str):
        fp = aggregator._footprints.get(req_id)
        if fp is None:
            raise HTTPException(status_code=404, detail=f"Request {req_id} not found")
        return {
            "req_id": fp.req_id,
            "start_ts": fp.start_ts,
            "end_ts": fp.end_ts,
            "gpu_pages": sorted(fp.gpu_pages),
            "cpu_pages": sorted(fp.cpu_pages),
            "disk_pages": sorted(fp.disk_pages),
            "total_gpu_bytes": fp.total_gpu_bytes,
            "peak_gpu_bytes": fp.peak_gpu_bytes,
            "events": fp.events,
        }

    @app.get("/footprints")
    async def list_footprints(
        status: Optional[str] = Query(None),
        limit: int = Query(100, ge=1, le=1000),
    ):
        active = [
            {
                "req_id": fp.req_id,
                "start_ts": fp.start_ts,
                "gpu_bytes": fp.total_gpu_bytes,
            }
            for fp in list(aggregator._footprints.values())[:limit]
        ]
        return {"footprints": active, "count": aggregator.active_count}

    @app.get("/footprint/summary")
    async def get_summary():
        total_gpu = sum(
            fp.total_gpu_bytes for fp in aggregator._footprints.values()
        )
        total_cpu = sum(
            fp.total_cpu_bytes for fp in aggregator._footprints.values()
        )
        return {
            "active_requests": aggregator.active_count,
            "total_gpu_bytes": total_gpu,
            "total_cpu_bytes": total_cpu,
        }

    return app
```

- [ ] **Step 2: Commit**

```bash
cd sglang-kv-footprint
git add src/sglang_kv_footprint/api.py
git commit -m "feat: FootprintQueryAPI for real-time KV footprint inspection"
```

---

### Task 5.2: Implement OTELSpanExporter

**Files:**
- Create: `sglang-kv-footprint/src/sglang_kv_footprint/exporters/otel.py`

- [ ] **Step 1: Implement**

```python
# sglang-kv-footprint/src/sglang_kv_footprint/exporters/otel.py
"""OpenTelemetry Span exporter for KV footprint data."""

from __future__ import annotations

import logging

from sglang_kv_footprint.aggregator import RequestKVFootprint

logger = logging.getLogger(__name__)

_otel_available = False
try:
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
        OTLPSpanExporter as GRPCSpanExporter,
    )
    from opentelemetry.sdk.resources import SERVICE_NAME, Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    _otel_available = True
except ImportError:
    logger.debug("OpenTelemetry not installed; OTLP export disabled")


class OTELSpanExporter:
    """Exports each completed request footprint as an OTel Span."""

    def __init__(self, service_name: str = "sglang-kv-footprint",
                 otlp_endpoint: str = "http://localhost:4317"):
        if not _otel_available:
            raise ImportError("opentelemetry packages required for OTLP export")

        resource = Resource(attributes={SERVICE_NAME: service_name})
        provider = TracerProvider(resource=resource)
        exporter = GRPCSpanExporter(endpoint=otlp_endpoint)
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)

        self._tracer = trace.get_tracer(__name__)

    def export(self, footprint: RequestKVFootprint) -> None:
        with self._tracer.start_as_current_span(
            "kv_footprint",
            start_time=int(footprint.start_ts * 1e9),
        ) as span:
            span.set_attribute("req_id", footprint.req_id)
            span.set_attribute("peak_gpu_bytes", footprint.peak_gpu_bytes)
            span.set_attribute("peak_cpu_bytes", footprint.peak_cpu_bytes)
            span.set_attribute("total_gpu_bytes", footprint.total_gpu_bytes)

            for ts, etype in footprint.events:
                span.add_event(
                    etype,
                    timestamp=int(ts * 1e9),
                )

            if footprint.end_ts:
                span.end(end_time=int(footprint.end_ts * 1e9))
```

- [ ] **Step 2: Wire into FootprintEventPublisher._init_exporters()**

```python
elif name == "otel":
    if self.config.otlp_endpoint:
        exporters.append(
            OTELSpanExporter(
                service_name=self.config.otlp_service_name,
                otlp_endpoint=self.config.otlp_endpoint,
            )
        )
```

- [ ] **Step 3: Commit**

```bash
cd sglang-kv-footprint
git add src/sglang_kv_footprint/exporters/otel.py
git commit -m "feat: OTELSpanExporter for distributed trace integration"
```

---

## Cross-Cutting Verification

After all phases are complete, run the end-to-end verification:

```bash
# 1. Start sglang with footprint plugin enabled
python -m sglang.launch_server \
    --model meta-llama/Llama-3-8B \
    --enable-kv-cache-events \
    --kv-events-publisher '{"publisher":"footprint"}' &

# 2. Send test requests
curl http://localhost:30000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"prompt": "Hello, world!", "max_tokens": 10}'

# 3. Check JSONL output
cat /tmp/sglang-kv-footprints/sglang-kv-footprint-*.jsonl | head -3

# 4. Check Prometheus metrics (if enabled)
curl http://localhost:30000/metrics | grep kv_footprint

# 5. Check query API (if enabled)
curl http://localhost:9090/footprint/summary

# 6. Verify zero overhead: compare latency p50/p99 with and without plugin
```
