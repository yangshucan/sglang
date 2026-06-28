# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SGLang is a high-performance LLM serving framework — a Python monorepo with C++/CUDA kernels, Rust routing/gRPC components, and a large test suite. It powers over 400,000 GPUs worldwide across NVIDIA, AMD, Intel, Google TPU, and Ascend NPU hardware.

**Package name:** `sglang` (PyPI), CLI entry point: `sglang`

## Environment Variables & Server Arguments

- All `SGLANG_*` env vars are defined and accessed through `python/sglang/srt/environ.py`. Never create env vars outside this module. See `.claude/skills/env-var-conventions/SKILL.md` before touching any env var.
- Server arguments are defined as a `@dataclasses.dataclass` in `python/sglang/srt/server_args.py`. New args must be added as dataclass fields with appropriate type annotations and argparse integration.

## Build & Run

```bash
# Install from source (editable, with test deps)
pip install -e "python/[test]" --no-build-isolation

# Build sgl-kernel (AOT C++/CUDA kernels)
cd sgl-kernel && bash build.sh

# Install pre-commit hooks
pre-commit install
```

## Lint & Format

Pre-commit runs automatically, but individual tools:

```bash
# Python formatting
ruff check --select=F401,F821 --fix <file>
black <file>
isort <file>

# C++/CUDA formatting
clang-format --style=file -i <file>

# Spell check
codespell --config .codespellrc

# Notebook cleanup
nbstripout --keep-output <notebook>

# Rust formatting
cd sgl-model-gateway && cargo +nightly fmt -- --check
cd experimental/sgl-router && cargo fmt -- --check
```

## Tests

### Running tests

```bash
# Single test file
python3 test/registered/core/test_srt_endpoint.py

# Single test method
python3 test/registered/core/test_srt_endpoint.py TestSRTEndpoint.test_simple_decode

# JIT kernel test
python3 python/sglang/jit_kernel/tests/test_add_constant.py

# Suite-based (with auto-partitioning for CI jobs)
python3 test/run_suite.py --hw cuda --suite base-b-test-1-gpu-small
python3 test/run_suite.py --hw cuda --suite base-b-test-1-gpu-small \
    --auto-partition-id 0 --auto-partition-size 4

# Nightly tests
python3 test/run_suite.py --hw cuda --suite nightly-1-gpu --nightly

# CPU tests
python3 test/run_suite.py --hw cpu --suite base-a-test-cpu
```

### Test structure

```
test/
├── registered/       # CI-discovered tests (most tests here)
├── manual/           # Non-CI tests for local/specialized use
├── run_suite.py      # CI runner — scans registered/ and JIT kernel dirs
└── srt/              # Legacy CI setup (being deprecated)
```

Every CI test file must call a registration function at module level:
```python
from sglang.test.ci.ci_register import register_cuda_ci
register_cuda_ci(est_time=80, stage="base-b", runner_config="1-gpu-small")
```

Tests support both `unittest` and `pytest`. CI runs with failfast. Each file must end with either `unittest.main()` or `pytest.main([__file__])`. See `test/README.md` and `.claude/skills/write-sglang-test/SKILL.md` for complete guidance.

### CI stages

Three sequential stages: **A** (pre-flight, ~3 min) → **B** (basic, ~30 min) → **C** (advanced, ~30 min). Kernel and multimodal-gen tests run in parallel with stage B. See `.claude/skills/ci-workflow-guide/SKILL.md`.

Suite selection by hardware need:
| Need | Suite |
|------|-------|
| No GPU | `base-a-test-cpu` |
| Small GPU (32GB) | `base-b-test-1-gpu-small` |
| Large GPU / Hopper | `base-b-test-1-gpu-large` |
| JIT kernel correctness | `base-b-kernel-unit-1-gpu-large` |
| Multi-GPU | `base-b-test-2-gpu-large`, `base-c-test-*` |
| Long-running/experimental | `nightly-*` |

## Key Dependencies

- `torch==2.11.0`, `transformers==5.8.1`
- `flashinfer_python[cu13]==0.6.12`, `flash-attn-4>=4.0.0b9`
- `xgrammar==0.2.1` (structured output)
- `sglang-kernel==0.4.3` (AOT CUDA kernels, the `sgl-kernel/` directory in this repo)
- `sgl-deep-gemm==0.1.2`, `quack-kernels>=0.4.1`
- `nvidia-cutlass-dsl[cu13]==4.5.2`
- Optional: `sglang[ray]` for Ray multi-node, `sglang[tracing]` for OTel, `sglang[http2]` for Granian HTTP/2

## High-Level Architecture

SGLang is an LLM inference engine with two major layers:

### 1. Frontend Language (`python/sglang/lang/`)
A composable DSL for LLM programming (`gen`, `select`, `function`, `system`/`user`/`assistant` roles). Compiled to an intermediate representation (IR). Supports multiple backends (OpenAI, Anthropic, LiteLLM, local SRT). This is the original "SGLang" concept — the DSL that gave the project its name.

### 2. Server Runtime — SRT (`python/sglang/srt/`)
The high-performance inference engine. Key subsystems:

- **Managers** (`managers/`): Orchestration layer.
  - `Scheduler` — Core scheduling loop. Receives requests, manages token pools, dispatches forward passes. One of three large-class-init components (see rules).
  - `TokenizerManager` — Tokenization/detokenization, communicates with Scheduler via ZMQ. Also a large-class-init component.
  - `DataParallelController` — Coordinates data-parallel replicas.

- **ModelRunner** (`model_executor/model_runner.py`): Executes forward passes, manages KV cache, CUDA graphs, and attention backends. The third large-class-init component.

- **KV Cache** (`mem_cache/`): RadixAttention-based prefix caching (`radix_cache.py`), HiCache hierarchical caching (`hicache_storage.py`), SWA sliding-window caches, chunk caches, and multiple memory pool implementations. The radix tree (`radix_cache.py`, C++ backed `radix_cache_cpp.py`) is central to SGLang's performance.

- **Models** (`models/`): ~198 model implementations. Each model matches a HuggingFace family. Models are composed from shared **Layers** (`layers/`): attention (FlashInfer/FlashAttention backends), linear layers (with quantization support), MoE, normalization, rotary embeddings, samplers, etc.

- **Distributed Inference** (`distributed/`): Tensor parallelism (TP), pipeline parallelism (PP), expert parallelism (EP), and data parallelism (DP) via `parallel_state.py`. Device communicators support NCCL, Mooncake (RDMA), NIXL, etc.

- **Prefill-Decode Disaggregation** (`disaggregation/`): Splits prefill and decode across different GPUs/clusters. Supports multiple transfer backends (Mooncake RDMA, NIXL, MORI, Ascend, fake for testing).

- **Speculative Decoding** (`speculative/`): Eagle, DFlash, n-gram, MTP (Multi-Token Prediction), and Frozen KV MTP workers. Uses a registry pattern (`spec_registry.py`).

- **Compilation** (`compilation/`): torch.compile integration with piecewise CUDA graphs, FX passes, and Inductor backends.

- **Entrypoints** (`entrypoints/`): HTTP server (FastAPI), gRPC server, OpenAI-compatible API (`entrypoints/openai/`), Anthropic-compatible API (`entrypoints/anthropic/`), Ollama-compatible API. The `Engine` class (`engine.py`) is the main Python entry point.

- **Connectors** (`connector/`): Model weight loading from S3, Azure Blob, Redis.

- **Session** (`session/`): Session-level context and streaming session management.

- **Observability** (`observability/`): Prometheus metrics, request timing stats, startup logging, CPU monitoring.

### 3. Kernel Libraries

- **`sgl-kernel/`**: AOT C++/CUDA kernels with Python bindings. Covers attention, MoE, quantization, GEMM, allreduce, etc. Separate CMake build. Published as `sglang-kernel` on PyPI. See `.claude/skills/add-sgl-kernel/SKILL.md`.

- **`python/sglang/jit_kernel/`**: Lightweight JIT Triton kernels. Runtime-compiled. For self-contained kernels that don't need the heavyweight AOT build. See `.claude/skills/add-jit-kernel/SKILL.md`.

### 4. Other Components

- **`sgl-model-gateway/`**: Rust-based model routing gateway (Cargo project).
- **`rust/sglang-grpc/`**: PyO3 Rust bindings for the gRPC protocol core.
- **`sglang-kv-footprint/`**: KV cache footprint analysis tool.
- **`docs_new/`**: Active documentation using Mintlify cookbook format.
- **`docs/`**: Legacy docs — **no new content allowed** (enforced by pre-commit).
- **`benchmark/`**: Model-specific benchmark configurations (~45 models).
- **`3rdparty/`**: Vendored third-party code.
- **`proto/`**: Protobuf definitions for internal protocols.

## Critical Rules

Before modifying the following, you MUST read the corresponding skill via the Skill tool:

| Component | Skill |
|-----------|-------|
| Speculative decoding (anything under `python/sglang/srt/speculative/`, related attention backends, scheduler accumulators, IPC fields, metrics, CLI flags) | `speculative-naming` |
| `Scheduler.__init__`, `TokenizerManager.__init__`, `ModelRunner.__init__` | `large-class-init-style` |
| Environment variables (adding/renaming `SGLANG_*`, touching `environ.py`) | `env-var-conventions` |
| Scripted runtime (anything related) | `scripted-runtime-notes` |

## Documentation

- Public docs: https://docs.sglang.io
- Contribution guide: https://docs.sglang.io/developer_guide/contribution_guide.html
