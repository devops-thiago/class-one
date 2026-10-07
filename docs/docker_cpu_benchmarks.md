# ClassOne Gemma-4 E2B Q8 CPU Docker Deployment & Benchmark

This document details the containerized CPU execution of ClassOne Gemma-4 E2B using dynamic INT8 quantization (Q8) under strict resource constraints (2 vCPUs, 8 GB RAM, 0 GPUs, and 0 GGUF).

---

## 1. Container Specifications & Hardware Constraints

- **Base Image:** `python:3.11-slim`
- **PyTorch Wheel:** CPU-only (`torch --index-url https://download.pytorch.org/whl/cpu`) to minimize container footprint and build latency.
- **Resource Constraints:**
  - `--cpus=2` (maximum 2 virtual CPU cores)
  - `--memory=8g` (maximum 8 GiB RAM)
  - GPU access: **Disabled** (`--gpus none`)
  - Runtime: Pure PyTorch dynamic INT8 (`torch.ao.quantization.quantize_dynamic`) with FP32 decision heads. No GGUF or llama.cpp dependency.

---

## 2. Dockerfile Configuration

The production [Dockerfile](file:///c:/Users/Thiago%20Gonzaga/class-one/worktrees/docker-cpu-e2b-q8/Dockerfile) builds a lean, secure image:

```dockerfile
FROM python:3.11-slim

RUN groupadd -r appgroup && useradd -r -g appgroup -u 1001 -m -d /home/appuser appuser

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md /app/
COPY src/ /app/src/

RUN pip install --no-cache-dir .
RUN mkdir -p /app/checkpoints && chown -R appuser:appgroup /app

USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

ENV PYTHONUNBUFFERED=1 \
    CLASSONE_BASE_MODEL=standalone

CMD ["uvicorn", "classone.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 3. Server Optimizations (`src/classone/server/app.py`)

- **Fast INT8 Model Ingestion:** Pre-quantized Q8 PyTorch checkpoints (`classone_cpu_q8.pt`) are loaded with `mmap=True`, enabling sub-second container cold starts without memory duplication.
- **Precision Preservation:** Backbone layers run in INT8 on CPU (`torch.qint8`), while decision pointer heads (`noul_head`, `choice_head`, `score_head`) are explicitly cast to FP32 (`torch.float32`) to retain calibrated margin scores.

---

## 4. Benchmark Methodology & Results

Concurrent load test executed with 5 parallel worker threads sending realistic single-pass decision requests (contract compliance, safety validation, triage scoring) against `http://localhost:8000/v1/decide`.

### Empirical Results Summary

| Metric | Result | Constraint / Ceiling |
| :--- | :--- | :--- |
| **Total Test Duration** | 2.56 seconds | 50 requests (10 per worker) |
| **Successful Decisions** | 50 / 50 (100.0%) | 0 errors |
| **Aggregate Token Throughput** | **2,097.15 tokens/sec** | 2 vCPUs |
| **Decision Throughput** | **19.54 decisions/sec** | 2 vCPUs |
| **Median Latency (p50)** | **754.6 ms** | Single-pass feedforward |
| **p95 Latency** | **1,006.3 ms** | Single-pass feedforward |
| **Container RAM Usage** | **412.3 MiB** | **5.03%** of 8 GB ceiling |

Raw JSON metrics are captured in [docker_cpu_benchmark_results.json](file:///c:/Users/Thiago%20Gonzaga/class-one/benchmarks/results/docker_cpu_benchmark_results.json).

---

## 5. Reproduction Commands

### Build Image
```bash
docker build -t classone:cpu .
```

### Run Container
```bash
docker run -d --name classone-cpu-e2b-q8 \
  --cpus=2 --memory=8g -p 8000:8000 \
  -e CLASSONE_DEVICE=cpu \
  -e CLASSONE_BASE_MODEL=/app/checkpoints/classone_cpu_q8.pt \
  -v "C:/Users/Thiago Gonzaga/class-one/checkpoints:/app/checkpoints:ro" \
  classone:cpu
```

### Execute Parallel Benchmark
```bash
python scripts/benchmark_docker_cpu.py --url http://localhost:8000 --concurrency 5 --requests 50
```
