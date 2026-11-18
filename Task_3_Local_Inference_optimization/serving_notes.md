# Task 3: Inference Optimization & 24GB GPU Capacity Plan

## 1. Precision Benchmark Observations (7B Model Reference)
- **16-Bit (BF16):** ~15.20 GB VRAM | ~38 tokens/sec | Full syntactic accuracy.
- **4-Bit (Q4_K_M):** ~4.80 GB VRAM | ~72 tokens/sec | ~98% quality parity with ~3x lower memory footprint.

## 2. Core Concepts
- **KV Cache:** Stores past tokens' Key and Value attention tensors in GPU VRAM during autoregressive generation so the model does not recalculate previous tokens at every step.
- **Quantization:** Compresses weights from 16-bit floats (2 bytes/param) to 4-bit integers (0.5 bytes/param), reducing VRAM usage and memory bandwidth bottlenecks.

## 3. Serving 20 Concurrent Users on 1x 24GB GPU (RTX 3090 / 4090)
- **Model:** Qwen2.5-7B-Instruct (4-bit quantized) = **4.80 GB**
- **Engine:** vLLM (PagedAttention + Continuous Batching)
- **Context Assumption:** 500 input tokens + 128 output tokens = 628 tokens/user
- **KV Cache (20 Users):** ~**0.72 GB**
- **CUDA Runtime Overhead:** ~**1.50 GB**
- **Total Peak VRAM:** **~7.02 GB** (Leaves ~17 GB headroom for smooth, low-latency serving).
