# v2: Optimization

**Goal:** Improve throughput and latency via model compression and batching.

### Features
- FP16/INT8/INT4 quantization (GPTQ, AWQ, BitsAndBytes)
- Dynamic batching & continuous batching
- PagedAttention / KV cache optimization
- Speculative decoding
- Prefix caching

### Structure
```
v2-optimization/
├── src/
│   ├── quant/         # Quantization backends
│   ├── batch/         # Batching engine
│   ├── cache/         # KV cache management
│   └── decode/        # Speculative decoding
├── benchmarks/
└── requirements.txt
```
