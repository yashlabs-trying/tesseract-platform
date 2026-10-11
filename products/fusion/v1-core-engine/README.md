# v1: Core Engine

**Goal:** Minimal inference server capable of loading models and serving predictions.

### Features
- Model registry and loader (HuggingFace, local)
- REST + gRPC inference API
- Token streaming (SSE)
- Request/response logging
- CLI for serving

### Structure
```
v1-core-engine/
├── src/
│   ├── core/          # Model loading, inference loop
│   ├── api/           # REST/gRPC endpoints
│   └── cli/           # CLI interface
├── tests/
└── requirements.txt
```
