# v3: Multi-Modal Fusion

**Goal:** Unify text, vision, and audio models under a single serving layer with cross-modal routing.

### Features
- Vision model support (LLaVA, CLIP, Stable Diffusion)
- Audio model support (Whisper, Bark)
- Cross-modal routing & fusion
- Multi-modal prompt templating
- Unified tokenization

### Structure
```
v3-multimodal/
├── src/
│   ├── vision/       # Image/video model handlers
│   ├── audio/        # Speech/audio model handlers
│   ├── fusion/       # Cross-modal pipeline
│   └── router/       # Modal-aware router
├── tests/
└── requirements.txt
```
