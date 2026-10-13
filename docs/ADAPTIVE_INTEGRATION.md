# Adaptive Integration (Optional, Disabled by Default)

Adaptive Quantization (`yashlabs-trying/tesseract-adaptive`) is a future
product under active development. The platform is designed to work **completely
without it**.

## Reserved interface

`integrations/adaptive/contracts.py` defines the only surface the platform knows:

```python
class AdaptivePrecisionProvider(Protocol):
    def list_profiles(self, model_id: str) -> list[PrecisionProfileRef]: ...
    def evaluate_constraints(self, request: WorkloadRequest) -> list[str]: ...
    def recommend_profile(self, request, capacity=None) -> PrecisionProfileRef | None: ...
    def validate_profile(self, profile: PrecisionProfileRef) -> bool: ...
    def plan_transition(self, current, target) -> dict: ...
    def plan_rollback(self, transition) -> dict: ...
```

`PrecisionProfileRef` mirrors `tesseract.platform.precision-profile.v1` and
covers BF16/FP16, int8, int4, mixed precision, quality-certificate references,
weight-memory estimates, backend compatibility and transition overhead.

## Providers shipped today

| Provider | When | Behaviour |
|---|---|---|
| `DisabledAdaptiveProvider` | default (`TESSERACT_ADAPTIVE_ENABLED=false`) | reports nothing available; platform deploys declared precision |
| `UnavailableAdaptiveProvider` | enabled but no URL | raises `IntegrationUnavailable` (never silently degrades) |
| `HttpAdaptiveProvider` | enabled with URL | scaffold; endpoint paths are placeholders until Adaptive publishes its service interface |

`build_adaptive_provider(settings)` selects the provider. **No provider
fabricates quantization results.**

## Target integration flow (future)

```
Adaptive: find a validated precision profile
        |
        v
Fusion:   recalculate memory and placement for the profile
        |
        v
Ignite:   deploy the selected profile
        |
        v
Foundry:  verify measured performance
```

When Adaptive is wired in:

1. `_resolve_precision` calls `recommend_profile` and `validate_profile`; a
   profile that fails validation is rejected and recorded, and the platform
   falls back to the model's declared precision.
2. The chosen `profile_ref` flows into the `Deployment` (`precision_profile_ref`)
   and into Ignite's `ReplicaGroup` (`spec.model.precision`).
3. `plan_transition` / `plan_rollback` produce the transition and reverse plans
   used by controlled-action mode.

## Connecting the real repository later

* Implement `AdaptivePrecisionProvider` in `tesseract-adaptive` (or a small
  sidecar) and expose it over HTTP.
* Point `TESSERACT_ADAPTIVE_URL` at it and set `TESSERACT_ADAPTIVE_ENABLED=true`.
* No coordinator changes are required — the coordinator depends only on the
  protocol.
* Contract tests (`tests/integration/test_adaptive_optional.py`) already prove
  the platform operates correctly when Adaptive is absent.

## Non-goals

* The platform does not implement quantization.
* The platform does not claim any quantization quality result.
* Adaptive remains independently maintainable and releasable.
