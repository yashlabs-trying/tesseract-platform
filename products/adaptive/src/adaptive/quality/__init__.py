"""Quality evaluation (Milestone 2).

Planned public interface::

    evaluate(architecture, profile, suite) -> QualityReport

Planned suites: perplexity, calibration reconstruction error, reference-vs-
quantized output divergence, task accuracy, long-context quality, structured
output correctness. Calibration and final evaluation datasets are kept
separate; reconstruction error alone is never accepted as proof of quality.
"""

from __future__ import annotations
