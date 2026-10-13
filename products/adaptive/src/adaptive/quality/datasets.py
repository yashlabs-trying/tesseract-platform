"""Token datasets and calibration/evaluation split isolation.

Calibration and final evaluation data must never be the same. The helpers here
enforce that invariant structurally so a quality report cannot accidentally
reuse its calibration set as the evaluation set.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum

from adaptive.errors import ConfigurationError


class DatasetSplit(str, Enum):
    calibration = "calibration"
    evaluation = "evaluation"


@dataclass(frozen=True)
class TokenDataset:
    """A named, pinned set of token-id sequences."""

    name: str
    split: DatasetSplit
    sequences: tuple[tuple[int, ...], ...]
    seed: int = 0

    def fingerprint(self) -> str:
        payload = {
            "name": self.name,
            "split": self.split.value,
            "seed": self.seed,
            "sequences": [list(seq) for seq in self.sequences],
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    def __len__(self) -> int:
        return len(self.sequences)


def ensure_disjoint(calibration: TokenDataset, evaluation: TokenDataset) -> None:
    """Raise unless the calibration and evaluation sets are distinct."""
    if calibration.split is not DatasetSplit.calibration:
        raise ConfigurationError("calibration dataset must use the calibration split")
    if evaluation.split is not DatasetSplit.evaluation:
        raise ConfigurationError("evaluation dataset must use the evaluation split")
    if calibration.fingerprint() == evaluation.fingerprint():
        raise ConfigurationError("calibration and evaluation datasets must be disjoint")
    cal_set = set(calibration.sequences)
    eval_set = set(evaluation.sequences)
    if cal_set and eval_set and cal_set & eval_set:
        raise ConfigurationError("calibration and evaluation sequences overlap")


def make_synthetic_dataset(
    name: str,
    split: DatasetSplit,
    *,
    vocab_size: int,
    num_sequences: int = 4,
    seq_len: int = 16,
    seed: int = 0,
) -> TokenDataset:
    """Build a deterministic pseudo-random token dataset for tests."""
    import random

    rng = random.Random(seed)
    sequences = tuple(
        tuple(rng.randrange(vocab_size) for _ in range(seq_len))
        for _ in range(num_sequences)
    )
    return TokenDataset(name=name, split=split, sequences=sequences, seed=seed)


def make_long_context_dataset(
    name: str,
    *,
    vocab_size: int,
    num_sequences: int = 2,
    seq_len: int = 64,
    seed: int = 0,
) -> TokenDataset:
    """Build an evaluation dataset of longer sequences for long-context checks."""
    return make_synthetic_dataset(
        name,
        DatasetSplit.evaluation,
        vocab_size=vocab_size,
        num_sequences=num_sequences,
        seq_len=seq_len,
        seed=seed,
    )
