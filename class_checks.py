"""Shared MISSING-CLASS POLICY enforcement (frozen architecture).

    For model-fitting splits, DOWN/NEUTRAL/UP must all have sufficient
    representation (validation.min_class_count). If training/internal-
    validation class coverage is insufficient: FAIL LOUDLY. Do not
    silently turn the problem into a two-class model. Do not silently
    change folds.

Used by every stage that fits a real model on a specific split (Stage 2H's
fold-local feature-importance model, Stage 2I's baselines/LogReg/LightGBM).
Kept as its own tiny module rather than duplicated in each — one source of
truth for what "insufficient" means and how it fails.
"""
from __future__ import annotations

from typing import Dict, Iterable

import numpy as np

CLASS_VALUES = (0, 1, 2)


class MissingClassError(ValueError):
    """Raised when a model-fitting split does not have min_class_count rows
    for every one of the 3 frozen classes (DOWN=0, NEUTRAL=1, UP=2)."""


def class_counts(y: Iterable) -> Dict[int, int]:
    y_arr = np.asarray(y)
    return {c: int(np.sum(y_arr == c)) for c in CLASS_VALUES}


def check_class_coverage(y: Iterable, min_class_count: int, context: str = "") -> Dict[int, int]:
    """Raise MissingClassError if any of the 3 classes has fewer than
    ``min_class_count`` rows in ``y``. Returns the per-class counts dict on
    success (callers can log it).

    This NEVER collapses to a 2-class problem, never drops the failing
    class, and never silently proceeds — the only two outcomes are "counts
    dict returned" or "MissingClassError raised".
    """
    if min_class_count < 1:
        raise ValueError(f"min_class_count must be >= 1, got {min_class_count}")
    counts = class_counts(y)
    insufficient = {c: n for c, n in counts.items() if n < min_class_count}
    if insufficient:
        prefix = f"{context}: " if context else ""
        raise MissingClassError(
            f"{prefix}insufficient class representation "
            f"(min_class_count={min_class_count}). Insufficient classes: {insufficient}. "
            f"Full counts: {counts}."
        )
    return counts
