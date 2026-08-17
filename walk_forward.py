"""Stage 2G — Purge-aware expanding-window outer walk-forward splits.

Frozen contract (Stage 1B/1C, restated in the Stage 2G task):
  - Expanding-window outer validation: outer_train grows fold over fold,
    outer_test moves forward in contiguous, non-overlapping blocks.
  - Before training, PURGE the trailing ``horizon_bars`` rows off the end
    of outer_train. Any training row within ``horizon_bars`` bars of
    outer_test's start has a target whose label window
    (``exit[t] = close[t+horizon_bars]``, see target_engineering.py)
    overlaps the test set — training on it would leak test-window price
    information into the model via its own label. Per config_schema's
    frozen rule, purge size is ALWAYS exactly ``horizon_bars`` — there is
    no independently configurable purge_bars (single source of truth).
  - EMBARGO (config_schema.ValidationConfig.embargo_bars, default 0): an
    additional buffer of rows immediately AFTER each outer_test block that
    is excluded from every LATER fold's outer_train. Even though a row
    right after a test block has no label overlap with that test block
    (embargo is not about label leakage), its FEATURES are backward-looking
    rolling/EMA windows that can pick up serial correlation from data
    right at the test boundary; embargoing it delays how soon that data
    re-enters training in a later, larger expanding window.
  - Outer test must never influence feature selection, scaling, training,
    hyperparameters, early stopping, or threshold selection for that same
    fold (enforced by later stages that consume these splits — Stage
    2H/2I/2K — not by this module, which only computes row-index splits).

This module works purely on row POSITION (0..n_rows-1), matching
target_engineering's/dataset_builder's row-position convention. It does not
read a DataFrame or touch any config object — callers pass plain ints
(mirroring config_schema.ValidationConfig / TargetConfig field names) so
this stays decoupled from config_schema until the runtime cutover.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


class WalkForwardError(ValueError):
    """Raised when the requested fold/row-count configuration cannot
    produce the requested number of valid folds (fails loudly rather than
    silently returning fewer folds than asked for, unless the caller
    explicitly opts into that via allow_fewer_folds=True)."""


@dataclass(frozen=True)
class WalkForwardSplit:
    fold: int                 # 1-indexed fold number
    train_idx: np.ndarray      # sorted row positions eligible for outer_train
    test_idx: np.ndarray       # sorted row positions in outer_test (contiguous)
    test_start: int            # first row position in outer_test
    test_end: int              # exclusive end of outer_test
    purge_start: int           # first row position purged (== test_start - horizon_bars)


def generate_walk_forward_splits(
    n_rows: int,
    n_folds: int,
    min_train_rows: int,
    min_test_rows: int,
    horizon_bars: int,
    embargo_bars: int = 0,
    allow_fewer_folds: bool = False,
) -> List[WalkForwardSplit]:
    """Generate purge-aware, expanding-window outer walk-forward splits.

    Test blocks are CONTIGUOUS and NON-OVERLAPPING, partitioning
    ``[min_train_rows, n_rows)`` into ``n_folds`` equal-sized blocks (the
    final block absorbs any remainder rows from integer-division). This
    guarantees full test coverage with no gaps and no duplicate test rows
    across folds — required by the Stage 2J OOF contract ("every eligible
    outer-test timestamp must appear ... exactly once").

    Raises WalkForwardError (fails loudly, never silently) if:
      - fewer than 1 valid fold can be produced under the given minimums, or
      - exactly ``n_folds`` valid folds cannot be produced and
        ``allow_fewer_folds`` is False (the default).
    """
    if n_folds < 1:
        raise WalkForwardError(f"n_folds must be >= 1, got {n_folds}")
    if min_train_rows < 1:
        raise WalkForwardError(f"min_train_rows must be >= 1, got {min_train_rows}")
    if min_test_rows < 1:
        raise WalkForwardError(f"min_test_rows must be >= 1, got {min_test_rows}")
    if horizon_bars < 1:
        raise WalkForwardError(f"horizon_bars must be >= 1, got {horizon_bars}")
    if embargo_bars < 0:
        raise WalkForwardError(f"embargo_bars must be >= 0, got {embargo_bars}")

    # The FIRST fold's outer_test must start `horizon_bars` rows after
    # min_train_rows, not exactly at min_train_rows: purge always removes
    # the trailing `horizon_bars` rows of outer_train before test, so
    # starting test exactly at min_train_rows would leave fold 1 with only
    # `min_train_rows - horizon_bars` usable training rows — always short
    # of the requested minimum, for ANY horizon_bars >= 1. Reserving the
    # extra `horizon_bars` rows up front means fold 1's post-purge
    # train_idx has EXACTLY min_train_rows rows (the tightest, most
    # honest fold 1 possible), and every later fold only grows from there.
    first_test_start = min_train_rows + horizon_bars
    available_for_test = n_rows - first_test_start
    if available_for_test < min_test_rows:
        raise WalkForwardError(
            f"Not enough rows for even one fold: n_rows={n_rows}, "
            f"min_train_rows={min_train_rows} + horizon_bars(purge)={horizon_bars} "
            f"leaves only {available_for_test} rows for testing, need "
            f">= min_test_rows={min_test_rows}"
        )

    base_test_size = max(min_test_rows, available_for_test // n_folds)

    splits: List[WalkForwardSplit] = []
    # Tracks (test_end, embargo_end) for every PRIOR fold so later folds can
    # exclude those rows from outer_train.
    prior_embargo_ranges: List[tuple] = []

    fold = 0
    test_start = first_test_start
    while test_start < n_rows and len(splits) < n_folds:
        remaining = n_rows - test_start
        is_last_requested_fold = (len(splits) == n_folds - 1)
        test_size = remaining if is_last_requested_fold else min(base_test_size, remaining)
        test_end = test_start + test_size
        if test_end - test_start < min_test_rows:
            break  # not enough rows left for a valid final block

        purge_start = test_start - horizon_bars
        raw_train_end = max(0, purge_start)

        train_mask = np.ones(raw_train_end, dtype=bool)
        for (prior_test_end, prior_embargo_end) in prior_embargo_ranges:
            lo = max(0, prior_test_end)
            hi = min(raw_train_end, prior_embargo_end)
            if hi > lo:
                train_mask[lo:hi] = False
        train_idx = np.nonzero(train_mask)[0]

        if len(train_idx) < min_train_rows:
            break  # purge/embargo ate too far into the required train size

        fold += 1
        splits.append(
            WalkForwardSplit(
                fold=fold,
                train_idx=train_idx,
                test_idx=np.arange(test_start, test_end),
                test_start=test_start,
                test_end=test_end,
                purge_start=purge_start,
            )
        )
        prior_embargo_ranges.append((test_end, test_end + embargo_bars))
        test_start = test_end

    if len(splits) == 0:
        raise WalkForwardError(
            f"Could not produce any valid fold from n_rows={n_rows}, n_folds={n_folds}, "
            f"min_train_rows={min_train_rows}, min_test_rows={min_test_rows}, "
            f"horizon_bars={horizon_bars}, embargo_bars={embargo_bars}"
        )
    if len(splits) < n_folds and not allow_fewer_folds:
        raise WalkForwardError(
            f"Requested n_folds={n_folds} but only {len(splits)} valid fold(s) fit "
            f"given n_rows={n_rows}, min_train_rows={min_train_rows}, "
            f"min_test_rows={min_test_rows}, horizon_bars={horizon_bars}, "
            f"embargo_bars={embargo_bars}. Pass allow_fewer_folds=True to accept "
            f"fewer folds instead of failing."
        )
    return splits
