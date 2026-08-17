"""Boundary tests for walk_forward.py (Stage 2G).

Expected fold boundaries below were hand-derived (see the module docstring
of walk_forward.py for the reasoning) and cross-checked with a standalone
script before being written into these assertions.
"""
import numpy as np
import pytest

from walk_forward import WalkForwardError, WalkForwardSplit, generate_walk_forward_splits


# ---------------------------------------------------------------------------
# Exact boundary arithmetic (hand-derived): n_rows=100, n_folds=3,
# min_train_rows=40, min_test_rows=10, horizon_bars=3, embargo_bars=0
# ---------------------------------------------------------------------------

def test_exact_fold_boundaries_hand_derived():
    splits = generate_walk_forward_splits(
        n_rows=100, n_folds=3, min_train_rows=40, min_test_rows=10, horizon_bars=3
    )
    assert len(splits) == 3

    f1, f2, f3 = splits
    assert (f1.test_start, f1.test_end) == (43, 62)
    assert (f2.test_start, f2.test_end) == (62, 81)
    assert (f3.test_start, f3.test_end) == (81, 100)

    assert f1.purge_start == 40
    assert f2.purge_start == 59
    assert f3.purge_start == 78

    assert len(f1.train_idx) == 40 and f1.train_idx.max() == 39
    assert len(f2.train_idx) == 59 and f2.train_idx.max() == 58
    assert len(f3.train_idx) == 78 and f3.train_idx.max() == 77


def test_fold_1_train_size_is_exactly_min_train_rows_not_short():
    # This is the specific off-by-one the purge-offset fix targets: without
    # reserving horizon_bars extra rows before fold 1's test starts, fold 1
    # would always end up with (min_train_rows - horizon_bars) usable rows,
    # silently under-delivering the requested minimum.
    splits = generate_walk_forward_splits(
        n_rows=200, n_folds=2, min_train_rows=50, min_test_rows=10, horizon_bars=7
    )
    assert len(splits[0].train_idx) == 50


# ---------------------------------------------------------------------------
# Purge correctness: no train row's label window can overlap the test start
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("horizon_bars", [1, 3, 5, 10])
def test_purge_removes_exactly_horizon_bars_rows_before_test(horizon_bars):
    splits = generate_walk_forward_splits(
        n_rows=300, n_folds=4, min_train_rows=60, min_test_rows=10, horizon_bars=horizon_bars
    )
    for s in splits:
        # The highest train row index must be strictly less than
        # test_start - horizon_bars is WRONG framing; it must be exactly
        # test_start - horizon_bars - 1 (i.e., purge_start - 1), proving no
        # train row falls inside [purge_start, test_start).
        assert s.purge_start == s.test_start - horizon_bars
        assert s.train_idx.max() == s.purge_start - 1
        # No train row anywhere in [purge_start, test_start).
        assert not np.any((s.train_idx >= s.purge_start) & (s.train_idx < s.test_start))


def test_a_training_row_whose_label_window_touches_test_is_never_included():
    # Direct simulation: a training row t has label window [t, t+h]; if
    # t+h >= test_start, that label overlaps test and t must be purged.
    horizon_bars = 5
    splits = generate_walk_forward_splits(
        n_rows=150, n_folds=2, min_train_rows=50, min_test_rows=10, horizon_bars=horizon_bars
    )
    for s in splits:
        for t in s.train_idx:
            assert t + horizon_bars < s.test_start, (
                f"train row {t} has label window reaching {t + horizon_bars}, "
                f"which overlaps test_start={s.test_start}"
            )


# ---------------------------------------------------------------------------
# Test-block coverage: contiguous, non-overlapping, exactly once each
# ---------------------------------------------------------------------------

def test_test_blocks_are_contiguous_and_non_overlapping():
    splits = generate_walk_forward_splits(
        n_rows=250, n_folds=5, min_train_rows=50, min_test_rows=10, horizon_bars=3
    )
    for prev, curr in zip(splits, splits[1:]):
        assert prev.test_end == curr.test_start

    all_test_rows = np.concatenate([s.test_idx for s in splits])
    assert len(all_test_rows) == len(set(all_test_rows.tolist())), "duplicate test rows across folds"

    # Full coverage from the first fold's test_start through n_rows.
    assert all_test_rows.min() == splits[0].test_start
    assert all_test_rows.max() == 250 - 1
    assert set(all_test_rows.tolist()) == set(range(splits[0].test_start, 250))


def test_expanding_window_train_only_grows():
    splits = generate_walk_forward_splits(
        n_rows=250, n_folds=5, min_train_rows=50, min_test_rows=10, horizon_bars=3
    )
    prev_len = 0
    for s in splits:
        assert len(s.train_idx) > prev_len
        prev_len = len(s.train_idx)


# ---------------------------------------------------------------------------
# Embargo: rows immediately after a PRIOR fold's test are excluded from
# LATER folds' train until embargo_bars have elapsed
# ---------------------------------------------------------------------------

def test_embargo_excludes_rows_right_after_a_prior_test_block():
    embargo_bars = 5
    splits = generate_walk_forward_splits(
        n_rows=100, n_folds=3, min_train_rows=40, min_test_rows=10,
        horizon_bars=3, embargo_bars=embargo_bars,
    )
    f1, f2, f3 = splits
    # f1's test_end == 62 (same arithmetic as the no-embargo case, since
    # embargo only affects train_idx of LATER folds, not test placement).
    assert f1.test_end == 62
    embargoed_rows = set(range(f1.test_end, f1.test_end + embargo_bars))
    # f3's raw train range reaches up to purge_start=78, which is beyond
    # f1's embargoed rows (62-66) -> they must be absent from f3.train_idx.
    assert embargoed_rows.issubset(set(range(0, f3.purge_start)))
    assert not embargoed_rows & set(f3.train_idx.tolist())
    # But every OTHER row in [0, purge_start) that is not embargoed by any
    # prior fold must still be present.
    expected_present = set(range(0, f3.purge_start)) - embargoed_rows
    assert expected_present == set(f3.train_idx.tolist())


def test_embargo_zero_is_a_true_no_op():
    with_embargo = generate_walk_forward_splits(
        n_rows=100, n_folds=3, min_train_rows=40, min_test_rows=10, horizon_bars=3, embargo_bars=0
    )
    without_embargo_kw = generate_walk_forward_splits(
        n_rows=100, n_folds=3, min_train_rows=40, min_test_rows=10, horizon_bars=3
    )
    for a, b in zip(with_embargo, without_embargo_kw):
        np.testing.assert_array_equal(a.train_idx, b.train_idx)
        assert (a.test_start, a.test_end) == (b.test_start, b.test_end)


def test_embargo_delays_reentry_by_exactly_embargo_bars():
    # Increasing embargo_bars by 1 must shrink the affected later fold's
    # train_idx by exactly 1 (as long as the embargo window stays fully
    # inside that fold's raw train range) — a precise boundary check, not
    # just "embargo removes some rows".
    kwargs = dict(n_rows=100, n_folds=3, min_train_rows=40, min_test_rows=10, horizon_bars=3)
    s5 = generate_walk_forward_splits(embargo_bars=5, **kwargs)
    s6 = generate_walk_forward_splits(embargo_bars=6, **kwargs)
    assert len(s5[2].train_idx) == len(s6[2].train_idx) + 1


# ---------------------------------------------------------------------------
# Infeasible configurations: fail loudly, not silently
# ---------------------------------------------------------------------------

def test_raises_when_not_enough_rows_for_a_single_fold():
    with pytest.raises(WalkForwardError):
        generate_walk_forward_splits(
            n_rows=45, n_folds=1, min_train_rows=40, min_test_rows=10, horizon_bars=3
        )


def test_raises_when_fewer_folds_fit_than_requested_by_default():
    # 100 rows, min_train_rows=40, horizon_bars=3 -> first test can start at
    # 43 at the earliest, leaving 57 rows. Requesting 20 folds forces a
    # base_test_size of max(10, 57//20)=10, but repeatedly shrinking train
    # via purge on every fold is not the failure mode here — the failure
    # mode tested is simply requesting more folds than the data supports
    # at all once min_test_rows dominates.
    with pytest.raises(WalkForwardError):
        generate_walk_forward_splits(
            n_rows=100, n_folds=20, min_train_rows=40, min_test_rows=10, horizon_bars=3
        )


def test_allow_fewer_folds_returns_fewer_without_raising():
    splits = generate_walk_forward_splits(
        n_rows=100, n_folds=20, min_train_rows=40, min_test_rows=10, horizon_bars=3,
        allow_fewer_folds=True,
    )
    assert 1 <= len(splits) < 20


def test_purge_eating_below_min_train_rows_reduces_or_fails():
    # A very large horizon_bars relative to min_train_rows and n_rows can
    # make later folds infeasible too (not just fold 1) if data runs out;
    # the point of this test is simply that the function never silently
    # returns a fold whose train_idx is shorter than min_train_rows.
    with pytest.raises(WalkForwardError):
        generate_walk_forward_splits(
            n_rows=60, n_folds=3, min_train_rows=40, min_test_rows=10, horizon_bars=15
        )


@pytest.mark.parametrize(
    "bad_kwargs",
    [
        dict(n_folds=0),
        dict(min_train_rows=0),
        dict(min_test_rows=0),
        dict(horizon_bars=0),
        dict(embargo_bars=-1),
    ],
)
def test_invalid_parameters_raise(bad_kwargs):
    kwargs = dict(n_rows=100, n_folds=3, min_train_rows=40, min_test_rows=10, horizon_bars=3, embargo_bars=0)
    kwargs.update(bad_kwargs)
    with pytest.raises(WalkForwardError):
        generate_walk_forward_splits(**kwargs)


# ---------------------------------------------------------------------------
# Determinism / immutability
# ---------------------------------------------------------------------------

def test_splits_are_deterministic_across_calls():
    kwargs = dict(n_rows=250, n_folds=5, min_train_rows=50, min_test_rows=10, horizon_bars=3, embargo_bars=2)
    a = generate_walk_forward_splits(**kwargs)
    b = generate_walk_forward_splits(**kwargs)
    for sa, sb in zip(a, b):
        np.testing.assert_array_equal(sa.train_idx, sb.train_idx)
        np.testing.assert_array_equal(sa.test_idx, sb.test_idx)


def test_split_dataclass_is_frozen():
    splits = generate_walk_forward_splits(
        n_rows=100, n_folds=1, min_train_rows=40, min_test_rows=10, horizon_bars=3
    )
    with pytest.raises(Exception):
        splits[0].fold = 999
