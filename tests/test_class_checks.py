"""Tests for class_checks.py (shared MISSING-CLASS POLICY enforcement)."""
import numpy as np
import pytest

from class_checks import MissingClassError, check_class_coverage, class_counts


def test_class_counts_basic():
    y = np.array([0, 0, 1, 1, 1, 2])
    assert class_counts(y) == {0: 2, 1: 3, 2: 1}


def test_check_class_coverage_passes_when_all_sufficient():
    y = np.array([0] * 10 + [1] * 10 + [2] * 10)
    counts = check_class_coverage(y, min_class_count=10)
    assert counts == {0: 10, 1: 10, 2: 10}


def test_check_class_coverage_raises_on_missing_class_entirely():
    y = np.array([0] * 10 + [1] * 10)  # no class 2 at all
    with pytest.raises(MissingClassError) as excinfo:
        check_class_coverage(y, min_class_count=5)
    assert "2" in str(excinfo.value)


def test_check_class_coverage_raises_on_underrepresented_class():
    y = np.array([0] * 10 + [1] * 10 + [2] * 3)  # class 2 has only 3 rows
    with pytest.raises(MissingClassError):
        check_class_coverage(y, min_class_count=5)


def test_check_class_coverage_boundary_is_inclusive():
    # Exactly min_class_count rows for the scarcest class must PASS (>=),
    # one fewer must FAIL.
    y_exact = np.array([0] * 10 + [1] * 10 + [2] * 5)
    check_class_coverage(y_exact, min_class_count=5)  # must not raise

    y_short = np.array([0] * 10 + [1] * 10 + [2] * 4)
    with pytest.raises(MissingClassError):
        check_class_coverage(y_short, min_class_count=5)


def test_check_class_coverage_never_collapses_to_two_class():
    # Regression guard for the explicit policy: even if two classes are
    # abundant, a missing third class must still raise, not silently
    # proceed as a 2-class problem.
    y = np.array([0] * 1000 + [1] * 1000)
    with pytest.raises(MissingClassError):
        check_class_coverage(y, min_class_count=1)


def test_invalid_min_class_count_raises():
    with pytest.raises(ValueError):
        check_class_coverage(np.array([0, 1, 2]), min_class_count=0)
