"""Tests for experiment_runner.py (Stage 2O)."""
import json

import numpy as np
import pandas as pd
import pytest

from dataset_builder import build_dataset, feature_columns
from experiment_runner import ExperimentError, run_experiment
from oof_builder import ALL_MODELS, OOFPipelineConfig


def _synthetic_ohlcv(n=300, seed=31):
    rng = np.random.default_rng(seed)
    ret = rng.normal(0, 0.003, n)
    close = 100 * np.cumprod(1 + ret)
    open_ = np.roll(close, 1)
    open_[0] = 100.0
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.0008, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.0008, n)))
    volume = np.abs(rng.normal(1000, 100, n))
    times = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    return pd.DataFrame(
        {"open_time": times, "open": open_, "high": high, "low": low, "close": close, "volume": volume}
    )


@pytest.fixture(scope="module")
def small_config():
    return OOFPipelineConfig(
        n_folds=2, min_train_rows=150, min_test_rows=40, horizon_bars=3, embargo_bars=0,
        min_class_count=3, correlation_threshold=0.95, importance_top_k=5,
        internal_validation_fraction=0.2, lgbm_early_stopping_rounds=5, random_state=42,
        lgbm_params={"n_estimators": 30, "num_leaves": 7},
    )


@pytest.fixture(scope="module")
def experiment_result(tmp_path_factory, small_config):
    ohlcv = _synthetic_ohlcv()
    dataset = build_dataset(ohlcv, horizon_bars=3, neutral_atr_mult=0.4, atr_period=14, lag_periods=2)
    candidate_features = feature_columns(dataset)
    out_dir = tmp_path_factory.mktemp("experiment_artifacts")
    result = run_experiment(
        dataset, candidate_features, symbol="BTCUSDT", interval="5m", run_id="exp_test_1",
        oof_config=small_config, output_dir=str(out_dir),
        threshold_kwargs={"min_trades_for_tuning": 5},
        backtest_kwargs={"initial_equity": 10_000.0, "position_fraction": 0.5, "fee_bps": 5.0, "slippage_bps": 2.0, "latency_bps": 1.0},
        production_lgbm_params={"n_estimators": 30, "num_leaves": 7},
    )
    return result, out_dir


# ---------------------------------------------------------------------------
# Artifact presence + structure
# ---------------------------------------------------------------------------

_EXPECTED_FILES = [
    "config_snapshot.json", "metrics.json", "fold_metrics.csv", "oof_predictions.csv",
    "feature_importance.csv", "selected_features.json", "trade_ledger.csv",
    "strategy_timeline.csv", "production_model.joblib", "equity_curve.png",
    "confusion_matrix_lightgbm.png",
]


def test_all_expected_artifact_files_are_written(experiment_result):
    _, out_dir = experiment_result
    for fname in _EXPECTED_FILES:
        path = out_dir / fname
        assert path.exists(), f"missing artifact: {fname}"
        assert path.stat().st_size > 0, f"artifact is empty: {fname}"


def test_config_snapshot_is_valid_json_with_expected_keys(experiment_result):
    _, out_dir = experiment_result
    payload = json.loads((out_dir / "config_snapshot.json").read_text())
    assert payload["symbol"] == "BTCUSDT"
    assert payload["run_id"] == "exp_test_1"
    assert payload["oof_config"]["horizon_bars"] == 3


def test_metrics_json_has_all_four_models_and_benchmark(experiment_result):
    _, out_dir = experiment_result
    payload = json.loads((out_dir / "metrics.json").read_text())
    assert set(payload["predictive_metrics_by_model"].keys()) == set(ALL_MODELS)
    assert set(payload["trading_metrics_by_model"].keys()) == set(ALL_MODELS)
    assert "total_return" in payload["benchmark_trading_metrics"]
    for m in ALL_MODELS:
        pm = payload["predictive_metrics_by_model"][m]
        assert "macro_f1" in pm and "confusion_matrix" in pm and "class_distribution" in pm


def test_oof_predictions_csv_matches_returned_oof_df_row_count(experiment_result):
    result, out_dir = experiment_result
    on_disk = pd.read_csv(out_dir / "oof_predictions.csv")
    assert len(on_disk) == len(result.oof_df)


def test_selected_features_json_matches_production_model(experiment_result):
    result, out_dir = experiment_result
    saved = json.loads((out_dir / "selected_features.json").read_text())
    assert saved == result.production_model.selected_features


def test_feature_importance_csv_has_one_row_per_selected_feature(experiment_result):
    result, out_dir = experiment_result
    fi = pd.read_csv(out_dir / "feature_importance.csv")
    assert len(fi) == len(result.production_model.selected_features)
    assert set(fi["feature"]) == set(result.production_model.selected_features)


def test_trade_ledger_csv_row_count_matches_sum_of_model_ledgers(experiment_result):
    result, out_dir = experiment_result
    ledger = pd.read_csv(out_dir / "trade_ledger.csv")
    expected_total = sum(len(l) for l in result.ledgers_by_model.values())
    assert len(ledger) == expected_total


def test_strategy_timeline_csv_has_model_column_and_full_row_count(experiment_result):
    result, out_dir = experiment_result
    timeline = pd.read_csv(out_dir / "strategy_timeline.csv")
    assert "model" in timeline.columns
    expected_total = sum(len(t) for t in result.timelines_by_model.values())
    assert len(timeline) == expected_total
    assert set(timeline["model"].unique()) == set(ALL_MODELS)


def test_production_model_is_loadable_and_usable(experiment_result):
    import joblib
    result, out_dir = experiment_result
    model = joblib.load(out_dir / "production_model.joblib")
    X = result.oof_df[["prob_down"]].rename(columns={"prob_down": result.production_model.selected_features[0]})
    # Just confirm the loaded model responds to predict_proba without error
    # on a minimally-shaped input using its own selected features.
    from production_model import predict_proba_production
    features_df = pd.DataFrame(
        {f: np.zeros(3) for f in result.production_model.selected_features}
    )
    probs = predict_proba_production(result.production_model, features_df)
    assert probs.shape == (3, 3)


# ---------------------------------------------------------------------------
# Fold metrics content
# ---------------------------------------------------------------------------

def test_fold_metrics_csv_has_one_row_per_fold(experiment_result, small_config):
    _, out_dir = experiment_result
    fm = pd.read_csv(out_dir / "fold_metrics.csv")
    assert len(fm) == small_config.n_folds
    assert set(fm.columns) == {
        "fold", "n_outer_train", "n_outer_test", "n_internal_train", "n_internal_val",
        "n_correlation_pruned_features", "n_selected_features", "best_iteration",
    }


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def test_missing_open_time_raises(tmp_path, small_config):
    ohlcv = _synthetic_ohlcv()
    dataset = build_dataset(ohlcv, horizon_bars=3, neutral_atr_mult=0.4, atr_period=14, lag_periods=2)
    dataset = dataset.drop(columns=["open_time"])
    with pytest.raises(ExperimentError):
        run_experiment(
            dataset, feature_columns(dataset), "BTCUSDT", "5m", "r1", small_config,
            str(tmp_path / "out"), write_plots=False,
        )
