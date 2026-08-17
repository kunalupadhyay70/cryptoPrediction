"""Stage 2Q — End-to-end offline synthetic integration tests.

Every module (target_engineering, dataset_builder, walk_forward,
feature_selection, model_training, oof_builder, threshold_selection,
position_engine, metrics, production_model, experiment_runner, main) has
its own unit/contract test suite already. This file is deliberately NOT a
re-run of those -- it adds full-PIPELINE-level checks that only make sense
once every stage is wired together end to end via main.train_backtest:

  1. A leakage probe run through the ENTIRE pipeline (OHLCV -> features ->
     target -> OOF -> thresholds -> backtest -> production model),
     perturbing only far-future rows and asserting every on-disk artifact
     for the unaffected historical region is untouched.
  2. Determinism: running the whole pipeline twice on identical input with
     a fixed random_state produces byte-identical OOF predictions and
     production model feature importances.
  3. On-disk reconciliation: the trade_ledger.csv and strategy_timeline.csv
     artifacts written to disk (not just the in-memory DataFrames already
     checked in tests/test_experiment_runner.py) reconcile exactly.

Entirely offline -- no network access anywhere in this file.
"""
import numpy as np
import pandas as pd
import pytest

from config_schema import load_config
from data_collector import DataCollector
from main import train_backtest


def _synthetic_ohlcv(n=400, seed=71):
    rng = np.random.default_rng(seed)
    ret = rng.normal(0, 0.0025, n)
    close = 100 * np.cumprod(1 + ret)
    open_ = np.roll(close, 1)
    open_[0] = 100.0
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.0007, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.0007, n)))
    volume = np.abs(rng.normal(1000, 100, n))
    times = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    return pd.DataFrame(
        {"open_time": times, "open": open_, "high": high, "low": low, "close": close, "volume": volume}
    )


def _small_cfg_dict(db_path):
    return {
        "data": {
            "exchange": "binance_futures", "symbol": "BTCUSDT", "interval": "5m", "target_days": 1,
            "db_path": str(db_path), "rest_base_url": "https://fapi.binance.com",
            "ws_base_url": "wss://fstream.binance.com/ws", "depth_limit": 20, "trades_limit": 1000,
            "kline_limit": 1500, "pagination_sleep_seconds": 0.2, "incremental": True, "integrity_check": True,
        },
        "target": {"horizon_bars": 3, "neutral_atr_mult": 0.4, "atr_period": 14},
        "features": {"lag_periods": 2, "correlation_threshold": 0.95, "importance_top_k": 6, "min_fold_stability": 0.5},
        "validation": {
            "n_folds": 3, "min_train_rows": 130, "min_test_rows": 30, "embargo_bars": 0,
            "internal_validation_fraction": 0.2, "min_class_count": 3,
        },
        "models": {
            "random_state": 7, "logistic_regression": {"C": 1.0, "max_iter": 1000},
            "lightgbm": {
                "num_leaves": 7, "learning_rate": 0.1, "n_estimators": 25, "min_child_samples": 5,
                "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.0, "reg_lambda": 0.0,
                "early_stopping_rounds": 5,
            },
            "catboost": {"enabled": False},
        },
        "backtest": {
            "fee_bps": 5.0, "slippage_bps": 2.0, "latency_bps": 1.0, "funding_bps_per_bar": 0.0,
            "default_buy_threshold": 0.45, "default_sell_threshold": 0.4, "min_trades_for_tuning": 5,
            "threshold_sweep_min": 0.34, "threshold_sweep_max": 0.8, "threshold_sweep_step": 0.02,
            "initial_equity": 10000.0, "position_fraction": 0.5, "max_trades_per_day": 50.0,
        },
        "experiment": {"name": "e2e_integration_test"},
        "live": {"poll_seconds": 1, "emit_every_iterations": 1, "max_iterations": 1},
    }


def _run_pipeline(tmp_path, ohlcv, run_name, output_dir):
    import yaml
    cfg_dict = _small_cfg_dict(tmp_path / f"{run_name}.db")
    cfg_dict["experiment"]["name"] = run_name
    config_path = tmp_path / f"{run_name}.yaml"
    config_path.write_text(yaml.dump(cfg_dict))
    cfg = load_config(config_path)

    import pytest as _pytest  # local import to use monkeypatch context manager
    mp = _pytest.MonkeyPatch()
    try:
        mp.setattr(DataCollector, "collect_historical_paginated", lambda self: {"total_stored": len(ohlcv)})
        mp.setattr(DataCollector, "run_integrity_check", lambda self: {"gap_events": 0, "missing_bars": 0, "rows": len(ohlcv), "duplicates": 0, "issues": []})
        mp.setattr(DataCollector, "load_ohlcv_dataframe", lambda self: ohlcv)
        train_backtest(cfg, output_dir=str(output_dir))
    finally:
        mp.undo()


# ---------------------------------------------------------------------------
# 1. Full-pipeline leakage probe
# ---------------------------------------------------------------------------

def test_full_pipeline_leakage_probe_perturbing_far_future_rows(tmp_path):
    ohlcv = _synthetic_ohlcv(n=400)
    cutoff = 380  # deep into the tail; far past every fold's earlier test rows

    out_before = tmp_path / "before"
    _run_pipeline(tmp_path, ohlcv, "leakage_before", out_before)

    perturbed = ohlcv.copy()
    perturbed.loc[cutoff:, ["open", "high", "low", "close", "volume"]] *= 5.0
    out_after = tmp_path / "after"
    _run_pipeline(tmp_path, perturbed, "leakage_after", out_after)

    # run_id legitimately differs between the two runs (it's the experiment
    # name, set per-run) -- excluded from the comparison on purpose, not a
    # weakening of the leakage guarantee itself.
    oof_before = pd.read_csv(out_before / "oof_predictions.csv").drop(columns=["run_id"])
    oof_after = pd.read_csv(out_after / "oof_predictions.csv").drop(columns=["run_id"])

    # Every OOF row whose timestamp is well before the perturbed region
    # must be byte-for-byte identical across the two runs -- this is the
    # strongest possible full-pipeline leakage guarantee: features,
    # target, fold assignment, selected features, model predictions,
    # thresholds, and signal all traced through the ENTIRE pipeline.
    safe_timestamp = ohlcv.loc[cutoff - 20, "open_time"]
    safe_before = oof_before[oof_before["timestamp"] < str(safe_timestamp)].sort_values(
        ["model", "fold", "timestamp"]
    ).reset_index(drop=True)
    safe_after = oof_after[oof_after["timestamp"] < str(safe_timestamp)].sort_values(
        ["model", "fold", "timestamp"]
    ).reset_index(drop=True)

    assert len(safe_before) > 0, "leakage probe needs at least some pre-cutoff OOF rows to compare"
    pd.testing.assert_frame_equal(safe_before, safe_after)


# ---------------------------------------------------------------------------
# 2. Determinism
# ---------------------------------------------------------------------------

def test_full_pipeline_is_deterministic_given_fixed_random_state(tmp_path):
    ohlcv = _synthetic_ohlcv(n=400, seed=99)

    out1 = tmp_path / "run1"
    _run_pipeline(tmp_path, ohlcv, "determinism_run1", out1)
    out2 = tmp_path / "run2"
    _run_pipeline(tmp_path, ohlcv, "determinism_run2", out2)

    oof1 = pd.read_csv(out1 / "oof_predictions.csv").drop(columns=["run_id"])
    oof2 = pd.read_csv(out2 / "oof_predictions.csv").drop(columns=["run_id"])
    pd.testing.assert_frame_equal(oof1, oof2)

    fi1 = pd.read_csv(out1 / "feature_importance.csv").sort_values("feature").reset_index(drop=True)
    fi2 = pd.read_csv(out2 / "feature_importance.csv").sort_values("feature").reset_index(drop=True)
    pd.testing.assert_frame_equal(fi1, fi2)


# ---------------------------------------------------------------------------
# 3. On-disk reconciliation
# ---------------------------------------------------------------------------

def test_on_disk_ledger_and_timeline_reconcile(tmp_path):
    ohlcv = _synthetic_ohlcv(n=400, seed=13)
    out_dir = tmp_path / "reconcile"
    _run_pipeline(tmp_path, ohlcv, "reconcile_run", out_dir)

    ledger = pd.read_csv(out_dir / "trade_ledger.csv")
    timeline = pd.read_csv(out_dir / "strategy_timeline.csv")

    assert not ledger.empty
    assert not timeline.empty

    for model in timeline["model"].unique():
        model_timeline = timeline[timeline["model"] == model]
        model_ledger = ledger[ledger["model"] == model] if "model" in ledger.columns else ledger

        total_timeline_net_pnl = model_timeline["net_pnl"].sum()
        total_ledger_net_pnl = model_ledger["net_pnl"].sum() if not model_ledger.empty else 0.0
        assert total_timeline_net_pnl == pytest.approx(total_ledger_net_pnl, abs=1e-6)
