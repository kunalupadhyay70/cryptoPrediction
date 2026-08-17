"""Tests for main.py (Stage 2P runtime cutover).

Entirely offline: the one end-to-end test monkeypatches
DataCollector.collect_historical_paginated / run_integrity_check /
load_ohlcv_dataframe so nothing here ever touches the network.
"""
import json

import numpy as np
import pandas as pd
import pytest

from config_schema import load_config
from data_collector import DataCollector
from main import (
    _backtest_kwargs, _lgbm_params, _oof_config_from_app_config, _threshold_kwargs,
    build_collector, train_backtest,
)
from oof_builder import OOFPipelineConfig


# ---------------------------------------------------------------------------
# The real config.yaml must validate against AppConfig -- a direct
# regression guard for the Stage 2P cutover (the whole point of this stage
# is that config.yaml and its runtime consumers now agree on one contract).
# ---------------------------------------------------------------------------

def test_repo_config_yaml_validates_against_app_config():
    cfg = load_config("config.yaml")
    assert cfg.data.symbol == "BTCUSDT"
    assert cfg.data.interval == "5m"
    assert cfg.target.horizon_bars == 3


def test_repo_config_yaml_has_no_legacy_sections():
    import yaml
    with open("config.yaml") as f:
        raw = yaml.safe_load(f)
    # These top-level/legacy keys must be gone -- config_schema.AppConfig's
    # extra="forbid" would already reject them at load time, but this test
    # documents the intent directly against the raw YAML, independent of
    # whether load_config happens to be called first.
    assert "exchange" not in raw
    assert "database" not in raw
    assert "collection" not in raw
    assert "feature_pruning" not in raw
    assert "thresholds" not in raw
    assert "signal" not in raw
    assert "label_mode" not in raw["target"]
    assert "bar_size_minutes" not in raw["backtest"]


# ---------------------------------------------------------------------------
# Wiring helpers: AppConfig -> the new pipeline modules' own config objects
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def cfg():
    return load_config("tests/fixtures/config_valid.yaml")


def test_oof_config_wiring_matches_app_config_fields(cfg):
    oc = _oof_config_from_app_config(cfg)
    assert isinstance(oc, OOFPipelineConfig)
    assert oc.n_folds == cfg.validation.n_folds
    assert oc.min_train_rows == cfg.validation.min_train_rows
    assert oc.min_test_rows == cfg.validation.min_test_rows
    # Purge size == target.horizon_bars, ALWAYS -- never a separate field.
    assert oc.horizon_bars == cfg.target.horizon_bars
    assert oc.embargo_bars == cfg.validation.embargo_bars
    assert oc.min_class_count == cfg.validation.min_class_count
    assert oc.correlation_threshold == cfg.features.correlation_threshold
    assert oc.importance_top_k == cfg.features.importance_top_k
    assert oc.internal_validation_fraction == cfg.validation.internal_validation_fraction
    assert oc.lgbm_early_stopping_rounds == cfg.models.lightgbm.early_stopping_rounds
    assert oc.random_state == cfg.models.random_state


def test_lgbm_params_wiring_exact_values(cfg):
    # tests/fixtures/config_valid.yaml deliberately uses distinctive
    # non-default-looking values -- a direct regression guard against the
    # Stage 0 bug where LightGBM hyperparameters were present in config.yaml
    # but silently never reached the model.
    params = _lgbm_params(cfg)
    assert params["num_leaves"] == 47
    assert params["learning_rate"] == pytest.approx(0.037)
    assert params["n_estimators"] == 777
    assert params["min_child_samples"] == 100
    assert "n_estimators" in params  # present here; OOFPipelineConfig's fold loop overrides it with best_iteration later


def test_threshold_kwargs_wiring(cfg):
    tk = _threshold_kwargs(cfg)
    assert tk["default_buy_threshold"] == cfg.backtest.default_buy_threshold
    assert tk["default_sell_threshold"] == cfg.backtest.default_sell_threshold
    assert tk["min_trades_for_tuning"] == cfg.backtest.min_trades_for_tuning


def test_backtest_kwargs_wiring(cfg):
    bk = _backtest_kwargs(cfg)
    assert bk["fee_bps"] == cfg.backtest.fee_bps
    assert bk["position_fraction"] == cfg.backtest.position_fraction
    assert bk["initial_equity"] == cfg.backtest.initial_equity


def test_build_collector_wiring(cfg):
    collector = build_collector(cfg)
    assert isinstance(collector, DataCollector)
    assert collector.config.symbol == cfg.data.symbol
    assert collector.config.kline_interval == cfg.data.interval
    assert collector.config.db_path == cfg.data.db_path


# ---------------------------------------------------------------------------
# End-to-end offline smoke test: train_backtest wired against a
# monkeypatched, network-free DataCollector.
# ---------------------------------------------------------------------------

def _synthetic_ohlcv(n=300, seed=41):
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


def test_train_backtest_end_to_end_offline(tmp_path, monkeypatch):
    small_cfg_dict = {
        "data": {
            "exchange": "binance_futures", "symbol": "BTCUSDT", "interval": "5m", "target_days": 1,
            "db_path": str(tmp_path / "test.db"), "rest_base_url": "https://fapi.binance.com",
            "ws_base_url": "wss://fstream.binance.com/ws", "depth_limit": 20, "trades_limit": 1000,
            "kline_limit": 1500, "pagination_sleep_seconds": 0.2, "incremental": True, "integrity_check": True,
        },
        "target": {"horizon_bars": 3, "neutral_atr_mult": 0.4, "atr_period": 14},
        "features": {"lag_periods": 2, "correlation_threshold": 0.95, "importance_top_k": 5, "min_fold_stability": 0.5},
        "validation": {
            "n_folds": 2, "min_train_rows": 150, "min_test_rows": 40, "embargo_bars": 0,
            "internal_validation_fraction": 0.2, "min_class_count": 3,
        },
        "models": {
            "random_state": 42, "logistic_regression": {"C": 1.0, "max_iter": 1000},
            "lightgbm": {
                "num_leaves": 7, "learning_rate": 0.1, "n_estimators": 30, "min_child_samples": 5,
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
        "experiment": {"name": "test_experiment"},
        "live": {"poll_seconds": 1, "emit_every_iterations": 1, "max_iterations": 1},
    }
    import yaml
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump(small_cfg_dict))
    cfg = load_config(config_path)

    ohlcv = _synthetic_ohlcv()
    monkeypatch.setattr(DataCollector, "collect_historical_paginated", lambda self: {"total_stored": len(ohlcv)})
    monkeypatch.setattr(DataCollector, "run_integrity_check", lambda self: {"gap_events": 0, "missing_bars": 0, "rows": len(ohlcv), "duplicates": 0, "issues": []})
    monkeypatch.setattr(DataCollector, "load_ohlcv_dataframe", lambda self: ohlcv)

    out_dir = tmp_path / "artifacts_out"
    train_backtest(cfg, output_dir=str(out_dir))

    assert (out_dir / "metrics.json").exists()
    assert (out_dir / "production_model.joblib").exists()
    assert (out_dir / "production_thresholds.json").exists()
    thresholds = json.loads((out_dir / "production_thresholds.json").read_text())
    assert thresholds["model"] == "lightgbm"
    assert 0.0 <= thresholds["buy_threshold"] <= 1.0
