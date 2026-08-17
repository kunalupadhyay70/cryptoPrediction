"""Tests for the canonical typed configuration (config_schema.py, Stage 2B).

Covers: successful parsing of a valid config, strict unknown-key rejection,
missing-required-field rejection, interval validation delegated to the
Stage 2A canonical authority (intervals.py), numeric-range validation, exact
preservation of LightGBM hyperparameters through the YAML -> AppConfig load
(the direct regression guard for the Stage 0 "config value silently ignored"
bug), and the YAML loader itself (including malformed/non-mapping YAML).
"""
import copy
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from config_schema import AppConfig, ConfigError, load_config

FIXTURES_DIR = Path(__file__).parent / "fixtures"
VALID_FIXTURE_PATH = FIXTURES_DIR / "config_valid.yaml"


@pytest.fixture
def valid_config_dict():
    """A fresh mutable copy of the canonical valid fixture as a plain dict,
    so individual tests can mutate it without affecting other tests."""
    with VALID_FIXTURE_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _dump(tmp_path: Path, data: dict, name: str = "config.yaml") -> Path:
    p = tmp_path / name
    with p.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f)
    return p


def _set_in(d: dict, dotted_key: str, value) -> dict:
    """Set a nested dict value via 'section.field' notation, in place."""
    section, field = dotted_key.split(".", 1)
    d[section][field] = value
    return d


def _del_in(d: dict, dotted_key: str) -> dict:
    section, field = dotted_key.split(".", 1)
    del d[section][field]
    return d


# ---------------------------------------------------------------------------
# Valid complete config
# ---------------------------------------------------------------------------

def test_valid_fixture_loads_via_load_config():
    cfg = load_config(VALID_FIXTURE_PATH)
    assert isinstance(cfg, AppConfig)


def test_valid_fixture_produces_typed_nested_models(valid_config_dict):
    cfg = AppConfig.model_validate(valid_config_dict)
    assert isinstance(cfg, AppConfig)
    assert cfg.data.exchange == "binance_futures"
    assert cfg.data.symbol == "BTCUSDT"
    assert cfg.data.interval == "5m"
    assert cfg.target.horizon_bars == 3
    assert cfg.features.include_orderbook_features is False
    assert cfg.validation.n_folds == 5
    assert cfg.models.random_state == 42
    assert cfg.models.catboost.enabled is False
    assert cfg.backtest.position_fraction == 1.0
    assert cfg.risk.enabled is False
    assert cfg.artifacts.root_dir == "artifacts"
    assert cfg.experiment.name == "stage2b_fixture"
    assert cfg.live.poll_seconds == 5
    assert cfg.live.emit_every_iterations == 12
    assert cfg.live.max_iterations == 100000


# ---------------------------------------------------------------------------
# Unknown key rejection (extra="forbid")
# ---------------------------------------------------------------------------

def test_unknown_key_in_nested_section_is_rejected(valid_config_dict):
    valid_config_dict["data"]["banana"] = 123
    with pytest.raises(ValidationError):
        AppConfig.model_validate(valid_config_dict)


def test_unknown_top_level_section_is_rejected(valid_config_dict):
    valid_config_dict["datta"] = {"foo": "bar"}  # typo'd section name
    with pytest.raises(ValidationError):
        AppConfig.model_validate(valid_config_dict)


def test_unknown_key_in_lightgbm_block_is_rejected(valid_config_dict):
    valid_config_dict["models"]["lightgbm"]["magical_parameter"] = 10
    with pytest.raises(ValidationError):
        AppConfig.model_validate(valid_config_dict)


def test_unknown_key_in_live_block_is_rejected(valid_config_dict):
    valid_config_dict["live"]["magical_option"] = True
    with pytest.raises(ValidationError):
        AppConfig.model_validate(valid_config_dict)


# ---------------------------------------------------------------------------
# purge_bars must NOT exist as an independent config field (Stage 2B-R FIX 1)
#
# The frozen architecture requires purge size == target.horizon_bars exactly.
# Allowing validation.purge_bars as its own field would create two sources
# of truth for a leakage-critical quantity. It must be rejected the same way
# any other unknown key is rejected.
# ---------------------------------------------------------------------------

def test_validation_purge_bars_is_rejected_as_unknown_field(valid_config_dict):
    valid_config_dict["validation"]["purge_bars"] = 3
    with pytest.raises(ValidationError) as excinfo:
        AppConfig.model_validate(valid_config_dict)
    assert "purge_bars" in str(excinfo.value)


def test_horizon_bars_alone_continues_to_validate_normally(valid_config_dict):
    # No validation.purge_bars anywhere in the fixture — target.horizon_bars
    # is the sole source of truth and must validate on its own without it.
    assert "purge_bars" not in valid_config_dict["validation"]
    cfg = AppConfig.model_validate(valid_config_dict)
    assert cfg.target.horizon_bars == 3


# ---------------------------------------------------------------------------
# live: config (Stage 2B-R FIX 2)
# ---------------------------------------------------------------------------

def test_live_config_survives_parsing_exactly(valid_config_dict):
    cfg = AppConfig.model_validate(valid_config_dict)
    assert cfg.live.poll_seconds == 5
    assert cfg.live.emit_every_iterations == 12
    assert cfg.live.max_iterations == 100000


@pytest.mark.parametrize(
    "dotted_key, bad_value",
    [
        ("live.poll_seconds", 0),
        ("live.emit_every_iterations", 0),
        ("live.max_iterations", 0),
    ],
)
def test_invalid_live_values_are_rejected(valid_config_dict, dotted_key, bad_value):
    _set_in(valid_config_dict, dotted_key, bad_value)
    with pytest.raises(ValidationError):
        AppConfig.model_validate(valid_config_dict)


def test_live_max_iterations_null_is_accepted_for_unlimited_operation(valid_config_dict):
    valid_config_dict["live"]["max_iterations"] = None
    cfg = AppConfig.model_validate(valid_config_dict)
    assert cfg.live.max_iterations is None


def test_live_missing_required_field_is_rejected(valid_config_dict):
    del valid_config_dict["live"]["poll_seconds"]
    with pytest.raises(ValidationError):
        AppConfig.model_validate(valid_config_dict)


def test_live_missing_entire_section_is_rejected(valid_config_dict):
    del valid_config_dict["live"]
    with pytest.raises(ValidationError):
        AppConfig.model_validate(valid_config_dict)


# ---------------------------------------------------------------------------
# Missing required field
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "dotted_key",
    [
        "data.symbol",
        "data.interval",
        "target.horizon_bars",
        "validation.n_folds",
        "backtest.fee_bps",
    ],
)
def test_missing_required_field_is_rejected(valid_config_dict, dotted_key):
    _del_in(valid_config_dict, dotted_key)
    with pytest.raises(ValidationError):
        AppConfig.model_validate(valid_config_dict)


def test_missing_required_lightgbm_field_is_rejected(valid_config_dict):
    del valid_config_dict["models"]["lightgbm"]["num_leaves"]
    with pytest.raises(ValidationError):
        AppConfig.model_validate(valid_config_dict)


def test_missing_entire_required_section_is_rejected(valid_config_dict):
    del valid_config_dict["models"]
    with pytest.raises(ValidationError):
        AppConfig.model_validate(valid_config_dict)


# ---------------------------------------------------------------------------
# Invalid interval — must go through the Stage 2A canonical validator
# ---------------------------------------------------------------------------

def test_invalid_interval_is_rejected(valid_config_dict):
    _set_in(valid_config_dict, "data.interval", "banana")
    with pytest.raises(ValidationError) as excinfo:
        AppConfig.model_validate(valid_config_dict)
    # The error message must originate from intervals.py's own diagnostic
    # (not a separately reinvented interval-validation message), proving the
    # canonical Stage 2A authority is actually being reused here.
    assert "Unsupported interval" in str(excinfo.value)


def test_valid_intervals_from_stage_2a_all_pass(valid_config_dict):
    from intervals import SUPPORTED_INTERVALS

    for interval in SUPPORTED_INTERVALS:
        d = copy.deepcopy(valid_config_dict)
        _set_in(d, "data.interval", interval)
        AppConfig.model_validate(d)  # must not raise


# ---------------------------------------------------------------------------
# Invalid numeric ranges
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "dotted_key, bad_value",
    [
        ("target.horizon_bars", 0),
        ("validation.n_folds", 1),
        ("models.lightgbm.learning_rate", 0.0),
        ("models.lightgbm.learning_rate", -0.01),
        ("models.lightgbm.subsample", 1.5),
        ("backtest.position_fraction", 1.5),
        ("validation.internal_validation_fraction", 1.0),
        ("validation.internal_validation_fraction", 0.0),
        ("target.atr_period", 1),
        ("features.correlation_threshold", 1.5),
        ("features.min_fold_stability", 0.0),
    ],
)
def test_invalid_numeric_range_is_rejected(valid_config_dict, dotted_key, bad_value):
    _set_in(valid_config_dict, dotted_key, bad_value)
    with pytest.raises(ValidationError):
        AppConfig.model_validate(valid_config_dict)


def test_threshold_sweep_min_must_be_less_than_max(valid_config_dict):
    valid_config_dict["backtest"]["threshold_sweep_min"] = 0.80
    valid_config_dict["backtest"]["threshold_sweep_max"] = 0.51
    with pytest.raises(ValidationError):
        AppConfig.model_validate(valid_config_dict)


def test_default_sell_threshold_must_be_less_than_buy_threshold(valid_config_dict):
    valid_config_dict["backtest"]["default_sell_threshold"] = 0.60
    valid_config_dict["backtest"]["default_buy_threshold"] = 0.55
    with pytest.raises(ValidationError):
        AppConfig.model_validate(valid_config_dict)


# ---------------------------------------------------------------------------
# Model hyperparameters preserved exactly through the parse
# ---------------------------------------------------------------------------

def test_lightgbm_hyperparameters_survive_parsing_exactly(valid_config_dict):
    # Regression guard for the Stage 0 bug class: "config.yaml contains a
    # value but the runtime object silently loses/ignores it." Here we
    # assert the loader boundary itself never drops or defaults over an
    # explicitly configured value.
    cfg = AppConfig.model_validate(valid_config_dict)
    assert cfg.models.lightgbm.num_leaves == 47
    assert cfg.models.lightgbm.learning_rate == pytest.approx(0.037)
    assert cfg.models.lightgbm.n_estimators == 777
    assert cfg.models.lightgbm.min_child_samples == 100
    assert cfg.models.lightgbm.subsample == pytest.approx(0.7)
    assert cfg.models.lightgbm.colsample_bytree == pytest.approx(0.7)
    assert cfg.models.lightgbm.reg_alpha == pytest.approx(0.5)
    assert cfg.models.lightgbm.reg_lambda == pytest.approx(2.0)
    assert cfg.models.lightgbm.early_stopping_rounds == 75


def test_distinctive_lightgbm_values_change_when_source_changes(valid_config_dict):
    # A stronger version of the above: change the source values to something
    # ELSE distinctive and confirm the parsed config tracks the change
    # exactly, rather than the test merely matching a coincidental default.
    valid_config_dict["models"]["lightgbm"]["num_leaves"] = 9001
    valid_config_dict["models"]["lightgbm"]["learning_rate"] = 0.0123
    valid_config_dict["models"]["lightgbm"]["n_estimators"] = 42
    cfg = AppConfig.model_validate(valid_config_dict)
    assert cfg.models.lightgbm.num_leaves == 9001
    assert cfg.models.lightgbm.learning_rate == pytest.approx(0.0123)
    assert cfg.models.lightgbm.n_estimators == 42


# ---------------------------------------------------------------------------
# YAML loader
# ---------------------------------------------------------------------------

def test_load_config_from_temporary_yaml_file(tmp_path, valid_config_dict):
    path = _dump(tmp_path, valid_config_dict)
    cfg = load_config(path)
    assert isinstance(cfg, AppConfig)
    assert cfg.data.interval == "5m"
    assert cfg.models.lightgbm.num_leaves == 47


def test_load_config_accepts_str_or_path(tmp_path, valid_config_dict):
    path = _dump(tmp_path, valid_config_dict)
    cfg_from_str = load_config(str(path))
    cfg_from_path = load_config(path)
    assert cfg_from_str == cfg_from_path


def test_load_config_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "does_not_exist.yaml")


def test_load_config_non_mapping_top_level_raises_config_error(tmp_path):
    path = tmp_path / "list_config.yaml"
    with path.open("w", encoding="utf-8") as f:
        f.write("- just\n- a\n- list\n")
    with pytest.raises(ConfigError):
        load_config(path)


def test_load_config_scalar_top_level_raises_config_error(tmp_path):
    path = tmp_path / "scalar_config.yaml"
    with path.open("w", encoding="utf-8") as f:
        f.write("just a string\n")
    with pytest.raises(ConfigError):
        load_config(path)


def test_load_config_malformed_yaml_syntax_raises(tmp_path):
    path = tmp_path / "broken.yaml"
    with path.open("w", encoding="utf-8") as f:
        f.write("data:\n  symbol: [unclosed\n  interval: 5m\n")
    with pytest.raises(yaml.YAMLError):
        load_config(path)


def test_load_config_with_invalid_field_raises_validation_error_not_swallowed(
    tmp_path, valid_config_dict
):
    # Ensure the loader does NOT catch a Pydantic error and substitute a
    # default — it must propagate ValidationError untouched.
    valid_config_dict["data"]["interval"] = "banana"
    path = _dump(tmp_path, valid_config_dict)
    with pytest.raises(ValidationError):
        load_config(path)
