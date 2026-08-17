"""Canonical typed configuration for the pipeline (Stage 2B).

This module is the SINGLE source of truth for what a valid ``config.yaml``
looks like. It replaces the scattered pattern of ``cfg["section"].get(key,
hardcoded_default)`` calls in ``main.py`` and the dataclasses-with-duplicated-
alias-fields pattern found in ``data_collector.py`` / ``feature_engineering.py``
/ ``model.py`` (Stage 0 audit).

Design rules (frozen by the Stage 1 architecture review):
  - Every configuration model is STRICT: unknown keys are rejected outright
    (``extra="forbid"``), not silently ignored. This is the direct fix for
    the Stage 0 finding that ``config.yaml``'s ``lgbm_*`` hyperparameters and
    several other keys were present but never consumed by any runtime
    object — with strict models, an unwired key becomes a hard failure at
    load time instead of a silent no-op.
  - Every concept has exactly ONE canonical field name. No dual-name aliasing
    (e.g. the old ``lgbm_num_leaves`` vs ``num_leaves`` / ``primary`` vs
    ``model_type`` pattern) is reproduced here.
  - Interval validation is NEVER reimplemented here. ``DataConfig.interval``
    is validated by delegating to ``intervals.interval_to_seconds`` — the
    Stage 2A canonical interval authority — so there is exactly one place in
    the whole repository that knows which interval strings are valid.

MIGRATION SEQUENCE (read this before adding a field): ``config_schema.py``
currently defines the TARGET canonical configuration contract for the
architecture frozen in Stage 1/1B/1C/1D. Runtime modules (``main.py``,
``model.py``, ``feature_engineering.py``, ``backtester.py``,
``signal_engine.py``) are intentionally NOT migrated to consume it yet,
because their legacy target/model/backtest semantics (global-percentile
vol_breakout labels, static regime-threshold signal config, model-type
switching, etc.) are still being replaced by later implementation stages.
This module must represent the FINAL architecture, not a wrapper that also
has to accommodate today's legacy runtime — do not add a canonical field
just to make the current application "fit" (see the explicit non-goals
list below). The real ``config.yaml`` cutover happens only once the
runtime consumers actually match this schema; until then this schema lives
beside the current runtime, unused by it. A canonical example that DOES
validate against this schema lives at ``tests/fixtures/config_valid.yaml``.

Deliberately NOT represented here (retired/reworked by later stages, not
carried forward as compatibility fields): ``target.label_mode``,
``target.vol_breakout_percentile``, ``target.deadzone_threshold``,
``signal.low_vol_buy``/``high_vol_buy`` (and the rest of the legacy
``signal:`` section), ``model.use_regime_models``, ``model.compare_stacking``,
``backtest.bar_size_minutes``, ``backtest.label_horizon_bars``. No
``to_legacy_dict()`` or other compatibility bridge exists or is planned —
this is a clean-cutover schema, not an adapter.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from intervals import InvalidIntervalError, interval_to_seconds


class ConfigError(Exception):
    """Raised for structural configuration problems that are not a Pydantic
    field-validation error — e.g. the YAML file's top level isn't a mapping.

    Field-level problems (missing required key, wrong type, out-of-range
    value, unknown key) are surfaced directly as ``pydantic.ValidationError``
    and are NEVER caught/replaced here — see ``load_config`` below.
    """


class _StrictModel(BaseModel):
    """Base class for every configuration section: reject unknown keys."""

    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# data:
# ---------------------------------------------------------------------------

class DataConfig(_StrictModel):
    exchange: str
    symbol: str
    interval: str
    target_days: int = Field(gt=0)
    db_path: str
    rest_base_url: str
    ws_base_url: str
    depth_limit: int = Field(gt=0)
    trades_limit: int = Field(gt=0)
    kline_limit: int = Field(gt=0)
    pagination_sleep_seconds: float = Field(ge=0)
    incremental: bool = True
    integrity_check: bool = True

    @model_validator(mode="after")
    def _validate_interval(self) -> "DataConfig":
        # Delegates to the Stage 2A canonical interval authority instead of
        # re-declaring a list of valid intervals here.
        try:
            interval_to_seconds(self.interval)
        except InvalidIntervalError as exc:
            raise ValueError(str(exc)) from exc
        return self


# ---------------------------------------------------------------------------
# target:
# ---------------------------------------------------------------------------

class TargetConfig(_StrictModel):
    """Configuration for the causal, ATR-scaled 3-class target frozen in
    Stage 1B/1C. This is NOT the legacy dead_zone/vol_breakout label config
    (see EXISTING CONFIG KEYS NOT YET MIGRATED in the Stage 2B report) — the
    new target logic itself is not implemented until a later stage; this
    model only validates its configuration shape.
    """

    horizon_bars: int = Field(gt=0)
    neutral_atr_mult: float = Field(gt=0)
    atr_period: int = Field(gt=1)


# ---------------------------------------------------------------------------
# features:
# ---------------------------------------------------------------------------

class FeatureConfig(_StrictModel):
    lag_periods: int = Field(gt=0, default=3)
    correlation_threshold: float = Field(gt=0, le=1)
    importance_top_k: int = Field(gt=0)
    min_fold_stability: float = Field(gt=0, le=1)
    # V1 default per the frozen architecture: order-book/trade-flow features
    # are disabled for historical training/backtesting until a real
    # historical microstructure backfill exists (Stage 0 finding — most rows
    # were zero placeholders). The historical order-book collection code
    # itself is NOT removed in this stage.
    include_orderbook_features: bool = False


# ---------------------------------------------------------------------------
# validation:
# ---------------------------------------------------------------------------

class ValidationConfig(_StrictModel):
    """NOTE — purge size is intentionally NOT a field here.

    The frozen architecture (Stage 1C §4/§6) requires purge_bars ==
    target.horizon_bars EXACTLY — allowing an independent
    ``validation.purge_bars`` would create two sources of truth for the
    same leakage-critical quantity, and a user could set them
    inconsistently (e.g. horizon_bars=3, purge_bars=1) and silently violate
    the no-leakage guarantee the walk-forward split exists to enforce.

    There is deliberately no field for it: the future walk-forward
    implementation must read the purge size directly from
    ``config.target.horizon_bars``, never from a separate validation-section
    value. If a config.yaml author writes ``validation.purge_bars: ...``,
    strict unknown-key rejection (extra="forbid") makes that a hard failure
    rather than a silently-ignored or silently-inconsistent setting.

    embargo_bars, by contrast, IS an independent field: Stage 1D fixed its
    default to 0 but explicitly kept it configurable, since (unlike purge)
    embargo is not derived from any other locked quantity.
    """

    n_folds: int = Field(ge=2)
    min_train_rows: int = Field(gt=0)
    min_test_rows: int = Field(gt=0)
    embargo_bars: int = Field(ge=0, default=0)
    internal_validation_fraction: float = Field(gt=0, lt=1, default=0.2)
    min_class_count: int = Field(gt=0, default=30)


# ---------------------------------------------------------------------------
# models:
# ---------------------------------------------------------------------------

class LogisticRegressionConfig(_StrictModel):
    C: float = Field(gt=0)
    max_iter: int = Field(gt=0)


class LightGBMConfig(_StrictModel):
    """LightGBM hyperparameters.

    Deliberately REQUIRED (no defaults) for every field below — this is the
    direct fix for the Stage 0 bug where config.yaml's lgbm_* values were
    present but silently ignored because main.py never read them into
    ModelConfig, which then silently used its own hardcoded defaults
    instead. With no defaults here, an incomplete lightgbm: block in
    config.yaml fails loudly at load time rather than silently falling back
    to values the operator never chose.
    """

    num_leaves: int = Field(gt=0)
    learning_rate: float = Field(gt=0)
    n_estimators: int = Field(gt=0)
    min_child_samples: int = Field(gt=0)
    subsample: float = Field(gt=0, le=1)
    colsample_bytree: float = Field(gt=0, le=1)
    reg_alpha: float = Field(ge=0)
    reg_lambda: float = Field(ge=0)
    early_stopping_rounds: int = Field(gt=0)


class CatBoostConfig(_StrictModel):
    """Optional secondary robustness-check model (Stage 1B §8). Disabled by
    default — only enable once the primary LightGBM pipeline is validated.
    """

    enabled: bool = False
    iterations: int = Field(gt=0, default=1000)
    learning_rate: float = Field(gt=0, default=0.02)
    depth: int = Field(gt=0, default=6)


class ModelsConfig(_StrictModel):
    random_state: int = 42
    logistic_regression: LogisticRegressionConfig
    lightgbm: LightGBMConfig
    catboost: CatBoostConfig = CatBoostConfig()


# ---------------------------------------------------------------------------
# backtest:
# ---------------------------------------------------------------------------

class BacktestConfig(_StrictModel):
    fee_bps: float = Field(ge=0)
    slippage_bps: float = Field(ge=0)
    latency_bps: float = Field(ge=0)
    funding_bps_per_bar: float = Field(ge=0)

    # Deterministic fallback thresholds used when a fold's threshold-tuning
    # pool doesn't clear min_trades_for_tuning (Stage 1D FIX 5).
    default_buy_threshold: float = Field(gt=0, lt=1)
    default_sell_threshold: float = Field(gt=0, lt=1)
    min_trades_for_tuning: int = Field(gt=0)

    threshold_sweep_min: float = Field(gt=0, lt=1)
    threshold_sweep_max: float = Field(gt=0, lt=1)
    threshold_sweep_step: float = Field(gt=0, lt=1)

    # Fixed-notional accounting engine (Stage 1D).
    initial_equity: float = Field(gt=0, default=1.0)
    position_fraction: float = Field(gt=0, le=1, default=1.0)

    max_trades_per_day: float = Field(gt=0)

    @model_validator(mode="after")
    def _validate_threshold_ordering(self) -> "BacktestConfig":
        if self.threshold_sweep_min >= self.threshold_sweep_max:
            raise ValueError(
                "threshold_sweep_min must be < threshold_sweep_max "
                f"(got min={self.threshold_sweep_min}, max={self.threshold_sweep_max})"
            )
        if self.default_sell_threshold >= self.default_buy_threshold:
            raise ValueError(
                "default_sell_threshold must be < default_buy_threshold "
                f"(got sell={self.default_sell_threshold}, buy={self.default_buy_threshold})"
            )
        return self


# ---------------------------------------------------------------------------
# risk: / artifacts: / experiment:
# ---------------------------------------------------------------------------

class RiskConfig(_StrictModel):
    """Reserved for Stage 9 (position sizing, stop-loss, drawdown controls).
    Kept intentionally empty beyond the enabled flag for V1.
    """

    enabled: bool = False


class ArtifactConfig(_StrictModel):
    root_dir: str = "artifacts"
    keep_last_n_experiments: int = Field(gt=0, default=20)


class ExperimentConfig(_StrictModel):
    name: str = "default"
    tags: List[str] = Field(default_factory=list)
    notes: str = ""


# ---------------------------------------------------------------------------
# live:
# ---------------------------------------------------------------------------

class LiveConfig(_StrictModel):
    """Production inference loop settings (main.py's `live` mode).

    Kept as its OWN section rather than folded into `experiment:` — these
    values control runtime inference behavior (how often to poll the
    exchange, how often to emit a signal, how long the loop may run), not
    experiment/run metadata. The frozen architecture retains a production
    inference path (Stage 1 §2's "Production Inference Pipeline"), so this
    section has a real, permanent home rather than being a legacy leftover.
    """

    poll_seconds: float = Field(gt=0)
    emit_every_iterations: int = Field(gt=0)
    # None means "run until externally stopped" (unlimited live operation is
    # intentionally supported); if given, it must be a positive bar count.
    max_iterations: Optional[int] = Field(default=None, gt=0)


# ---------------------------------------------------------------------------
# Top-level AppConfig
# ---------------------------------------------------------------------------

class AppConfig(_StrictModel):
    data: DataConfig
    target: TargetConfig
    features: FeatureConfig
    validation: ValidationConfig
    models: ModelsConfig
    backtest: BacktestConfig
    risk: RiskConfig = RiskConfig()
    artifacts: ArtifactConfig = ArtifactConfig()
    experiment: ExperimentConfig = ExperimentConfig()
    live: LiveConfig


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_config(path: Union[str, Path]) -> AppConfig:
    """Read, parse, and validate a config.yaml into a typed AppConfig.

    Failure modes are all loud and untouched:
      - missing file            -> FileNotFoundError (from open())
      - malformed YAML syntax   -> yaml.YAMLError (from yaml.safe_load())
      - top level isn't a mapping -> ConfigError (raised here, explicitly)
      - any field/type/range/unknown-key problem -> pydantic.ValidationError

    Nothing here catches a Pydantic ValidationError and substitutes a
    default — an invalid or incomplete configuration always fails the load.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ConfigError(
            f"Top-level content of {path} must be a YAML mapping (dict), "
            f"got {type(raw).__name__}: {raw!r}"
        )

    return AppConfig.model_validate(raw)
