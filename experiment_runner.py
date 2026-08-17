"""Stage 2O — Experiment artifacts.

Orchestrates Stages 2D-2N end-to-end (OOF generation -> past-only
thresholds/signals -> predictive metrics -> per-model backtests -> trading
metrics -> passive-long benchmark -> production model) and persists every
artifact needed to reproduce and audit the experiment: config snapshot,
metrics.json, fold_metrics.csv, oof_predictions.csv, feature_importance.csv,
selected_features.json, trade_ledger.csv, strategy_timeline.csv, plots, and
the serialized production model.

This module contains NO new modeling/accounting logic of its own -- every
number it writes comes from a single call into the already-tested Stage
2D-2N modules, so this stage's job is wiring and persistence, not
computation.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from metrics import compute_passive_long_benchmark, compute_predictive_metrics, compute_trading_metrics
from oof_builder import ALL_MODELS, MODEL_LIGHTGBM, OOFPipelineConfig, generate_oof_predictions
from position_engine import run_backtest
from production_model import ProductionModelResult, train_production_model
from threshold_selection import apply_thresholds_and_signals


class ExperimentError(ValueError):
    pass


@dataclass
class ExperimentArtifacts:
    output_dir: Path
    oof_df: pd.DataFrame
    fold_metrics_df: pd.DataFrame
    predictive_metrics_by_model: Dict[str, object]
    trading_metrics_by_model: Dict[str, object]
    benchmark_trading_metrics: object
    production_model: ProductionModelResult
    timelines_by_model: Dict[str, pd.DataFrame]
    ledgers_by_model: Dict[str, pd.DataFrame]


def _bars_from_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    if "open_time" not in dataset.columns:
        raise ExperimentError("dataset is missing open_time; cannot build bars for backtesting")
    return dataset.rename(columns={"open_time": "timestamp"})[["timestamp", "open", "close"]]


def run_experiment(
    dataset: pd.DataFrame,
    feature_candidate_cols: List[str],
    symbol: str,
    interval: str,
    run_id: str,
    oof_config: OOFPipelineConfig,
    output_dir: str,
    min_fold_stability: float = 0.6,
    threshold_kwargs: Optional[dict] = None,
    backtest_kwargs: Optional[dict] = None,
    production_lgbm_params: Optional[dict] = None,
    write_plots: bool = True,
) -> ExperimentArtifacts:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # --- Stage 2J/2K: OOF predictions + past-only thresholds/signals. ---
    oof_df, fold_infos = generate_oof_predictions(
        dataset, feature_candidate_cols, symbol, interval, run_id, oof_config,
    )
    fold1 = next((fi for fi in fold_infos if fi.fold == 1), None)
    if fold1 is None or fold1.fold1_internal_val is None:
        raise ExperimentError("fold 1 did not produce fold1_internal_val; cannot select thresholds")
    threshold_kwargs = dict(threshold_kwargs or {})
    oof_df = apply_thresholds_and_signals(oof_df, fold1.fold1_internal_val, **threshold_kwargs)

    # --- Stage 2M: predictive metrics, per model (OOF table only). ---
    predictive_metrics_by_model = {}
    for model in ALL_MODELS:
        sub = oof_df[oof_df["model"] == model]
        predictive_metrics_by_model[model] = compute_predictive_metrics(
            sub["actual_class"].to_numpy(), sub["pred_class"].to_numpy(),
            sub["prob_down"].to_numpy(), sub["prob_neutral"].to_numpy(), sub["prob_up"].to_numpy(),
        )

    # --- Stage 2L/2M: per-model backtest + trading metrics. ---
    bars = _bars_from_dataset(dataset)
    backtest_kwargs = dict(backtest_kwargs or {})
    horizon_bars = oof_config.horizon_bars
    timelines_by_model, ledgers_by_model, trading_metrics_by_model = {}, {}, {}
    for model in ALL_MODELS:
        sub = oof_df[oof_df["model"] == model][["timestamp", "fold", "signal"]]
        timeline, ledger = run_backtest(bars, sub, model=model, horizon_bars=horizon_bars, **backtest_kwargs)
        n_actionable = int(sub["signal"].isin(["BUY", "SELL"]).sum())
        timelines_by_model[model] = timeline
        ledgers_by_model[model] = ledger
        trading_metrics_by_model[model] = compute_trading_metrics(timeline, ledger, n_actionable)

    # --- Passive-long benchmark over the EXACT same OOF evaluation window. ---
    window_start, window_end = oof_df["timestamp"].min(), oof_df["timestamp"].max()
    window_bars = bars[(bars["timestamp"] >= window_start) & (bars["timestamp"] <= window_end)].reset_index(drop=True)
    # compute_passive_long_benchmark always uses full notional (a passive
    # long holds the whole position, unlike the strategy's tunable
    # position_fraction) -- pass through only the cost/funding/equity
    # kwargs it actually accepts.
    _benchmark_kwargs = {
        k: v for k, v in backtest_kwargs.items()
        if k in ("initial_equity", "fee_bps", "slippage_bps", "latency_bps", "funding_bps_per_bar")
    }
    _, _, benchmark_trading_metrics = compute_passive_long_benchmark(window_bars, **_benchmark_kwargs)

    # --- Stage 2N: production model (future inference only). ---
    fold_selected_features = [fi.selected_features for fi in fold_infos]
    fold_best_iterations = [fi.best_iteration for fi in fold_infos]
    production = train_production_model(
        dataset, feature_candidate_cols, fold_selected_features, fold_best_iterations,
        min_fold_stability=min_fold_stability, min_class_count=oof_config.min_class_count,
        lgbm_params=production_lgbm_params, random_state=oof_config.random_state,
    )

    fold_metrics_df = pd.DataFrame([
        {
            "fold": fi.fold, "n_outer_train": fi.n_outer_train, "n_outer_test": fi.n_outer_test,
            "n_internal_train": fi.n_internal_train, "n_internal_val": fi.n_internal_val,
            "n_correlation_pruned_features": len(fi.correlation_pruned_features),
            "n_selected_features": len(fi.selected_features), "best_iteration": fi.best_iteration,
        }
        for fi in fold_infos
    ])

    _write_artifacts(
        out, oof_df, fold_metrics_df, predictive_metrics_by_model, trading_metrics_by_model,
        benchmark_trading_metrics, production, timelines_by_model, ledgers_by_model,
        oof_config, symbol, interval, run_id, write_plots,
    )

    return ExperimentArtifacts(
        output_dir=out, oof_df=oof_df, fold_metrics_df=fold_metrics_df,
        predictive_metrics_by_model=predictive_metrics_by_model,
        trading_metrics_by_model=trading_metrics_by_model,
        benchmark_trading_metrics=benchmark_trading_metrics, production_model=production,
        timelines_by_model=timelines_by_model, ledgers_by_model=ledgers_by_model,
    )


# ---------------------------------------------------------------------------
# Artifact persistence
# ---------------------------------------------------------------------------

def _jsonify(x):
    if isinstance(x, dict):
        return {str(k): _jsonify(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonify(v) for v in x]
    if isinstance(x, np.ndarray):
        return _jsonify(x.tolist())
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        v = float(x)
        return None if np.isnan(v) else v
    if isinstance(x, float) and np.isnan(x):
        return None
    if isinstance(x, Path):
        return str(x)
    return x


def _dataclass_to_jsonable(obj):
    if not is_dataclass(obj):
        raise ExperimentError(f"expected a dataclass metrics object, got {type(obj)}")
    return _jsonify(asdict(obj))


def _write_artifacts(
    out: Path, oof_df, fold_metrics_df, predictive_metrics_by_model, trading_metrics_by_model,
    benchmark_trading_metrics, production, timelines_by_model, ledgers_by_model,
    oof_config, symbol, interval, run_id, write_plots,
):
    config_snapshot = {
        "symbol": symbol, "interval": interval, "run_id": run_id,
        "oof_config": _jsonify(asdict(oof_config)),
    }
    (out / "config_snapshot.json").write_text(json.dumps(config_snapshot, indent=2))

    metrics_payload = {
        "predictive_metrics_by_model": {
            m: _dataclass_to_jsonable(pm) for m, pm in predictive_metrics_by_model.items()
        },
        "trading_metrics_by_model": {
            m: _dataclass_to_jsonable(tm) for m, tm in trading_metrics_by_model.items()
        },
        "benchmark_trading_metrics": _dataclass_to_jsonable(benchmark_trading_metrics),
    }
    (out / "metrics.json").write_text(json.dumps(metrics_payload, indent=2))

    fold_metrics_df.to_csv(out / "fold_metrics.csv", index=False)
    oof_df.to_csv(out / "oof_predictions.csv", index=False)

    importances = getattr(production.model, "feature_importances_", None)
    if importances is not None:
        pd.DataFrame({"feature": production.selected_features, "importance": importances}).to_csv(
            out / "feature_importance.csv", index=False
        )
    (out / "selected_features.json").write_text(json.dumps(production.selected_features, indent=2))

    ledger_frames = [ledger.copy() for ledger in ledgers_by_model.values() if not ledger.empty]
    combined_ledger = pd.concat(ledger_frames, ignore_index=True) if ledger_frames else pd.DataFrame()
    combined_ledger.to_csv(out / "trade_ledger.csv", index=False)

    timeline_frames = []
    for model, timeline in timelines_by_model.items():
        tf = timeline.copy()
        tf.insert(1, "model", model)
        timeline_frames.append(tf)
    combined_timeline = pd.concat(timeline_frames, ignore_index=True) if timeline_frames else pd.DataFrame()
    combined_timeline.to_csv(out / "strategy_timeline.csv", index=False)

    import joblib
    joblib.dump(production.model, out / "production_model.joblib")

    if write_plots:
        _write_plots(out, timelines_by_model, predictive_metrics_by_model)


def _write_plots(out: Path, timelines_by_model, predictive_metrics_by_model):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    for model, timeline in timelines_by_model.items():
        if timeline.empty:
            continue
        ax.plot(pd.to_datetime(timeline["timestamp"]), timeline["equity_after"], label=model)
    ax.set_title("Equity Curve by Model")
    ax.set_xlabel("Time")
    ax.set_ylabel("Equity")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "equity_curve.png")
    plt.close(fig)

    if MODEL_LIGHTGBM in predictive_metrics_by_model:
        cm = predictive_metrics_by_model[MODEL_LIGHTGBM].confusion_matrix
        fig2, ax2 = plt.subplots(figsize=(4, 4))
        im = ax2.imshow(cm, cmap="Blues")
        ax2.set_title("LightGBM Confusion Matrix (OOF)")
        ax2.set_xlabel("Predicted")
        ax2.set_ylabel("Actual")
        ax2.set_xticks([0, 1, 2])
        ax2.set_yticks([0, 1, 2])
        ax2.set_xticklabels(["DOWN", "NEUTRAL", "UP"])
        ax2.set_yticklabels(["DOWN", "NEUTRAL", "UP"])
        for (i, j), v in np.ndenumerate(cm):
            ax2.text(j, i, str(v), ha="center", va="center")
        fig2.colorbar(im, ax=ax2)
        fig2.tight_layout()
        fig2.savefig(out / "confusion_matrix_lightgbm.png")
        plt.close(fig2)
