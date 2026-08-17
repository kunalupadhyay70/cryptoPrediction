"""Stage 2K — Past-only threshold selection + signal computation.

Frozen contract (restated):
  Fold 1: thresholds are chosen from fold 1's OWN internal_validation
    predictions (never outer_test). If fold 1 has too few candidate trades
    to tune reliably, fall back to deterministic config-supplied thresholds.
  Fold k >= 2: thresholds are chosen using ONLY OOF predictions from outer
    folds 1..k-1 (never the current outer_test, never later folds).
  Production/live thresholds (computed once from the FULL historical OOF
    after evaluation) are a separate, later step (Stage 2N/2O territory) —
    this module only implements the per-fold, past-only mechanism used to
    populate buy_threshold/sell_threshold/signal on the OOF table itself.

SIGNAL RULE (frozen, implemented verbatim in compute_signal):
  BUY  if prob_up   >= buy_threshold  AND prob_up   > prob_down AND prob_up   > prob_neutral
  SELL if prob_down >= sell_threshold AND prob_down > prob_up   AND prob_down > prob_neutral
  else HOLD

Baselines A (majority-class) and B (persistence) are deterministic one-hot
rule-based classifiers -- per the frozen "never threshold-tune deterministic
baselines" rule, they are NOT touched by this module. Their signal is a
direct pred_class -> {DOWN: SELL, NEUTRAL: HOLD, UP: BUY} mapping, applied
by the orchestration function below with threshold columns left NaN.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

CLASS_DOWN, CLASS_NEUTRAL, CLASS_UP = 0, 1, 2
SIGNAL_BUY, SIGNAL_SELL, SIGNAL_HOLD = "BUY", "SELL", "HOLD"

# Baselines A/B: deterministic, never threshold-tuned.
_BASELINE_MODELS_NO_THRESHOLD = ("baseline_majority_class", "baseline_persistence")


class ThresholdSelectionError(ValueError):
    pass


@dataclass(frozen=True)
class ThresholdResult:
    buy_threshold: float
    sell_threshold: float
    buy_used_fallback: bool
    sell_used_fallback: bool
    buy_n_trades: int
    sell_n_trades: int
    buy_score: float
    sell_score: float


def compute_signal(prob_down: np.ndarray, prob_neutral: np.ndarray, prob_up: np.ndarray,
                    buy_threshold: float, sell_threshold: float) -> np.ndarray:
    """Vectorized, frozen SIGNAL RULE. Returns an object array of
    "BUY"/"SELL"/"HOLD" strings, one per row."""
    prob_down = np.asarray(prob_down, dtype=float)
    prob_neutral = np.asarray(prob_neutral, dtype=float)
    prob_up = np.asarray(prob_up, dtype=float)

    buy = (prob_up >= buy_threshold) & (prob_up > prob_down) & (prob_up > prob_neutral)
    sell = (prob_down >= sell_threshold) & (prob_down > prob_up) & (prob_down > prob_neutral)
    # BUY and SELL conditions are mutually exclusive by construction (each
    # requires prob_up/prob_down to be the strict max; both can't hold at
    # once), but resolve any theoretical overlap by preferring HOLD-safety:
    # BUY takes precedence only when SELL doesn't also fire.
    out = np.full(prob_down.shape, SIGNAL_HOLD, dtype=object)
    out[buy & ~sell] = SIGNAL_BUY
    out[sell & ~buy] = SIGNAL_SELL
    return out


def _sweep_one_side(
    probs: np.ndarray,
    tradable_return: np.ndarray,
    side: str,
    threshold_sweep_min: float,
    threshold_sweep_max: float,
    threshold_sweep_step: float,
    min_trades_for_tuning: int,
    default_threshold: float,
    cost_one_way: float,
) -> ThresholdResult:
    """Sweep a single side's threshold grid, scoring each candidate by a
    Sharpe-like objective (mean net return / std net return over the
    trades it would trigger, net of a symmetric round-trip cost estimate).
    Falls back to ``default_threshold`` (with used_fallback=True and
    score=nan) if no swept threshold clears ``min_trades_for_tuning``
    trades.
    """
    if side not in ("buy", "sell"):
        raise ThresholdSelectionError(f"side must be 'buy' or 'sell', got {side!r}")
    if threshold_sweep_min >= threshold_sweep_max:
        raise ThresholdSelectionError("threshold_sweep_min must be < threshold_sweep_max")
    if threshold_sweep_step <= 0:
        raise ThresholdSelectionError("threshold_sweep_step must be > 0")

    prob_down, prob_neutral, prob_up = probs[:, 0], probs[:, 1], probs[:, 2]
    side_probs = prob_up if side == "buy" else prob_down
    other_max = np.maximum(prob_down, prob_neutral) if side == "buy" else np.maximum(prob_up, prob_neutral)

    grid = np.arange(threshold_sweep_min, threshold_sweep_max + 1e-12, threshold_sweep_step)
    round_trip_cost = 2.0 * cost_one_way

    best_thr, best_score, best_n = None, -np.inf, 0
    for thr in grid:
        triggered = (side_probs >= thr) & (side_probs > other_max)
        n_trades = int(triggered.sum())
        if n_trades < min_trades_for_tuning:
            continue
        raw_returns = tradable_return[triggered]
        directional_returns = raw_returns if side == "buy" else -raw_returns
        net_returns = directional_returns - round_trip_cost
        mean_r = float(np.mean(net_returns))
        std_r = float(np.std(net_returns, ddof=0))
        score = mean_r / std_r if std_r > 1e-12 else (np.inf if mean_r > 0 else -np.inf)
        if score > best_score:
            best_thr, best_score, best_n = float(thr), score, n_trades

    if best_thr is None:
        return ThresholdResult(
            buy_threshold=default_threshold if side == "buy" else np.nan,
            sell_threshold=default_threshold if side == "sell" else np.nan,
            buy_used_fallback=(side == "buy"), sell_used_fallback=(side == "sell"),
            buy_n_trades=0 if side == "buy" else 0, sell_n_trades=0 if side == "sell" else 0,
            buy_score=np.nan, sell_score=np.nan,
        )
    return ThresholdResult(
        buy_threshold=best_thr if side == "buy" else np.nan,
        sell_threshold=best_thr if side == "sell" else np.nan,
        buy_used_fallback=False if side == "buy" else False,
        sell_used_fallback=False if side == "sell" else False,
        buy_n_trades=best_n if side == "buy" else 0,
        sell_n_trades=best_n if side == "sell" else 0,
        buy_score=best_score if side == "buy" else np.nan,
        sell_score=best_score if side == "sell" else np.nan,
    )


def select_threshold(
    prob_down: np.ndarray,
    prob_neutral: np.ndarray,
    prob_up: np.ndarray,
    tradable_return: np.ndarray,
    threshold_sweep_min: float = 0.34,
    threshold_sweep_max: float = 0.80,
    threshold_sweep_step: float = 0.02,
    min_trades_for_tuning: int = 10,
    default_buy_threshold: float = 0.40,
    default_sell_threshold: float = 0.40,
    cost_one_way: float = 0.0,
) -> ThresholdResult:
    """Combine independent buy-side / sell-side sweeps (over the SAME
    candidate probability/return rows -- e.g. one fold's internal_validation
    predictions, or the concatenation of prior folds' OOF predictions) into
    a single ThresholdResult.
    """
    probs = np.column_stack([
        np.asarray(prob_down, dtype=float),
        np.asarray(prob_neutral, dtype=float),
        np.asarray(prob_up, dtype=float),
    ])
    tradable_return = np.asarray(tradable_return, dtype=float)
    if len(probs) != len(tradable_return):
        raise ThresholdSelectionError("probs and tradable_return must have the same length")

    # Rows with NaN tradable_return (e.g. unlabeled tail rows) cannot be
    # scored as trades; exclude them from the sweep entirely rather than
    # letting a NaN silently poison the mean/std.
    valid = ~np.isnan(tradable_return)
    probs, tradable_return = probs[valid], tradable_return[valid]

    buy_res = _sweep_one_side(
        probs, tradable_return, "buy", threshold_sweep_min, threshold_sweep_max, threshold_sweep_step,
        min_trades_for_tuning, default_buy_threshold, cost_one_way,
    )
    sell_res = _sweep_one_side(
        probs, tradable_return, "sell", threshold_sweep_min, threshold_sweep_max, threshold_sweep_step,
        min_trades_for_tuning, default_sell_threshold, cost_one_way,
    )
    return ThresholdResult(
        buy_threshold=buy_res.buy_threshold, sell_threshold=sell_res.sell_threshold,
        buy_used_fallback=buy_res.buy_used_fallback, sell_used_fallback=sell_res.sell_used_fallback,
        buy_n_trades=buy_res.buy_n_trades, sell_n_trades=sell_res.sell_n_trades,
        buy_score=buy_res.buy_score, sell_score=sell_res.sell_score,
    )


def apply_thresholds_and_signals(
    oof_df: pd.DataFrame,
    fold1_internal_val_by_model: dict,
    threshold_sweep_min: float = 0.34,
    threshold_sweep_max: float = 0.80,
    threshold_sweep_step: float = 0.02,
    min_trades_for_tuning: int = 10,
    default_buy_threshold: float = 0.40,
    default_sell_threshold: float = 0.40,
    cost_one_way: float = 0.0,
) -> pd.DataFrame:
    """Second pass over the Stage 2J OOF table: fills buy_threshold /
    sell_threshold / signal per (model, fold), strictly past-only.

    - Baselines A/B: no tuning; signal = direct pred_class mapping;
      thresholds left NaN.
    - Model in {logistic_regression, lightgbm}, fold == 1: thresholds
      selected from ``fold1_internal_val_by_model[model]`` (that model's
      fold-1 internal_validation probs/tradable_return -- never outer_test).
    - Model in {logistic_regression, lightgbm}, fold k >= 2: thresholds
      selected from this model's OWN OOF rows in folds 1..k-1 only.
    """
    out = oof_df.copy()
    tunable_models = [m for m in out["model"].unique() if m not in _BASELINE_MODELS_NO_THRESHOLD]

    # --- Baselines: direct one-hot mapping, no thresholds. ---
    baseline_mask = out["model"].isin(_BASELINE_MODELS_NO_THRESHOLD)
    if baseline_mask.any():
        pred = out.loc[baseline_mask, "pred_class"].to_numpy()
        sig = np.full(pred.shape, SIGNAL_HOLD, dtype=object)
        sig[pred == CLASS_UP] = SIGNAL_BUY
        sig[pred == CLASS_DOWN] = SIGNAL_SELL
        out.loc[baseline_mask, "signal"] = sig
        # buy_threshold/sell_threshold remain NaN for baselines (already NaN
        # from Stage 2J; nothing to set).

    for model in tunable_models:
        model_mask = out["model"] == model
        folds = sorted(out.loc[model_mask, "fold"].unique())
        for fold in folds:
            fold_mask = model_mask & (out["fold"] == fold)
            if fold == 1:
                src = fold1_internal_val_by_model.get(model)
                if src is None:
                    raise ThresholdSelectionError(
                        f"fold1_internal_val_by_model missing entry for tunable model {model!r}"
                    )
                probs, tret = src["probs"], src["tradable_return"]
            else:
                prior_mask = model_mask & (out["fold"] < fold)
                prior = out.loc[prior_mask]
                if prior.empty:
                    raise ThresholdSelectionError(
                        f"model {model!r} fold {fold} has no prior-fold OOF rows to select thresholds from"
                    )
                probs = prior[["prob_down", "prob_neutral", "prob_up"]].to_numpy()
                tret = prior["tradable_return"].to_numpy()

            result = select_threshold(
                probs[:, 0], probs[:, 1], probs[:, 2], tret,
                threshold_sweep_min=threshold_sweep_min, threshold_sweep_max=threshold_sweep_max,
                threshold_sweep_step=threshold_sweep_step, min_trades_for_tuning=min_trades_for_tuning,
                default_buy_threshold=default_buy_threshold, default_sell_threshold=default_sell_threshold,
                cost_one_way=cost_one_way,
            )
            out.loc[fold_mask, "buy_threshold"] = result.buy_threshold
            out.loc[fold_mask, "sell_threshold"] = result.sell_threshold
            rows = out.loc[fold_mask]
            sig = compute_signal(
                rows["prob_down"].to_numpy(), rows["prob_neutral"].to_numpy(), rows["prob_up"].to_numpy(),
                result.buy_threshold, result.sell_threshold,
            )
            out.loc[fold_mask, "signal"] = sig

    return out
