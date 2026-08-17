"""Stage 2J — Canonical OOF (out-of-fold) prediction generation.

Orchestrates walk_forward.py (Stage 2G) + feature_selection.py (Stage 2H) +
model_training.py (Stage 2I) into the canonical LONG-format OOF table. This
is the ONLY place historical model/backtest performance is allowed to be
computed from — see the frozen "OOF PREDICTIONS" rule: the production
model must never predict its own historical training rows for reported
metrics, and every row in the returned table is, by construction, a
prediction on a row that fold's model never trained on.

Per-fold procedure (all fold-local; outer_test is NEVER touched before its
own model is frozen):
  1. Take the fold's outer_train / outer_test row positions from
     walk_forward.generate_walk_forward_splits (purge + embargo already
     applied to outer_train).
  2. Drop unlabeled rows (ATR warm-up / tail rows with no target_class)
     from BOTH outer_train and outer_test — an unlabeled test row has no
     ground truth to score against, so it is not an "eligible" OOF row.
  3. Chronological 80/20 split of the (labeled) outer_train into
     internal_train / internal_validation.
  4. Correlation-prune candidate features on internal_train only, then
     select features by LightGBM importance (internal_train to fit,
     internal_validation only for early stopping) -> selected_features +
     best_iteration (Stage 2H).
  5. Refit LightGBM on ALL of outer_train with n_estimators=best_iteration
     (fixed, not re-tuned against outer_test) and predict outer_test.
     Baseline C (Logistic Regression) is fit the same way, on the same
     selected_features, for a fair comparison. Baselines A/B are
     deterministic rule-based baselines (Stage 2I) needing no fitting
     beyond a majority-vote count.
  6. Every model's predictions for this fold are appended as LONG-format
     rows to the OOF table.

buy_threshold / sell_threshold / signal are left NaN/None by this module —
they are computed in a strictly past-only way by Stage 2K (threshold
selection), which needs completed OOF probabilities from prior folds
before it can compute fold k's thresholds. Populating them here, before
enough prior-fold OOF exists, would be premature (and for fold 1
specifically requires internal-validation predictions this module does not
retain). Stage 2K fills these columns in as a second pass over this
table.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from class_checks import check_class_coverage
from feature_selection import prune_correlated_features, select_features_by_lgbm_importance
from model_training import (
    CLASS_ORDER, fit_lightgbm_multiclass_probs, fit_logistic_regression_probs,
    majority_class_baseline_probs,
)
from walk_forward import generate_walk_forward_splits

MODEL_BASELINE_A = "baseline_majority_class"
MODEL_BASELINE_B = "baseline_persistence"
MODEL_BASELINE_C = "logistic_regression"
MODEL_LIGHTGBM = "lightgbm"
ALL_MODELS = (MODEL_BASELINE_A, MODEL_BASELINE_B, MODEL_BASELINE_C, MODEL_LIGHTGBM)

OOF_COLUMNS = [
    "timestamp", "symbol", "interval", "fold", "model",
    "actual_class", "prob_down", "prob_neutral", "prob_up", "pred_class",
    "buy_threshold", "sell_threshold", "signal",
    "close_t", "next_open", "exit_close", "tradable_return",
    "atr", "volatility_band", "vol_regime", "run_id",
]


class OOFError(ValueError):
    pass


@dataclass
class OOFPipelineConfig:
    n_folds: int
    min_train_rows: int
    min_test_rows: int
    horizon_bars: int
    embargo_bars: int
    min_class_count: int
    correlation_threshold: float
    importance_top_k: int
    internal_validation_fraction: float
    lgbm_early_stopping_rounds: int = 50
    random_state: int = 42
    lgbm_params: dict = field(default_factory=dict)


@dataclass
class FoldInfo:
    fold: int
    n_outer_train: int
    n_outer_test: int
    n_internal_train: int
    n_internal_val: int
    correlation_pruned_features: List[str]
    selected_features: List[str]
    best_iteration: int
    # Populated ONLY for fold == 1 (None for every later fold): the raw
    # ingredients Stage 2K's threshold selection needs for its "fold 1: use
    # fold 1 internal validation predictions" rule. Keyed by the two
    # threshold-tunable model names (lightgbm, logistic_regression); each
    # value is {"probs": (n_val, 3) canonical-order array, "tradable_return":
    # (n_val,) array}. Later folds don't need this — Stage 2K sources their
    # threshold data directly from already-completed prior-fold OOF rows in
    # the returned oof_df.
    fold1_internal_val: dict = None


def _chronological_internal_split(
    labeled_train_idx: np.ndarray, internal_validation_fraction: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Split a SORTED (ascending, i.e. chronological) row-position array
    into a leading internal_train portion and a trailing internal_validation
    portion. Both portions are guaranteed non-empty."""
    n = len(labeled_train_idx)
    split_point = int(round(n * (1 - internal_validation_fraction)))
    split_point = max(1, min(n - 1, split_point))
    return labeled_train_idx[:split_point], labeled_train_idx[split_point:]


def _persistence_probs_for_rows(close: pd.Series, row_positions: np.ndarray) -> np.ndarray:
    """Baseline B scored per-row (not assuming contiguity — the final
    fold's test block can have trailing unlabeled rows removed, so
    row_positions may skip some positions)."""
    proba = np.zeros((len(row_positions), 3))
    for i, t in enumerate(row_positions):
        if t < 1:
            raise OOFError(f"persistence baseline needs a t-1 row; got row position {t}")
        delta = close.iloc[int(t)] - close.iloc[int(t) - 1]
        cls = 2 if delta > 0 else (0 if delta < 0 else 1)
        proba[i, CLASS_ORDER.index(cls)] = 1.0
    return proba


def generate_oof_predictions(
    dataset: pd.DataFrame,
    feature_candidate_cols: List[str],
    symbol: str,
    interval: str,
    run_id: str,
    config: OOFPipelineConfig,
) -> Tuple[pd.DataFrame, List[FoldInfo]]:
    """Build the canonical LONG-format OOF table for all 4 models across
    all outer folds. Returns (oof_df, fold_infos) — fold_infos carries the
    per-fold selected_features/best_iteration needed by Stage 2N's
    production-model retraining (median best_iteration, stability-selected
    features)."""
    required_cols = {"close", "open", "target_class", "entry_price", "exit_price", "atr", "target_band"}
    missing = required_cols - set(dataset.columns)
    if missing:
        raise OOFError(f"dataset is missing required columns: {sorted(missing)}")

    n_rows = len(dataset)
    splits = generate_walk_forward_splits(
        n_rows=n_rows,
        n_folds=config.n_folds,
        min_train_rows=config.min_train_rows,
        min_test_rows=config.min_test_rows,
        horizon_bars=config.horizon_bars,
        embargo_bars=config.embargo_bars,
    )

    labeled_mask = dataset["target_class"].notna().to_numpy()
    has_open_time = "open_time" in dataset.columns
    has_vol_regime = "vol_regime" in dataset.columns

    X_all = dataset[feature_candidate_cols]
    y_all_float = dataset["target_class"]

    all_rows: List[dict] = []
    fold_infos: List[FoldInfo] = []

    for split in splits:
        outer_train_idx = split.train_idx[labeled_mask[split.train_idx]]
        outer_test_idx = split.test_idx[labeled_mask[split.test_idx]]

        if len(outer_train_idx) == 0 or len(outer_test_idx) == 0:
            raise OOFError(
                f"Fold {split.fold}: no labeled (eligible) rows in outer_train or "
                f"outer_test after excluding ATR warm-up / tail-unlabeled rows"
            )

        internal_train_idx, internal_val_idx = _chronological_internal_split(
            outer_train_idx, config.internal_validation_fraction
        )

        y_internal_train = y_all_float.iloc[internal_train_idx].astype(int).to_numpy()
        y_internal_val = y_all_float.iloc[internal_val_idx].astype(int).to_numpy()

        pruned = prune_correlated_features(
            dataset.iloc[internal_train_idx], feature_candidate_cols, config.correlation_threshold
        )
        importance_result = select_features_by_lgbm_importance(
            X_all.iloc[internal_train_idx][pruned], y_internal_train,
            X_all.iloc[internal_val_idx][pruned], y_internal_val,
            candidate_features=pruned,
            top_k=config.importance_top_k,
            min_class_count=config.min_class_count,
            random_state=config.random_state,
            early_stopping_rounds=config.lgbm_early_stopping_rounds,
            lgbm_params=config.lgbm_params,
        )
        selected_features = importance_result.selected_features
        best_iteration = importance_result.best_iteration

        fold1_internal_val = None
        if split.fold == 1:
            # Stage 2K needs fold 1's threshold selection sourced from
            # fold 1's OWN internal_validation predictions (never from
            # outer_test). LightGBM's come free from the already-fitted
            # importance-selection model above; Logistic Regression needs
            # one extra (cheap) fit on internal_train, scored on
            # internal_validation — mirroring exactly the same
            # internal_train -> internal_validation split used for
            # LightGBM, on the same selected_features.
            logreg_internal_val_probs = fit_logistic_regression_probs(
                X_all.iloc[internal_train_idx][selected_features], y_internal_train,
                X_all.iloc[internal_val_idx][selected_features],
                min_class_count=config.min_class_count, random_state=config.random_state,
            )
            internal_val_tradable_return = (
                dataset["tradable_return"].iloc[internal_val_idx].to_numpy()
            )
            fold1_internal_val = {
                MODEL_LIGHTGBM: {
                    "probs": importance_result.internal_val_probs,
                    "tradable_return": internal_val_tradable_return,
                },
                MODEL_BASELINE_C: {
                    "probs": logreg_internal_val_probs,
                    "tradable_return": internal_val_tradable_return,
                },
            }

        y_outer_train = y_all_float.iloc[outer_train_idx].astype(int).to_numpy()
        X_outer_train = X_all.iloc[outer_train_idx][selected_features]
        X_outer_test = X_all.iloc[outer_test_idx][selected_features]

        model_probs: Dict[str, np.ndarray] = {
            MODEL_BASELINE_A: majority_class_baseline_probs(y_outer_train, len(outer_test_idx)),
            MODEL_BASELINE_B: _persistence_probs_for_rows(dataset["close"], outer_test_idx),
            MODEL_BASELINE_C: fit_logistic_regression_probs(
                X_outer_train, y_outer_train, X_outer_test,
                min_class_count=config.min_class_count, random_state=config.random_state,
            ),
            MODEL_LIGHTGBM: fit_lightgbm_multiclass_probs(
                X_outer_train, y_outer_train, X_outer_test,
                min_class_count=config.min_class_count, n_estimators=best_iteration,
                lgbm_params=config.lgbm_params, random_state=config.random_state,
            ),
        }

        for model_name, proba in model_probs.items():
            for i, t in enumerate(outer_test_idx):
                row = dataset.iloc[int(t)]
                p_down, p_neutral, p_up = proba[i]
                all_rows.append({
                    "timestamp": row["open_time"] if has_open_time else int(t),
                    "symbol": symbol,
                    "interval": interval,
                    "fold": split.fold,
                    "model": model_name,
                    "actual_class": int(row["target_class"]),
                    "prob_down": float(p_down),
                    "prob_neutral": float(p_neutral),
                    "prob_up": float(p_up),
                    "pred_class": int(np.argmax(proba[i])),
                    "buy_threshold": np.nan,
                    "sell_threshold": np.nan,
                    "signal": None,
                    "close_t": float(row["close"]),
                    "next_open": float(row["entry_price"]) if pd.notna(row["entry_price"]) else np.nan,
                    "exit_close": float(row["exit_price"]) if pd.notna(row["exit_price"]) else np.nan,
                    "tradable_return": float(row["tradable_return"]) if pd.notna(row["tradable_return"]) else np.nan,
                    "atr": float(row["atr"]) if pd.notna(row["atr"]) else np.nan,
                    "volatility_band": float(row["target_band"]) if pd.notna(row["target_band"]) else np.nan,
                    "vol_regime": (float(row["vol_regime"]) if has_vol_regime and pd.notna(row["vol_regime"]) else np.nan),
                    "run_id": run_id,
                })

        fold_infos.append(FoldInfo(
            fold=split.fold,
            n_outer_train=len(outer_train_idx),
            n_outer_test=len(outer_test_idx),
            n_internal_train=len(internal_train_idx),
            n_internal_val=len(internal_val_idx),
            correlation_pruned_features=pruned,
            selected_features=selected_features,
            best_iteration=best_iteration,
            fold1_internal_val=fold1_internal_val,
        ))

    oof_df = pd.DataFrame(all_rows, columns=OOF_COLUMNS)
    validate_oof_invariants(oof_df)
    return oof_df, fold_infos


def validate_oof_invariants(oof_df: pd.DataFrame) -> None:
    """Enforce the frozen OOF contract. Raises OOFError with a specific
    diagnostic on the first violation found; never silently repairs."""
    missing_cols = [c for c in OOF_COLUMNS if c not in oof_df.columns]
    if missing_cols:
        raise OOFError(f"OOF table is missing required columns: {missing_cols}")

    if oof_df.empty:
        raise OOFError("OOF table is empty")

    # One row per (timestamp, model, run_id).
    dupe_mask = oof_df.duplicated(subset=["timestamp", "model", "run_id"], keep=False)
    if dupe_mask.any():
        offenders = oof_df.loc[dupe_mask, ["timestamp", "model", "run_id"]].drop_duplicates()
        raise OOFError(f"Duplicate (timestamp, model, run_id) rows found:\n{offenders.head(10)}")

    # Probability invariant: prob_down + prob_neutral + prob_up ~= 1.
    prob_sum = oof_df[["prob_down", "prob_neutral", "prob_up"]].sum(axis=1)
    bad = ~np.isclose(prob_sum.to_numpy(), 1.0, atol=1e-6)
    if bad.any():
        raise OOFError(
            f"{int(bad.sum())} row(s) have prob_down+prob_neutral+prob_up != 1 "
            f"(example sums: {prob_sum[bad].head(5).tolist()})"
        )

    # Every model must cover exactly the same set of eligible (timestamp, fold) rows.
    per_model_keys = {
        model: set(zip(group["timestamp"], group["fold"]))
        for model, group in oof_df.groupby("model")
    }
    models = sorted(per_model_keys)
    if len(models) < len(ALL_MODELS):
        raise OOFError(f"Expected {len(ALL_MODELS)} models in OOF table, found {len(models)}: {models}")
    reference = per_model_keys[models[0]]
    for m in models[1:]:
        if per_model_keys[m] != reference:
            only_in_ref = reference - per_model_keys[m]
            only_in_m = per_model_keys[m] - reference
            raise OOFError(
                f"Model {m!r} does not cover the same eligible (timestamp, fold) set as "
                f"{models[0]!r}. Missing from {m!r}: {list(only_in_ref)[:5]}. "
                f"Extra in {m!r}: {list(only_in_m)[:5]}."
            )

    # actual_class must always be one of the 3 frozen classes.
    if not oof_df["actual_class"].isin(CLASS_ORDER).all():
        raise OOFError("actual_class contains a value outside {0, 1, 2}")
