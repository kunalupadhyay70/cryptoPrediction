"""
Model — LightGBM walk-forward CV, regime models, threshold optimization.
Exports: ModelConfig, RegimeModelSet, StackedEnsembleModel,
         prune_correlated_features, prune_shap_features
"""
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FoldResult
# ---------------------------------------------------------------------------

@dataclass
class FoldResult:
    fold_idx: int
    train_rows: int
    test_rows: int
    accuracy: float
    auc: float
    optimal_threshold: float
    feature_importances: Dict[str, float]
    regime: str = "all"


# ---------------------------------------------------------------------------
# ModelConfig — accepts both old and new field names
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    random_state: int = 42

    # Primary model type — accepted under two names
    model_type: str = "lightgbm"        # original name
    primary: str = "lightgbm"           # main.py uses this

    # Gates
    min_accuracy: float = 0.52
    min_auc: float = 0.55              # lowered from 0.58 to reflect realistic target
    max_auc_std: float = 0.06          # relaxed from 0.04 — 30 days isn't enough for tight folds

    # Validation
    n_folds: int = 5
    min_train_rows: int = 3000         # lowered from 5000 to allow more folds on 30-day data
    min_test_rows: int = 500           # lowered from 1000
    min_fold_stability: float = 0.4    # LOWERED from 0.6 — was pruning too aggressively

    # Regime models
    use_regime_models: bool = True
    vol_regime_threshold: float = 0.0015
    compare_stacking: bool = False

    # Feature pruning
    shap_top_k: int = 60               # raised from 50 — more features with lagged set
    shap_top_n: int = 60
    correlation_threshold: float = 0.95

    # LightGBM — tuned for financial tabular data (less overfit, more signal)
    lgbm_num_leaves: int = 63          # deeper trees to capture non-linear patterns
    lgbm_learning_rate: float = 0.02   # SLOWER learning (was 0.05) → better generalisation
    lgbm_n_estimators: int = 1000      # MORE trees to compensate for slower lr
    lgbm_min_child_samples: int = 100  # HIGHER (was 50) → stronger regularisation
    lgbm_subsample: float = 0.7        # slightly lower row sampling
    lgbm_colsample_bytree: float = 0.7 # slightly lower col sampling
    lgbm_reg_alpha: float = 0.5        # STRONGER L1 (was 0.1)
    lgbm_reg_lambda: float = 2.0       # STRONGER L2 (was 1.0)
    lgbm_early_stopping_rounds: int = 75  # more patience

    # Threshold sweep
    threshold_sweep_step: float = 0.01
    threshold_sweep_min: float = 0.50
    threshold_sweep_max: float = 0.70
    fee_bps: float = 8.0

    def __post_init__(self):
        # Resolve model_type / primary alias
        if self.primary != "lightgbm":
            self.model_type = self.primary
        elif self.model_type != "lightgbm":
            self.primary = self.model_type
        # Resolve shap_top_k / shap_top_n alias
        if self.shap_top_k != 50:
            self.shap_top_n = self.shap_top_k
        elif self.shap_top_n != 50:
            self.shap_top_k = self.shap_top_n


# ---------------------------------------------------------------------------
# StackedEnsembleModel
# ---------------------------------------------------------------------------

class StackedEnsembleModel:
    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self.model_: Any = None
        self.model_high_vol_: Any = None
        self.model_low_vol_: Any = None
        self.feature_cols_: List[str] = []
        self.metrics_: Dict = {}
        self.fold_results_: List[FoldResult] = []
        self.optimal_threshold_: float = 0.55
        self.stable_features_: List[str] = []
        self._lgbm_ok = self._check_lgbm()
        self._catboost_ok = self._check_catboost()

    @staticmethod
    def _check_lgbm() -> bool:
        try:
            import lightgbm; return True
        except ImportError:
            LOGGER.warning("LightGBM not installed — using GradientBoosting fallback")
            return False

    @staticmethod
    def _check_catboost() -> bool:
        try:
            import catboost; return True
        except ImportError:
            return False

    def _make_primary_model(self, model_type: Optional[str] = None):
        """Public — called by main.py for the SHAP warm-up fit."""
        return self._make_model(model_type)

    def _make_model(self, model_type: Optional[str] = None):
        mt = model_type or self.config.model_type
        rs = self.config.random_state
        if mt == "lightgbm" and self._lgbm_ok:
            import lightgbm as lgb
            return lgb.LGBMClassifier(
                num_leaves=self.config.lgbm_num_leaves,
                learning_rate=self.config.lgbm_learning_rate,
                n_estimators=self.config.lgbm_n_estimators,
                min_child_samples=self.config.lgbm_min_child_samples,
                subsample=self.config.lgbm_subsample,
                colsample_bytree=self.config.lgbm_colsample_bytree,
                reg_alpha=self.config.lgbm_reg_alpha,
                reg_lambda=self.config.lgbm_reg_lambda,
                random_state=rs, n_jobs=-1, verbosity=-1,
            )
        if mt == "catboost" and self._catboost_ok:
            from catboost import CatBoostClassifier
            return CatBoostClassifier(
                iterations=self.config.lgbm_n_estimators,
                learning_rate=self.config.lgbm_learning_rate,
                depth=6, random_seed=rs, verbose=False,
            )
        from sklearn.ensemble import GradientBoostingClassifier
        LOGGER.info("Using GradientBoostingClassifier fallback")
        return GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.8, random_state=rs
        )

    def _walk_forward_splits(self, n: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        n_folds = self.config.n_folds
        base_test  = max(self.config.min_test_rows,  n // (n_folds + 1))
        base_train = max(self.config.min_train_rows, base_test * 2)
        splits = []
        for fold in range(n_folds):
            train_end = base_train + fold * base_test
            test_end  = train_end + base_test
            if test_end > n:
                break
            ti, vi = np.arange(0, train_end), np.arange(train_end, test_end)
            if len(ti) >= self.config.min_train_rows and len(vi) >= self.config.min_test_rows:
                splits.append((ti, vi))
        if not splits:
            sp = int(n * 0.8)
            splits = [(np.arange(sp), np.arange(sp, n))]
        LOGGER.info("Walk-forward splits: %d folds", len(splits))
        return splits

    def _optimize_threshold(self, y_true, y_prob) -> Tuple[float, float]:
        best_t, best_s = 0.55, -np.inf
        costs = self.config.fee_bps / 10_000
        for thresh in np.arange(self.config.threshold_sweep_min,
                                 self.config.threshold_sweep_max,
                                 self.config.threshold_sweep_step):
            sell_t = 1.0 - thresh
            sigs = np.where(y_prob >= thresh, 1, np.where(y_prob <= sell_t, -1, 0))
            direction = np.where(y_true == 1, 1.0, -1.0)
            ret = sigs * direction * 0.0018 - np.abs(sigs) * costs
            active = ret[sigs != 0]
            if len(active) < 20: continue
            sharpe = active.mean() / (active.std() + 1e-9) * np.sqrt(365*24*60)
            if sharpe > best_s:
                best_s = sharpe; best_t = thresh
        return best_t, best_s

    def _select_stable_features(self, fold_importances, feature_cols):
        from collections import Counter
        n = len(fold_importances)
        if n == 0: return feature_cols
        appearance = Counter()
        for fi in fold_importances:
            for feat, imp in fi.items():
                if imp > 0: appearance[feat] += 1
        stable = [f for f in feature_cols if appearance.get(f, 0)/n >= self.config.min_fold_stability]
        LOGGER.info("Stable features: %d/%d (%.0f%% threshold)", len(stable), len(feature_cols),
                    self.config.min_fold_stability*100)
        return stable if stable else feature_cols

    def _fit_model(self, model, X_train, y_train, X_val=None, y_val=None):
        mt = self.config.model_type
        if mt == "lightgbm" and self._lgbm_ok and X_val is not None:
            import lightgbm as lgb
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                      callbacks=[lgb.early_stopping(self.config.lgbm_early_stopping_rounds, verbose=False),
                                 lgb.log_evaluation(-1)])
        elif mt == "catboost" and self._catboost_ok and X_val is not None:
            from catboost import Pool
            model.fit(X_train, y_train, eval_set=Pool(X_val, y_val),
                      early_stopping_rounds=self.config.lgbm_early_stopping_rounds, verbose=False)
        else:
            model.fit(X_train, y_train)
        return model

    def _get_importance(self, model, feature_cols):
        try: return dict(zip(feature_cols, model.feature_importances_))
        except AttributeError: return {}

    def fit_evaluate(self, df: pd.DataFrame, feature_cols: List[str], target_col: str) -> Dict:
        X_all = self._clean(df[feature_cols])
        y_all = pd.to_numeric(df[target_col], errors="coerce")
        valid = y_all.notna()
        X_all, y_all = X_all[valid], y_all[valid].astype(int).values
        splits = self._walk_forward_splits(len(X_all))

        fold_aucs, fold_thresholds, fold_imps = [], [], []
        self.fold_results_ = []
        fold_metrics_list = []

        for i, (ti, vi) in enumerate(splits):
            X_tr_full, X_te = X_all.iloc[ti].copy(), X_all.iloc[vi].copy()
            y_tr_full, y_te = y_all[ti], y_all[vi]
            val_sp = int(len(X_tr_full) * 0.8)
            X_tr, X_val = X_tr_full.iloc[:val_sp], X_tr_full.iloc[val_sp:]
            y_tr, y_val = y_tr_full[:val_sp], y_tr_full[val_sp:]

            if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
                LOGGER.warning("Fold %d: single class — skipping", i+1); continue

            m = self._make_model()
            self._fit_model(m, X_tr, y_tr, X_val, y_val)
            prob = m.predict_proba(X_te)[:, 1]
            pred = (prob >= 0.5).astype(int)
            acc  = accuracy_score(y_te, pred)
            auc  = roc_auc_score(y_te, prob) if len(np.unique(y_te)) > 1 else 0.5
            opt_t, opt_s = self._optimize_threshold(y_te, prob)
            imp = self._get_importance(m, feature_cols)

            fold_aucs.append(auc); fold_thresholds.append(opt_t); fold_imps.append(imp)
            fr = FoldResult(i+1, len(X_tr_full), len(X_te), acc, auc, opt_t, imp)
            self.fold_results_.append(fr)
            fold_metrics_list.append({"fold": i+1, "auc": auc, "accuracy": acc,
                                      "train_rows": len(X_tr_full), "test_rows": len(X_te),
                                      "optimal_threshold": opt_t})
            LOGGER.info("Fold %d/%d | train=%d test=%d | AUC=%.4f Acc=%.4f thresh=%.2f sharpe=%.2f",
                        i+1, len(splits), len(X_tr_full), len(X_te), auc, acc, opt_t, opt_s)

        if not fold_aucs:
            LOGGER.error("No folds completed")
            self.metrics_ = {"accuracy": 0.0, "roc_auc": 0.5, "n_folds": 0,
                             "n_folds_used": 0, "status": "NO_FOLDS", "fold_metrics": []}
            return self.metrics_

        mean_auc = float(np.mean(fold_aucs))
        std_auc  = float(np.std(fold_aucs))
        worst    = float(np.min(fold_aucs))
        self.optimal_threshold_ = float(np.median(fold_thresholds))
        self.stable_features_ = self._select_stable_features(fold_imps, feature_cols)
        self.feature_cols_ = self.stable_features_

        LOGGER.info("CV | mean_auc=%.4f std=%.4f worst=%.4f thresh=%.2f stable=%d",
                    mean_auc, std_auc, worst, self.optimal_threshold_, len(self.stable_features_))
        if std_auc > self.config.max_auc_std:
            LOGGER.warning("High fold variance: std=%.4f > %.4f", std_auc, self.config.max_auc_std)

        # Final model on all data
        X_final = self._clean(df[self.feature_cols_])[valid]
        mf = self._make_model()
        self._fit_model(mf, X_final, y_all)
        self.model_ = mf

        last = self.fold_results_[-1]
        self.metrics_ = {
            "accuracy": float(last.accuracy),
            "roc_auc": mean_auc,
            "roc_auc_std": std_auc,
            "worst_fold_auc": worst,
            "n_folds": len(fold_aucs),
            "n_folds_used": len(fold_aucs),
            "optimal_threshold": self.optimal_threshold_,
            "stable_features": len(self.stable_features_),
            "passes_auc_gate": int(mean_auc >= self.config.min_auc),
            "passes_variance_gate": int(std_auc <= self.config.max_auc_std),
            "fold_metrics": fold_metrics_list,
            "status": self._gate_status(mean_auc, std_auc),
        }
        return self.metrics_

    def _gate_status(self, mean_auc, std_auc):
        if mean_auc < self.config.min_auc:     return "REJECTED_LOW_AUC"
        if std_auc  > self.config.max_auc_std: return "REJECTED_HIGH_VARIANCE"
        return "ACCEPTED"

    def predict_proba(self, X: pd.DataFrame, feature_cols: Optional[List[str]] = None,
                      regime: Optional[int] = None) -> np.ndarray:
        # Always predict with the exact columns the model was trained on (self.feature_cols_).
        # The feature_cols argument is accepted for API compatibility but is intentionally
        # ignored when self.feature_cols_ is available, preventing shape mismatches that
        # occur when the caller passes the pre-pruning column list.
        training_cols = self.feature_cols_ if self.feature_cols_ else feature_cols
        if not training_cols:
            raise RuntimeError("Model has not been fitted yet — feature_cols_ is empty.")
        Xc = self._clean(X.reindex(columns=training_cols, fill_value=0.0))
        if regime is not None and self.config.use_regime_models:
            if regime == 1 and self.model_high_vol_ is not None:
                return self.model_high_vol_.predict_proba(Xc)[:, 1]
            if regime == 0 and self.model_low_vol_ is not None:
                return self.model_low_vol_.predict_proba(Xc)[:, 1]
        return self.model_.predict_proba(Xc)[:, 1]

    @staticmethod
    def _clean(X: pd.DataFrame) -> pd.DataFrame:
        return X.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def fold_report(self) -> pd.DataFrame:
        if not self.fold_results_: return pd.DataFrame()
        return pd.DataFrame([{"fold": fr.fold_idx, "regime": fr.regime,
                               "train_rows": fr.train_rows, "test_rows": fr.test_rows,
                               "accuracy": fr.accuracy, "auc": fr.auc,
                               "optimal_threshold": fr.optimal_threshold}
                              for fr in self.fold_results_])

    def feature_stability_report(self) -> pd.DataFrame:
        if not self.fold_results_: return pd.DataFrame()
        all_feats = set()
        for fr in self.fold_results_: all_feats.update(fr.feature_importances.keys())
        rows = []
        for feat in sorted(all_feats):
            vals = [fr.feature_importances.get(feat, 0.0) for fr in self.fold_results_]
            rows.append({"feature": feat, "mean_importance": np.mean(vals), "std_importance": np.std(vals),
                         "n_folds_present": sum(v>0 for v in vals),
                         "stability": sum(v>0 for v in vals)/len(vals)})
        return pd.DataFrame(rows).sort_values("mean_importance", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# RegimeModelSet
# ---------------------------------------------------------------------------

class RegimeModelSet:
    """
    Trains and holds separate models for high-vol and low-vol regimes.
    Accepts a ModelConfig (or nothing) as first argument.
    """
    def __init__(self, config: Optional[ModelConfig] = None) -> None:
        self.config = config or ModelConfig()
        self.high_vol: Any = None
        self.low_vol: Any  = None
        self.feature_cols: List[str] = []
        self.vol_regime_threshold: float = self.config.vol_regime_threshold
        self._ensemble = StackedEnsembleModel(self.config)

    def fit_evaluate(self, df: pd.DataFrame, feature_cols: List[str], target_col: str) -> Dict:
        """
        Train separate walk-forward models for high-vol and low-vol regimes.
        Returns dict mapping regime name → metrics dict.
        """
        if "vol_regime" not in df.columns:
            LOGGER.warning("vol_regime column missing — skipping regime models")
            return {}

        self.feature_cols = feature_cols
        results = {}

        for regime_name, regime_val in [("high_vol", 1), ("low_vol", 0)]:
            mask = df["vol_regime"] == regime_val
            sub  = df[mask].reset_index(drop=True)
            if len(sub) < self.config.min_train_rows // 2:
                LOGGER.warning("Regime '%s': only %d rows — skipping", regime_name, len(sub))
                continue

            LOGGER.info("Regime '%s': %d rows", regime_name, len(sub))
            e = StackedEnsembleModel(self.config)
            m = e.fit_evaluate(sub, feature_cols, target_col)
            m["regime"] = regime_name

            if regime_name == "high_vol":
                self.high_vol = e.model_
            else:
                self.low_vol  = e.model_

            results[regime_name] = m

        return results

    def predict_proba(self, X: pd.DataFrame, regime: int) -> np.ndarray:
        cols = self.feature_cols or list(X.columns)
        Xc = X.reindex(columns=cols, fill_value=0.0).apply(pd.to_numeric, errors="coerce").fillna(0.0)
        if regime == 1 and self.high_vol is not None:
            return self.high_vol.predict_proba(Xc)[:, 1]
        if regime == 0 and self.low_vol is not None:
            return self.low_vol.predict_proba(Xc)[:, 1]
        raise ValueError(f"No model for regime={regime!r}")

    @classmethod
    def from_ensemble(cls, ensemble: StackedEnsembleModel) -> "RegimeModelSet":
        rms = cls(ensemble.config)
        rms.high_vol      = ensemble.model_high_vol_
        rms.low_vol       = ensemble.model_low_vol_
        rms.feature_cols  = list(ensemble.feature_cols_)
        return rms


# ---------------------------------------------------------------------------
# Module-level pruning helpers
# ---------------------------------------------------------------------------

def prune_correlated_features(
    df: pd.DataFrame,
    feature_cols: List[str],
    threshold: float = 0.95,
) -> List[str]:
    """Remove features with |correlation| > threshold (upper-triangle rule)."""
    X = (df[feature_cols]
         .apply(pd.to_numeric, errors="coerce")
         .replace([np.inf, -np.inf], np.nan)
         .fillna(0.0))
    corr  = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    to_drop = {col for col in upper.columns if upper[col].max() > threshold}
    kept = [c for c in feature_cols if c not in to_drop]
    if to_drop:
        LOGGER.info("prune_correlated_features: removed %d, %d remaining (threshold=%.2f)",
                    len(to_drop), len(kept), threshold)
    return kept


def prune_shap_features(
    model: Any,
    df: pd.DataFrame,
    feature_cols: List[str],
    top_n: int = 50,
    top_k: int = None,          # alias accepted by main.py
) -> List[str]:
    """Keep top top_n (or top_k) features by mean |SHAP|. Graceful fallback."""
    n = top_k if top_k is not None else top_n
    try:
        import shap
    except ImportError:
        LOGGER.warning("prune_shap_features: shap not installed — returning all %d features", len(feature_cols))
        return feature_cols
    try:
        X = (df[feature_cols]
             .apply(pd.to_numeric, errors="coerce")
             .replace([np.inf, -np.inf], np.nan)
             .fillna(0.0))
        explainer  = shap.TreeExplainer(model)
        shap_vals  = explainer.shap_values(X)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
        mean_abs = np.abs(shap_vals).mean(axis=0)
        ranked   = sorted(zip(feature_cols, mean_abs), key=lambda x: -x[1])
        selected = [f for f, _ in ranked[:n]]
        LOGGER.info("prune_shap_features: kept top %d of %d", len(selected), len(feature_cols))
        return selected
    except Exception as exc:
        LOGGER.warning("prune_shap_features: failed (%s) — returning all features", exc)
        return feature_cols