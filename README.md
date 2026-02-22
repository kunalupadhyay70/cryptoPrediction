# BTCUSDT 1-Minute Direction ML System (v2 — Upgraded)

Production-grade modular ML pipeline for predicting next 1-minute BTCUSDT candle direction. Refactored for statistical robustness, minimal overfitting, and realistic out-of-sample performance.

---

## What Changed vs v1

| Area | v1 | v2 |
|------|----|----|
| Data size | 1440 candles (24h) | 30–90 days (~43k–130k rows) |
| Collection | Single API call | Paginated + incremental |
| Labels | Noisy binary close-to-close | Configurable: dead_zone / vol_breakout / regime_conditioned |
| Validation | Simple 80/20 split | Walk-forward expanding window (5 folds, no leakage) |
| Model | 3-model stacking (overfit risk) | LightGBM with early stopping; CatBoost fallback |
| Order book | Best bid/ask only | 10-level imbalances, microprice, depth slope, delta imbalance |
| Feature pruning | None | Correlation filter (>0.95) + SHAP importance |
| Thresholds | Static hardcoded | Swept per fold, optimized for Sharpe |
| Backtest execution | At current close (lookahead) | At next bar OPEN (no lookahead) |
| Position flips | Not modelled | Charged extra one-way cost |
| Regime models | None | Separate high-vol / low-vol models |
| AUC target | 0.73 (unrealistic) | 0.58 stable across folds (realistic) |
| Sharpe target | 3.0 (overfit) | 1.5 after full costs |
| Artifacts | CSV only | Fold report, feature stability, equity curve, config snapshot |

---

## Architecture

```
exchange_clients.py     Binance / Delta REST + WS adapter
data_collector.py       Paginated historical + incremental + integrity checks
feature_engineering.py  OHLCV + 10-level OB + trade flow + configurable labels
model.py                LightGBM + walk-forward CV + regime models + threshold opt
backtester.py           Realistic costs, execute-on-open, PnL distribution
signal_engine.py        Regime-aware signals with fold-optimized thresholds
main.py                 Orchestration: train/backtest + live loop
config.yaml             All parameters config-driven
```

---

## Label Modes

Set `target.label_mode` in `config.yaml`:

| Mode | Description |
|------|-------------|
| `dead_zone` | +1 if ret > +0.05%, 0 if < -0.05%, NaN otherwise (removes noise) |
| `vol_breakout` | 1 if next 3-bar absolute move > 70th percentile |
| `regime_conditioned` | Dead zone with wider threshold in high-vol regime |

---

## Validation

Walk-forward expanding window — **no future data leakage**:

```
Fold 1:  Train [0 → 5000]      Test [5000 → 6000]
Fold 2:  Train [0 → 6000]      Test [6000 → 7000]
Fold 3:  Train [0 → 7000]      Test [7000 → 8000]
...
```

Reports: mean AUC, std AUC, worst-fold AUC. Rejects if std > 0.04.

---

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Collect 30 days + train + backtest
python main.py --mode train_backtest --config config.yaml

# Live signal generation
python main.py --mode live --config config.yaml
```

### Config key settings

```yaml
collection:
  days_history: 30     # Minimum; set 90 for full robustness
  incremental: true    # Re-runs only fetch new candles

target:
  label_mode: dead_zone   # Recommended starting point

model:
  type: lightgbm          # Recommended
  use_regime_models: true
```

---

## Artifacts (saved to `artifacts/`)

| File | Contents |
|------|----------|
| `model_dataset.csv` | Full feature + label dataset |
| `backtest_report.csv` | Sharpe, drawdown, win rate, PnL distribution |
| `fold_report.csv` | Per-fold AUC, accuracy, optimal threshold |
| `feature_stability.csv` | Feature importance mean/std across folds |
| `equity_curve.csv` | Cumulative equity (1 = starting capital) |
| `config_snapshot.json` | Exact config used for this run |

---

## Performance Targets (Realistic)

| Metric | Target |
|--------|--------|
| Mean fold AUC | > 0.58 |
| Fold AUC std | < 0.04 |
| Sharpe (after costs) | > 1.5 |
| Max drawdown | < 20% |
| Trades per day | > 5 |

---

## Notes

- **Data integrity**: `data_collector.py` validates no gaps, duplicates, or ordering violations.
- **Regime models**: automatically disabled if insufficient regime-specific training data.
- **SHAP pruning**: requires `shap` installed; falls back gracefully if not available.
- **LightGBM fallback**: falls back to `sklearn.GradientBoostingClassifier` if not installed.
- **CatBoost**: optional alternative; set `model.type: catboost` in config.
