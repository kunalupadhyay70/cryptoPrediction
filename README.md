# BTCUSDT Perpetual Futures — 3-Class Direction Model (V1)

A leak-safe, walk-forward-validated ML pipeline for predicting the
3-bar-ahead direction (DOWN / NEUTRAL / UP) of BTCUSDT perpetual futures on
5-minute bars, with a realistic fixed-notional backtest engine and a
passive-long benchmark. Built OHLCV-only — no historical order-book or
trade-flow features (the repository has no real historical microstructure
data to train on honestly).

This is a from-scratch rebuild of an earlier, methodologically unsound
prototype (see `docs/INTERVIEW_NOTES.md` for the full list of leakage bugs
that prototype had and how this version avoids each one).

## Architecture

```
intervals.py            Canonical interval string <-> seconds/ms authority
config_schema.py         Canonical, strict (extra="forbid") Pydantic config contract
data_collector.py         Paginated + incremental OHLCV collection, integrity checks, OHLCV loader

target_engineering.py     Causal ATR-scaled 3-class target (frozen formula, see below)
dataset_builder.py        Causal OHLCV-only feature pipeline (rolling/ewm/shift only, no bfill)
walk_forward.py           Purge-aware expanding-window walk-forward splits
class_checks.py           Shared MISSING-CLASS POLICY (fail loudly, never silently collapse classes)
feature_selection.py      Fold-local correlation pruning + LightGBM importance selection
model_training.py         Baselines (majority/persistence), Logistic Regression, LightGBM
oof_builder.py             Orchestrates the above into the canonical out-of-fold prediction table
threshold_selection.py    Past-only per-fold threshold selection + the frozen SIGNAL RULE
position_engine.py         FLAT/LONG/SHORT state machine + fixed-notional accounting
metrics.py                  Predictive metrics, trading metrics, passive-long benchmark
production_model.py        Median-best-iteration, stability-selected-feature refit for future inference
experiment_runner.py        Orchestrates the full pipeline end to end + writes every artifact
main.py                     CLI entrypoint (train_backtest / live), config.yaml -> AppConfig -> pipeline
config.yaml                  The one canonical configuration file
```

Every module above has its own test file in `tests/` (e.g.
`target_engineering.py` <-> `tests/test_target_engineering.py`).
`tests/test_integration_e2e.py` additionally proves the *composition* of
all these modules doesn't leak, is deterministic, and reconciles on disk.

## The frozen target definition

For each bar `t`:

```
entry[t]  = open[t+1]
exit[t]   = close[t+h]                       (h = horizon_bars, default 3)
tradable_return[t] = exit[t] / entry[t] - 1

TR[t]  = max(high[t]-low[t], |high[t]-close[t-1]|, |low[t]-close[t-1]|)
ATR[t] = rolling mean of TR, causal (only rows <= t)
band[t] = neutral_atr_mult * ATR[t] / close[t]

class[t] = UP    if tradable_return[t] >  band[t]
         = DOWN  if tradable_return[t] < -band[t]
         = NEUTRAL otherwise (exact ties go to NEUTRAL, strict inequality)
```

NEUTRAL rows are never filtered out. Tail rows without a full `t+h` window
and ATR warm-up rows stay genuinely unlabeled (`NaN`) — never
back-filled.

## Validation methodology

Expanding-window walk-forward with **purge = horizon_bars** (never a
separately configurable value — deriving it from anything else would let a
config author silently reintroduce leakage) and a configurable **embargo**
(default 0). Within each fold's `outer_train`: a chronological 80/20 split
into `internal_train` / `internal_validation`; correlation pruning and
LightGBM-importance feature selection both run on `internal_train` only,
with `internal_validation` used solely for early stopping. The selected
features and `best_iteration` are then frozen and the model is refit on
**all** of `outer_train` before predicting `outer_test`.

Every model's out-of-fold predictions land in one canonical LONG-format
table (`oof_predictions.csv`) — this table, and only this table, is the
source of every reported historical metric. The production model (refit on
all historical data for future inference) never contributes to it.

## Models

Baseline A (majority-class), Baseline B (causal persistence — `sign(close[t]
- close[t-1])`, never the actual future label), Baseline C (multinomial
Logistic Regression, scaler fit on train only), and the primary LightGBM
multiclass model. Baselines are deterministic and are never
threshold-tuned.

## Threshold selection & signals

Fold 1 uses fold 1's own internal-validation predictions; fold `k>=2` uses
only OOF predictions from folds `1..k-1` — never the current or a future
fold. `BUY` if `prob_up >= buy_threshold AND prob_up` is the strict max
across classes; `SELL` symmetric on `prob_down`; else `HOLD`. Separately,
a production/live threshold is computed once from the **full** historical
OOF table and is never used to recompute historical performance.

## Backtest accounting

A single engine (`position_engine.py`) produces both the equity timeline
and the trade ledger from the same per-bar walk, which is what guarantees
they reconcile exactly. FLAT / LONG / SHORT, one position at a time, no
pyramiding, no early reversal. `notional = equity_at_entry *
position_fraction`, one-way costs `(fee+slippage+latency)/10000` charged
on notional at both entry and exit, funding charged per held bar. A
passive-long benchmark reuses the exact same engine over the exact same
OOF evaluation window, with the same cost/funding treatment — never a
zero-cost benchmark.

## Running it

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Train + backtest against config.yaml (collects data first if the
# configured exchange is reachable; otherwise trains on whatever OHLCV is
# already stored locally).
python main.py --mode train_backtest --config config.yaml

# Production inference loop (requires a completed train_backtest run first
# — loads artifacts/<experiment.name>/production_model.joblib +
# selected_features.json + production_thresholds.json).
python main.py --mode live --config config.yaml
```

### Running a real, longer BTCUSDT experiment

`config.yaml` ships with `data.target_days: 60` for fast iteration. For a
real evaluation, edit `config.yaml` (or copy it) and set
`data.target_days` to at least 180, preferably ~365, then run the same
command above. In this sandbox, `https://fapi.binance.com` is not reachable
(outbound network is restricted), so this has not been run against real
data from here — see `docs/INTERVIEW_NOTES.md` under "Real-data status"
for the exact reproducible command and what to expect.

## Artifacts (written to `artifacts/<experiment.name>/`)

| File | Contents |
|------|----------|
| `config_snapshot.json` | Exact config used for this run |
| `metrics.json` | Predictive + trading metrics per model, plus the passive-long benchmark |
| `fold_metrics.csv` | Per-fold row counts, selected feature count, best_iteration |
| `oof_predictions.csv` | The canonical out-of-fold prediction table (sole source of historical metrics) |
| `feature_importance.csv` | Production model's feature importances |
| `selected_features.json` | Production model's stability-selected feature list |
| `trade_ledger.csv` | Every closed trade, all models, one row per trade |
| `strategy_timeline.csv` | Per-bar equity timeline, all models |
| `production_thresholds.json` | Buy/sell thresholds computed once from full historical OOF (future inference only) |
| `production_model.joblib` | The serialized production model (future inference only — never used for reported historical performance) |
| `equity_curve.png`, `confusion_matrix_lightgbm.png` | Plots |

## Tests

```bash
pytest -q
```

354+ tests across every module, including leakage-perturbation probes,
hand-derived-expected-value tests, boundary tests, and full-pipeline
integration tests. See `docs/INTERVIEW_NOTES.md` for the engineering
report this project was built against.
