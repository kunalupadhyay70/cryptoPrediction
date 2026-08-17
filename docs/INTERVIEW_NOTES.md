# Engineering Report — CryptoPrediction V1

This is the consolidated report for the autonomous rebuild of this
repository's ML pipeline (Stages 0 through 2Q), written for an audience
evaluating the engineering and ML methodology directly.

## PROJECT STATUS

**COMPLETE** for all software-engineering work reachable offline.
**BLOCKED_EXTERNAL** for one specific item only: running the pipeline
against real, current BTCUSDT data, because this sandbox cannot reach
`https://fapi.binance.com` (outbound network is restricted here — see
"Real-data status" below). Everything else — architecture, every pipeline
stage, the accounting engine, metrics, artifacts, the config/runtime
cutover, and end-to-end offline integration tests — is implemented, tested,
and committed.

## STAGES COMPLETED

Stage 0 (repository audit) through Stage 2Q (end-to-end offline synthetic
integration tests), inclusive. Stage 2R (real-data experiment) is blocked
per above; a reproducible command is provided. Stage 2S (this document +
README + repository cleanup) is in progress as part of this same report.

## COMMITS

One local commit per completed stage, on branch `stage-2a-interval-utility`
(never pushed — no push authorization in this environment):

```
40fcd4f Stage 2B: strict Pydantic configuration foundation
203a4df Stage 2C: interval-aware DataCollector integrity checking
97bdb54 Stage 2D: causal, ATR-scaled 3-class target (isolated module)
c1b68d7 Stage 2E/2F: causal OHLCV-only feature pipeline + target integration
a9b3740 Stage 2G: purge-aware expanding-window outer walk-forward splits
499fe15 Stage 2H: fold-local feature selection (correlation pruning + LGBM importance)
897ab95 Stage 2I: baselines + multinomial Logistic Regression + LightGBM multiclass
58a3209 Stage 2J: canonical OOF prediction generation + invariant enforcement
3b30b82 Stage 2K: past-only threshold selection and signal computation
f4e4c54 Stage 2L: FLAT/LONG/SHORT position engine + fixed-notional accounting
bae52db Stage 2M: predictive metrics, trading metrics, passive-long benchmark
b97d10e Stage 2N: production model training (future inference only)
ec6f52d Stage 2O: experiment artifacts (orchestration + persistence)
72622d1 Stage 2K/2P prep: production threshold selection + OHLCV loader
e2d19b0 Stage 2P: canonical config/runtime cutover
b2af066 Stage 2Q: end-to-end offline synthetic integration tests
<pending> Stage 2S: repository cleanup + README + this document
```

(Stages 0/1/1B-1D/2A were completed in the earlier human-reviewed phase of
this project, before autonomous operation began; their commits — including
`40fcd4f`/`203a4df` above, which were re-verified/re-delivered under
autonomous operation — precede this list.)

## FINAL TEST RESULT

**354 passed, 0 failed** (`pytest -q`), spanning 17 source modules and 19
test files. No skipped tests, no xfails.

## FINAL ARCHITECTURE

See the README's architecture diagram. In one sentence: OHLCV in ->
causal features + causal ATR-scaled 3-class target ->
purge-aware expanding walk-forward with fold-local feature selection ->
four models (2 deterministic baselines, Logistic Regression, LightGBM) ->
canonical out-of-fold prediction table -> past-only threshold selection and
signals -> a single FLAT/LONG/SHORT accounting engine producing both the
equity timeline and trade ledger -> predictive + trading metrics and a
real (non-zero-cost) passive-long benchmark -> a separately-refit
production model for future inference only -> one orchestrator that writes
every artifact -> a `config.yaml`/`main.py` runtime that drives all of it
through one strict, canonical configuration contract.

## FILES CREATED

`target_engineering.py`, `dataset_builder.py`, `walk_forward.py`,
`class_checks.py`, `feature_selection.py`, `model_training.py`,
`oof_builder.py`, `threshold_selection.py`, `position_engine.py`,
`metrics.py`, `production_model.py`, `experiment_runner.py`,
`docs/INTERVIEW_NOTES.md` (this file), plus one test file per module under
`tests/`, plus `tests/test_main.py` and `tests/test_integration_e2e.py`.

## FILES MODIFIED

`config.yaml` (rewritten to the canonical schema), `main.py` (rewritten to
drive the new pipeline), `data_collector.py` (added `load_ohlcv_dataframe`;
earlier stages also fixed interval-aware pagination/gap-detection),
`config_schema.py` / `intervals.py` (earlier stages), `requirements.txt`
(added `matplotlib`, `joblib`), `.gitignore` (added `*.db`, `/data/`,
`/artifacts/`), `README.md` (rewritten).

## FILES DELETED

`model.py`, `feature_engineering.py`, `backtester.py`, `signal_engine.py`
— the legacy pipeline these superseded. Confirmed dead (no remaining
import anywhere in the repository, including tests) before deletion.
`exchange_clients.py` was kept — `data_collector.py` still uses it.

## LEAKAGE PROTECTIONS

- **Target/feature construction**: every feature uses `.rolling(window,
  min_periods=window)`, `.ewm(adjust=False)`, or `.shift(+k)` only —
  never `.shift(-k)`, `bfill`, `ffill`, or `fillna`. Enforced by both a
  leakage-perturbation test (perturb rows after a cutoff, assert earlier
  rows bit-for-bit unchanged) and source-code inspection tests.
- **Walk-forward**: purge = `horizon_bars` (never independently
  configurable — a fixed off-by-one bug here, caught during self-review
  before any downstream stage used it, meant fold 1 always had too few
  training rows for any horizon >= 1; fixed and regression-tested).
  Embargo excludes a configurable window after each fold's test block from
  later folds' training data.
- **Feature selection**: correlation pruning and LightGBM importance
  selection are both fold-local, structurally (they only ever receive the
  exact rows the caller sliced out — no access to the full dataset or
  `outer_test`), not just by convention.
- **Model fitting**: `internal_validation` is used only for early
  stopping, never for gradient updates; the final per-fold model is
  refit on all of `outer_train` with a frozen `n_estimators`, then
  predicts `outer_test` only. `StandardScaler` is proven (via a
  monkeypatch spy) to never see test rows.
- **Threshold selection**: strictly past-only per fold (fold 1 uses its
  own internal-validation predictions; fold k>=2 uses only OOF from folds
  `1..k-1`), verified by independently recomputing each fold's threshold
  from the same raw ingredients outside the orchestration function, and by
  perturbing a fold's own OOF rows and confirming its threshold doesn't
  change.
- **OOF table**: the sole source of every reported historical metric;
  invariants (row uniqueness, probability-sum, cross-model timestamp
  coverage) are enforced on every build, not just tested once.
- **Production model**: trained on ALL historical data and explicitly
  never contributes a row to the OOF table or any reported historical
  metric — it exists only for future inference.
- **Full-pipeline probe**: `tests/test_integration_e2e.py` runs the
  entire pipeline twice, perturbing only far-future rows, and asserts
  every pre-perturbation OOF row is unchanged — proving the composition of
  all stages doesn't leak, not just each stage individually.

## MODELS

Baseline A (majority-class, deterministic one-hot), Baseline B (causal
persistence — `sign(close[t]-close[t-1])`, never the actual future label),
Baseline C (multinomial Logistic Regression via `lbfgs`, scaler fit on
train only), and the primary LightGBM multiclass model (fixed
`n_estimators` from fold-local early stopping). CatBoost is deliberately
not wired in for V1 (frozen as "must not block V1").

## OOF EVALUATION

Canonical LONG-format table, one row per `(timestamp, model, run_id)`,
covering every eligible (labeled) timestamp for every model equally.
Probabilities are always reindexed to `(DOWN, NEUTRAL, UP)` via each
estimator's own `classes_` attribute — never assumed positional. Predictive
metrics (Macro F1 primary; Balanced Accuracy, Macro ROC-AUC, Log Loss
secondary; confusion matrix; per-class precision/recall; class
distribution) are computed directly from this table and never touch the
backtest engine's outputs.

## BACKTEST ACCOUNTING

A single per-bar walk (`position_engine.run_backtest`) produces both the
equity timeline and trade ledger, which is what guarantees they reconcile
exactly — proven directly in tests, both on the in-memory DataFrames and
on the CSV files actually written to disk. `notional = equity_at_entry *
position_fraction`; `gross_pnl` is true mark-to-market per held bar (not a
lump sum at exit); one-way costs are charged on notional at both entry and
exit; funding is charged per held bar (symmetric for LONG/SHORT — a
documented simplification). One position at a time; no pyramiding; no
early reversal (a signal firing while already in a position is dropped,
not executed).

## RESULTS

No real-data run has been performed (see "Real-data status" below) — per
the frozen "no false claims" rule, this report makes no performance claims
(accuracy, Sharpe, AUC, profitability, or otherwise) for real BTCUSDT
data, because none exist yet. Every number produced so far comes from
synthetic fixtures used purely to prove the pipeline's mechanics are
correct (leakage-free, reconciling, deterministic), not to represent real
trading performance in any way.

## BASELINE COMPARISON

Not yet meaningful without real data — the OOF table and metrics pipeline
report all four models (2 baselines + Logistic Regression + LightGBM)
side by side automatically on every run, so a baseline comparison is a
structural property of the pipeline (`metrics.json`'s
`predictive_metrics_by_model` / `trading_metrics_by_model` sections), not
something that needs separate implementation once real data is available.

## KNOWN LIMITATIONS

- **V1 is OHLCV-only.** No historical order-book/trade-flow features —
  the repository never had real historical microstructure data to train
  on. Live order-book collection code (`data_collector.py`'s
  `collect_orderbook_snapshot` / `collect_orderbook_ws`) is untouched and
  isolated for future work.
- **Funding is symmetric for LONG/SHORT** in the accounting engine — a
  documented simplification; real perpetual futures funding is
  asymmetric by side and regime.
- **CatBoost is not wired in.** Frozen as optional/secondary and
  explicitly must not block V1.
- **The backtest engine assumes a positionally contiguous bar series**
  (bar `i+1` immediately follows bar `i` in time) — the same assumption
  `target_engineering`/`dataset_builder` already make via `.shift(-h)`.
  A dataset with real timestamp gaps (e.g. an exchange outage) is not
  specially handled; `run_integrity_check` will surface such gaps but
  nothing currently refuses to backtest across one.
- **Live mode's feature computation recomputes the full feature pipeline
  on every poll** (`build_dataset` on the whole stored history) rather
  than an incremental/streaming update — correct, but not the most
  efficient design for a tight polling loop.
- **No real-data run has happened yet** — see below.

## REAL-DATA STATUS

`https://fapi.binance.com` is not reachable from this sandbox (confirmed
via a live request during Stage 2R: `ProxyError` / `403 Forbidden` on the
outbound proxy, consistent with the Stage 0 finding). No real-data results
are fabricated anywhere in this repository or report.

**Reproducible command** (run from a machine with real network access):

```bash
git clone <this repo> && cd cryptoPrediction
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Edit config.yaml: set data.target_days to >= 180 (preferably ~365) for
# a real evaluation window; the shipped default (60) is for fast local
# iteration only.

python main.py --mode train_backtest --config config.yaml
```

This collects real BTCUSDT 5-minute perpetual futures OHLCV from Binance,
runs the complete Stage 2D-2O pipeline against it, and writes every
artifact listed in the README under `artifacts/<experiment.name>/`.

## HOW TO RUN LOCALLY

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q   # 354 passed
```

## HOW TO TRAIN

```bash
python main.py --mode train_backtest --config config.yaml
```

Trains and evaluates the entire pipeline; writes all artifacts (including
the production model) to `artifacts/<experiment.name>/`.

## HOW TO BACKTEST

Backtesting is not a separate step — `train_backtest` above runs the OOF
generation, threshold selection, and the position-engine backtest for
every model (plus the passive-long benchmark) as part of one call, and
writes `trade_ledger.csv` / `strategy_timeline.csv` / `metrics.json`.

## HOW TO RUN LIVE/INFERENCE

```bash
python main.py --mode live --config config.yaml
```

Requires a completed `train_backtest` run first (loads
`artifacts/<experiment.name>/production_model.joblib`,
`selected_features.json`, and `production_thresholds.json`). Polls the
configured exchange, recomputes features on the latest data, and logs a
BUY/SELL/HOLD signal using the exact same `threshold_selection.compute_signal`
function used throughout the offline pipeline.

## REMAINING WORK

1. Run the pipeline against real BTCUSDT data once network access is
   available (Stage 2R, blocked here — see above) and evaluate whether
   LightGBM actually beats the baselines and the passive-long benchmark
   out of sample. Keep the result honest either way.
2. Consider CatBoost as a secondary robustness check once the primary
   LightGBM pipeline is validated on real data (already gated `enabled:
   false` in `config_schema.CatBoostConfig`).
3. Consider incremental/streaming feature computation for `live` mode if
   polling frequency needs to increase materially.
4. Revisit the symmetric-funding simplification if real perpetual futures
   funding asymmetry turns out to matter materially for the strategy's
   economics.
5. Historical order-book/trade-flow features remain future work, gated on
   acquiring real historical microstructure data.
