# Live RL Battery Arbitrage — AEMO NEM (VIC1)

A reinforcement-learning agent that makes real charge/discharge decisions for a
**simulated** battery against live AEMO electricity prices, every 30 minutes,
with performance tracked publicly against a rule-based baseline and a
perfect-foresight bound.

> ⚠️ **Simulated paper trading only.** No real money, no market participation,
> no battery hardware. Prices are real AEMO dispatch data; every trade is
> hypothetical. This is a research/portfolio project, not financial advice.

**Live dashboard:** https://probabilistic-electricity-price-forecasting-battery-arbitrage.streamlit.app/

---

## Architecture

```
                        ┌──────────────────────────────────────────────┐
 AEMO 5MIN API ───────► │        LIVE TICK (GitHub Actions, */30)      │
 (NEMWEB fallback)      │ fetch prices ─► build features ─► quantile   │
 Open-Meteo solar ────► │ forecasts ─► RL policy (numpy) ─► settle     │
                        │ battery P&L ─► append ledger                 │
                        └──────────────────┬───────────────────────────┘
 AEMO monthly CSVs                         │ push to `live-data` branch
 (2023 → now) ─► 30-min dataset ─► train   ▼
 forecaster + PPO (5 seeds) ─► models/  Streamlit dashboard (public)
```

- **Data**: 5-minute VIC1 dispatch price/demand (AEMO visualisation API live,
  monthly archive CSVs for history, NEMWEB DispatchIS as live fallback);
  solar irradiance from Open-Meteo (forecast + archive). Everything resampled
  to 30-minute intervals labelled by interval start, NEM market time (UTC+10).
- **Forecasting**: LightGBM quantile regression (q05/q50/q95) at 30 and 60 min
  horizons, split-conformally calibrated to 90% coverage. The forecast
  distribution is part of the agent's state, not a trading rule.
- **RL**: PPO (Stable-Baselines3) on a Gymnasium env. Trained across 5 seeds;
  the deployed "primary" is the **median**-validation seed, and all seeds are
  paper-traded live so seed variance stays visible.
- **Live loop**: a GitHub Actions cron job every 30 minutes runs one tick and
  publishes ledger/state to the `live-data` branch. Inference is pure
  numpy + LightGBM (policies are exported from torch at deploy time, with a
  parity test).

## RL formulation

| Component | Choice |
|---|---|
| State (24 dims) | battery SoC; time-of-day + day-of-week encodings; last 6 prices + rolling stats + 7-day percentile (asinh-normalized); conformal quantile forecasts for the settling and next interval; current + next-3h solar irradiance; lagged demand |
| Actions | Discrete(5): charge 100%/50%, hold, discharge 50%/100% of 50 kW |
| Reward | realized interval P&L: `energy_sold·p − energy_bought·p − 1% fee·|p| − $0.03/kWh degradation` (identical accounting for all strategies) |
| Battery | 100 kWh / 50 kW, 90% one-way efficiency |
| Episode | 1 week (336 intervals), random start + random initial SoC |

**Baselines** (same costs, same data):
- *Threshold*: charge below the 30th / discharge above the 70th percentile of
  training prices, using the last completed interval's price.
- *Perfect foresight*: LP over the realized price path (continuous power) — an
  upper bound no causal policy can reach.

## Honesty guarantees

- A decision for interval T uses only data from before T (enforced by a
  no-lookahead test on the feature builder). The one exception is solar
  irradiance at/after T, which is an exogenous *weather forecast* available at
  decision time (backtests use archived actuals as a proxy — a documented
  simplification).
- Walk-forward evaluation (train 1y → calibrate 30d → trade 90d out-of-sample,
  rolling), not a single split.
- Results are reported as mean ± std across seeds; the deployed seed is the
  median validation performer, and all seeds trade live in parallel.
- If the live tick misses intervals (scheduler jitter, API outage), strategies
  hold or catch up using only information predating each interval.

## Repo layout

```
src/
  config.py               battery/cost/action/paths configuration
  Battery_model.py        SoC/power/efficiency battery simulator
  data_feeds/             live AEMO price feed (+ NEMWEB fallback), Open-Meteo solar
  rl/
    features.py           THE feature/state builder (shared by train + live)
    forecast.py           LightGBM quantile + conformal forecaster (persistable)
    env.py                Gymnasium env + shared settlement accounting
    baselines.py          threshold policy
    perfect_foresight.py  LP upper bound (scipy HiGHS)
    train.py              PPO training
    policy_export.py      torch → numpy policy export (parity-checked)
    evaluate.py           metrics + walk-forward orchestration
  live/                   decision loop, ledger, persisted state
scripts/
  build_dataset.py        backfill AEMO months + weather → data/dataset_30min.csv
  run_backtest.py         walk-forward backtest + report figures
  train_agent.py          train 5 seeds, export deployment bundle
  live_tick.py            one scheduled live tick
app/streamlit_app.py      dashboard (live + backtest + legacy pages)
models/deployed/          committed deployment bundle (forecaster, policies, meta)
artifacts/backtest/       walk-forward report (CSVs + figures)
artifacts/live/           ledger, state, caches (updated on `live-data` branch)
.github/workflows/live-loop.yml   the 30-min scheduler
```

## Reproduce

```bash
pip install -r requirements.txt -r requirements-rl.txt

# 1. Build the 30-min dataset (downloads AEMO months + solar archive)
python scripts/build_dataset.py

# 2. Walk-forward backtest (3 seeds x ~7 folds; several hours on CPU)
python scripts/run_backtest.py --seeds 3 --timesteps 150000

# 3. Train + export the deployment bundle
python scripts/train_agent.py --seeds 5 --timesteps 300000

# 4. Run one live tick locally
python scripts/live_tick.py

# Tests (env invariants, no-lookahead, LP sanity, export parity)
python -m pytest tests/ -q
```

The scheduled loop needs no secrets: GitHub Actions runs
`scripts/live_tick.py` on `requirements-live.txt` (no torch) and force-pushes
`artifacts/live/` to the `live-data` branch, which the dashboard reads via
raw URLs — so the Streamlit app never redeploys on data updates.

## Results

See the **RL Backtest** page on the dashboard, or `artifacts/backtest/`
(`wf_cumulative_pnl.png`, `wf_fold_profits.png`, `wf_fold_summary.csv`) for the
current walk-forward report: cumulative P&L, Sharpe, max drawdown, cycles, and
profit-per-cycle for the RL agent (per seed) vs both baselines.

The original conformal-policy project this evolved from is preserved under the
legacy dashboard pages and `pipelines/` + `docs/Data_pipeline.md`.

## Author

Shoaib Kabir
