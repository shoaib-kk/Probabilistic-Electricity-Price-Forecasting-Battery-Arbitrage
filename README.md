# Battery Arbitrage with Conformal Forecasting

End-to-end pipeline for battery arbitrage on Australian NEM price data: collect → clean → explore → forecast → simulate. Combines uncertainty-aware quantile regression with conformal prediction and benchmarks against simple threshold policies.

---

## Highlights
- Automated AEMO data ingestion with integrity checks and chunked merges
- Leakage-safe cleaning, gap handling, and feature engineering for 5-minute data
- Exploratory analysis: seasonal decomposition, hourly/weekly structure, spike diagnostics
- Baselines: persistence, seasonal naive, ARIMA
- Probabilistic models: LightGBM quantile regression + conformal calibration
- Multi-horizon trading policies and battery simulator (efficiency, power, SOC constraints)
- Backtesting with profit, drawdown, and cycle metrics

---

## Pipeline at a Glance
1) **Collect**: download monthly AEMO price and demand CSVs; validate completeness; merge with SQLite to deduplicate and sort.
2) **Clean**: enforce time grid, fill small gaps, drop improbable values, engineer lags/seasonality/rolling stats; leakage-safe train/test split.
3) **Explore**: hourly averages, heatmaps (hour x day), weekly decomposition, spike summaries.
4) **Forecast**: LightGBM quantile models + conformal intervals across horizons (e.g., 30/60 min); ARIMA baseline.
5) **Simulate**: battery model with charge/discharge efficiency, power caps, SOC buffers; baseline percentile policy vs. aggressive conformal policy.
6) **Backtest**: cumulative profit, terminal value, drawdown, equivalent cycles, profit per cycle, directional accuracy.

---

## Strategies
- **Baseline threshold**: charge below 30th percentile, discharge above 70th percentile of training prices.
- **Conformal multi-horizon**: quantile forecasts + conformal adjustment; selects actions based on median vs. current price with costs/fees and horizon weighting.

---

## Results (illustrative run)

| Policy                  | Total Profit | Max Drawdown | Equivalent Cycles | Profit per Cycle |
|-------------------------|--------------|--------------|-------------------|------------------|
| Baseline Threshold      | $1639.88     | $89.08       | 127.7             | $12.84           |
| Conformal Multi-Horizon | $829.92      | $241.30      | 225.2             | $3.66            |

Notes:
- Threshold policy is a strong benchmark.
- Conformal policy trades more, with higher drawdown and lower profit per cycle; tuning horizons/edges can shift this balance.

---

## Quickstart
1) Install: `pip install -r requirements.txt`
2) Collect and clean: run the data scripts (or `Main.py`) to produce `PRICE_AND_DEMAND_FULL_VIC1.csv` and cleaned train/test CSVs.
3) Explore: run analysis/visualisation scripts to inspect seasonality and spikes.
4) Forecast and backtest: train quantile models, then run arbitrage simulation to compare baseline vs. conformal policies.
5) Build Streamlit artifacts: `python artifact_loader.py` (writes CSVs under `./artifacts` for cloud deployment).

---

## Roadmap
- FastAPI prediction endpoint and model persistence
- Streamlit dashboard for monitoring
- Scheduled retraining and data refresh

---

## Author
Shoaib Kabir


