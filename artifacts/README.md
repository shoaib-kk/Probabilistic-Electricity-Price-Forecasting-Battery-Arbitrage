This folder stores precomputed CSV artifacts used by the Streamlit app.

Required files:
- `actions.csv`
- `backtest_summary.csv`
- `cumulative_profit.csv`
- `conformal_forecast.csv`
- `final_metrics.csv`

Optional walk-forward validation outputs:
- `wfv_fold_summary.csv`
- `wfv_actions.csv`
- `wfv_forecast_diagnostics.csv`

Generate/update these files with:

```powershell
python -m pipelines.artifact_loader
```

Optional flags:

```powershell
python -m pipelines.artifact_loader --horizons 30,60 --fee-rate 0.01 --target-coverage 0.90
```

Walk-forward generation:

```powershell
python -m pipelines.artifact_loader --walk-forward --horizons 60,120 --wfv-train-days 180 --wfv-calibration-days 30 --wfv-test-days 14 --wfv-step-days 14
```

Notes:
- Build artifacts offline, then deploy only the app + `artifacts/` to Streamlit Community Cloud.
- The app is designed to read artifacts only (no in-app training).
