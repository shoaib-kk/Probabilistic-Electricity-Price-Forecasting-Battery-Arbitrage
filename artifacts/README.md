This folder stores precomputed CSV artifacts used by the Streamlit app.

Required files:
- `actions.csv`
- `backtest_summary.csv`
- `cumulative_profit.csv`
- `conformal_forecast.csv`
- `final_metrics.csv`

Generate/update these files with:

```powershell
python artifact_loader.py
```

Optional flags:

```powershell
python artifact_loader.py --horizons 30,60 --fee-rate 0.01 --target-coverage 0.90
```

Notes:
- Build artifacts offline, then deploy only the app + `artifacts/` to Streamlit Community Cloud.
- The app is designed to read artifacts only (no in-app training).
