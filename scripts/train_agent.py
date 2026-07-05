"""Train and export the deployed models for the live paper-trading loop.

Window layout (ending at the last row of data/dataset_30min.csv):
    [ train (365d) ][ cal (30d) ][ validation (30d) ]

- Forecaster: fit on train, conformally calibrated on cal.
- PPO: one agent per seed, trained on train+cal, evaluated on validation.
- Exports to models/deployed/: forecaster.joblib, policy_seed{k}.npz
  (numpy inference, parity-checked against SB3), meta.json, and
  val_summary.csv. Also seeds the live price cache with recent history so
  the first live tick has enough data to build features.

Usage: python scripts/train_agent.py [--seeds 5] [--timesteps 300000]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("train_agent")

from src import config
from src.rl import features as F
from src.rl.baselines import thresholds_from_prices
from src.rl.env import make_env_data, rollout_policy
from src.rl.evaluate import ledger_metrics, rl_policy_from_model
from src.rl.forecast import PriceForecaster
from src.rl.policy_export import NumpyPolicy, check_parity, export_policy
from src.rl.train import train_ppo


def seed_live_price_cache() -> None:
    """Prefill the live cache with recent 5-minute rows from the merged CSV."""
    df = pd.read_csv(
        config.MERGED_PRICE_CSV,
        usecols=["SETTLEMENTDATE", "RRP", "TOTALDEMAND"],
        parse_dates=["SETTLEMENTDATE"],
    ).rename(
        columns={"SETTLEMENTDATE": "settlement", "RRP": "rrp", "TOTALDEMAND": "demand"}
    )
    cutoff = df["settlement"].max() - pd.Timedelta(days=config.PRICE_CACHE_DAYS)
    recent = df[df["settlement"] > cutoff]
    config.PRICE_CACHE_CSV.parent.mkdir(parents=True, exist_ok=True)
    recent.to_csv(config.PRICE_CACHE_CSV, index=False)
    logger.info("Seeded live price cache: %d rows to %s",
                len(recent), recent["settlement"].max())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--train-days", type=int, default=365)
    parser.add_argument("--cal-days", type=int, default=30)
    parser.add_argument("--val-days", type=int, default=30)
    args = parser.parse_args()

    df30 = pd.read_csv(config.DATASET_30MIN_CSV, index_col=0, parse_dates=True)
    end = df30.index.max()
    val_start = end - pd.Timedelta(days=args.val_days)
    cal_start = val_start - pd.Timedelta(days=args.cal_days)
    train_start = cal_start - pd.Timedelta(days=args.train_days)
    logger.info("Windows: train %s | cal %s | val %s -> %s",
                train_start, cal_start, val_start, end)

    feature_frame = F.build_feature_frame(df30)
    idx = feature_frame.index
    train_rows = idx[(idx >= train_start) & (idx < cal_start)]
    cal_rows = idx[(idx >= cal_start) & (idx < val_start)]

    forecaster = PriceForecaster().fit(feature_frame, df30["rrp"], train_rows, cal_rows)

    span = idx >= train_start
    forecasts = forecaster.predict(feature_frame.loc[span])
    env_data = make_env_data(df30.loc[span], feature_frame.loc[span], forecasts)
    train_data = env_data.slice(train_start, val_start)  # train + cal for the agent
    val_data = env_data.slice(val_start, end + pd.Timedelta(minutes=30))
    logger.info("EnvData: train=%d val=%d rows", len(train_data), len(val_data))

    config.DEPLOYED_DIR.mkdir(parents=True, exist_ok=True)
    val_metrics: dict[int, dict] = {}
    for seed in range(args.seeds):
        logger.info("Training PPO seed %d (%d steps)...", seed, args.timesteps)
        model = train_ppo(train_data, seed=seed, total_timesteps=args.timesteps)
        ledger = rollout_policy(val_data, rl_policy_from_model(model))
        val_metrics[seed] = ledger_metrics(ledger)
        logger.info("  seed %d val profit=$%.2f sharpe=%.2f",
                    seed, val_metrics[seed]["total_profit"],
                    val_metrics[seed]["sharpe_daily_ann"])
        path = config.DEPLOYED_DIR / f"policy_seed{seed}.npz"
        export_policy(model, path)
        check_parity(model, NumpyPolicy.load(path),
                     obs_dim=train_data.obs.shape[1] + 1)
        del model

    profits = {s: m["total_profit"] for s, m in val_metrics.items()}
    ranked = sorted(profits, key=profits.get)
    primary_seed = ranked[len(ranked) // 2]  # median seed: no cherry-picking
    low, high = thresholds_from_prices(df30.loc[train_rows, "rrp"])

    forecaster.save(config.DEPLOYED_DIR / "forecaster.joblib")
    meta = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_end": str(end),
        "seeds": list(range(args.seeds)),
        "primary_seed": primary_seed,
        "timesteps": args.timesteps,
        "thresholds": {"low_mwh": low, "high_mwh": high},
        "obs_columns": F.obs_column_names(),
        "val_window": [str(val_start), str(end)],
        "val_metrics": {str(k): v for k, v in val_metrics.items()},
    }
    (config.DEPLOYED_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    pd.DataFrame(val_metrics).T.rename_axis("seed").to_csv(
        config.DEPLOYED_DIR / "val_summary.csv"
    )
    seed_live_price_cache()
    logger.info("Deployment bundle written to %s (primary seed %d)",
                config.DEPLOYED_DIR, primary_seed)


if __name__ == "__main__":
    main()
