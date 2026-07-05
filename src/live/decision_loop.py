"""One tick of the live paper-trading loop.

Every scheduled run: fetch latest AEMO prices and solar forecast, rebuild
features, let each deployed strategy (RL seeds + threshold baseline) act
on every not-yet-processed 30-minute interval, settle P&L, and append to
the ledger. Missed runs are caught up interval by interval using only
information that predates each interval, so late execution never creates
lookahead. If feature history is insufficient, strategies hold.

Inference dependencies are deliberately light: numpy policies (no torch),
LightGBM forecaster, pandas.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.Battery_model import Battery
from src.data_feeds.aemo_live import resample_to_30min, update_price_cache
from src.data_feeds.weather import hourly_to_30min, update_weather_cache
from src.live.ledger import append_rows
from src.live.state import load_state, save_state
from src.rl import features as F
from src.rl.env import settle_interval
from src.rl.forecast import PriceForecaster
from src.rl.policy_export import NumpyPolicy

logger = logging.getLogger(__name__)

ACTION_CHARGE_FULL, ACTION_HOLD, ACTION_DISCHARGE_FULL = 0, 2, 4


def load_deployment(deploy_dir: Path = config.DEPLOYED_DIR):
    meta = json.loads((deploy_dir / "meta.json").read_text())
    forecaster = PriceForecaster.load(deploy_dir / "forecaster.joblib")
    policies = {
        f"rl_seed{s}": NumpyPolicy.load(deploy_dir / f"policy_seed{s}.npz")
        for s in meta["seeds"]
    }
    return meta, forecaster, policies


def build_market_frame(
    prices_5min: pd.DataFrame, weather_hourly: pd.DataFrame
) -> pd.DataFrame:
    df30 = resample_to_30min(prices_5min)
    if len(weather_hourly):
        ghi30 = hourly_to_30min(weather_hourly)
        df30["ghi"] = ghi30.reindex(df30.index).interpolate(limit=4).fillna(0.0)
    else:
        logger.warning("No weather data available; using ghi=0")
        df30["ghi"] = 0.0
    df30["demand"] = df30["demand"].ffill()
    return df30.dropna(subset=["rrp", "demand"])


def extend_with_future_ghi(
    df30: pd.DataFrame, weather_hourly: pd.DataFrame, steps: int = 6
) -> pd.DataFrame:
    """Append future rows carrying only forecast GHI (rrp/demand NaN).

    The ghi_fut_3h feature at the newest settled interval needs irradiance
    for the NEXT few intervals; that is a weather forecast, which is
    legitimately available at decision time. Without this, the frontier
    row always has NaN features and the agent could never act live.
    """
    future_idx = pd.date_range(
        df30.index[-1] + pd.Timedelta(minutes=30), periods=steps, freq="30min"
    )
    future = pd.DataFrame(index=future_idx, columns=df30.columns, dtype=float)
    if len(weather_hourly):
        ghi30 = hourly_to_30min(weather_hourly)
        future["ghi"] = ghi30.reindex(future_idx).interpolate(limit=4).fillna(0.0)
    else:
        future["ghi"] = 0.0
    return pd.concat([df30, future])


def run_tick(deploy_dir: Path = config.DEPLOYED_DIR) -> dict:
    meta, forecaster, policies = load_deployment(deploy_dir)
    strategy_names = list(policies) + ["threshold"]
    low_thr = meta["thresholds"]["low_mwh"]
    high_thr = meta["thresholds"]["high_mwh"]

    prices = update_price_cache()
    if prices.empty:
        logger.error("No price data available (fetch failed, cache empty)")
        return {"status": "no_data", "intervals_processed": 0}
    weather = update_weather_cache()
    df30 = build_market_frame(prices, weather)
    if df30.empty:
        return {"status": "no_complete_intervals", "intervals_processed": 0}

    state = load_state(strategy_names=strategy_names)
    last = pd.Timestamp(state["last_processed"]) if state["last_processed"] else None
    if last is not None:
        pending = df30.index[df30.index > last]
    else:
        pending = df30.index[-1:]  # first ever tick: start from the latest interval
    pending = pending[-config.MAX_CATCHUP_INTERVALS:]
    if len(pending) == 0:
        return {"status": "up_to_date", "intervals_processed": 0}

    features = F.build_feature_frame(extend_with_future_ghi(df30, weather))
    forecasts = forecaster.predict(features)
    obs_mat, obs_idx = F.build_obs_matrix(features, forecasts)
    obs_pos = {ts: i for i, ts in enumerate(obs_idx)}
    lag1 = df30["rrp"].shift(1)
    battery_cfg = config.BatteryConfig()
    costs = config.CostConfig()
    decided_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    rows: list[dict] = []
    for T in pending:
        price = float(df30.loc[T, "rrp"])
        obs_row = obs_mat[obs_pos[T]] if T in obs_pos else None
        fc = forecasts.loc[T] if T in forecasts.index else None
        for name in strategy_names:
            st = state["strategies"][name]
            if obs_row is None:
                action_idx = ACTION_HOLD  # not enough history to act honestly
            elif name == "threshold":
                p = float(lag1.loc[T]) if pd.notna(lag1.loc[T]) else None
                if p is None:
                    action_idx = ACTION_HOLD
                elif p < low_thr:
                    action_idx = ACTION_CHARGE_FULL
                elif p > high_thr:
                    action_idx = ACTION_DISCHARGE_FULL
                else:
                    action_idx = ACTION_HOLD
            else:
                obs = np.concatenate(([np.float32(st["soc"])], obs_row))
                action_idx = policies[name].predict(obs)

            battery = Battery(
                capacity_kwh=battery_cfg.capacity_kwh,
                max_power_kw=battery_cfg.max_power_kw,
                charge_efficiency=battery_cfg.charge_efficiency,
                discharge_efficiency=battery_cfg.discharge_efficiency,
                initial_soc=st["soc"],
            )
            result = settle_interval(battery, action_idx, price, costs)
            st["soc"] = result["soc"]
            st["cumulative_profit"] += result["profit"]
            rows.append(
                {
                    "interval_start": T,
                    "strategy": name,
                    "action": result["action"],
                    "power_kw": result["power_kw"],
                    "energy_bought_kwh": result["energy_bought_kwh"],
                    "energy_sold_kwh": result["energy_sold_kwh"],
                    "soc": result["soc"],
                    "price_mwh": price,
                    "profit": result["profit"],
                    "cumulative_profit": st["cumulative_profit"],
                    "h1_q05": None if fc is None else fc.get("h1_q05"),
                    "h1_q50": None if fc is None else fc.get("h1_q50"),
                    "h1_q95": None if fc is None else fc.get("h1_q95"),
                    "decided_at_utc": decided_at,
                    "catch_up": bool(T != pending[-1]),
                }
            )
        state["last_processed"] = str(T)

    append_rows(rows)
    save_state(state)
    latest = {r["strategy"]: r["action"] for r in rows if r["interval_start"] == pending[-1]}
    summary = {
        "status": "ok",
        "intervals_processed": int(len(pending)),
        "latest_interval": str(pending[-1]),
        "latest_actions": latest,
    }
    logger.info("Tick complete: %s", summary)
    return summary
