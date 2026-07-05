"""Evaluation: strategy metrics and walk-forward backtesting of the RL
agent against the threshold baseline and the perfect-foresight bound.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.rl import features as F
from src.rl.baselines import make_threshold_policy, thresholds_from_prices
from src.rl.env import EnvData, make_env_data, rollout_policy
from src.rl.forecast import PriceForecaster
from src.rl.perfect_foresight import solve_perfect_foresight
from src.rl.train import train_ppo

logger = logging.getLogger(__name__)


def ledger_metrics(
    ledger: pd.DataFrame, capacity_kwh: float = config.BatteryConfig().capacity_kwh
) -> dict:
    daily = ledger["profit"].resample("D").sum()
    std = float(daily.std())
    sharpe = float(daily.mean() / std * np.sqrt(365.0)) if std > 0 else 0.0
    cum = ledger["cumulative_profit"]
    max_dd = float((cum.cummax() - cum).max())
    throughput = float(
        ledger["energy_bought_kwh"].sum() + ledger["energy_sold_kwh"].sum()
    )
    cycles = throughput / (2.0 * capacity_kwh) if capacity_kwh > 0 else 0.0
    total = float(cum.iloc[-1]) if len(cum) else 0.0
    return {
        "total_profit": total,
        "sharpe_daily_ann": sharpe,
        "max_drawdown": max_dd,
        "equivalent_cycles": cycles,
        "profit_per_cycle": total / cycles if cycles > 0 else 0.0,
        "n_intervals": int(len(ledger)),
    }


def rl_policy_from_model(model):
    """Adapt an SB3 model (or anything with .predict) to the rollout interface."""

    def policy(idx: int, soc: float, obs_row: np.ndarray) -> int:
        obs = np.concatenate(([np.float32(soc)], obs_row)).astype(np.float32)
        action, _ = model.predict(obs, deterministic=True)
        return int(action)

    return policy


def rl_policy_from_numpy(numpy_policy):
    def policy(idx: int, soc: float, obs_row: np.ndarray) -> int:
        obs = np.concatenate(([np.float32(soc)], obs_row))
        return numpy_policy.predict(obs)

    return policy


def prepare_fold_data(
    df30: pd.DataFrame,
    feature_frame: pd.DataFrame,
    fold: dict,
) -> tuple[EnvData, EnvData, PriceForecaster, tuple[float, float]]:
    """Fit the forecaster for one fold and assemble train/test EnvData."""
    idx = feature_frame.index
    train_rows = idx[(idx >= fold["train_start"]) & (idx < fold["train_end"])]
    cal_rows = idx[(idx >= fold["cal_start"]) & (idx < fold["cal_end"])]

    forecaster = PriceForecaster().fit(
        feature_frame, df30["rrp"], train_rows, cal_rows
    )
    span = (idx >= fold["train_start"]) & (idx < fold["test_end"])
    forecasts = forecaster.predict(feature_frame.loc[span])
    env_data = make_env_data(df30.loc[span], feature_frame.loc[span], forecasts)

    train_data = env_data.slice(fold["train_start"], fold["cal_end"])
    test_data = env_data.slice(fold["test_start"], fold["test_end"])
    thresholds = thresholds_from_prices(df30.loc[train_rows, "rrp"])
    return train_data, test_data, forecaster, thresholds


def run_walk_forward(
    df30: pd.DataFrame,
    *,
    train_days: int = 365,
    cal_days: int = 30,
    test_days: int = 90,
    step_days: int = 90,
    seeds: list[int] = (0, 1, 2),
    timesteps: int = 150_000,
    out_dir: Path = config.BACKTEST_DIR,
    max_folds: int | None = None,
) -> pd.DataFrame:
    """Walk-forward backtest. Writes fold summary + stitched ledgers to out_dir."""
    from src.walk_forward import build_time_folds

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Building feature frame over %d intervals...", len(df30))
    feature_frame = F.build_feature_frame(df30)

    folds = build_time_folds(
        df30.index,
        train_days=train_days,
        calibration_days=cal_days,
        test_days=test_days,
        step_days=step_days,
    )
    if max_folds:
        folds = folds[:max_folds]
    logger.info("Running %d folds, seeds=%s, timesteps=%d", len(folds), seeds, timesteps)

    summary_rows: list[dict] = []
    stitched: dict[str, list[pd.DataFrame]] = {}

    for fold in folds:
        fid = fold["fold_id"]
        logger.info(
            "Fold %d: test %s -> %s", fid, fold["test_start"], fold["test_end"]
        )
        train_data, test_data, _, (low, high) = prepare_fold_data(
            df30, feature_frame, fold
        )
        if len(test_data) == 0 or len(train_data) <= config.EPISODE_STEPS:
            logger.warning("Fold %d skipped: not enough data", fid)
            continue

        ledgers: dict[str, pd.DataFrame] = {}
        ledgers["threshold"] = rollout_policy(
            test_data, make_threshold_policy(test_data, low, high)
        )
        ledgers["perfect_foresight"] = solve_perfect_foresight(test_data)
        for seed in seeds:
            model = train_ppo(train_data, seed=seed, total_timesteps=timesteps)
            ledgers[f"rl_seed{seed}"] = rollout_policy(
                test_data, rl_policy_from_model(model)
            )
            del model

        for name, ledger in ledgers.items():
            metrics = ledger_metrics(ledger)
            summary_rows.append(
                {
                    "fold_id": fid,
                    "strategy": name,
                    "test_start": fold["test_start"],
                    "test_end": fold["test_end"],
                    **metrics,
                }
            )
            logger.info(
                "  %-18s profit=$%9.2f sharpe=%6.2f dd=$%8.2f cycles=%6.1f",
                name,
                metrics["total_profit"],
                metrics["sharpe_daily_ann"],
                metrics["max_drawdown"],
                metrics["equivalent_cycles"],
            )
            stitched.setdefault(name, []).append(ledger)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "wf_fold_summary.csv", index=False)

    cum_frames = {}
    for name, parts in stitched.items():
        joined = pd.concat(parts).sort_index()
        joined["cumulative_profit"] = joined["profit"].cumsum()
        joined.to_csv(out_dir / f"wf_ledger_{name}.csv")
        cum_frames[name] = joined["cumulative_profit"]
    pd.DataFrame(cum_frames).to_csv(out_dir / "wf_cumulative_pnl.csv")

    return summary
