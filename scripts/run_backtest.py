"""Walk-forward backtest: RL agent vs threshold baseline vs perfect foresight.

Writes to artifacts/backtest/:
    wf_fold_summary.csv     per-fold, per-strategy (per-seed for RL) metrics
    wf_ledger_<name>.csv    stitched per-interval ledgers over all test windows
    wf_cumulative_pnl.csv   stitched cumulative P&L per strategy
    wf_cumulative_pnl.png / wf_fold_profits.png  report figures

Usage: python scripts/run_backtest.py [--seeds 3] [--timesteps 150000]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("run_backtest")

from src import config
from src.rl.evaluate import run_walk_forward


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--timesteps", type=int, default=150_000)
    parser.add_argument("--train-days", type=int, default=365)
    parser.add_argument("--cal-days", type=int, default=30)
    parser.add_argument("--test-days", type=int, default=90)
    parser.add_argument("--step-days", type=int, default=90)
    parser.add_argument("--max-folds", type=int, default=None)
    args = parser.parse_args()

    df30 = pd.read_csv(config.DATASET_30MIN_CSV, index_col=0, parse_dates=True)
    summary = run_walk_forward(
        df30,
        train_days=args.train_days,
        cal_days=args.cal_days,
        test_days=args.test_days,
        step_days=args.step_days,
        seeds=list(range(args.seeds)),
        timesteps=args.timesteps,
        max_folds=args.max_folds,
    )

    # Report figures (same entity colors as the dashboard).
    import matplotlib

    matplotlib.use("Agg")
    from app.live_dashboard import plot_backtest_cum, plot_fold_profits

    cum = pd.read_csv(
        config.BACKTEST_DIR / "wf_cumulative_pnl.csv", index_col=0, parse_dates=True
    )
    plot_backtest_cum(cum).savefig(
        config.BACKTEST_DIR / "wf_cumulative_pnl.png", dpi=150
    )
    plot_fold_profits(summary).savefig(
        config.BACKTEST_DIR / "wf_fold_profits.png", dpi=150
    )

    rl = summary[summary["strategy"].str.startswith("rl_seed")]
    per_seed = rl.groupby("strategy")["total_profit"].sum()
    thr = summary.loc[summary["strategy"] == "threshold", "total_profit"].sum()
    pf = summary.loc[summary["strategy"] == "perfect_foresight", "total_profit"].sum()
    logger.info("=== Walk-forward totals over all test windows ===")
    logger.info("RL agent : $%.2f mean, $%.2f std across %d seeds",
                per_seed.mean(), per_seed.std(), len(per_seed))
    logger.info("Threshold: $%.2f", thr)
    logger.info("Perfect foresight bound: $%.2f", pf)


if __name__ == "__main__":
    main()
