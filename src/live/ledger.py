"""Append-only decision/P&L ledger for the live paper-trading loop."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src import config

LEDGER_COLUMNS = [
    "interval_start",
    "strategy",
    "action",
    "power_kw",
    "energy_bought_kwh",
    "energy_sold_kwh",
    "soc",
    "price_mwh",
    "profit",
    "cumulative_profit",
    "h1_q05",
    "h1_q50",
    "h1_q95",
    "decided_at_utc",
    "catch_up",
]


def append_rows(rows: list[dict], path: Path = config.LEDGER_CSV) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)[LEDGER_COLUMNS]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


def load_ledger(path: Path = config.LEDGER_CSV) -> pd.DataFrame:
    if not Path(path).exists():
        return pd.DataFrame(columns=LEDGER_COLUMNS)
    return pd.read_csv(path, parse_dates=["interval_start", "decided_at_utc"])
