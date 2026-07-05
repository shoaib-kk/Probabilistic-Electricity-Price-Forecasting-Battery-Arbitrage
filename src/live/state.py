"""Persisted paper-trading state: one battery per strategy plus a
high-water mark of processed intervals (idempotency across ticks)."""

from __future__ import annotations

import json
from pathlib import Path

from src import config


def default_state(strategy_names: list[str]) -> dict:
    return {
        "last_processed": None,
        "strategies": {
            name: {"soc": config.BatteryConfig().initial_soc, "cumulative_profit": 0.0}
            for name in strategy_names
        },
    }


def load_state(path: Path = config.STATE_JSON, strategy_names: list[str] | None = None) -> dict:
    if Path(path).exists():
        state = json.loads(Path(path).read_text())
        # New strategies (e.g. after retraining with more seeds) start fresh.
        for name in strategy_names or []:
            state["strategies"].setdefault(
                name,
                {"soc": config.BatteryConfig().initial_soc, "cumulative_profit": 0.0},
            )
        return state
    return default_state(strategy_names or [])


def save_state(state: dict, path: Path = config.STATE_JSON) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(state, indent=2, default=str))
