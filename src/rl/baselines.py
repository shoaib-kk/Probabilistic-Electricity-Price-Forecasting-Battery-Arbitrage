"""Rule-based baseline policies, expressed in the same rollout interface
as the RL agent so all strategies share one settlement code path.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.rl.env import EnvData

ACTION_CHARGE_FULL = 0
ACTION_HOLD = 2
ACTION_DISCHARGE_FULL = 4


def thresholds_from_prices(
    train_prices_mwh: pd.Series, low_quantile: float = 0.3, high_quantile: float = 0.7
) -> tuple[float, float]:
    return (
        float(train_prices_mwh.quantile(low_quantile)),
        float(train_prices_mwh.quantile(high_quantile)),
    )


def make_threshold_policy(data: EnvData, low_mwh: float, high_mwh: float):
    """Charge below the low threshold, discharge above the high threshold.

    Decides on the last COMPLETED interval's price (the settling interval's
    price is not known at decision time), mirroring the repo's original
    threshold strategy but without its price-known-at-decision shortcut.
    """

    def policy(idx: int, soc: float, obs_row: np.ndarray) -> int:
        p = data.price_lag1_mwh[idx]
        if np.isnan(p):
            return ACTION_HOLD
        if p < low_mwh:
            return ACTION_CHARGE_FULL
        if p > high_mwh:
            return ACTION_DISCHARGE_FULL
        return ACTION_HOLD

    return policy
