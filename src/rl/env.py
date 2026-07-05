"""Gymnasium environment for battery arbitrage at 30-minute cadence.

Observation: [soc] + normalized market state row (see src.rl.features).
Action: Discrete(5) -> (charge/hold/discharge, fraction of max power).
Reward: realized interval P&L in dollars (fees + degradation included,
identical accounting to src.arbitrage_sim.run_arbitrage_simulation),
scaled by reward_scale for optimizer stability.
"""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from src import config
from src.Battery_model import Battery
from src.rl import features as F


@dataclass
class EnvData:
    """Aligned, NaN-free market data ready for the env or a policy rollout."""

    timestamps: pd.DatetimeIndex
    prices_mwh: np.ndarray  # settlement price of each interval
    price_lag1_mwh: np.ndarray  # last completed interval's price (for rules)
    obs: np.ndarray  # float32 [n, obs_dim - 1], soc excluded

    def __len__(self) -> int:
        return len(self.prices_mwh)

    def slice(self, start: pd.Timestamp, end: pd.Timestamp) -> "EnvData":
        mask = (self.timestamps >= start) & (self.timestamps < end)
        return EnvData(
            timestamps=self.timestamps[mask],
            prices_mwh=self.prices_mwh[mask],
            price_lag1_mwh=self.price_lag1_mwh[mask],
            obs=self.obs[mask],
        )


def make_env_data(
    df: pd.DataFrame, feature_frame: pd.DataFrame, forecasts: pd.DataFrame
) -> EnvData:
    """Assemble EnvData from the 30-min market frame + features + forecasts."""
    mat, idx = F.build_obs_matrix(feature_frame, forecasts)
    prices = df.loc[idx, "rrp"].to_numpy(dtype=float)
    lag1 = df["rrp"].shift(1).loc[idx].to_numpy(dtype=float)
    return EnvData(
        timestamps=pd.DatetimeIndex(idx),
        prices_mwh=prices,
        price_lag1_mwh=lag1,
        obs=mat,
    )


def settle_interval(
    battery: Battery,
    action_idx: int,
    price_mwh: float,
    costs: config.CostConfig,
    dt_hours: float = config.DT_HOURS,
) -> dict:
    """Apply one action and return the P&L breakdown for the interval."""
    verb, frac = config.ACTIONS[action_idx]
    price_kwh = price_mwh / 1000.0
    power_kw = frac * battery.max_power_kw
    if verb == "hold" or power_kw <= 0.0:
        bought, sold, soc = 0.0, 0.0, battery.soc
    else:
        bought, sold, soc = battery.step(verb, power_kw, dt_hours)
    fee = costs.fee_rate * (bought + sold) * abs(price_kwh)
    degradation = costs.degradation_cost_per_kwh * (bought + sold)
    profit = sold * price_kwh - bought * price_kwh - fee - degradation
    return {
        "action": config.ACTION_NAMES[action_idx],
        "power_kw": power_kw if verb != "hold" else 0.0,
        "energy_bought_kwh": bought,
        "energy_sold_kwh": sold,
        "soc": soc,
        "fee": fee,
        "degradation": degradation,
        "profit": profit,
    }


class BatteryArbitrageEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        data: EnvData,
        battery_cfg: config.BatteryConfig = config.BatteryConfig(),
        costs: config.CostConfig = config.CostConfig(),
        episode_steps: int = config.EPISODE_STEPS,
        random_start: bool = True,
        random_soc: bool = True,
        reward_scale: float = 0.01,
    ):
        super().__init__()
        if len(data) < episode_steps + 1:
            raise ValueError(
                f"Dataset has {len(data)} rows; need > episode_steps={episode_steps}"
            )
        self.data = data
        self.battery_cfg = battery_cfg
        self.costs = costs
        self.episode_steps = episode_steps
        self.random_start = random_start
        self.random_soc = random_soc
        self.reward_scale = reward_scale

        obs_dim = 1 + data.obs.shape[1]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(len(config.ACTIONS))
        self._battery: Battery | None = None
        self._idx = 0
        self._steps_done = 0

    def _obs(self) -> np.ndarray:
        return np.concatenate(
            ([np.float32(self._battery.soc)], self.data.obs[self._idx])
        ).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self.random_start:
            high = len(self.data) - self.episode_steps - 1
            self._idx = int(self.np_random.integers(0, high + 1))
        else:
            self._idx = 0
        soc = (
            float(self.np_random.uniform(0.2, 0.8))
            if self.random_soc
            else self.battery_cfg.initial_soc
        )
        self._battery = Battery(
            capacity_kwh=self.battery_cfg.capacity_kwh,
            max_power_kw=self.battery_cfg.max_power_kw,
            charge_efficiency=self.battery_cfg.charge_efficiency,
            discharge_efficiency=self.battery_cfg.discharge_efficiency,
            initial_soc=soc,
        )
        self._steps_done = 0
        return self._obs(), {"timestamp": self.data.timestamps[self._idx]}

    def step(self, action):
        result = settle_interval(
            self._battery, int(action), float(self.data.prices_mwh[self._idx]), self.costs
        )
        info = {
            **result,
            "timestamp": self.data.timestamps[self._idx],
            "price_mwh": float(self.data.prices_mwh[self._idx]),
        }
        self._idx += 1
        self._steps_done += 1
        terminated = (
            self._steps_done >= self.episode_steps or self._idx >= len(self.data)
        )
        reward = result["profit"] * self.reward_scale
        obs = self._obs() if not terminated else np.zeros(
            self.observation_space.shape, dtype=np.float32
        )
        return obs, reward, terminated, False, info


def rollout_policy(
    data: EnvData,
    policy_fn,
    battery_cfg: config.BatteryConfig = config.BatteryConfig(),
    costs: config.CostConfig = config.CostConfig(),
) -> pd.DataFrame:
    """Run any policy over the full dataset sequentially (evaluation mode).

    policy_fn(idx, soc, obs_row) -> action index, where obs_row is the
    normalized state row WITHOUT soc (same layout the env appends to).
    Returns a per-interval ledger DataFrame with cumulative P&L.
    """
    battery = Battery(
        capacity_kwh=battery_cfg.capacity_kwh,
        max_power_kw=battery_cfg.max_power_kw,
        charge_efficiency=battery_cfg.charge_efficiency,
        discharge_efficiency=battery_cfg.discharge_efficiency,
        initial_soc=battery_cfg.initial_soc,
    )
    records = []
    cum = 0.0
    for i in range(len(data)):
        action_idx = int(policy_fn(i, battery.soc, data.obs[i]))
        result = settle_interval(battery, action_idx, float(data.prices_mwh[i]), costs)
        cum += result["profit"]
        records.append(
            {
                "timestamp": data.timestamps[i],
                "price_mwh": float(data.prices_mwh[i]),
                **result,
                "cumulative_profit": cum,
            }
        )
    return pd.DataFrame.from_records(records).set_index("timestamp")
