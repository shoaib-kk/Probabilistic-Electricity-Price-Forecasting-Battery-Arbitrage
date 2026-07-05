"""Shared feature/state builder for the forecaster, the RL env, and the
live loop. Everything flows through this module so the state the agent
was trained on and the state it sees live cannot diverge.

Input frame convention: 30-minute intervals indexed by interval START
(naive NEM market time) with columns:
    rrp     -- volume-weighted-ish price for the interval, $/MWh
    demand  -- regional demand, MW
    ghi     -- solar irradiance for the interval, W/m^2 (exogenous
               weather forecast proxy; the only columns allowed to look
               at rows >= T are ghi-derived, because weather forecasts
               are legitimately available at decision time)

A feature row at index T may otherwise only use price/demand data from
intervals strictly before T. The decision for interval T settles at
rrp[T], which is therefore also the h=1 forecast target.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PRICE_SCALE = 100.0
GHI_SCALE = 1000.0
PCT_WINDOW = 336  # 7 days of 30-minute intervals

PRICE_LAGS = [1, 2, 3, 4, 5, 6, 12, 24, 48, 336]
ROLL_WINDOWS_MEAN = [6, 24, 48]
ROLL_WINDOWS_STD = [24, 48]


def nprice(x):
    """Symmetric log-like squash for prices in $/MWh (handles -1000..17500)."""
    return np.arcsinh(np.asarray(x, dtype=float) / PRICE_SCALE)


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Full tabular feature set for the quantile forecaster."""
    out = pd.DataFrame(index=df.index)
    rrp = df["rrp"]
    past = rrp.shift(1)

    for k in PRICE_LAGS:
        out[f"rrp_lag_{k}"] = rrp.shift(k)
    for w in ROLL_WINDOWS_MEAN:
        out[f"rrp_roll_mean_{w}"] = past.rolling(w).mean()
    for w in ROLL_WINDOWS_STD:
        out[f"rrp_roll_std_{w}"] = past.rolling(w).std()
    out["rrp_roll_min_48"] = past.rolling(48).min()
    out["rrp_roll_max_48"] = past.rolling(48).max()
    out["rrp_pct_7d"] = past.rolling(PCT_WINDOW).apply(
        lambda a: float(np.mean(a <= a[-1])), raw=True
    )

    demand = df["demand"]
    out["demand_lag_1"] = demand.shift(1)
    out["demand_lag_2"] = demand.shift(2)
    out["demand_lag_48"] = demand.shift(48)
    out["demand_roll_mean_48"] = demand.shift(1).rolling(48).mean()

    hours = df.index.hour + df.index.minute / 60.0
    out["hour_sin"] = np.sin(2 * np.pi * hours / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * hours / 24.0)
    dow = df.index.dayofweek
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    out["is_weekend"] = (dow >= 5).astype(float)

    ghi = df["ghi"]
    out["ghi"] = ghi
    out["ghi_fut_3h"] = pd.concat(
        [ghi.shift(-k) for k in range(1, 7)], axis=1
    ).mean(axis=1)

    return out


def target_for_horizon(rrp: pd.Series, horizon: int) -> pd.Series:
    """h=1 is the interval being decided (rrp[T]); h=2 is the next one."""
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    return rrp.shift(-(horizon - 1))


def forecast_columns(horizons: list[int], quantiles: list[float]) -> list[str]:
    return [f"h{h}_q{int(q * 100):02d}" for h in horizons for q in quantiles]


# Observation layout for the RL agent. The env prepends SoC as element 0.
OBS_FORECAST_COLS = forecast_columns([1, 2], [0.05, 0.5, 0.95])


def obs_column_names() -> list[str]:
    return (
        ["soc"]
        + ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend"]
        + [f"nprice_lag_{k}" for k in [1, 2, 3, 4, 5, 6]]
        + ["nprice_roll_mean_24", "nprice_roll_std_24", "rrp_pct_7d"]
        + [f"n_{c}" for c in OBS_FORECAST_COLS]
        + ["ghi_norm", "ghi_fut_norm", "ndemand"]
    )


def build_obs_matrix(
    features: pd.DataFrame, forecasts: pd.DataFrame
) -> tuple[np.ndarray, pd.Index]:
    """Normalized observation rows (without SoC), aligned on shared index.

    Returns (matrix float32 [n, obs_dim - 1], index of usable rows). Rows
    with any NaN (feature warmup, missing forecasts) are dropped.
    """
    idx = features.index.intersection(forecasts.index)
    f = features.loc[idx]
    fc = forecasts.loc[idx]

    cols = [
        f["hour_sin"],
        f["hour_cos"],
        f["dow_sin"],
        f["dow_cos"],
        f["is_weekend"],
    ]
    cols += [nprice(f[f"rrp_lag_{k}"]) for k in [1, 2, 3, 4, 5, 6]]
    cols += [
        nprice(f["rrp_roll_mean_24"]),
        np.log1p(f["rrp_roll_std_24"].clip(lower=0.0) / PRICE_SCALE),
        f["rrp_pct_7d"],
    ]
    cols += [nprice(fc[c]) for c in OBS_FORECAST_COLS]
    cols += [
        f["ghi"] / GHI_SCALE,
        f["ghi_fut_3h"] / GHI_SCALE,
        (f["demand_lag_1"] / 1000.0 - 5.5) / 1.5,
    ]

    mat = np.column_stack([np.asarray(c, dtype=float) for c in cols])
    ok = ~np.isnan(mat).any(axis=1)
    return mat[ok].astype(np.float32), idx[ok]
