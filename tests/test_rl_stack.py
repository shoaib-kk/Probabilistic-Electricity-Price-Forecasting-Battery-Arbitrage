"""Correctness tests for the RL battery arbitrage stack."""

import numpy as np
import pandas as pd
import pytest

from src import config
from src.data_feeds.aemo_live import _parse_mms_price_rows, resample_to_30min
from src.rl import features as F
from src.rl.baselines import make_threshold_policy
from src.rl.env import (
    BatteryArbitrageEnv,
    EnvData,
    make_env_data,
    rollout_policy,
    settle_interval,
)
from src.rl.perfect_foresight import solve_perfect_foresight
from src.Battery_model import Battery


def synthetic_frame(days: int = 21, seed: int = 0) -> pd.DataFrame:
    """30-min market frame with a daily price cycle plus noise."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=days * 48, freq="30min")
    hours = idx.hour + idx.minute / 60.0
    price = 80 + 60 * np.sin(2 * np.pi * (hours - 18) / 24) + rng.normal(0, 10, len(idx))
    demand = 5500 + 800 * np.sin(2 * np.pi * (hours - 18) / 24)
    ghi = np.clip(900 * np.sin(np.pi * (hours - 6) / 12), 0, None)
    return pd.DataFrame({"rrp": price, "demand": demand, "ghi": ghi}, index=idx)


def fake_forecasts(df: pd.DataFrame) -> pd.DataFrame:
    """Deterministic stand-in forecasts so env tests avoid LightGBM."""
    out = pd.DataFrame(index=df.index)
    for h in [1, 2]:
        base = df["rrp"].shift(1)
        out[f"h{h}_q05"] = base - 30
        out[f"h{h}_q50"] = base
        out[f"h{h}_q95"] = base + 30
    return out


@pytest.fixture(scope="module")
def env_data():
    df = synthetic_frame()
    feats = F.build_feature_frame(df)
    return make_env_data(df, feats, fake_forecasts(df))


def test_obs_layout_matches_names(env_data):
    assert env_data.obs.shape[1] + 1 == len(F.obs_column_names())
    assert not np.isnan(env_data.obs).any()


def test_env_soc_bounds_and_accounting(env_data):
    env = BatteryArbitrageEnv(env_data, episode_steps=100, random_start=False,
                              random_soc=False, reward_scale=1.0)
    obs, _ = env.reset(seed=42)
    assert obs.shape == env.observation_space.shape
    total = 0.0
    for i in range(100):
        action = [0, 4, 2, 1, 3][i % 5]
        obs, reward, terminated, truncated, info = env.step(action)
        assert 0.0 <= info["soc"] <= 1.0
        # Reward must equal the P&L identity from the info breakdown.
        price_kwh = info["price_mwh"] / 1000.0
        expected = (
            info["energy_sold_kwh"] * price_kwh
            - info["energy_bought_kwh"] * price_kwh
            - info["fee"]
            - info["degradation"]
        )
        assert reward == pytest.approx(expected, abs=1e-9)
        total += reward
        if terminated:
            break
    assert terminated


def test_hold_action_is_free(env_data):
    battery = Battery(100.0, 50.0, 0.9, 0.9, 0.5)
    result = settle_interval(battery, 2, 150.0, config.CostConfig())
    assert result["profit"] == 0.0
    assert battery.soc == 0.5


def test_no_lookahead_in_features():
    """Perturbing future prices must not change past feature rows."""
    df = synthetic_frame()
    feats_before = F.build_feature_frame(df)
    t_cut = df.index[800]
    df2 = df.copy()
    df2.loc[df2.index > t_cut, "rrp"] = 9999.0
    df2.loc[df2.index > t_cut, "demand"] = 9999.0
    feats_after = F.build_feature_frame(df2)
    ghi_cols = ["ghi", "ghi_fut_3h"]  # exogenous weather forecast, exempt
    check = [c for c in feats_before.columns if c not in ghi_cols]
    rows = feats_before.index <= t_cut
    pd.testing.assert_frame_equal(
        feats_before.loc[rows, check], feats_after.loc[rows, check]
    )


def test_perfect_foresight_dominates_threshold(env_data):
    test = env_data.slice(env_data.timestamps[400], env_data.timestamps[-1])
    pf = solve_perfect_foresight(test)
    thr = rollout_policy(test, make_threshold_policy(test, 40.0, 120.0))
    assert pf["cumulative_profit"].iloc[-1] >= thr["cumulative_profit"].iloc[-1]
    assert (pf["soc"] >= -1e-9).all() and (pf["soc"] <= 1 + 1e-9).all()
    assert pf["cumulative_profit"].iloc[-1] > 0


def test_rollout_matches_env_accounting(env_data):
    """Same fixed action sequence through env and rollout gives equal P&L."""
    env = BatteryArbitrageEnv(env_data, episode_steps=50, random_start=False,
                              random_soc=False, reward_scale=1.0)
    env.reset(seed=0)
    env_total = 0.0
    actions = [(i * 3) % 5 for i in range(50)]
    for a in actions:
        _, r, *_ = env.step(a)
        env_total += r
    ledger = rollout_policy(
        EnvData(
            timestamps=env_data.timestamps[:50],
            prices_mwh=env_data.prices_mwh[:50],
            price_lag1_mwh=env_data.price_lag1_mwh[:50],
            obs=env_data.obs[:50],
        ),
        lambda i, soc, obs: actions[i],
    )
    assert ledger["profit"].sum() == pytest.approx(env_total, abs=1e-9)


def test_resample_drops_incomplete_blocks():
    stamps = pd.date_range("2024-01-01 00:05", periods=11, freq="5min")
    df = pd.DataFrame({"settlement": stamps, "rrp": range(11), "demand": 5000.0})
    out = resample_to_30min(df)
    # 11 stamps = one complete block (00:05..00:30) + partial second block.
    assert len(out) == 1
    assert out.index[0] == pd.Timestamp("2024-01-01 00:00")
    assert out["rrp"].iloc[0] == pytest.approx(np.mean([0, 1, 2, 3, 4, 5]))


def test_mms_parser():
    raw = (
        'C,SETP.WORLD,DISPATCHIS,AEMO,PUBLIC\n'
        'I,DISPATCH,PRICE,4,SETTLEMENTDATE,RUNNO,REGIONID,INTERVENTION,RRP\n'
        'D,DISPATCH,PRICE,4,"2026/07/05 12:30:00",1,VIC1,0,88.5\n'
        'D,DISPATCH,PRICE,4,"2026/07/05 12:30:00",1,VIC1,1,99.9\n'
        'D,DISPATCH,PRICE,4,"2026/07/05 12:30:00",1,NSW1,0,120.0\n'
    ).encode()
    rows = _parse_mms_price_rows(raw, "VIC1")
    assert len(rows) == 1
    assert rows[0]["rrp"] == 88.5


def test_policy_export_parity(env_data):
    from src.rl.policy_export import NumpyPolicy, check_parity, export_policy
    from src.rl.train import train_ppo

    model = train_ppo(env_data, seed=0, total_timesteps=2048, n_envs=1)
    path = config.MODELS_DIR / "test_export.npz"
    export_policy(model, path)
    np_policy = NumpyPolicy.load(path)
    check_parity(model, np_policy, obs_dim=env_data.obs.shape[1] + 1, n=300)
    path.unlink()
