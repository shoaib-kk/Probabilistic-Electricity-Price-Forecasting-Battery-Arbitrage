"""PPO training on the battery arbitrage environment (Stable-Baselines3)."""

from __future__ import annotations

import logging

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from src import config
from src.rl.env import BatteryArbitrageEnv, EnvData

logger = logging.getLogger(__name__)


def train_ppo(
    train_data: EnvData,
    seed: int,
    total_timesteps: int = 200_000,
    n_envs: int = 4,
    battery_cfg: config.BatteryConfig = config.BatteryConfig(),
    costs: config.CostConfig = config.CostConfig(),
    reward_scale: float = 0.1,
    verbose: int = 0,
) -> PPO:
    def make_env():
        env = BatteryArbitrageEnv(
            train_data,
            battery_cfg=battery_cfg,
            costs=costs,
            random_start=True,
            random_soc=True,
            reward_scale=reward_scale,
        )
        return Monitor(env)

    vec_env = DummyVecEnv([make_env for _ in range(n_envs)])
    model = PPO(
        "MlpPolicy",
        vec_env,
        policy_kwargs=dict(net_arch=[128, 128]),
        learning_rate=3e-4,
        n_steps=256,
        batch_size=256,
        gamma=0.995,
        gae_lambda=0.95,
        ent_coef=0.01,
        seed=seed,
        device="cpu",
        verbose=verbose,
    )
    model.learn(total_timesteps=total_timesteps, progress_bar=False)
    vec_env.close()
    return model
