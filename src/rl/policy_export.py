"""Export a trained SB3 PPO policy to plain numpy weights.

The live loop only needs deterministic argmax inference, so shipping the
MLP as an .npz lets the scheduled runner skip installing torch and
stable-baselines3 entirely. Parity with SB3's deterministic predict() is
enforced by check_parity (also exercised in tests).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def export_policy(model, path: Path) -> None:
    """Extract Linear layers from the PPO actor into an .npz file."""
    import torch  # local import: only needed at export time

    policy = model.policy
    layers = []
    for module in policy.mlp_extractor.policy_net:
        if isinstance(module, torch.nn.Linear):
            layers.append(
                (module.weight.detach().cpu().numpy(), module.bias.detach().cpu().numpy())
            )
        elif not isinstance(module, torch.nn.Tanh):
            raise ValueError(f"Unsupported layer in policy_net: {type(module)}")
    action_net = policy.action_net
    if not isinstance(action_net, torch.nn.Linear):
        raise ValueError(f"Unsupported action_net: {type(action_net)}")
    layers.append(
        (action_net.weight.detach().cpu().numpy(), action_net.bias.detach().cpu().numpy())
    )

    arrays = {}
    for i, (w, b) in enumerate(layers):
        arrays[f"W{i}"] = w
        arrays[f"b{i}"] = b
    arrays["n_layers"] = np.array(len(layers))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


class NumpyPolicy:
    """Deterministic argmax inference for an exported PPO actor.

    Hidden layers use tanh (SB3 MlpPolicy default); the final layer emits
    action logits.
    """

    def __init__(self, layers: list[tuple[np.ndarray, np.ndarray]]):
        self.layers = layers

    @staticmethod
    def load(path: Path) -> "NumpyPolicy":
        data = np.load(path)
        n = int(data["n_layers"])
        layers = [(data[f"W{i}"], data[f"b{i}"]) for i in range(n)]
        return NumpyPolicy(layers)

    def predict(self, obs: np.ndarray) -> int:
        x = np.asarray(obs, dtype=np.float64)
        for w, b in self.layers[:-1]:
            x = np.tanh(w @ x + b)
        w, b = self.layers[-1]
        logits = w @ x + b
        return int(np.argmax(logits))


def check_parity(model, numpy_policy: NumpyPolicy, obs_dim: int, n: int = 500) -> None:
    """Raise if SB3 and the numpy export disagree on any random observation."""
    rng = np.random.default_rng(0)
    obs_batch = rng.normal(0, 2, size=(n, obs_dim)).astype(np.float32)
    for obs in obs_batch:
        sb3_action, _ = model.predict(obs, deterministic=True)
        np_action = numpy_policy.predict(obs)
        if int(sb3_action) != np_action:
            raise AssertionError(
                f"Policy export mismatch: sb3={int(sb3_action)} numpy={np_action}"
            )
