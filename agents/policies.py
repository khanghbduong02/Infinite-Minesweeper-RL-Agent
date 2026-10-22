"""Load trained PPO / NumPy-DQN checkpoints as action functions.

An action function has signature act(obs, mask) -> int. load_policy returns
(act, window_size, flag_actions) for both policy types.
"""
from __future__ import annotations

import numpy as np


def load_policy(kind: str, path: str):
    """Return (act, window_size), plus flag_actions for flag-capable DQNs."""
    if kind == "ppo":
        from sb3_contrib import MaskablePPO

        model = MaskablePPO.load(path, device="cpu")
        window = model.observation_space["obs"].shape[1]

        def act(obs, mask):
            a, _ = model.predict(obs, action_masks=mask, deterministic=True)
            return int(a)

        return act, window

    if kind == "dqn":
        from agents.numpy_dqn import CNN, MLP, arch_of

        d = np.load(path)
        window = int(d["window"])
        rng = np.random.default_rng(0)
        flag_actions = bool(d["flag_actions"]) if "flag_actions" in d.files else False
        if arch_of(d) == "cnn":
            q = CNN(window, d["c1b"].shape[0], rng, flag_actions)
        else:
            q = MLP(d["W1"].shape[0], d["W1"].shape[1], d["W2"].shape[1], rng,
                    flag_actions)
        q.set_params([d[n] for n in q.NAMES])

        def act(obs, mask):
            qs, _ = q.forward(obs["obs"].ravel().astype(np.float64)[None, :])
            return int(np.argmax(np.where(mask, qs[0], -1e9)))

        return (act, window, flag_actions) if flag_actions else (act, window)

    raise ValueError(f"unknown policy kind: {kind!r} (expected 'ppo' or 'dqn')")
