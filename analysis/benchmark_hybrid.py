"""Compare PPO/DQN policies and the DQN-plus-rules hybrid on matched seeds."""
from __future__ import annotations

import numpy as np

from agents.policies import load_policy
from experiments.eval import run_episodes, run_policy_episodes


def main():
    rng = np.random.default_rng(20261021)
    models = {}
    policies = (
        ("PPO 9x9 50k", "ppo", "ppo_mines_9x9_varied_50k.zip", False),
        ("DQN 9x9 seed 17", "dqn", "dqn_cnn_flags_double_500.npz", False),
        ("PPO 15x15 100k", "ppo", "ppo_mines_15x15_context_100k.zip", False),
        ("DQN 15x15 raw", "dqn", "dqn_cnn_flags_15x15_context_200.npz", False),
        ("DQN 15x15 + rules", "dqn", "dqn_cnn_flags_15x15_context_200.npz", True),
    )
    for name, kind, path, hybrid in policies:
        loaded = load_policy(kind, path)
        act, window = loaded[:2]
        flags = loaded[2] if len(loaded) == 3 else False
        lengths, hit = run_policy_episodes(
            act, window, 500, 0.16, flag_actions=flags,
            deduction_override_enabled=hybrid,
        )
        models[name] = np.asarray(lengths)
        print(f"{name}{' + rules' if hybrid else ''}: mean={np.mean(lengths):.1f} "
              f"median={np.median(lengths):.1f} "
              f"max={max(lengths)} mine_hit={hit:.3f}", flush=True)

    baselines = {}
    for name in ("random", "rules", "csp"):
        lengths, _, _ = run_episodes(name, 500, 0.16, seed0=0)
        baselines[name] = np.asarray(lengths)
        print(f"{name}: mean={np.mean(lengths):.1f} median={np.median(lengths):.1f} "
              f"max={max(lengths)}", flush=True)

    for name, lengths in models.items():
        for baseline, values in baselines.items():
            delta = lengths - values
            boot = delta[rng.integers(0, 500, size=(20_000, 500))].mean(1)
            low, high = np.quantile(boot, [0.025, 0.975])
            print(f"{name} - {baseline}: delta={delta.mean():.1f} "
                  f"paired_95%_CI=[{low:.1f}, {high:.1f}] "
                  f"win={np.sum(delta > 0)}/500", flush=True)


if __name__ == "__main__":
    main()
