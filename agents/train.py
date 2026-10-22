"""Phase 3/4: PPO with action masking on the window env.

Requires torch + sb3 + sb3-contrib (installing in minesweeper-rl env).
Usage:
    conda activate minesweeper-rl
    python -m agents.train --timesteps 200000 --window 15 --density 0.16
"""
from __future__ import annotations

import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=200_000)
    ap.add_argument("--window", type=int, default=15)
    ap.add_argument("--density", type=float, default=0.16)
    ap.add_argument("--reward", type=str, default="dense",
                    choices=["dense", "cascade", "sparse"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", type=str, default="ppo_mines.pt")
    args = ap.parse_args()

    try:
        from sb3_contrib import MaskablePPO
    except ImportError as e:
        raise SystemExit(
            "sb3-contrib not installed yet in this env. Wait for the "
            "background pip install or run: pip install torch stable-baselines3 "
            "sb3-contrib --index-url (cpu torch)") from e

    from env.gym_wrapper import MinesweeperWindowEnv

    env = MinesweeperWindowEnv(window_size=args.window, p=args.density,
                               seed=args.seed, reward_mode=args.reward)
    model = MaskablePPO("MultiInputPolicy", env, verbose=1, seed=args.seed)
    model.learn(total_timesteps=args.timesteps)
    model.save(args.save)
    print(f"saved -> {args.save}")


if __name__ == "__main__":
    main()
