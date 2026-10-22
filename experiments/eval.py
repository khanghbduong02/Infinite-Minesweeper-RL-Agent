"""Evaluate baselines: mean/median revealed, survival, guess stats.

Usage:
    conda activate minesweeper-rl
    python -m experiments.eval --episodes 200 --density 0.16
"""
from __future__ import annotations

import argparse
import random
import statistics

import numpy as np

from agents.policies import load_policy
from env.gym_wrapper import MinesweeperWindowEnv
from env.mines import InfiniteBoard
from solvers.baselines import random_move, rules_move, csp_move
from solvers.common import trivial_deductions


def run_episodes(agent: str, episodes: int, p: float, seed0: int = 0,
                 max_steps: int = 300):
    rng = random.Random(seed0)
    lengths, guesses, guess_wins = [], [], []
    for ep in range(episodes):
        b = InfiniteBoard(seed=seed0 + ep, p=p, max_steps=max_steps)
        b.reveal(0, 0)
        g = 0
        while not b.dead and not b.over_budget():
            if agent == "random":
                c = random_move(b, rng)
                was_guess = True
            elif agent == "rules":
                c, was_guess = rules_move(b, rng)
            else:
                c, was_guess, _ = csp_move(b, rng)
            g += was_guess
            _, hit = b.reveal(*c)
            if hit:
                guess_wins.append(0)
                break
            if was_guess:
                guess_wins.append(1)
        lengths.append(len(b.revealed))
        guesses.append(g)
    return lengths, guesses, guess_wins


def deduction_override(env, obs, act):
    """Use model scores only among forced-safe reveals; use it normally to guess."""
    safe, mines = trivial_deductions(env.board.revealed, env.board.flags)
    for cell in mines:
        env.board.flag(*cell)
    safe, _ = trivial_deductions(env.board.revealed, env.board.flags)
    safe = {cell for cell in safe
            if cell not in env.board.revealed and cell not in env.board.flags}
    obs = env._obs()
    mask = obs["action_mask"].astype(bool)
    if safe:
        safe_mask = np.zeros_like(mask)
        for action in range(env.n_cells):
            if env._coords(action) in safe:
                safe_mask[action] = True
        if (mask & safe_mask).any():
            mask &= safe_mask
    return act(obs, mask)


def run_policy_episodes(act, window: int, episodes: int, p: float, seed0: int = 0,
                        max_steps: int = 300, max_revealed: int = 2000,
                        flag_actions: bool = False, deduction_override_enabled: bool = False):
    """Same boards and budget as run_episodes, driven by a trained policy."""
    env = MinesweeperWindowEnv(window_size=window, p=p, max_steps=max_steps,
                               max_revealed=max_revealed, flag_actions=flag_actions)
    lengths, hits = [], 0
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed0 + ep)
        term = trunc = False
        while not (term or trunc):
            action = (deduction_override(env, obs, act) if deduction_override_enabled
                      else act(obs, obs["action_mask"].astype(bool)))
            obs, _, term, trunc, _ = env.step(action)
        lengths.append(len(env.board.revealed))
        hits += term
    return lengths, hits / episodes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--density", type=float, default=0.16)
    ap.add_argument("--policy", default="all",
                    choices=["all", "random", "rules", "csp", "ppo", "dqn"],
                    help="'all' runs the three baselines; ppo/dqn need --model")
    ap.add_argument("--model", type=str, default=None,
                    help="ppo: saved MaskablePPO model; dqn: .npz weights")
    ap.add_argument("--deduction-override", action="store_true",
                    help="let the model break ties among forced-safe reveals and choose guesses otherwise")
    args = ap.parse_args()
    if args.policy in ("ppo", "dqn"):
        if not args.model:
            ap.error(f"--policy {args.policy} needs --model PATH")
        loaded_policy = load_policy(args.policy, args.model)
        act, window = loaded_policy[:2]
        flag_actions = loaded_policy[2] if len(loaded_policy) == 3 else False
        L, hit_rate = run_policy_episodes(
            act, window, args.episodes, args.density, flag_actions=flag_actions,
            deduction_override_enabled=args.deduction_override)
        print(f"{args.policy:6s} mean={statistics.mean(L):7.1f} "
              f"median={statistics.median(L):7.1f} max={max(L):5d} "
              f"mine_hit_rate={hit_rate:.3f} window={window} "
              f"deduction_override={args.deduction_override}")
        return
    agents = ("random", "rules", "csp") if args.policy == "all" else (args.policy,)
    for agent in agents:
        L, G, W = run_episodes(agent, args.episodes, args.density)
        print(f"{agent:6s} mean={statistics.mean(L):7.1f} "
              f"median={statistics.median(L):7.1f} max={max(L):5d} "
              f"avg_guesses={statistics.mean(G):5.1f} "
              f"guess_survive={statistics.mean(W) if W else 0:.3f}")


if __name__ == "__main__":
    main()
