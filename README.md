# Infinite Minesweeper RL Agent

Train a reinforcement learning agent to play Minesweeper on an **unbounded grid**, and test whether it can beat a classical constraint-solver baseline, especially on forced guesses.

**Difficulty: 1/3 (easiest)** | **Compute: CPU is enough, GPU optional**

---

## 1. Motivation

Standard Minesweeper is mostly solved by constraint reasoning: a solver deduces safe cells and mines from the numbers, and computes mine probabilities when it must guess. Infinite Minesweeper removes the board edge, so the agent must:

- act on a **local view** of a world with no global boundary,
- decide **where to explore next** (which frontier to open),
- handle **forced guesses**, where the quality of the guess decides the run,
- generalize across a **very large state space** instead of memorizing a fixed board size.

The question is where learning adds value over exact reasoning.

## 2. Research Questions

1. Can an RL agent match a constraint-satisfaction (CSP) solver on *deterministic* moves (safe/mine deductions)?
2. Can it match or beat the solver's **guessing policy**? The baseline guesses the cell with the lowest exact mine probability. Can RL learn useful long-term guessing, such as picking guesses that open up more information afterwards?
3. Does a policy trained on a local window (e.g. 15x15) generalize to **larger and denser** worlds than it saw in training?

## 3. Game Definition

- Grid: unbounded 2D, generated lazily in chunks (e.g. 16x16). Each cell is a mine with probability `p` (default ~0.15-0.20), decided deterministically from `(seed, x, y)`.
- First click and its neighbors are always safe.
- Actions: reveal a cell, flag a cell.
- Episode ends on hitting a mine, or after a step/cell budget.

> Pin down exactly how "infinite Minesweeper" is defined in your version (mine density, safe-start rule, whether flags are required) and keep it fixed. Write it in `docs/rules.md`.

## 4. RL Formulation

| Component | Design |
|---|---|
| Agent | CNN policy over a local window |
| Observation | `(C, H, W)` window centered on the agent's focus point. Channels: hidden, revealed number (one-hot 0-8), flagged, out-of-window mask |
| Actions | Choose a cell inside the window to reveal (`H*W` actions) with action masking for already-revealed cells. Optionally a "shift window" action |
| Reward | +small for each newly revealed safe cell, +larger for cells opened via a zero cascade, large negative on mine hit, small step cost. Compare against sparse "survive N cells" reward |
| Objective | Maximize safe cells revealed before hitting a mine (survival length) |

Key design choice: **window + masking** keeps the action space finite. Compare this against a "frontier-selection" action space (pick one of the frontier cells).

## 5. Approach

- Environment: custom Gymnasium env, written from scratch, with chunked lazy generation.
- Algorithms: PPO with action masking (Stable-Baselines3 / sb3-contrib `MaskablePPO`), then your own PPO or DQN implementation for one comparison.
- Baselines (all must be implemented):
  1. Random legal move.
  2. Rule-based solver (trivial deductions only: "number equals flagged neighbors", "number equals hidden neighbors").
  3. CSP/probabilistic solver (enumerate consistent mine configurations on each frontier component, guess lowest probability).
  4. Optional: imitation learning from the CSP solver, then RL fine-tuning.

## 6. Experiments

| # | Experiment | Compares |
|---|---|---|
| E1 | Sanity: finite 9x9 / 16x16 board | RL vs baselines on a known benchmark |
| E2 | Infinite world, fixed density | Random vs rules vs CSP vs RL |
| E3 | Reward ablation | Dense vs sparse vs cascade-bonus |
| E4 | Observation window size | 9x9 vs 15x15 vs 21x21 |
| E5 | Generalization | Train at density p, test at p+/-0.05 |
| E6 | Guess quality | Only evaluate on states where no safe deduction exists |
| E7 | Imitation init | RL from scratch vs warm start from CSP solver |

Use at least 5 seeds for RL runs and 1000+ evaluation episodes per agent.

## 7. Evaluation

- **Primary:** mean/median safe cells revealed per episode; survival curve (fraction of episodes alive after N reveals).
- **Guess analysis:** number of forced guesses per episode, and success rate of guesses vs. the solver's theoretical probability.
- **Learning curves** with confidence intervals over seeds.
- **Qualitative:** heatmaps of the agent's action probabilities on hand-picked board states (classic 1-2-1, 1-2-2-1 patterns).
- **Success criteria:** the agent clearly beats rule-based play, approaches the CSP solver on deterministic moves, and the analysis explains where it falls short or surpasses it. A negative result with solid analysis is fine.

## 8. Roadmap

| Phase | Goal |
|---|---|
| 1 | Environment, rules doc, unit tests (cascade, chunk generation, safe start) |
| 2 | Random, rule-based and CSP baselines |
| 3 | Gymnasium wrapper, action masking, first PPO runs on finite boards |
| 4 | Infinite-world training, reward design |
| 5 | Ablations (reward, window size) |
| 6 | Generalization and guess-quality experiments |
| 7 | Imitation init, extra seeds, plots |
| 8 | Report and demo |

## 9. Risks

- Agent learns only trivial deductions and stalls. Mitigation: imitation warm start, curriculum on density.
- Long episodes make training slow. Mitigation: cap episode length, vectorized envs, NumPy-optimized reveal logic.
- CSP solver is the hardest baseline to implement. Mitigation: restrict to frontier components with a size cap.

## 10. Repository Layout

```
infinite-minesweeper-rl/
  env/            # InfiniteMinesweeperEnv, chunk generator
  solvers/        # random, rules, csp
  agents/         # PPO configs, custom PPO
  experiments/    # one script/config per experiment
  analysis/       # plots, notebooks
  docs/rules.md
  README.md
```

## 11. Initial Resources

- Schulman et al. (2017), *Proximal Policy Optimization Algorithms*
- Huang & Ontanon (2022), *A Closer Look at Invalid Action Masking in Policy Gradient Algorithms*
- Kaye (2000), *Minesweeper is NP-complete*
- Gymnasium, Stable-Baselines3, sb3-contrib (`MaskablePPO`)
- Existing Minesweeper solver write-ups, for CSP and probability computation ideas