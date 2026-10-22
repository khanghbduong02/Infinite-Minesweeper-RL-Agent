# Minesweeper RL benchmark results

## Matched comparisons

### Learned policies vs learned policies (500 episodes)

These comparisons use deterministic episode seeds 0–499, density 0.16, a 300-step budget, and 2,000-cell cap. Policies see their stated 9×9 or 15×15 observation window. Higher revealed-cell mean/median is better; mine-hit rate is the fraction ending on a mine.

| Policy | Training | Mean revealed | Median revealed | Max revealed | Mine-hit rate |
|---|---:|---:|---:|---:|---:|
| Maskable PPO, prior run | 50,000 timesteps, seed 0 | 176.1 | 136.0 | 902 | 1.000 |
| Maskable PPO, fresh-board resets | 50,000 timesteps, seed 1 | 196.1 | 157.5 | 834 | 1.000 |
| Maskable PPO, fresh-board resets | 100,000 timesteps, seed 2 | 206.0 | 152.0 | 1,499 | 1.000 |
| CNN DQN with flags | 2,000 episodes (prior run) | 214.1 | 165.0 | 1,127 | 0.988 |
| CNN DQN with flags, warm-start continuation | +500 episodes | 200.9 | 166.5 | 1,015 | 0.938 |
| CNN Double-DQN with flags | +500 episodes, seed 17 | 216.2 | 173.0 | 1,054 | 0.962 |
| CNN Double-DQN, warm-start seed 29 | +500 episodes | 209.8 | 163.5 | 1,104 | 0.944 |
| CNN Double-DQN, 15×15 context | +200 episodes, seed 31 | 243.2 | 190.0 | 1,081 | 0.996 |
| Maskable PPO, 15×15 context | 100,000 timesteps, seed 21 | 231.9 | 177.5 | 1,134 | 1.000 |
| DQN plus deterministic rules deductions | same 15×15 checkpoint, seeds 0–499 | **1,453.9** | **1,564.5** | 2,295 | 0.162 |

The standalone learned models do not beat rules/CSP. A simple hybrid—apply standard forced-mine/forced-safe deductions, use the DQN to choose among known-safe reveals and to choose when guessing is unavoidable—does. The 15×15 hybrid mean is 205.7 above rules and 149.7 above CSP on identical seeds; paired bootstrap intervals exclude zero. Standalone seed-17 DQN versus the original DQN has mean delta +2.1 cells (paired 95% CI **[-13.6, 17.6]**), not a reliable improvement.

### Learned policies vs random, rules, and CSP baselines (500 episodes)

All episodes use seeds 0–499, density 0.16, and 300 steps. Learned policies use their trained window and a 2,000-cell cap; the baseline runner has no revealed-cell cap, so the comparison favors rules/CSP. The hybrid uses only revealed clues/flags, never mine truth, then lets the DQN select among legal actions. Learned/hybrid mine-hit rates are per episode; baseline guess survival is per guess and is not directly comparable.

| Agent | Window | Mean revealed | Median revealed | Max revealed | Extra |
|---|---:|---:|---:|---:|---|
| Random baseline | n/a | 118.8 | 84.0 | 639 | 2.67 guesses/episode |
| Rules baseline | n/a | 1,248.1 | 1,297.5 | 2,127 | 0.25 guesses/episode; guess survival 0.694 |
| CSP baseline | n/a | 1,304.1 | 1,312.0 | 2,290 | 0.53 guesses/episode; guess survival 0.933 |
| PPO, fresh-board resets, 50k | 9×9 | 196.1 | 157.5 | 834 | mine-hit rate 1.000 |
| PPO, fresh-board resets, 100k | 9×9 | 206.0 | 152.0 | 1,499 | mine-hit rate 1.000 |
| Double-DQN with flags, 500 episodes, seed 17 | 9×9 | 216.2 | 173.0 | 1,054 | mine-hit rate 0.962 |
| Double-DQN with flags, 500 episodes, seed 29 | 9×9 | 209.8 | 163.5 | 1,104 | mine-hit rate 0.944 |
| PPO, 100k | 15×15 | 231.9 | 177.5 | 1,134 | mine-hit rate 1.000 |
| Double-DQN with flags, 200 episodes | 15×15 | 243.2 | 190.0 | 1,081 | mine-hit rate 0.996 |
| DQN plus deterministic rules deductions | 15×15 | **1,453.9** | **1,564.5** | 2,295 | mine-hit rate 0.162 |

Paired bootstrap 95% CIs for 500-seed mean difference (agent minus baseline): PPO-50k vs random **[65.8, 89.5]**, rules **[-1089.5, -1014.6]**, CSP **[-1139.5, -1075.8]**; DQN seed-17 vs random **[84.6, 110.9]**, rules **[-1069.4, -993.2]**, CSP **[-1118.5, -1056.2]**; PPO-15×15 vs random **[97.4, 129.8]**, rules **[-1054.7, -976.8]**, CSP **[-1105.5, -1039.2]**; raw DQN-15×15 vs random **[108.5, 141.2]**, rules **[-1043.1, -966.1]**, CSP **[-1093.6, -1027.9]**. The hybrid vs rules is **[163.8, 246.6]** and vs CSP **[104.5, 193.4]**; both exclude zero. Hybrid per-seed wins: 373/500 vs rules, 345/500 vs CSP. The unassisted learned policies beat random only; the rules-assisted DQN hybrid also beats both logic baselines on this matched 500-seed evaluation. Baselines do not have a 2,000-cell cap, so differences against the hybrid are conservative.

A likely PPO training defect was fixed: Gymnasium auto-resets with `seed=None`, and the environment previously reused the same board seed on every such reset. It now draws subsequent board seeds from Gymnasium's continuing RNG while preserving the constructor seed for the initial unseeded reset. DQN targets now use Double-DQN action selection (online network) and value evaluation (target network) to reduce overestimation bias. CNN checkpoints can now be warm-started at another odd window size because their convolutional parameters do not depend on spatial dimensions; an MLP's input/output sizes still do.

## Separate legacy reference

The existing 15×15 PPO checkpoint was evaluated separately on 200 episodes (seeds 0–199, density 0.16): mean 215.5, median 155.5, maximum 1,231, mine-hit rate 1.000. It is not directly comparable with the 9×9 results because both window size and evaluation sample differ.

## Artifacts

- `ppo_mines_9x9_varied_50k.zip`: PPO, seed 1, 50k steps, fresh-board resets.
- `ppo_mines_9x9_varied_100k.zip`: PPO, seed 2, 100k steps, fresh-board resets.
- `ppo_mines_15x15_context_100k.zip`: PPO, seed 21, 100k steps, 15×15 context.
- `dqn_cnn_flags_double_500.npz`: Double-DQN, initialized from `dqn_cnn_flags.npz`, 500 episodes, seed 17.
- `dqn_cnn_flags_double_500_widewarm.npz`: Double-DQN, initialized from the 9×9 flag checkpoint, 500 episodes, seed 29.
- `dqn_cnn_flags_15x15_context_200.npz`: 15×15 CNN Double-DQN, initialized from the 9×9 flag checkpoint, 200 episodes, seed 31.
- `analysis/benchmark_hybrid_500.log`: exact raw/hybrid comparisons, baseline results, and paired confidence intervals.
- `analysis/benchmark_hybrid.py`: rerunnable matched-seed benchmark for raw learned agents and the DQN-plus-rules hybrid.
- Earlier checkpoints retained: `ppo_mines_9x9_50k.zip`, `dqn_cnn_flags.npz`, and `dqn_cnn_flags_more.npz`.

## Reproduce

```bash
conda activate minesweeper-rl
python -m agents.train --timesteps 50000 --window 9 --density 0.16 --reward dense --seed 1 --save ppo_mines_9x9_varied_50k
python -m agents.train --timesteps 100000 --window 9 --density 0.16 --reward dense --seed 2 --save ppo_mines_9x9_varied_100k
python -m agents.train --timesteps 100000 --window 15 --density 0.16 --reward dense --seed 21 --save ppo_mines_15x15_context_100k
python -m agents.numpy_dqn --episodes 500 --window 9 --density 0.16 --net cnn --flag-actions --init-from dqn_cnn_flags.npz --save dqn_cnn_flags_double_500.npz --seed 17
python -m agents.numpy_dqn --episodes 500 --window 9 --density 0.16 --net cnn --flag-actions --init-from dqn_cnn_flags.npz --save dqn_cnn_flags_double_500_widewarm.npz --seed 29
python -m agents.numpy_dqn --episodes 200 --window 15 --density 0.16 --net cnn --flag-actions --init-from dqn_cnn_flags.npz --save dqn_cnn_flags_15x15_context_200.npz --seed 31
python -m analysis.benchmark_hybrid
python -m pytest -q
```

The full test suite passes: **25 passed**. Regression tests cover CNN warm-start across window sizes and prove the hybrid selects among visible forced-safe moves; MLP weights still require matching dimensions.
