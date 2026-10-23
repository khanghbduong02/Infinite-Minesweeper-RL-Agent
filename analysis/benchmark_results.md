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

A first attempt to improve pure policies changed the loss emphasis, not the safety logic: PPO gained a selectable spatial CNN extractor and adjustable mine-hit penalty, and DQN gained the same penalty plus lower-epsilon fine-tuning controls. This was not an improvement in the measured runs. On a separate 500-board set (seeds 1000–1499), PPO MLP 9×9/50k averaged 197.3 cells (mine-hit 1.000), while the new PPO CNN 9×9/50k model with penalty 50 averaged 182.1 (1.000). The 100-episode DQN fine-tune with penalty 50 averaged 258.5 raw cells versus 224.6 for the original checkpoint, but on 500 boards it averaged 258.5 versus the original 219.7 and still hit a mine in every episode. Do not treat the smaller 100-episode hybrid follow-up (fine-tuned 1,628.8 vs original 1,370.0, seed range 2000–2099) as decisive; it is only a pilot, not a matched 500-board evaluation.

A separate safety-mode test asks how long each policy can reveal cells without guessing. The earlier runtime shield used only single-clue rules, so it could pause despite a safe move implied by overlapping clues. The play path now runs exact constraint DP (memoized residual clue-count states) before selecting: it flags mines with posterior probability 1, takes posterior-zero cells even when multiple clues are needed, then otherwise chooses the least-risk move across the frontier and unconstrained cells. In `--safe-only`, the same risk estimate pauses if all cells have nonzero probability; clicking Continue explicitly disables the shield and resumes the lowest-risk guess. Infinite-play viewer mode has no action/reveal budget unless `--moves` is supplied; it ends on mine hit or user Pause. This still is not a guarantee of winning: some boards require a guess, exact components above 25 cells or the DP state cap use conservative approximations, and the solver trusts observed clues and flags.

Historical tests (seeds 3000–3099) measured the old single-clue shield: DQN 15×15 averaged 1,414 revealed cells before budget/abstention, pausing on uncertainty in 19% of boards; fine-tuned DQN 1,405.9 / 24%; PPO MLP 1,136.5 / 38%; PPO CNN 438.0 / 98%. Those rates are not measurements of the new DP solver. Pure policy runs still show mine hits in nearly all episodes (original/fine-tuned DQN means 219.7/258.5; PPO MLP/CNN 197.3/182.1), so use the solver-policy hybrid rather than a raw network. These are single-training-seed experiments, not proof of architectural superiority.

After replacing brute-force assignment enumeration and single-clue-only filtering with memoized constraint DP, a fresh 100-episode CSP sample (seeds 0–99, density 0.16, the evaluator's 300-action cap) averaged 1,297.6 revealed cells, median 1,352.5, max 2,077, and 12.6 guesses per board; 99.2% of guesses survived. This is a small smoke benchmark, not a matched claim of improvement over the earlier 500-board runs.

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
- `ppo_mines_9x9_cnn_50k.zip`: experimental PPO CNN run, seed 41, 50k steps, mine penalty 50 (did not improve the matched test score).
- `dqn_cnn_flags_15x15_penalty50_100.npz`: experimental 100-episode DQN fine-tune, seed 41, mine penalty 50 (did not improve pure-policy mine survival).
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
python -m experiments.eval --policy dqn --model dqn_cnn_flags_15x15_context_200.npz --episodes 500 --density 0.16 --safe-only
python -m experiments.play --agent dqn --model dqn_cnn_flags_15x15_context_200.npz --safe-only
python -m pytest -q
```

Regression tests cover CNN warm-start across window sizes, mine-penalty wiring, and selection/abstention around logically forced-safe moves; MLP weights still require matching dimensions. `--safe-only` never guesses, but its guarantee assumes clue/flag state is correct and intentionally pauses instead of solving uncertain positions.
