# Learning to Play Minesweeper: A Beginner’s Guide

This guide explains the agents in this repository: what reinforcement learning (RL) means here, how the game is represented, what training changes, how to run and evaluate a model, and why the strongest player combines a learned policy with Minesweeper logic.

## The short version

The board is an unbounded grid. A trained policy does **not** receive the entire board: it sees an 11-channel image of a fixed-size window around its current focus. It chooses an action in that window. During training, the game reports a reward, and the policy gradually adjusts its action values to favor actions that lead to better future rewards.

There are two different kinds of intelligence in the current hybrid:

- **The neural policy** learns preferences from experience. It can rank legal actions, but a policy trained by itself is not guaranteed to follow Minesweeper logic.
- **The constraint solver** reasons from the visible clues and flags. In learned-agent play it uses the current observation window, filters out higher-risk choices, and allows the policy to choose among the remaining actions.

The solver is not RL. Combining it with RL is useful because logical deduction is reliable for rules, while a learned policy can make a choice when multiple similarly safe options remain.

## 1. The game and its observations

`env/mines.py` implements the game. Mine locations are generated deterministically from the game seed, coordinates, and mine density `p`. That makes the same seed reproduce the same board. The starting cell and its immediate neighborhood are guaranteed safe. Revealing a zero also reveals a connected area of safe cells.

`env/gym_wrapper.py` turns the game into a Gymnasium environment for training:

- **Observation:** an `(11, H, W)` array, where `H` and `W` are the window size. The channels represent hidden cells, flags, and one channel for each clue number 0 through 8. For example, a 15×15 observation has shape `(11, 15, 15)`.
- **Action:** reveal one cell in the window. Flag-capable DQN models have a second action for flagging each cell.
- **Action mask:** a list marking which actions are currently legal. The learner and PPO use this mask so they do not intentionally choose cells that are already revealed or flagged.
- **Focus:** after a reveal, the focus moves to that cell and a new local window is observed. There is no separate “move the camera” action.

The game itself can continue beyond the observation window. The policy has to act from local information, just as a player would when looking through a small viewport.

## 2. What reinforcement learning means here

In ordinary programming, we might write explicit instructions such as “if the clue equals the number of neighboring flags, reveal the other neighbors.” In RL, we instead let an agent try actions and learn from their outcomes.

A training interaction is:

1. The environment provides an observation and legal-action mask.
2. The policy chooses an action.
3. The environment reveals a cell or places a flag, then returns a reward and the next observation.
4. Training adjusts the policy so actions associated with better long-term outcomes become more likely.

One action’s reward is not the whole objective. A move can be worthwhile because it reveals information that helps later moves. The algorithm therefore estimates **return**: the reward now plus discounted future rewards. The discount factor `gamma` controls how much later rewards matter.

A trained model is not a list of boards it has memorized. It is a set of numerical weights that maps observations to action scores. With a CNN, the same filters are used at different board positions, helping the model recognize similar local patterns in different parts of the window.

## 3. The NumPy DQN

The current practical learned-agent path is a hand-written **Deep Q-Network (DQN)** in `agents/numpy_dqn.py`. It uses NumPy rather than PyTorch for its network and training calculations.

### Q-values and action choice

The network estimates a **Q-value** for every action: an estimate of how good that action is, including its future consequences. The chosen action is the legal action with the highest Q-value.

Training also needs exploration. **Epsilon-greedy** means that with probability epsilon the agent tries a random legal action; otherwise it selects the action with the largest current Q-value. Early exploration gives the model varied experience. Epsilon decreases during training so later choices rely more on what it has learned. The command-line `--eps0` and `--eps1` set the starting and ending exploration values.

### Replay memory and stable targets

DQN stores recent transitions—observation, action, reward, next observation, and whether the episode ended—in a replay buffer. It trains on randomly sampled batches from this memory instead of only the latest action. This makes updates less dependent on the order of consecutive moves.

It maintains an online network and a slower-updated target network. This project uses **Double DQN** targets: the online network chooses the next legal action, and the target network estimates its value. This helps reduce over-optimistic values that can result from taking a maximum over noisy estimates. Adam updates the weights, and gradients are clipped before applying each update. Training updates occur every few environment steps (`--update-every`, default 4), rather than after every step.

### MLP versus CNN

- **MLP:** flattens the whole observation into a vector. It is simple, but its input and output sizes depend on the window dimensions.
- **CNN:** applies three 3×3 convolution layers and produces a score at every board position. Shared filters recognize local patterns in multiple positions. Because its filters do not depend on the board width or height, a CNN checkpoint can be warm-started at a different odd window size; an MLP checkpoint cannot.

The CNN’s local receptive field helps identify nearby clue patterns. It does not replace the exact solver, and a model’s successful training loss does not prove that it plays Minesweeper safely.

## 4. PPO is another training route

`agents/train.py` also supports **Proximal Policy Optimization (PPO)** through Stable-Baselines3 and `sb3-contrib`'s `MaskablePPO`. PPO is a policy-gradient method: it directly adjusts a probability distribution over actions while limiting how sharply one update changes the policy. The action mask prevents invalid actions from being selected.

PPO is an alternative experiment, not the NumPy DQN used by the strongest current play command. The optional `agents/ppo_cnn.py` extractor gives PPO a spatial CNN; the default PPO extractor is MLP. PPO needs its PyTorch/SB3 dependencies installed, while the custom DQN path uses NumPy.

## 5. Rewards: what is the agent being taught?

The environment’s dense reward is approximately:

- positive reward for each newly revealed safe cell,
- a small per-action cost (`step_cost`),
- a negative terminal reward when a mine is hit (`mine_penalty`, default 10).

The environment also has `cascade` and `sparse` reward modes. Cascade mode gives an additional bonus when one reveal opens several cells; sparse mode rewards the number of opened cells without the dense mode’s step cost. DQN’s mine penalty can be changed using `--mine-penalty`.

A reward is a training signal, not a direct guarantee about behavior. Raising the mine penalty may sound like it should make a model safer, but it can also make optimization harder or shift tradeoffs in unexpected ways. The experiments in `analysis/benchmark_results.md` include penalty changes that did **not** improve pure-policy mine survival. Measure the result instead of assuming a reward change worked.

## 6. Why the solver hybrid helps

`solvers/baselines.py` contains the rule and constraint solvers. A clue gives a constraint: for example, if a clue says 2 and one adjacent cell is already flagged, then exactly one of the remaining hidden neighbors is a mine.

For small connected groups of constraints, the solver uses memoized dynamic programming to compute mine probabilities. Each state records the next cell to consider and the remaining mine count for each clue. Impossible partial assignments are pruned; repeated states reuse cached results. The mine density prior is included when weighting assignments. If the component is too large or exceeds the computation budget, it falls back to conservative approximate risk estimates; approximations are not treated as proof that a cell is safe or a mine.

For a learned agent, local inference reads clues and flags from the **current observation**. Hidden cells just outside the window that border an edge clue are treated as unknowns, not as if the solver had seen the rest of the board. The solver can identify certain mines to flag, certain-safe reveals, and the lowest-risk visible moves. The DQN chooses among allowed actions. In `--safe-only` mode, learned play pauses instead of guessing when no safe visible reveal is established; clicking Continue in the viewer explicitly permits the lowest-risk guess.

This has important limits:

- Minesweeper sometimes requires a guess; the hybrid does not guarantee a win.
- Local inference cannot use a deduction that exists only elsewhere on the explored board.
- Wrong manual flags can invalidate deductions, so the solver assumes its supplied flags are correct.
- Exact inference has component-size and state-count limits; the fallback estimates may be less accurate.

## 7. Training, playing, and evaluation

Activate the project’s Conda environment before Python commands:

```bash
conda activate minesweeper-rl
```

Train a CNN DQN with reveal and flag actions, saving to a new checkpoint:

```bash
python -m agents.numpy_dqn --episodes 2000 --window 9 --density 0.16 --net cnn --flag-actions --save dqn_cnn_flags_learn.npz --seed 0
```

The existing 15×15 checkpoint used for the current DQN play example is `dqn_cnn_flags_15x15_context_200.npz`. Evaluate a trained checkpoint on reproducible boards:

```bash
python -m experiments.eval --policy dqn --model dqn_cnn_flags_15x15_context_200.npz --episodes 200 --density 0.16
```

Test the model without the logical override, or with the observation-window solver restricting its action choices:

```bash
python -m experiments.eval --policy dqn --model dqn_cnn_flags_15x15_context_200.npz --episodes 200 --density 0.16 --deduction-override
```

Play with the current DQN/solver hybrid, or enable safe-only pauses:

```bash
python -m experiments.play --agent dqn --model dqn_cnn_flags_15x15_context_200.npz
python -m experiments.play --agent dqn --model dqn_cnn_flags_15x15_context_200.npz --safe-only
```

Optional PPO training uses the same environment:

```bash
python -m agents.train --timesteps 100000 --window 15 --density 0.16 --extractor mlp --save ppo_mines_15x15
python -m agents.train --timesteps 100000 --window 15 --density 0.16 --extractor cnn --save ppo_mines_15x15_cnn
```

Run the tests after changing code:

```bash
python -m pytest -q
```

Training can take a long time. Begin with fewer episodes or timesteps to check that the command and environment work before starting a large run. Use a new `--save` path for experiments so you do not overwrite a useful checkpoint.

## 8. How to tell whether an experiment helped

Use the same evaluation seeds, mine density, action budget, and revealed-cell budget when comparing two checkpoints. Otherwise the comparison may mostly reflect easier boards or different stopping conditions. The project evaluator reports revealed-cell mean, median, maximum, and mine-hit rate for learned policies. For solver baselines, it also reports guesses and guess survival.

Read `analysis/benchmark_results.md` for the recorded experiments. It distinguishes standalone learned policies from older solver-assisted comparisons and notes where sample sizes differ. In particular, earlier matched results show that the standalone learned models often hit a mine, while a rules-assisted DQN performed much better. Those earlier numbers are not a benchmark of every later code change; run a fresh matched evaluation before claiming a new improvement.

A good learning loop is:

1. Change one training setting or algorithm detail.
2. Save the result to a distinct checkpoint.
3. Evaluate it on the same held-out seeds as a reference checkpoint.
4. Compare survival and revealed-cell results, not just training reward.
5. Repeat across multiple training seeds before drawing a broad conclusion.

## 9. Where to look in the code

- `env/mines.py` — deterministic mines, revealing, flood fill, flags, and chording.
- `env/gym_wrapper.py` — local observation tensor, action masks, rewards, reset, and step.
- `agents/numpy_dqn.py` — NumPy MLP/CNN, replay memory, DQN and Double-DQN training.
- `agents/train.py` and `agents/ppo_cnn.py` — PPO training entry point and optional CNN extractor.
- `agents/policies.py` — loads saved PPO or DQN models for play/evaluation.
- `solvers/common.py` and `solvers/baselines.py` — clue constraints, probability inference, and baseline solvers.
- `experiments/eval.py` — repeatable policy and baseline evaluation.
- `experiments/play.py` — interactive game viewer and learned-agent/solver hybrid.
- `analysis/benchmark_results.md` — measured results and caveats.

## Small glossary

- **Episode:** one game from reset until mine hit or a configured limit.
- **Policy:** the rule used to choose an action from an observation.
- **Q-value:** estimated long-term value of taking one action in one state.
- **Reward:** numerical feedback from the environment after an action.
- **Exploration:** trying actions to gather experience rather than always choosing the current favorite.
- **Replay buffer:** memory of past training transitions used for randomized updates.
- **Target network:** a slowly updated copy used to make DQN learning targets more stable.
- **Action mask:** legal/illegal indicators supplied to the policy.
- **CSP:** constraint-satisfaction problem; here, finding mine layouts consistent with visible clues.
- **Hybrid:** the learned policy and solver working together, with the solver limiting risk and the policy ranking permitted moves.
