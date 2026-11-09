# Infinite Minesweeper RL Agent — the complete guide

Everything about this project in one file: the rules it implements, a from-scratch
introduction to the RL parts, how the solver and the hybrid actually work, every measured
number with its caveats, and an honest list of where the remaining headroom is and what to do
about it.

If you read only one section, read **§1**. If you plan to change code, read **§22** first.

---

## §1. Start here: what this is, and what it found

The project trains an agent to play **Infinite Minesweeper** — Minesweeper on an unbounded
grid — and tests whether reinforcement learning can beat a classical constraint solver,
especially on the forced guesses that decide a run.

It cannot, and the way it fails is the interesting result. Three components are stacked:

| Layer | What it is | Mean cells (500 boards) | Mine-hit rate |
|---|---|---:|---:|
| 1. Learned policy alone | NumPy Double-DQN, 15×15 window, no solver | **243.2** | 0.996 |
| 2. Constraint solver masked into the policy | Exact inference; the network only orders equally-safe cells | **1,453.9** | 0.162 |
| 3. Classical baselines | Random / single-clue rules / CSP | 118.8 / 1,248.1 / 1,304.1 | — |

So the headline number is **"exact inference with a learned tie-breaker"**, not "an RL agent
that plays Minesweeper". The learned part is roughly 6× worse than the solver on its own, and
its marginal contribution inside the hybrid has never been isolated — the control that would
do it is described in §17 and is about twenty lines of code.

The negative result is real and reproducible: learned policies beat only the random baseline,
and the solver-first hybrid beats both logical baselines on matched seeds. What has *not* been
shown is that RL contributes anything to that hybrid.

Numbers are seeds 0–499, density `p = 0.16`, 300-step budget, 2,000-cell cap for learned
policies, single training seed. Read §15 before quoting any of them.

---

# Part I — The game

## §2. The game this repository implements

This is a **clean-room implementation**. Nothing needs to be downloaded or owned. The Steam
game is a *later* target for a screen-capture adapter, not a dependency.

### §2.1 Rules as implemented

- **Grid:** unbounded 2D integer lattice, generated lazily in 16×16 chunks.
- **Mines:** each cell `(x, y)` is a mine independently with probability `p` (default `0.16`),
  decided deterministically by `(seed, x, y)` via
  `random.Random(f"{seed}:{x},{y}").random() < p`. Same seed and coordinates → same board,
  which is what makes evaluation reproducible.
- **Safe start:** the 3×3 block around `(0, 0)` is always safe. `reset()` auto-reveals
  `(0, 0)`, so an episode never dies on move one.
- **Numbers:** a revealed cell shows the count of mined neighbours (0–8, Moore
  neighbourhood).
- **Zero cascade:** revealing a 0 flood-fills all connected zeros and their bordering numbers.
- **Actions (RL):** reveal one cell inside the local window (`Discrete(H*W)`). The NumPy DQN
  can optionally add one flag action per cell (`--flag-actions`); PPO defaults to
  reveal-only.
- **Chording:** revealing all hidden neighbours of a satisfied clue at once. Implemented in
  the game and in the viewer.
- **Episode end:** mine hit (`terminated=True`), or the step / revealed-cell budget
  (`truncated=True`; environment defaults `max_steps=500`, `max_revealed=2000`; the evaluation
  harness uses 300 steps and the same 2,000-cell cap).
- **Observation:** `(C=11, H, W)` float32 window centred on the focus (default 15×15).
  Channel 0 = hidden, channel 1 = flagged, channels 2–10 = one-hot clue values 0–8. These are
  binary planes, not colours. The action mask is separate from the image.
- **Rewards (`reward_mode`):**
  - `dense` (default): `+1.0` per newly revealed safe cell, `-0.02` per action, `-10.0` on a
    mine.
  - `cascade`: dense plus `+0.5` per extra cell when one action opens several cells.
  - `sparse`: an alternative reward shape for the E3 ablation; it did not change the
    conclusions.
  - Flag actions return `+1.0` for a correct flag against `-0.17` for a wrong one, minus the
    step cost.

### §2.2 What the Steam game does differently

From the store page and listings (Oct 2026), the commercial game is **zoned**: 8×8 zones with
exactly **10 mines each**, not a uniform density. Hitting a mine **locks the zone temporarily**
instead of ending the run; 64 cleared zones complete an area and reveal artwork across six
themes. Backgrounds are night-sky/colourful art, which matters for any future OCR. Controls
are mouse-only.

Why this matters: zone totals are a *strong extra constraint* (a zone with 10 mines and 40
revealed cells tells you a lot about the other 24), so the uniform-`p` model is a
simplification of the real game — not just a different difficulty setting.

### §2.3 The extractor plan (future, not v1)

Keep the clean environment as-is and add an adapter rather than retraining on pixels:

```
Steam window → screen capture → grid calibrate (zoom / spacing / viewport tracking)
  → digit OCR on revealed numbers (+ flag/cover classifier)
  → BoardObservation (same (C,H,W) tensor format as this env)
  → reuse solvers + RL policy (possibly fine-tune on noisy OCR)
  → optional click-through actuator (reveal / flag at screen coordinates)
```

The interface to implement then is `adapters/steam/` with `capture.py`, `calibrate.py`,
`recognize.py` (background suppression, digit classifier, confidence threshold → "unknown"
instead of a misread) and `to_observation.py`. `adapters/steam/INTERFACE.md` describes it. No
Steam purchase, install or asset copying is needed until that phase.

## §3. The zone simulator already in the repo

`env/zone_mines.py` implements the zoned rules provisionally: `ZONE_SIZE = 8`,
`MINES_PER_ZONE = 10`, `ZoneBoard`, `ZoneMinesweeperEnv`. A mine hit locks the zone
(`locked_zones`, `locked_cells`), locked cells become unplayable and are excluded from the
solver's candidate set, and zone mine totals are fed to the solver as extra constraints
(`extra_constraints`, `zone_risks`).

`analysis/benchmark_zones.py` evaluates it; `tests/test_zone_mines.py` covers it. This is the
closest thing in the repo to the actual game, and §18 explains why it is also where the solver
is most likely to degrade.

---

# Part II — Reinforcement learning, from scratch

## §4. What the agent sees and does

`env/mines.py` implements the board; `env/gym_wrapper.py` turns it into a Gymnasium
environment.

- **Observation:** `(11, H, W)`. Channel 0 is 1 for hidden cells, 1 is 1 for flags, and 2–10
  are one-hot clue values. A hidden cell is never confused with a revealed zero.
- **Action:** reveal one cell in the window. Flag-capable models get a second action head.
- **Action mask:** a separate `MultiBinary(H*W)` entry marking legal actions (hidden and
  unflagged). The mask is not an image channel, and it is what keeps the action space honest.
- **Focus:** after a reveal, the focus moves to that cell and the window re-centres. There is
  no camera action; the viewport follows play.
- **No "outside the board" channel:** the board is unbounded, so the window edge is a viewport
  boundary, not a game boundary.

## §5. What "reinforcement learning" means here

Instead of writing rules like "if the clue equals the number of neighbouring flags, reveal the
rest", the agent acts and learns from outcomes:

1. The environment gives an observation and a legal-action mask.
2. The policy picks an action.
3. The environment reveals/flags, then returns a reward and the next observation.
4. Training adjusts the policy so actions with better long-term outcomes become more likely.

One move's reward is not the objective. A move can be good because it reveals information that
makes later moves safe. Training therefore estimates **return**: reward now plus discounted
future reward, with `gamma` controlling how much the future counts.

A trained model is not a memorised board. It is a set of weights mapping observations to
action scores. A CNN reuses the same filters at every position, so it can recognise local clue
patterns anywhere in the window.

## §6. The NumPy DQN (`agents/numpy_dqn.py`)

The working learned path on this machine is a hand-written Deep Q-Network in pure NumPy,
because the machine's WDAC policy blocks torch's native DLLs. It is not vanilla DQN, but it is
not modern either.

**Q-values and exploration.** The network estimates a Q-value per action — the expected
discounted return of taking it. Play takes the highest legal Q. Training uses ε-greedy
exploration: with probability ε pick a random legal action, otherwise the greedy one; ε decays
(`--eps0`, `--eps1`) so later choices rely on what was learned.

**Replay and stable targets.** Transitions (observation, action, reward, next observation,
done, next mask) go into a replay buffer of 20,000, and updates sample random batches, which
decouples learning from move order. An online network and a slower target network are
maintained, and this project uses **Double-DQN** targets: the online net chooses the next
action, the target net evaluates it (`double_dqn_targets`). Adam updates the weights with
gradient clipping, one update every 4 environment steps (`--update-every`).

**MLP vs CNN.** The MLP flattens the window, so its input and output sizes depend on window
dimensions. The CNN applies three same-padded 3×3 convolutions and outputs **one Q value per
cell** — a cell-indexed, fully convolutional action head with a 7×7 receptive field. Weights
are shared across positions, and because the filters do not depend on spatial size, a CNN
checkpoint can be warm-started at another odd window size (`--init-from`); an MLP cannot.

**What is not in it:** n-step returns, prioritised replay, duelling heads, a distributional
head, frame stacking, vectorised environments.

**Its measured ceiling, alone: 243.2 mean cells with a 0.996 mine-hit rate.** It has learned
"don't click an open cell" and very little else.

## §7. PPO is the other training route (`agents/train.py`)

Stable-Baselines3 plus `sb3-contrib`'s `MaskablePPO`, optionally with the spatial extractor in
`agents/ppo_cnn.py` (default is MLP). PPO is a policy-gradient method: it adjusts a
distribution over actions directly while limiting how far one update moves the policy. The
mask prevents illegal actions.

PPO is an *alternative experiment*, not the path behind the strongest play command. It needs
torch; the NumPy DQN does not. Measured PPO results are in §14 — the best 15×15 run reached
231.9 mean with a 1.000 mine-hit rate, i.e. below the DQN and nowhere near the solver.

## §8. Rewards: what the agent is being taught

Dense reward per action is roughly: `+1` per newly revealed safe cell, `-0.02` step cost, `-10`
on a mine. `cascade` adds a bonus for opening several cells at once; `sparse` is an alternative
shape kept for the reward ablation.

Reward is a training signal, not a guarantee. Raising the mine penalty sounds like it should
make a model safer; in this repo it did not. The penalty-50 fine-tune reached 258.5 mean raw
cells on 500 boards — higher than the 219.7 of the checkpoint it started from — and still hit a
mine in **every** episode. A 100-board hybrid follow-up (1,628.8 vs 1,370.0) is a pilot, not a
matched result. Treat "I changed the reward and it feels safer" as unverified until measured on
held-out seeds.

---

# Part III — The solver, and the hybrid

## §9. Constraint inference (`solvers/baselines.py`, `solvers/common.py`)

A clue is a constraint: "this cell says 3, so exactly 3 of its hidden neighbours are mines".
Flags reduce the unknown count. For small connected groups of constraints (**components**),
the solver computes exact posterior marginals:

- it enumerates mine assignments consistent with every clue,
- memoised dynamic programming over `(next cell, remaining mine count per clue)` prunes
  impossible partial assignments and reuses repeated states,
- the mine-density prior `p` weights assignments,
- the result is a probability per hidden cell.

**Hard limits.** Exact inference only runs when a component has **≤ 25 cells** and the DP
stays under a **100,000-state cap**. Otherwise it falls back to a single-clue
`need / len(cells)` estimate. That fallback is why late-game guesses can be *not* the true
minimum-risk cell, and why the solver cannot prove safety there.

From the probabilities, play can derive three things: certain mines (probability 1), certain
safe cells (probability 0), and the minimum-risk candidate set. Flagging certain mines before
acting also enables chording.

## §10. Solver memory: the unresolved-clue outline

Early versions answered every question by rereading every clue ever opened. Correct, but the
cost grew with how long the game had run: after 1,000 revealed cells, each decision rescanned
all of them even though almost none could teach anything new.

**Game state vs solver memory.** The board keeps the full `revealed` map, because the game
needs it. A *decision* needs only clues that still border an unknown cell. A clue whose eight
neighbours are all revealed or flagged can never fire again, so it is settled and can be
forgotten.

**The outline.** `InfiniteBoard.active_clues` holds only unsettled clues — the boundary of the
explored region, which is exactly what the agent must reason about.

**Incremental repair, not rebuild.** `reveal` refreshes newly opened cells plus their ring;
`flag` refreshes the toggled cell plus its ring, so unflagging can bring a retired clue back;
zone play refreshes after a lock; `solvers.baselines.solver_clues(board)` rebuilds only if the
revealed map or flag set was replaced wholesale.

**Occupancy is still checked against the full board** via `all_revealed=`, so an edge clue never
mistakes an already-open but out-of-window cell for an unknown.

**Why it is safe.** A settled clue is provably useless, and two regressions pin it down:
`test_active_clues_track_only_unresolved_outline_and_flag_changes` and
`test_compact_clue_solver_matches_full_history_inference` (same candidate set as full-history
inference on a board with retired interior clues).

**Measured effect — cost only, not strength.** Best of 3 repetitions, mean over 4 seeds,
density 0.16, 15×15 window, CSP-driven play, candidate sets asserted identical at every
checkpoint:

| Revealed cells | Active clues | Full-history | Compact outline | Ratio |
|---:|---:|---:|---:|---:|
| 200 | 73 | 2.70 ms | 2.40 ms | 1.13× |
| 402 | 153 | 8.02 ms | 7.04 ms | 1.14× |
| 820 | 262 | 10.80 ms | 9.07 ms | 1.19× |
| 1,048 | 346 | 13.65 ms | 10.77 ms | 1.27× |
| 1,349 | 395 | 12.58 ms | 8.74 ms | 1.44× |

Absolute milliseconds are machine-specific. A second run measured 1.37× and 1.53× at the two
largest checkpoints, so treat the saving at 1,000+ cells as roughly **1.3–1.5×**, growing with
history rather than being constant. Read it as "same decisions, less repeated work" — never as
"plays better".

## §11. The hybrid: `deduction_override` (`experiments/eval.py:52`)

This is where the score comes from. In order, it:

1. Runs the exact solver over the whole board (`--global-inference`) or the local window.
2. Flags every cell with posterior mine probability 1.
3. Builds candidates as the cells whose risk equals the **minimum on the board**
   (`probability <= risk + 1e-12`).
4. Rewrites the action mask so only candidates are legal reveals, and only proven mines are
   legal flag actions.
5. Calls the network to pick among the survivors.

Three consequences matter more than the network itself:

- **At risk 0, the network cannot lose.** Every legal action is proven safe; the network is
  choosing an *order*, not a safety level. That choice affects how fast the frontier expands,
  which does show up in cells revealed per board.
- **At risk 0.16, the network cannot pick a safer cell.** The tier is already the best
  available guess, so RL cannot "learn better guessing" through this mask — only better
  ordering within the tier.
- **The tier is often a single cell.** Then the network has no decision at all. This is why the
  hybrid's score tracks solver quality, not policy quality.

## §12. The no-guess shield: `safe_policy_action` (`eval.py:83`)

`--safe-only` takes only proven-safe moves and pauses the episode the moment a guess would be
required. In the viewer, clicking Continue explicitly permits the solver's lowest-risk guess.
This measures pure deduction depth with zero guessing risk.

Historical shield numbers (seeds 3000–3099, *old single-clue shield*, not the current DP
solver): DQN 15×15 averaged 1,414 cells and paused on uncertainty in 19% of boards; PPO MLP
1,136.5 / 38%; PPO CNN 438.0 / 98%. Those pause rates do **not** apply to the current exact
solver.

## §13. Zone-aware solving

With `--zone-rules`, every zone whose hidden cells touch the candidate window or the frontier
contributes a constraint: `mines_left = mines_per_zone - flags_in_zone` over the zone's hidden
cells. If `mines_left == 0` the zone is proven safe; if it equals the hidden count, every hidden
cell is proven a mine; otherwise it is passed to the DP as an extra constraint and used as a
fallback risk.

Locked zones are skipped, since their cells can no longer be played. This is the mechanism that
makes the zoned game *easier* informationally than uniform-`p` — and the mechanism most at risk
from the 25-cell exactness cliff, because a zone constraint touches up to 64 cells (§18.2).

---

# Part IV — Measured results

## §14. The numbers

`analysis/benchmark_results.md` is the full log. Everything below is from it: seeds 0–499,
density 0.16, 300-step budget, 2,000-cell cap for learned policies.

### §14.1 Against the classical baselines

| Agent | Window | Mean | Median | Max | Extra |
|---|---:|---:|---:|---:|---|
| Random | — | 118.8 | 84.0 | 639 | 2.67 guesses/episode |
| Rules (single-clue) | — | 1,248.1 | 1,297.5 | 2,127 | 0.25 guesses/episode, survival 0.694 |
| CSP | — | 1,304.1 | 1,312.0 | 2,290 | 0.53 guesses/episode, survival 0.933 |
| PPO, 50k, seed 1 | 9×9 | 196.1 | 157.5 | 834 | mine-hit 1.000 |
| PPO, 100k, seed 2 | 9×9 | 206.0 | 152.0 | 1,499 | mine-hit 1.000 |
| Double-DQN, seed 17 | 9×9 | 216.2 | 173.0 | 1,054 | mine-hit 0.962 |
| Double-DQN, seed 29 | 9×9 | 209.8 | 163.5 | 1,104 | mine-hit 0.944 |
| PPO, 100k, seed 21 | 15×15 | 231.9 | 177.5 | 1,134 | mine-hit 1.000 |
| Double-DQN, 200 ep, seed 31 | 15×15 | 243.2 | 190.0 | 1,081 | mine-hit 0.996 |
| **DQN + solver mask** | 15×15 | **1,453.9** | **1,564.5** | 2,295 | mine-hit 0.162 |

### §14.2 What is statistically supported

- Hybrid − rules = **+205.7**, paired bootstrap 95% CI **[163.8, 246.6]**.
- Hybrid − CSP = **+149.7**, CI **[104.5, 193.4]**.
- Per-seed wins: **373/500** vs rules, **345/500** vs CSP.
- Raw DQN 15×15 vs random **[108.5, 141.2]** — it beats random.
- Raw DQN 15×15 vs rules ≈ **−1,000**, CI excludes zero — it loses badly, and so do PPO and the
  other DQN checkpoints, to both logical baselines.
- DQN seed-17 vs the original DQN: **+2.1** cells, CI **[−13.6, 17.6]** — not a reliable
  improvement, and a good example of why paired intervals matter.

### §14.3 Earlier learned-policy variants

For completeness, from the same log: MaskablePPO 176.1/196.1/206.0 across 50k–100k runs;
CNN DQN with flags 214.1; warm-start continuation 200.9; Double-DQN 216.2 and 209.8; 15×15
context 243.2; PPO 15×15 231.9. None of these beat the hybrid, and none beat the logical
baselines alone.

## §15. Caveats that change what those numbers mean

1. **Single training seed.** DQN seed 31, PPO seeds 1/2/21. The README asks for 5 seeds for RL
   runs. Until that exists, architecture comparisons are anecdotes.
2. **Baselines have no cell cap.** Learned policies stop at 2,000 revealed cells; the baseline
   runner does not. That makes the hybrid's advantage *conservative* (right direction) but
   means the means are not the same quantity for everyone.
3. **The metric is saturated.** Mine-hit 0.162 with a 2,000-cell cap means roughly **84% of
   hybrid episodes end on the budget, not on a mine**. The mean is mostly "did this run survive
   to the cap", and the variance comes from the 16% that die early.
4. **Guess survival is per guess; mine-hit is per episode.** They are not comparable, and the
   results file says so.
5. **There is a measurement inconsistency in the record.** The 500-board CSP run reports 0.53
   guesses/episode at 0.933 survival. A separate 100-board CSP sample reports 12.6 guesses per
   board at 0.992 survival. Both cannot describe the same quantity: 12.6 guesses at 99.2%
   survival implies a ~0.06% per-guess risk, implausible next to 0.933. Until the counting is
   reconciled, treat 12.6 as a bug, not a finding.

## §16. Observation-encoding pilot

A pilot compared 11-channel one-hot against a 3-channel compact encoding (hidden, flagged, `clue/8`) with otherwise identical CNN settings and 300 training
episodes each, evaluated on seeds 0–499:

| Encoding | Raw mean | Median | Mine-hit | Hybrid mean* | Hybrid mine-hit* |
|---|---:|---:|---:|---:|---:|
| Compact (3 ch) | 237.9 | 187.5 | 0.996 | 1,472.6 | 0.470 |
| One-hot (11 ch) | 225.3 | 170.0 | 0.992 | 1,487.8 | 0.410 |

*Hybrid columns are a 100-board pilot (seeds 0–99), not the 500-board matched set.

Compact scored 12.6 more raw cells; one-hot had a slightly lower mine-hit rate and did better in
the hybrid pilot. **No winner.** One training seed per encoding, not paired at the training-seed
level, and one pilot run. Keep one-hot as the default; treat compact as an open experiment.

## §17. What is *not* measured

### §17.1 The missing control: solver mask + random tie-break

The hybrid's +149.7 over CSP is **not** evidence that the network's tie-breaking works. The two
paths differ in more than tie-breaking:

- different candidate construction (window + frontier vs `csp_probabilities`),
- the certain-mine flagging loop runs before every hybrid action,
- CSP picks with `rng.choice`; the hybrid picks with a network.

The isolating control is trivial: **same solver, same mask, random pick within the tier.** About
twenty lines in `experiments/eval.py` — add `--random-tie-break` and, inside
`deduction_override`, replace `act(obs, mask)` with `rng.choice(np.flatnonzero(mask))`. Then
compare on seeds 0–499 with the same paired bootstrap the hybrid benchmark already uses.

Until that exists, the only defensible claim is:

> The solver-first mask reaches 1,453.9; the network's marginal contribution is unmeasured.

If the control lands within ~30 cells, the honest conclusion is that the project demonstrated
solver dominance and the RL component is decoration — which is a result worth writing down, and
not the result the README implies.

### §17.2 No PPO hybrid

A PPO-plus-solver hybrid has never been evaluated. The hybrid claim is DQN-specific by
omission, not by experiment.

---

# Part V — Where it stands, and what to do

## §18. Where the headroom actually is

### §18.1 Deaths, not average play

Since ~84% of hybrid episodes end on the budget, the score is dominated by the 16% that die. A
death requires a guess. So the levers are (a) needing fewer guesses and (b) surviving the ones
you need — not better average move quality.

### §18.2 The solver's exactness cliff

Exact marginals require **≤ 25 cells** per component and **≤ 100,000 DP states**; otherwise the
solver uses a single-clue `need / len(cells)` estimate. On an unbounded board the frontier grows
— 395 active clues at 1,349 revealed cells — so late-game guesses are frequently not the true
minimum-risk cell, and the fallback cannot prove safety.

The zone variant makes this acute: a zone constraint touches up to 64 cells, so
`solver_max_cells` becomes `max(25, 64) = 64` while the state cap stays at 100,000. That
combination very likely exceeds the cap and lands in the crude fallback exactly where the zone
constraint is most valuable. Measuring this is cheap (§21).

### §18.3 Guessing is myopic, and τ does not exist

There is no risk-threshold parameter anywhere. Two decisions are hard-coded:

- `csp_candidates`: `blind_risk = board.p`, so a fresh cell is preferred whenever
  `p <= frontier_risk`. The blind-vs-frontier tradeoff is pinned at exactly `0.16`.
- Safe-only play: pause when `risk > 0.0`, i.e. τ = 0.

A tunable τ is cheap and worth having as a baseline, but it is a *myopic* rule: a frontier cell
at 0.22 that resolves four clues can be better than a fresh cell at 0.16 that resolves nothing,
and no single threshold expresses that. That is precisely research question 2 — "can RL learn
useful long-term guessing, such as picking guesses that open up more information afterwards?" —
and it remains unanswered, because the current mask removes the network's ability to make the
trade at all.

### §18.4 The metric is compressed

At a 2,000-cell cap, differences between good policies get truncated. A survival curve (fraction
alive after N reveals) and cells-per-action would show differentiation the single mean hides.

## §19. If you want to improve it: ordered plan

Ordered by expected value per hour, not by novelty. Each step decides whether the next is worth
doing.

| # | Change | Why here | Metric that would show it |
|---|---|---|---|
| 1 | Solver mask + random tie-break control (§17.1) | You cannot report a hybrid result without it | paired delta vs current hybrid, seeds 0–499 |
| 2 | Instrument guesses: forced guesses/board, tier size, share outside the exact min-risk tier, DP fallback rate | Tells you whether τ or the fallback is the real loss | counts per board |
| 3 | Remove the exactness cliff: raise the cap, factor disconnected sub-components, mark approximate results as approximate instead of trusting `need/len` | Late guesses become actually-minimum-risk | guess survival, mine-hit rate |
| 4 | Information-aware guess ranking: `P(safe) × expected newly revealed`, or posterior-sample expectimax over the top-k tier | Directly attacks the 16%; needs no training | mine-hit rate, cells/board at fixed cap |
| 5 | τ sweep and top-k tier width, against #4 | Cheap learned-free baseline for the new guesser | sweep table on identical seeds |
| 6 | Feed the risk map as input channels, or behaviour-clone the solver then fine-tune (E7) | Only now does the network have a job it can win | hybrid vs solver-only, paired CIs |

**Do not** try another reward penalty, another encoding, another DQN variant (duelling, n-step,
PER, C51), or a PPO-vs-DQN comparison. The penalty-50 and encoding experiments already failed to
improve anything, and both raw architectures sit ~6× below the solver. Optimising the part that
is already not the bottleneck is the reliable way to spend a month and learn nothing.

## §20. Should you keep going?

Honest verdict: **the research question this repo was built around is answered, and the answer
is a negative result about RL.**

- Exact inference dominates. The learned policy is 6× worse alone and, even in the best hybrid,
  may be contributing nothing measurable.
- The interesting follow-up — "does RL earn its keep when the search is approximate?" — is real,
  but testing it needs boards where the DP cap actually binds and a fallback-vs-fallback+RL
  comparison. That is a new experiment, not more epochs on this one.
- One training seed means no "better" claim is defensible anyway.

Three legitimate ways to close this out:

1. **Park it.** Ship the results table and these findings, push, done. The repo then reads as a
   complete, honest negative result — more valuable to a reader than another 2,000 episodes.
2. **Make it defensible.** Run the headline hybrid on 5 training seeds and report the spread.
   Worth it only if the repo will be read critically by someone else.
3. **Pivot to a tool.** The zone rules exist because that is the game being played. A solver
   assistant is a different, user-facing project where the exact DP is the whole product and RL
   is dead weight. That is the version with a reason to keep going.

What is not worth doing is another algorithm. The signal in this setup has been extracted.

---

# Part VI — Working with the repo

## §21. Commands

```bash
conda activate minesweeper-rl

# --- sanity ---------------------------------------------------------------
python -m pytest -q

# --- train ----------------------------------------------------------------
# CNN DQN with reveal + flag actions, from scratch
python -m agents.numpy_dqn --episodes 2000 --window 15 --density 0.16 --net cnn --flag-actions --save models/dqn_cnn_flags_learn.npz --seed 0
# warm-start an existing CNN checkpoint at a new window
python -m agents.numpy_dqn --episodes 200 --window 15 --density 0.16 --net cnn --flag-actions --init-from models/dqn_cnn_flags.npz --save models/dqn_cnn_flags_15x15_context_200.npz --seed 31
# PPO (needs a working torch install)
python -m agents.train --timesteps 100000 --window 15 --density 0.16 --extractor cnn --save models/ppo_mines_15x15_cnn

# --- evaluate -------------------------------------------------------------
# all three classical baselines
python -m experiments.eval --policy all --episodes 500
# raw policy, no solver
python -m experiments.eval --policy dqn --model models/dqn_cnn_flags_15x15_context_200.npz --episodes 500 --density 0.16
# solver-first hybrid (the headline configuration)
python -m experiments.eval --policy dqn --model models/dqn_cnn_flags_15x15_context_200.npz --episodes 500 --density 0.16 --global-inference
# windowed deductions instead of whole-board inference
python -m experiments.eval --policy dqn --model models/dqn_cnn_flags_15x15_context_200.npz --episodes 200 --deduction-override
# never guess (pauses instead); reports stopped_uncertain
python -m experiments.eval --policy dqn --model models/dqn_cnn_flags_15x15_context_200.npz --episodes 500 --safe-only
# how often a guess is required, and whether the pick was in the solver's min-risk tier
python -m experiments.eval --policy dqn --model models/dqn_cnn_flags_15x15_context_200.npz --episodes 200 --global-inference --frontier-diagnostics

# --- zone rules (the provisional real game) -------------------------------
python -m experiments.eval --policy dqn --model models/dqn_cnn_flags_15x15_context_200.npz --episodes 100 --zone-rules
python -m analysis.benchmark_zones --episodes 20 --seed 0

# --- benchmarks -----------------------------------------------------------
python -m analysis.benchmark_hybrid            # matched 500-board hybrid vs baselines
python -m analysis.bench_frontier_latency      # solver decision cost

# --- play -----------------------------------------------------------------
python -m experiments.play --agent dqn --model models/dqn_cnn_flags_15x15_context_200.npz
python -m experiments.play --agent dqn --model models/dqn_cnn_flags_15x15_context_200.npz --safe-only
```

Training is slow. Start with a short run to prove the command and environment work, and always
use a new `--save` path so you never overwrite a useful checkpoint.

**Encoding pilot, if you want to repeat it:**

```bash
conda activate minesweeper-rl
python -m agents.numpy_dqn --episodes 100 --window 15 --density 0.16 --seed 20261024 --net cnn --flag-actions --encoding compact --save models/dqn_compact_pilot.npz
python -m agents.numpy_dqn --episodes 200 --window 15 --density 0.16 --seed 20261025 --net cnn --flag-actions --encoding compact --init-from models/dqn_compact_pilot.npz --save models/dqn_compact_300ep.npz
python -m experiments.eval --policy dqn --model models/dqn_compact_300ep.npz --episodes 500 --density 0.16
```

## §22. Evaluation discipline

Use the same seeds, density, action budget and revealed-cell budget whenever comparing two
checkpoints; otherwise the comparison mostly reflects easier boards or different stopping
conditions. The evaluator reports mean, median, max and mine-hit rate for learned policies, and
guesses plus guess survival for solver baselines.

The loop that produces trustworthy claims:

1. Change one setting or algorithm detail.
2. Save to a distinct checkpoint.
3. Evaluate on the same held-out seeds as a reference checkpoint.
4. Compare revealed cells **and** mine-hit rate / guess survival — not training reward.
5. Repeat across multiple training seeds before drawing a broad conclusion.
6. Report a paired interval on per-seed differences; a mean without one is a story.

If a change is not measured on identical episode seeds with a paired interval, it is a **pilot**,
not a result. Say so when writing it down.

## §23. Code map

| Path | What lives there |
|---|---|
| `env/mines.py` | Deterministic mines, reveal, flood fill, flags, chording, active-clue outline |
| `env/gym_wrapper.py` | Observation tensor, action mask, rewards, reset/step |
| `env/zone_mines.py` | Zoned rules: 8×8 zones, 10 mines each, zone locking |
| `solvers/common.py` | Constraint representation and probability DP |
| `solvers/baselines.py` | Random, single-clue rules, CSP, global inference, candidate selection |
| `agents/numpy_dqn.py` | NumPy MLP/CNN, replay, DQN/Double-DQN training, checkpoints |
| `agents/train.py`, `agents/ppo_cnn.py` | PPO entry point and optional CNN extractor |
| `agents/policies.py` | Loads PPO/DQN checkpoints as `act(obs, mask)` |
| `experiments/eval.py` | Baselines, policy evaluation, solver-first mask, safe-only shield, diagnostics |
| `experiments/play.py` | Interactive viewer and the solver/agent hybrid |
| `analysis/benchmark_hybrid.py` | Matched-seed hybrid-vs-baseline benchmark with paired CIs |
| `analysis/benchmark_zones.py` | Zone-rules benchmark |
| `analysis/bench_frontier_latency.py` | Solver decision-cost measurement |
| `analysis/benchmark_results.md` | The full results log |
| `adapters/steam/INTERFACE.md` | Future screen-capture adapter interface (§2.3) |

## §24. Glossary

- **Episode** — one game from reset until a mine hit or a configured limit.
- **Policy** — the rule choosing an action from an observation.
- **Q-value** — estimated long-term value of one action in one state.
- **Return** — reward now plus discounted future reward; `gamma` sets the discount.
- **Replay buffer** — memory of past transitions used for randomised updates.
- **Target network** — a slowly updated copy that stabilises DQN targets.
- **Double DQN** — choose the next action with the online net, evaluate it with the target net.
- **Action mask** — legal/illegal indicators supplied alongside the observation.
- **Component** — a connected group of clue constraints sharing hidden cells; exact inference is
  per component.
- **Posterior marginal** — probability a hidden cell is a mine, summed over all mine layouts
  consistent with the clues and the density prior.
- **DP cap** — the 100,000-state budget beyond which inference stops being exact and falls back
  to `need / len(cells)`.
- **Tier** — the set of cells at the minimum posterior risk; the hybrid's mask restricts play to
  it.
- **Forced guess** — a state where every hidden cell has nonzero risk, so any reveal can lose.
- **Chording** — revealing all hidden neighbours of a satisfied clue at once.
- **Zone** — an 8×8 tile with exactly 10 mines under the zone rules; its total is an extra
  constraint.
- **Hybrid** — solver selects the risk tier, policy orders permitted moves.

## §25. Known rough edges

Honest list of things a reader will trip over:

- **The network's contribution is unmeasured** (§17.1) while the README describes a hybrid
  result. This is the biggest documentation/evidence gap in the repo.
- **`analysis/benchmark_results.md`** still recommends the hybrid without that caveat, and
  contains the contradictory CSP guess counts described in §15, item 5.
- **`env/zone_mines.py`** defines `_legal_mask` twice; the second definition wins.
- **Zone inference** probably exhausts the DP cap on 64-cell zone constraints (§18.2), so zone
  play may be using the crude fallback where the constraint matters most. Unverified.
- **No 5-seed training runs**, so no architecture claim in this repo is statistically supported.
- **The hybrid benchmark's baselines are uncapped**, which is conservative for the hybrid but
  means "mean revealed" is not one quantity across rows.
