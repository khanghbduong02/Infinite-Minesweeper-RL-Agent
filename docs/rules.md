# Infinite Minesweeper — fixed rules for this repo

This is our **own clean-room implementation**. Nothing needs to be downloaded.
The Steam game (`Infinite Minesweeper`, app 5178500, ~$5, JSM games) is a
**later** target for a screen-capture number extractor, not a dependency.

## 1. Our implemented game (what RL trains on)

- **Grid:** unbounded 2D integer lattice, generated lazily in 16×16 chunks.
- **Mines:** each cell `(x, y)` is a mine independently with probability `p`
  (default `0.16`), decided deterministically by `(seed, x, y)` via
  `random.Random(f"{seed}:{x},{y}").random() < p`. Same seed + coords → same mine.
- **Safe start:** the 3×3 block around `(0, 0)` is always safe. `reset()`
  auto-reveals `(0, 0)` so episode 0 never dies instantly.
- **Numbers:** revealed cell shows count of mined neighbors (0–8, Moore).
- **Zero cascade:** revealing a 0 flood-fills all connected zeros + border numbers.
- **Actions (RL):** reveal-only, one cell inside the local window
  (`Discrete(H*W)`). Flags exist on the board state but are **solver-internal**
  in v1 — RL does not flag (documented simplification).
- **Episode end:** mine hit (`terminated=True`), or step / revealed-cell budget
  (`truncated=True`, defaults: `max_steps=500`, `max_revealed=2000`).
- **Rewards (`reward_mode`):**
  - `dense` (default): +1.0 per newly revealed safe cell, −10.0 on mine,
    −0.02 step cost.
  - `cascade`: dense + extra +0.5 per cell beyond the first when one action
    opens ≥2 cells (rewards zero-cascade seeking).
  - `sparse`: 0 per step, +1.0 per safe reveal only at end? Actually +1 per
    reveal with no mine penalty shaping except −10 on death and no step cost —
    use for E3 ablation.
- **Observation:** `(C=11, H, W)` window centered on focus (default 15×15):
  channels = `[hidden, flagged, num0…num8]`, float32 in {0,1}.
  Gym wrapper adds `action_mask: MultiBinary(H*W)` (1 = hidden & unflagged).
  Focus starts at `(0,0)` and jumps to the last revealed cell (window follows).

## 2. What the Steam game does differently (extractor phase, NOT v1)

Research (Oct 2026, store page + listings):

- Board is zoned: **8×8 zones with exactly 10 mines each**, not uniform-p.
- Mine hit **locks the zone temporarily** instead of ending the run.
- 64 cleared zones complete an area and reveal gallery artwork; 6 themes.
- Night-sky frontier + generative-AI art backgrounds ("Colorful" tag) — this is
  why you wanted a plain board for training and OCR later.
- Mouse-only controls (reveal / flag / chord via clicks).

## 3. Extractor plan (future, when you own the game)

Keep the clean env as-is; add an **adapter**, don't retrain on pixels:

```
Steam window → screen capture → grid calibrate (zoom/spacing/viewport track)
  → digit OCR on revealed numbers (+ flag/cover classifier)
  → BoardObservation (same (C,H,W) tensor format as our env)
  → reuse solvers + RL policy (possibly fine-tune on noisy OCR)
  → optional click-through actuator (reveal/flag at screen coords)
```

Interface to implement then: `adapters/steam/` with
`capture.py`, `calibrate.py`, `recognize.py` (OCR + colorful-background
robustness: background suppression, digit template/CNN classifier, confidence
threshold → "unknown" instead of misread), `to_observation.py`.
No Steam purchase, install, or asset copying is needed until that phase.
