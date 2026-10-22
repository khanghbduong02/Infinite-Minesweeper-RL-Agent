# Steam adapter (future — needs the purchased game, not started)

Target: `Infinite Minesweeper` (Steam app 5178500, JSM games, ~$5).
Key differences vs our env: 8×8 zones with exactly 10 mines each, mine hit
locks the zone instead of ending the run, night-sky/AI-art backgrounds.

Planned modules (implement when you own the game):
- `capture.py` — grab the game window (mss / WinAPI), track viewport + zoom.
- `calibrate.py` — grid spacing/origin from clicks or template match.
- `recognize.py` — digit classifier for revealed numbers + cover/flag states,
  with background suppression and a confidence threshold (unknown > misread).
- `to_observation.py` — output the same `(11,H,W)` tensor + action mask the
  solvers and RL policy already consume, so everything here is reused.
- `actuate.py` (optional) — click-through reveal/flag at screen coords.

Contract: `to_observation(board_view) -> (tensor, mask, focus)` identical in
shape/semantics to `MinesweeperWindowEnv._obs()`. See `docs/rules.md` §3.
