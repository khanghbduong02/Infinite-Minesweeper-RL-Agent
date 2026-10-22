"""Gymnasium wrapper: local-window Minesweeper with masked reveal/optional flag actions.

Observation: Dict {obs: Box(0,1,(11,H,W)), action_mask: MultiBinary(n_actions)}.
Implements action_masks() for sb3-contrib MaskablePPO.
"""
from __future__ import annotations

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover
    gym = None  # type: ignore

from env.mines import InfiniteBoard

N_NUM = 9  # numbers 0..8
N_CH = 2 + N_NUM  # hidden, flagged, num0..8


class MinesweeperWindowEnv(gym.Env if gym else object):  # type: ignore
    metadata = {"render_modes": []}

    def __init__(self, window_size: int = 15, p: float = 0.16, seed: int = 0,
                 max_steps: int | None = 500, max_revealed: int | None = 2000,
                 reward_mode: str = "dense", step_cost: float = 0.02,
                 mine_penalty: float = 10.0, flag_actions: bool = False):
        super().__init__()
        assert window_size % 2 == 1, "window_size must be odd"
        self.window_size = window_size
        self.hw = window_size // 2
        self.p = p
        self.seed_val = seed
        self.max_steps = max_steps
        self.max_revealed = max_revealed
        self.reward_mode = reward_mode
        self.step_cost = step_cost
        self.mine_penalty = mine_penalty
        self.flag_actions = flag_actions
        self.n_cells = window_size * window_size
        self.n_actions = self.n_cells * (2 if flag_actions else 1)
        self.action_space = spaces.Discrete(self.n_actions)
        self.observation_space = spaces.Dict({
            "obs": spaces.Box(0.0, 1.0, shape=(N_CH, window_size, window_size),
                              dtype=np.float32),
            "action_mask": spaces.MultiBinary(self.n_actions),
        })
        self.board: InfiniteBoard | None = None
        self.focus = (0, 0)
        self._rng = np.random.default_rng(seed)

    # -- MaskablePPO hook -------------------------------------------------
    def _legal_mask(self) -> np.ndarray:
        mask = np.zeros(self.action_space.n, dtype=bool)
        assert self.board is not None
        for i in range(self.n_cells):
            dx, dy = i % self.window_size - self.hw, i // self.window_size - self.hw
            c = (self.focus[0] + dx, self.focus[1] + dy)
            if c not in self.board.revealed and c not in self.board.flags:
                mask[i] = True
            if (self.flag_actions and c not in self.board.revealed
                    and c not in self.board.safe and c not in self.board.flags):
                mask[self.n_cells + i] = True  # agents may place each flag only once
        return mask

    def action_masks(self) -> np.ndarray:
        mask = self._legal_mask()
        if not mask.any():
            mask[:] = True  # avoid empty mask; step() ends the episode (no legal move)
        return mask

    # -- core --------------------------------------------------------------
    def _coords(self, action: int) -> tuple[int, int]:
        cell = action % self.n_cells
        dx, dy = cell % self.window_size - self.hw, cell // self.window_size - self.hw
        return (self.focus[0] + dx, self.focus[1] + dy)

    def _obs(self) -> dict:
        assert self.board is not None
        H = W = self.window_size
        t = np.zeros((N_CH, H, W), dtype=np.float32)
        for iy in range(H):
            for ix in range(W):
                x, y = self.focus[0] + ix - self.hw, self.focus[1] + iy - self.hw
                if (x, y) in self.board.flags:
                    t[1, iy, ix] = 1.0
                elif (x, y) in self.board.revealed:
                    t[2 + self.board.revealed[(x, y)], iy, ix] = 1.0
                else:
                    t[0, iy, ix] = 1.0
        return {"obs": t, "action_mask": self.action_masks().astype(np.int8)}

    def reset(self, *, seed=None, options=None):
        first_reset = seed is None and self.board is None
        # A no-seed initial reset uses the constructor seed; later auto-resets
        # draw fresh board seeds from Gym's continuing RNG.
        super().reset(seed=self.seed_val if first_reset else seed)
        s = (self.seed_val if first_reset else
             int(self.np_random.integers(0, np.iinfo(np.int32).max))
             if seed is None else seed)
        self.board = InfiniteBoard(seed=s, p=self.p,
                                   max_steps=self.max_steps,
                                   max_revealed=self.max_revealed)
        self.focus = (0, 0)
        self.board.reveal(0, 0)  # safe start, gives initial numbers
        return self._obs(), {"focus": self.focus}

    def step(self, action):
        assert self.board is not None
        action = int(action)
        flag_action = self.flag_actions and action >= self.n_cells
        x, y = self._coords(action)
        cell = (x, y)
        if flag_action:
            # Agent flags are one-way. Toggling a correct flag off and on would
            # let the policy farm the positive correct-flag reward indefinitely.
            if (cell in self.board.revealed or cell in self.board.safe
                    or cell in self.board.flags):
                self.board.steps += 1
                return self._obs(), -0.1, False, self.board.over_budget(), {
                    "invalid": True, "flag_action": True,
                    "flagged": False, "focus": self.focus
                }
            self.board.flag(x, y)
            correct = self.board.mine_at(x, y)
            self.board.steps += 1
            # At p=0.16 this is near-neutral for a blind flag, rewarding flags
            # only when clues make the cell more likely to be a mine than baseline.
            reward = (1.0 if correct else -0.17) - self.step_cost
            return self._obs(), reward, False, self.board.over_budget(), {
                "flag_action": True, "flagged": True,
                "correct": correct, "cell": cell, "focus": self.focus
            }
        if (x, y) in self.board.revealed or (x, y) in self.board.flags:
            # Masked agents never do this; unmasked ones can. Count it as a step
            # so episodes always terminate, and end when no legal move remains
            # (window fully revealed) instead of looping forever.
            self.board.steps += 1
            obs = self._obs()
            trunc = self.board.over_budget() or not self._legal_mask().any()
            return obs, -0.1, False, trunc, {"invalid": True, "focus": self.focus}
        newly, hit = self.board.reveal(x, y)
        self.focus = (x, y)
        n = len(newly)
        if hit:
            rew = -self.mine_penalty
            return self._obs(), rew, True, False, {"hit_mine": True, "new": 0,
                    "revealed": len(self.board.revealed), "focus": self.focus}
        if self.reward_mode == "cascade":
            rew = float(n) + (0.5 * (n - 1) if n > 1 else 0.0) - self.step_cost
        elif self.reward_mode == "sparse":
            rew = float(n)
        else:  # dense
            rew = float(n) - self.step_cost
        term, trunc = False, self.board.over_budget()
        info = {"hit_mine": False, "new": n, "revealed": len(self.board.revealed),
                "focus": self.focus}
        return self._obs(), rew, term, trunc, info
