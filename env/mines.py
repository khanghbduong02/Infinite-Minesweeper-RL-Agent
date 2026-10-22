"""Core infinite-minesweeper logic: deterministic lazy minefield + reveal/cascade.

No gymnasium/torch dependency here so baselines and tests stay lightweight.
Mine truth: random.Random((seed, x, y)).random() < p  (deterministic).
"""
from __future__ import annotations

from collections import deque
import random

CHUNK = 16
NEIGHBORS = [(dx, dy) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dx, dy) != (0, 0)]


def is_mine(seed: int, x: int, y: int, p: float) -> bool:
    return random.Random(f"{seed}:{x},{y}").random() < p


def neighbor_mine_count(seed: int, x: int, y: int, p: float, safe: set[tuple[int, int]]) -> int:
    n = 0
    for dx, dy in NEIGHBORS:
        c = (x + dx, y + dy)
        if c in safe:
            continue
        if is_mine(seed, c[0], c[1], p):
            n += 1
    return n


class InfiniteBoard:
    """Minimal mutable board state. Coordinates are unbounded ints."""

    def __init__(self, seed: int = 0, p: float = 0.16,
                 safe_center: tuple[int, int] = (0, 0), safe_radius: int = 1,
                 max_steps: int | None = 500, max_revealed: int | None = 2000):
        self.seed = seed
        self.p = p
        cx, cy = safe_center
        self.safe: set[tuple[int, int]] = {
            (cx + dx, cy + dy)
            for dx in range(-safe_radius, safe_radius + 1)
            for dy in range(-safe_radius, safe_radius + 1)
        }
        self.max_steps = max_steps
        self.max_revealed = max_revealed
        self.revealed: dict[tuple[int, int], int] = {}
        self.flags: set[tuple[int, int]] = set()
        self.steps = 0
        self.dead = False

    # -- queries ---------------------------------------------------------
    def mine_at(self, x: int, y: int) -> bool:
        if (x, y) in self.safe:
            return False
        return is_mine(self.seed, x, y, self.p)

    def number_at(self, x: int, y: int) -> int:
        return neighbor_mine_count(self.seed, x, y, self.p, self.safe)

    @property
    def revealed_count(self) -> int:
        return len(self.revealed)

    # -- actions ---------------------------------------------------------
    def flag(self, x: int, y: int) -> None:
        if (x, y) in self.revealed:
            return
        if (x, y) in self.flags:
            self.flags.discard((x, y))
        else:
            self.flags.add((x, y))

    def reveal(self, x: int, y: int) -> tuple[list[tuple[int, int]], bool]:
        """Reveal one cell. Returns (newly_revealed, hit_mine). Safe-start
        cells never contain mines. Zero cells flood-fill."""
        if self.dead or (x, y) in self.revealed or (x, y) in self.flags:
            return [], False
        self.steps += 1
        if self.mine_at(x, y):
            self.dead = True
            return [], True
        newly: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = {(x, y)}
        queue: deque[tuple[int, int]] = deque([(x, y)])
        cap = max(5000, self.max_revealed or 0)
        while queue:
            cx, cy = queue.popleft()
            if (cx, cy) in self.revealed or (cx, cy) in self.flags:
                continue
            if self.mine_at(cx, cy):
                continue  # never expand into a mine during cascade
            num = self.number_at(cx, cy)
            self.revealed[(cx, cy)] = num
            newly.append((cx, cy))
            if len(newly) >= cap:
                break  # degenerate open area (e.g. p=0); stop expanding
            if num == 0:
                for dx, dy in NEIGHBORS:
                    nb = (cx + dx, cy + dy)
                    if nb not in self.revealed and nb not in self.flags and nb not in seen:
                        seen.add(nb)
                        queue.append(nb)
        return newly, False

    def over_budget(self) -> bool:
        return ((self.max_steps is not None and self.steps >= self.max_steps)
                or (self.max_revealed is not None
                    and len(self.revealed) >= self.max_revealed))

    # -- chunk helper (for tests / future rendering) ---------------------
    def chunk_mines(self, cx: int, cy: int) -> list[list[bool]]:
        """Return CHUNK×CHUNK mine grid for chunk (cx, cy)."""
        grid = []
        for dy in range(CHUNK):
            row = []
            for dx in range(CHUNK):
                x, y = cx * CHUNK + dx, cy * CHUNK + dy
                row.append(self.mine_at(x, y))
            grid.append(row)
        return grid
