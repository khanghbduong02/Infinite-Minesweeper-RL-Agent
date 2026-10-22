import random

import numpy as np

from env.mines import InfiniteBoard
from solvers.baselines import rules_move, csp_move, random_move
from solvers.common import trivial_deductions


def _board_with_pattern():
    # hand-built state: revealed 1 with one hidden neighbor unflagged etc.
    b = InfiniteBoard(seed=0, p=0.0)
    b.revealed = {(0, 0): 1}
    b.flags = {(1, 1)}
    return b


def test_trivial_safe_deduction():
    b = _board_with_pattern()
    # number 1, one flagged neighbor -> all other hidden neighbors safe
    safe, mines = trivial_deductions(b.revealed, b.flags)
    assert (1, 1) not in safe
    assert (0, 1) in safe and (1, 0) in safe


def test_rules_flags_mines():
    b = InfiniteBoard(seed=0, p=0.0)
    b.revealed = {(0, 0): 8}
    # 8 with 8 hidden neighbors -> all mines
    rng = random.Random(0)
    cell, guess = rules_move(b, rng)
    assert len(b.flags) == 8  # flagged all around


def test_csp_runs_and_returns_frontier():
    b = InfiniteBoard(seed=5, p=0.16)
    b.reveal(0, 0)
    rng = random.Random(1)
    cell, guess, prob = csp_move(b, rng)
    assert cell not in b.revealed and cell not in b.flags
    assert 0.0 <= prob <= 1.0


def test_learned_policy_can_use_rules_to_take_known_safe_reveals():
    from env.gym_wrapper import MinesweeperWindowEnv
    from experiments.eval import deduction_override

    env = MinesweeperWindowEnv(window_size=5, p=0.16, seed=3)
    env.reset(seed=3)
    env.board.revealed = {(0, 0): 0}
    env.board.flags.clear()
    obs = env._obs()
    action = deduction_override(env, obs, lambda _obs, mask: int(mask.nonzero()[0][-1]))
    assert env._coords(action) not in env.board.revealed
    assert env._coords(action) in {(x, y) for x in range(-1, 2)
                                   for y in range(-1, 2)
                                   if (x, y) not in env.board.revealed}


def test_all_baselines_survive_first_moves():
    for fn in (lambda b, r: random_move(b, r),
               lambda b, r: rules_move(b, r)[0],
               lambda b, r: csp_move(b, r)[0]):
        b = InfiniteBoard(seed=3, p=0.16)
        b.reveal(0, 0)
        rng = random.Random(0)
        for _ in range(5):
            c = fn(b, rng)
            newly, hit = b.reveal(*c)
            if hit:
                break
        assert len(b.revealed) >= 1
