import random

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


def test_constraint_dp_detects_forced_safe_cells_from_overlapping_clues():
    from solvers.baselines import _enumerate_probs

    a, b, c = (0, 0), (1, 0), (2, 0)
    probabilities = _enumerate_probs(
        [((a, b), 1), ((b, c), 1), ((a, b, c), 1)], {a, b, c}, p=0.16)
    assert probabilities == {a: 0.0, b: 1.0, c: 0.0}
    # Independent constraints: exact posterior equals the prior conditioned on
    # exactly one mine, not uniform assignment weighting.
    weighted = _enumerate_probs([((a, b, c), 1)], {a, b, c}, p=0.1)
    assert all(abs(value - 1 / 3) < 1e-12 for value in weighted.values())


def test_csp_prefers_unconstrained_prior_risk_over_high_risk_frontier():
    from solvers.baselines import csp_candidates

    board = InfiniteBoard(seed=0, p=0.16)
    board.revealed = {(0, 0): 5}
    candidates, risk = csp_candidates(board, random.Random(0))
    assert risk == 0.16
    assert len(candidates) == 1
    cell = next(iter(candidates))
    assert cell not in board.revealed
    assert max(abs(cell[0]), abs(cell[1])) > 1


def test_window_solver_uses_only_observed_clues_and_returns_local_moves():
    from env.gym_wrapper import MinesweeperWindowEnv
    from solvers.baselines import window_csp_candidates

    env = MinesweeperWindowEnv(window_size=5, p=0.16, seed=3)
    env.reset(seed=3)
    env.board.revealed = {(0, 0): 0}
    env.board.flags.clear()
    obs = env._obs()
    expected = window_csp_candidates(obs, env.focus, env.p)

    # Change arbitrarily large off-screen board state without changing the input
    # observation: local inference must return the same answer and never read it.
    env.board.revealed[(10_000, 10_000)] = 8
    env.board.flags.add((10_001, 10_000))
    actual = window_csp_candidates(obs, env.focus, env.p)
    assert actual == expected
    candidates, risk, certain_mines = actual
    assert risk == 0.0
    assert candidates
    assert all(-2 <= x <= 2 and -2 <= y <= 2 for x, y in candidates)
    assert not certain_mines


def test_window_solver_keeps_outside_window_cells_unknown_for_edge_clues():
    from env.gym_wrapper import MinesweeperWindowEnv
    from solvers.baselines import window_csp_candidates

    env = MinesweeperWindowEnv(window_size=5, p=0.16, seed=3)
    env.reset(seed=3)
    # A clue on the right edge has neighbors outside the observation. One known
    # flag accounts for its single mine; the visible unflagged neighbors are safe.
    env.board.revealed = {(2, 0): 1}
    env.board.flags = {(1, 0)}
    candidates, risk, _ = window_csp_candidates(env._obs(), env.focus, env.p)
    assert risk == 0.0
    assert (2, 1) in candidates


def test_window_solver_flags_only_mines_proven_by_visible_clues():
    from env.gym_wrapper import MinesweeperWindowEnv
    from solvers.baselines import window_csp_candidates

    env = MinesweeperWindowEnv(window_size=5, p=0.16, seed=3)
    env.reset(seed=3)
    env.board.revealed = {(0, 0): 8}
    env.board.flags.clear()
    candidates, risk, certain_mines = window_csp_candidates(env._obs(), env.focus, env.p)
    neighbors = {(x, y) for x in (-1, 0, 1) for y in (-1, 0, 1) if (x, y) != (0, 0)}
    assert certain_mines == neighbors
    assert not candidates & neighbors
    assert risk == env.p


def test_learned_policy_can_use_rules_to_take_known_safe_reveals():
    from env.gym_wrapper import MinesweeperWindowEnv
    from experiments.eval import deduction_override, safe_policy_action

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

    env.board.revealed = {(0, 0): 1}
    env.board.flags.clear()
    env._obs()
    assert safe_policy_action(env, lambda _obs, mask: int(mask.nonzero()[0][0])) is None
    env.board.revealed = {(0, 0): 0}
    env.board.flags.clear()
    env._obs()
    safe_action = safe_policy_action(env, lambda _obs, mask: int(mask.nonzero()[0][0]))
    assert safe_action is not None
    assert env._coords(safe_action) in {(x, y) for x in range(-1, 2)
                                        for y in range(-1, 2)
                                        if (x, y) not in env.board.revealed}


def test_exact_csp_shield_does_not_abstain_when_overlapping_clues_prove_safe():
    from env.gym_wrapper import MinesweeperWindowEnv
    from experiments.eval import safe_policy_action

    env = MinesweeperWindowEnv(window_size=5, p=0.16, seed=3)
    env.reset(seed=3)
    env.board.revealed = {(0, 0): 0}
    env.board.flags.clear()
    action = safe_policy_action(
        env, lambda _obs, mask: int(mask.nonzero()[0][-1]))
    assert action is not None
    assert env._coords(action) in {(x, y) for x in range(-1, 2)
                                   for y in range(-1, 2)}
    assert env._coords(action) not in env.board.revealed


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
