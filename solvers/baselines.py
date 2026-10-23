"""Baselines: random / rules / CSP-probability. All operate on InfiniteBoard."""
from __future__ import annotations

import random
from functools import lru_cache

from solvers.common import (
    frontier_cells, trivial_deductions, frontier_components, hidden_neighbors,
)


def random_move(board, rng, window=None):
    """Random hidden unflagged cell: prefer frontier, else window, else nearby."""
    fr = [c for c in frontier_cells(board.revealed, board.flags)]
    if fr:
        return rng.choice(fr)
    if window is not None:
        fx, fy, hw = window
        cands = [(fx + dx, fy + dy) for dy in range(-hw, hw + 1)
                 for dx in range(-hw, hw + 1)
                 if (fx + dx, fy + dy) not in board.revealed
                 and (fx + dx, fy + dy) not in board.flags]
        if cands:
            return rng.choice(cands)
    # fallback: random hidden neighbor ring around a revealed cell
    if board.revealed:
        bx, by = rng.choice(list(board.revealed))
    else:
        bx, by = 0, 0
    for _ in range(100):
        c = (bx + rng.randint(-4, 4), by + rng.randint(-4, 4))
        if c not in board.revealed and c not in board.flags:
            return c
    return (bx + 5, by)


def rules_move(board, rng):
    """Trivial deductions first; flags known mines; else random frontier."""
    safe, mines = trivial_deductions(board.revealed, board.flags)
    for m in mines:
        board.flags.add(m)
    safe, mines = trivial_deductions(board.revealed, board.flags)
    safe = {c for c in safe if c not in board.revealed and c not in board.flags}
    if safe:
        return sorted(safe)[rng.randrange(len(safe))], False
    fr = [c for c in frontier_cells(board.revealed, board.flags)]
    if not fr:
        return random_move(board, rng), True
    return rng.choice(fr), True  # forced guess


def _enumerate_probs(comp_cons, comp_cells, cap=100_000, p=0.16):
    """Exact per-cell mine probabilities by memoized constraint DP.

    Each state is the next cell plus residual mine counts for every clue. The
    returned probabilities are weighted by the board's independent mine prior.
    Return None when the component/state budget is exceeded, {} if inconsistent.
    """
    cells = sorted(comp_cells)
    if len(cells) > 25:
        return None
    # Assign highly constrained cells first to make residual-state pruning bite.
    degree = {cell: sum(cell in constraint for constraint, _ in comp_cons)
              for cell in cells}
    cells.sort(key=lambda cell: (-degree[cell], cell))
    index = {cell: i for i, cell in enumerate(cells)}
    cons = [(tuple(index[cell] for cell in constraint), need)
            for constraint, need in comp_cons]
    n = len(cells)
    uses = [tuple(j for j, (members, _) in enumerate(cons) if i in members)
            for i in range(n)]
    remaining = [tuple(sum(i >= pos for i in members) for members, _ in cons)
                 for pos in range(n + 1)]
    initial = tuple(need for _, need in cons)
    visited = 0

    @lru_cache(maxsize=None)
    def solve(pos, needs):
        nonlocal visited
        if visited >= cap:
            return None
        visited += 1
        if any(need < 0 or need > remaining[pos][j]
               for j, need in enumerate(needs)):
            return 0.0
        if pos == n:
            return 1.0 if not any(needs) else 0.0
        one_needs = list(needs)
        for j in uses[pos]:
            one_needs[j] -= 1
        zero = solve(pos + 1, needs)
        one = solve(pos + 1, tuple(one_needs))
        if zero is None or one is None:
            return None
        return (1.0 - p) * zero + p * one

    total = solve(0, initial)
    if total is None:
        return None
    if total <= 0.0:
        return {}

    # Forward prefix weights plus cached backward totals yield exact marginals
    # without storing a full probability vector in every DP state.
    prefix = {initial: 1.0}
    mine_mass = [0.0] * n
    for pos in range(n):
        following = {}
        for needs, mass in prefix.items():
            one_needs = list(needs)
            for j in uses[pos]:
                one_needs[j] -= 1
            one_needs = tuple(one_needs)
            zero_tail = solve(pos + 1, needs)
            one_tail = solve(pos + 1, one_needs)
            if zero_tail is None or one_tail is None:
                return None
            mine_mass[pos] += mass * p * one_tail
            if zero_tail:
                following[needs] = following.get(needs, 0.0) + mass * (1.0 - p)
            if one_tail:
                following[one_needs] = following.get(one_needs, 0.0) + mass * p
        prefix = following
    return {cell: mine_mass[index[cell]] / total for cell in cells}


def _probabilities_from_clues(revealed, flags, p, known_safe=(), cap=100_000,
                              max_cells=25):
    """Infer probabilities from only the supplied clues and flags."""
    known_safe = set(known_safe)
    while True:
        safe, mines = trivial_deductions(revealed, flags)
        flags.update(mines)
        probs = {cell: 0.0 for cell in (safe | known_safe)
                 if cell not in revealed and cell not in flags}
        for comp_cons, comp_cells in frontier_components(revealed, flags):
            exact = (_enumerate_probs(comp_cons, comp_cells, cap, p)
                     if len(comp_cells) <= max_cells else None)
            if exact is not None and exact:
                probs.update(exact)
            else:
                # Approximate probabilities are never treated as proofs.
                for cells, need in comp_cons:
                    estimate = min(1.0 - 1e-6,
                                   max(1e-6, need / max(1, len(cells))))
                    for cell in cells:
                        probs[cell] = max(probs.get(cell, 1e-6), estimate)
        certain_mines = {cell for cell, probability in probs.items()
                         if probability >= 1.0 - 1e-12}
        new_mines = certain_mines - flags
        if not new_mines:
            return probs
        flags.update(new_mines)


def csp_probabilities(board, cap=100_000, max_cells=25):
    """Infer frontier mine probabilities; flag only logically certain mines."""
    return _probabilities_from_clues(board.revealed, board.flags, board.p,
                                     board.safe, cap, max_cells)


def _observation_state(observation, center):
    """Decode only revealed clues and flags visible in one observation tensor."""
    image = observation["obs"]
    height, width = image.shape[1:]
    if height != width:
        raise ValueError("Minesweeper observation windows must be square")
    half = width // 2
    revealed, flags, hidden = {}, set(), set()
    for iy in range(height):
        for ix in range(width):
            cell = (center[0] + ix - half, center[1] + iy - half)
            if image[1, iy, ix] > 0.5:
                flags.add(cell)
            elif image[0, iy, ix] > 0.5:
                hidden.add(cell)
            else:
                number = next((value for value in range(9)
                               if image[2 + value, iy, ix] > 0.5), None)
                if number is not None:
                    revealed[cell] = number
    return revealed, flags, hidden


def window_csp_candidates(observation, center, p, cap=100_000, max_cells=25):
    """Rank legal reveals using only clue/flag state in the supplied window.

    Clues at the edge still constrain every neighboring hidden cell, including
    unobserved cells just outside the action window; those cells are unknowns,
    not facts read from the board. Return (best_cells, risk, certain_mines),
    with both action candidates and returned flags restricted to this window.
    """
    revealed, visible_flags, hidden = _observation_state(observation, center)
    flags = set(visible_flags)
    probs = _probabilities_from_clues(revealed, flags, p, cap=cap,
                                      max_cells=max_cells)
    certain_mines = (flags - visible_flags) & hidden
    candidates = hidden - flags
    if not candidates:
        return set(), 1.0, certain_mines
    risks = {cell: probs.get(cell, p) for cell in candidates}
    risk = min(risks.values())
    best = {cell for cell, probability in risks.items() if probability <= risk + 1e-12}
    return best, risk, certain_mines


def chordable_clue_in_window(observation, center):
    """Find a fully visible clue whose matching flags permit a local chord."""
    revealed, flags, hidden = _observation_state(observation, center)
    visible = set(revealed) | flags | hidden
    for cell, number in sorted(revealed.items()):
        x, y = cell
        neighbors = {(x + dx, y + dy) for dx, dy in
                     ((-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0),
                      (-1, 1), (0, 1), (1, 1))}
        if not neighbors <= visible:
            continue
        if sum(neighbor in flags for neighbor in neighbors) == number:
            if neighbors & hidden:
                return cell
    return None


def _unconstrained_cell(board, rng, center=None, frontier=None):
    """Find a fresh cell with prior risk p, outside every currently known clue."""
    if center is None:
        center = next(iter(board.revealed), (0, 0))
    blocked = set(board.revealed) | set(board.flags)
    blocked |= frontier if frontier is not None else frontier_cells(board.revealed, board.flags)
    cx, cy = center
    for radius in range(1, 1000):
        ring = [(cx + dx, cy + dy)
                for dy in range(-radius, radius + 1)
                for dx in range(-radius, radius + 1)
                if max(abs(dx), abs(dy)) == radius]
        candidates = [cell for cell in ring if cell not in blocked]
        if candidates:
            return rng.choice(candidates)
    raise RuntimeError("failed to find an unexplored cell")


def csp_candidates(board, rng, center=None, cap=100_000, max_cells=25):
    """Return minimum-risk candidate cells and their posterior mine probability."""
    probs = csp_probabilities(board, cap=cap, max_cells=max_cells)
    forced_safe = {cell for cell, probability in probs.items() if probability <= 1e-12}
    if forced_safe:
        return forced_safe, 0.0
    frontier = set(probs)
    blind = _unconstrained_cell(board, rng, center, frontier)
    blind_risk = board.p
    frontier_risk = min(probs.values(), default=1.0)
    if blind_risk <= frontier_risk:
        return {blind}, blind_risk
    best = min(probs.values())
    return {cell for cell, probability in probs.items()
            if probability <= best + 1e-12}, best


def chordable_clue(board):
    """Return a clue whose correctly-proven flags allow a free safe chord."""
    for cell, number in sorted(board.revealed.items()):
        if sum(nb in board.flags for nb in hidden_neighbors(cell, board.revealed, set())) == number:
            if hidden_neighbors(cell, board.revealed, board.flags):
                return cell
    return None


def csp_probabilities(board, cap=100_000, max_cells=25):
    """Infer frontier mine probabilities; flag only logically certain mines."""
    return _probabilities_from_clues(board.revealed, board.flags, board.p,
                                     board.safe, cap, max_cells)


def csp_move(board, rng, cap=100_000, max_cells=25, center=None):
    """Choose a proven-safe cell, otherwise the minimum-posterior-risk cell."""
    candidates, risk = csp_candidates(board, rng, center, cap, max_cells)
    return rng.choice(sorted(candidates)), risk > 0.0, risk
