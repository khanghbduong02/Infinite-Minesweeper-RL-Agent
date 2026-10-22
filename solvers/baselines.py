"""Baselines: random / rules / CSP-probability. All operate on InfiniteBoard."""
from __future__ import annotations

import random
from itertools import product

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


def _enumerate_probs(comp_cons, comp_cells, cap=100_000):
    cells = sorted(comp_cells)
    idx = {c: i for i, c in enumerate(cells)}
    cons_idx = [([idx[c] for c in cs], need) for cs, need in comp_cons]
    n = len(cells)
    if n > 25:  # too big: skip exact enumeration
        return None
    total, mine_counts = 0, [0] * n
    # iterate all assignments, early-exit if over cap
    count = 0
    for bits in product((0, 1), repeat=n):
        count += 1
        if count > cap:
            return None
        ok = True
        for cs, need in cons_idx:
            if sum(bits[i] for i in cs) != need:
                ok = False
                break
        if ok:
            total += 1
            for i, b in enumerate(bits):
                mine_counts[i] += b
    if total == 0:
        return {}
    return {c: mine_counts[idx[c]] / total for c in cells}


def csp_move(board, rng, cap=20_000, max_cells=15):
    """Exact-probability baseline: deductions + lowest-probability guess.

    Returns (cell, was_guess, prob). Enumerates each frontier component;
    oversized components fall back to need/hidden local estimate.
    """
    safe, mines = trivial_deductions(board.revealed, board.flags)
    for m in mines:
        board.flags.add(m)
    safe, mines = trivial_deductions(board.revealed, board.flags)
    safe = {c for c in safe if c not in board.revealed and c not in board.flags}
    if safe:
        c = sorted(safe)[rng.randrange(len(safe))]
        return c, False, 0.0
    probs: dict = {}
    for comp_cons, comp_cells in frontier_components(board.revealed, board.flags):
        if len(comp_cells) <= max_cells:
            exact = _enumerate_probs(comp_cons, comp_cells, cap)
            if exact is not None:
                probs.update(exact)
                continue
        for cs, need in comp_cons:  # fallback local estimate
            for c in cs:
                est = need / max(1, len(cs))
                if c in probs:
                    probs[c] = max(probs[c], est)
                else:
                    probs[c] = est
    if not probs:
        return random_move(board, rng), True, 1.0
    best = min(probs.values())
    cands = sorted([c for c, pr in probs.items() if pr == best])
    return rng.choice(cands), True, best
