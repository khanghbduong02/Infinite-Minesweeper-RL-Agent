"""Shared helpers: frontier + constraints from revealed numbers only.

Solvers never see mine truth; they only use revealed numbers + flags.
"""
from __future__ import annotations

from env.mines import NEIGHBORS


def hidden_neighbors(cell, revealed, flags):
    x, y = cell
    out = []
    for dx, dy in NEIGHBORS:
        nb = (x + dx, y + dy)
        if nb not in revealed and nb not in flags:
            out.append(nb)
    return out


def frontier_cells(revealed, flags):
    """Hidden unflagged cells adjacent to at least one revealed number."""
    fr = set()
    for cell in revealed:
        for c in hidden_neighbors(cell, revealed, flags):
            fr.add(c)
    return fr


def constraints(revealed, flags):
    """List of (cells_tuple, need) where need = mines among cells."""
    cons = []
    for cell, num in revealed.items():
        hid = hidden_neighbors(cell, revealed, flags)
        if not hid:
            continue
        x, y = cell
        flagged = sum((x + dx, y + dy) in flags for dx, dy in NEIGHBORS)
        need = num - flagged
        cons.append((tuple(hid), need))
    return cons


def trivial_deductions(revealed, flags):
    """Rule-based: number==flagged -> rest safe; number==flagged+hidden -> mines."""
    safe, mines = set(), set()
    for cell, num in revealed.items():
        hid = hidden_neighbors(cell, revealed, flags)
        if not hid:
            continue
        x, y = cell
        flagged = sum((x + dx, y + dy) in flags for dx, dy in NEIGHBORS)
        if num == flagged:
            safe.update(hid)
        elif num == flagged + len(hid):
            mines.update(hid)
    return safe, mines


def frontier_components(revealed, flags):
    """Split frontier constraints into connected components (via shared cells)."""
    cons = constraints(revealed, flags)
    if not cons:
        return []
    cell_to_cons: dict = {}
    for i, (cells, _) in enumerate(cons):
        for c in cells:
            cell_to_cons.setdefault(c, []).append(i)
    seen = [False] * len(cons)
    comps = []
    for i in range(len(cons)):
        if seen[i]:
            continue
        stack, seen[i] = [i], True
        comp_idx, comp_cells = set(), set()
        while stack:
            j = stack.pop()
            comp_idx.add(j)
            cells, _ = cons[j]
            comp_cells.update(cells)
            for c in cells:
                for k in cell_to_cons[c]:
                    if not seen[k]:
                        seen[k] = True
                        stack.append(k)
        comps.append(([cons[j] for j in comp_idx], comp_cells))
    return comps
