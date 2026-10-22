"""Matplotlib viewer for the plain (no colorful background) board.

Usage:
    from env.viewer import render_window
    render_window(board, focus, window_size=15)  # shows plot
    render_window(board, focus, save="shot.png") # headless save
"""
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np

CELL_COLORS = ["#ef9a9a", "#c8cdd2", "#c8cdd2"] + ["#f5f5f0"] * 9
CELL_COLORS_BY_STATE = {"mine": CELL_COLORS[0], "hidden": CELL_COLORS[1], "flag": CELL_COLORS[2]}
CELL_CMAP = ListedColormap(CELL_COLORS).with_extremes(bad=CELL_COLORS_BY_STATE["hidden"])
CELL_NORM = BoundaryNorm(np.arange(-3.5, 9, 1), CELL_CMAP.N)
NUMBER_COLORS = {
    0: "#455a64", 1: "#1976d2", 2: "#388e3c", 3: "#d32f2f",
    4: "#303f9f", 5: "#7b1f1f", 6: "#00838f", 7: "#212121", 8: "#616161",
}


def window_array(board, focus, window_size=15):
    hw = window_size // 2
    arr = np.full((window_size, window_size), -2.0)  # -2 hidden, -1 flag, 0-8
    for iy in range(window_size):
        for ix in range(window_size):
            x, y = focus[0] + ix - hw, focus[1] + iy - hw
            if (x, y) in board.flags:
                arr[iy, ix] = -1.0
            elif (x, y) in board.revealed:
                arr[iy, ix] = float(board.revealed[(x, y)])
    return arr


def render_window(board, focus, window_size=15, save=None, title=None):
    arr = window_array(board, focus, window_size)
    fig, ax = plt.subplots(figsize=(5, 5))
    hw = window_size // 2
    shown = np.array(arr, copy=True)
    if board.dead:
        for iy in range(window_size):
            for ix in range(window_size):
                x, y = focus[0] + ix - hw, focus[1] + iy - hw
                if board.mine_at(x, y):
                    shown[iy, ix] = -3
    im = ax.imshow(shown, cmap=CELL_CMAP, norm=CELL_NORM)
    for iy in range(window_size):
        for ix in range(window_size):
            v = shown[iy, ix]
            if v == -2:
                continue
            if v == -1:
                ax.plot([ix - 0.12, ix - 0.12], [iy - 0.22, iy + 0.20],
                        color="#5d4037", linewidth=2)
                ax.add_patch(plt.Polygon([(ix - 0.10, iy - 0.20), (ix + 0.20, iy - 0.08),
                                          (ix - 0.10, iy + 0.02)], closed=True,
                                         facecolor="#d32f2f", edgecolor="#b71c1c"))
            elif v >= 0:
                ax.text(ix, iy, str(int(v)), ha="center", va="center",
                        color=NUMBER_COLORS[int(v)], fontweight="bold", fontsize=7)
            elif v == -3:
                ax.scatter([ix], [iy], marker="*", s=120, color="#212121")
    ax.plot(hw, hw, "r+", ms=12, mew=2)  # focus
    ax.set_title(title or f"revealed={len(board.revealed)} focus={focus}")
    fig.colorbar(im, ax=ax, shrink=0.8)
    if save:
        fig.savefig(save, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return arr
