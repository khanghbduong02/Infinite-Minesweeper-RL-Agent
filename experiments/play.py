"""Watch an agent play Infinite Minesweeper in a live Matplotlib window.

Usage:
    conda activate minesweeper-rl
    python -m experiments.play --agent dqn --model dqn_cnn.npz

Click Play to advance actions, Pause to stop, Replay for the same seed, or
New Game to advance to the next reproducible seed. Safe-only pauses at guesses;
click Continue to explicitly resume with the lowest-risk guess.
"""
from __future__ import annotations

import argparse
import random
import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon, Rectangle
from matplotlib.widgets import Button

from agents.policies import load_policy
from env.gym_wrapper import MinesweeperWindowEnv
from env.mines import InfiniteBoard
from env.viewer import CELL_CMAP, CELL_COLORS_BY_STATE, CELL_NORM, NUMBER_COLORS, window_array
from solvers.baselines import (random_move, rules_move, csp_move, chordable_clue,
                               window_csp_candidates, chordable_clue_in_window)

PLAYBACK_SPEED = 1.0  # 1x = no extra pacing delay; 0.5x = half native rate
PLAYBACK_FRAME_MS = 10  # GUI event-loop poll interval in ms


def observed_action_rate(durations):
    """Actions per second from wall-clock time spent between completed actions."""
    return len(durations) / sum(durations) if durations and sum(durations) > 0 else None


class PlaySession:
    def __init__(self, agent, model, density, seed, window, max_actions,
                 safe_only=False):
        self.agent, self.model = agent, model
        self.density, self.seed = density, seed
        self.window, self.max_actions = window, max_actions
        self.safe_only = safe_only
        if safe_only and agent not in ("ppo", "dqn"):
            raise ValueError("--safe-only requires a PPO or DQN policy")
        self.waiting_for_safe_move = False
        self.rng = random.Random(seed)
        self.act = None
        self.env = None
        self.flag_actions = False
        if agent in ("ppo", "dqn"):
            if not model:
                raise ValueError(f"--agent {agent} needs --model PATH")
            loaded_policy = load_policy(agent, model)
            self.act, self.window = loaded_policy[:2]
            self.flag_actions = loaded_policy[2] if len(loaded_policy) == 3 else False
            self.env = MinesweeperWindowEnv(window_size=self.window, p=density, seed=seed,
                                            max_steps=max_actions, max_revealed=max_actions,
                                            flag_actions=self.flag_actions)
        self.reset()

    @property
    def board(self):
        return self.env.board if self.env else self._board

    @property
    def focus(self):
        return self.env.focus if self.env else self._focus

    def reset(self, new_game=False):
        if new_game:
            self.seed += 1
        self.moves, self.points, self.done = 0, 0.0, False
        self.last_action, self.last_new, self.last_hit = None, 0, False
        self.last_step_seconds = None
        self.last_guess_risk = None
        self.waiting_for_safe_move = False
        self.message = "Ready"
        self.rng = random.Random(self.seed)
        if self.env:
            self.obs, _ = self.env.reset(seed=self.seed)
        else:
            self._board = InfiniteBoard(seed=self.seed, p=self.density,
                                        max_steps=self.max_actions,
                                        max_revealed=self.max_actions)
            self._board.reveal(0, 0)
            self._focus = (0, 0)


    def _agent_chord(self):
        clue = (chordable_clue_in_window(self.obs, self.focus) if self.env
                else chordable_clue(self.board))
        if clue is None:
            return False
        newly, hit, used = self.board.chord(*clue)
        if not used:
            return False
        if self.env:
            self.env.focus = clue
            self.obs = self.env._obs()
            reward = (-self.env.mine_penalty if hit else len(newly) - self.env.step_cost)
            self.done = hit or self.board.over_budget()
        else:
            self._focus = clue
            reward = -10.0 if hit else len(newly) - 0.02
            self.done = hit or (self.max_actions is not None
                                and self.moves + 1 >= self.max_actions)
        self.points += reward
        self.moves += 1
        self.last_action, self.last_new, self.last_hit = clue, len(newly), hit
        self.message = "Mine hit" if hit else f"Auto-chord revealed {len(newly)} cells"
        return True

    def advance(self):
        if self.done or self.agent == "human":
            return
        if self.env:
            mask = self.obs["action_mask"].astype(bool)
            candidates, risk, certain_mines = window_csp_candidates(
                self.obs, self.focus, self.density, cap=100_000, max_cells=25)
            for mine in certain_mines:
                self.board.flags.add(mine)
            if certain_mines:
                self.obs = self.env._obs()
                mask = self.obs["action_mask"].astype(bool)
            if self._agent_chord():
                return
            if self.safe_only and risk > 0.0:
                self.waiting_for_safe_move = True
                self.message = "Paused: no logically guaranteed-safe move; Continue to guess"
                return
            # The local solver never reads or recenters to hidden board state.
            self.obs = self.env._obs()
            mask = self.obs["action_mask"].astype(bool)
            if self.flag_actions:
                flag_mask = np.zeros(self.env.n_cells, dtype=bool)
                for action in range(self.env.n_cells):
                    flag_mask[action] = self.env._coords(action) in certain_mines
                mask[self.env.n_cells:] &= flag_mask
            else:
                mask[self.env.n_cells:] = False
            # Let the learned model rank only equally low-risk legal actions.
            candidate_mask = np.zeros(self.env.n_cells, dtype=bool)
            for action in range(self.env.n_cells):
                candidate_mask[action] = self.env._coords(action) in candidates
            mask[:self.env.n_cells] &= candidate_mask
            if not mask.any():
                raise RuntimeError("local CSP selected no legal reveal or certain mine")
            started = time.perf_counter()
            self.obs, reward, term, trunc, info = self.env.step(self.act(self.obs, mask))
            self.last_guess_risk = risk
            self.last_step_seconds = time.perf_counter() - started
            self.points += reward
            self.last_action = info.get("cell", info["focus"])
            self.last_new = info.get("new", 0)
            self.last_hit = term
            self.moves += 1
            self.done = term or trunc or (self.max_actions is not None and self.moves >= self.max_actions)
            if info.get("flag_action"):
                self.message = (("Flagged mine" if info["correct"] else "Incorrect flag")
                                if info["flagged"] else "Flag removed")
            elif term:
                self.message = "Mine hit"
            else:
                self.message = (f"Guess ({risk:.1%} mine risk)" if risk > 0.0
                                else "Proven-safe move")
                if risk > 0.0:
                    self.safe_only = False
        else:
            if self.agent == "random":
                cell = random_move(self.board, self.rng)
            elif self.agent == "rules":
                cell, _ = rules_move(self.board, self.rng)
            else:
                cell, _, _ = csp_move(self.board, self.rng)
                if self._agent_chord():
                    return
            started = time.perf_counter()
            newly, hit = self.board.reveal(*cell)
            self.last_step_seconds = time.perf_counter() - started
            self._focus = cell
            self.last_action, self.last_new, self.last_hit = cell, len(newly), hit
            self.moves += 1
            self.points += -10.0 if hit else len(newly) - 0.02
            self.done = hit or (self.max_actions is not None and self.moves >= self.max_actions)
            self.message = "Mine hit" if hit else "Playing"
        if self.max_actions is not None and self.moves >= self.max_actions and self.message not in ("Mine hit", "Budget reached"):
            self.message = "Action limit reached"

    def click_chord(self, ix, iy, center=None):
        """Reveal all hidden neighbors when a revealed clue's flags match it."""
        if (self.agent != "human" or self.done
                or not (0 <= ix < self.window and 0 <= iy < self.window)):
            return False
        cx, cy = center or self.focus
        x, y = cx + ix - self.window // 2, cy + iy - self.window // 2
        clue = self.board.revealed.get((x, y))
        if clue is None:
            return False
        flagged = sum((x + dx, y + dy) in self.board.flags
                      for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dx, dy) != (0, 0))
        if flagged != clue:
            self.message = f"Chord needs {clue} flags; found {flagged}"
            return True
        targets = [(x + dx, y + dy) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                   if (dx, dy) != (0, 0)
                   if (x + dx, y + dy) not in self.board.revealed
                   and (x + dx, y + dy) not in self.board.flags]
        if not targets:
            return True
        started = time.perf_counter()
        newly, hit, used = self.board.chord(x, y)
        if not used:
            return False
        self.last_step_seconds = time.perf_counter() - started
        self._focus = (x, y)
        self.last_action = next((cell for cell in targets
                                 if self.board.dead and self.board.mine_at(*cell)), (x, y))
        self.last_new, self.last_hit = len(newly), hit
        self.moves += 1
        self.points += -10.0 if hit else len(newly) - 0.02
        self.done = hit or (self.max_actions is not None and self.moves >= self.max_actions)
        self.message = "Mine hit" if hit else f"Chord revealed {len(newly)} cells"
        return True

    def click_cell(self, ix, iy, flag=False, center=None):
        """Apply a board click at local window position (ix, iy)."""
        if not (0 <= ix < self.window and 0 <= iy < self.window):
            return False
        if self.done and not flag:
            return False
        cx, cy = center or self.focus
        cell = (cx + ix - self.window // 2, cy + iy - self.window // 2)
        if flag:
            if cell in self.board.revealed or cell in self.board.safe:
                return False
            if cell in self.board.flags:
                self.board.flag(*cell)
            else:
                self.board.flags.add(cell)
            if self.env:
                self.obs = self.env._obs()
            self.last_action, self.last_new = cell, 0
            self.message = "Flagged" if cell in self.board.flags else "Unflagged"
            return True
        if cell in self.board.revealed or cell in self.board.flags:
            return False
        if self.env:
            started = time.perf_counter()
            self.obs, reward, term, trunc, info = self.env.step(iy * self.window + ix)
            self.last_step_seconds = time.perf_counter() - started
            self.points += reward
            self.last_action = info.get("cell", info["focus"])
            self.last_new = info.get("new", 0)
            self.last_hit = term
            self.moves += 1
            self.done = term or trunc or (self.max_actions is not None and self.moves >= self.max_actions)
        else:
            self.board.flags.discard(cell)
            started = time.perf_counter()
            newly, hit = self.board.reveal(*cell)
            self.last_step_seconds = time.perf_counter() - started
            self._focus = cell
            self.last_action, self.last_new, self.last_hit = cell, len(newly), hit
            self.moves += 1
            self.points += -10.0 if hit else len(newly) - 0.02
            self.done = hit or (self.max_actions is not None and self.moves >= self.max_actions)
        self.message = "Mine hit" if self.last_hit else "Playing"
        return True


def build_window(session):
    if PLAYBACK_SPEED <= 0:
        raise ValueError("PLAYBACK_SPEED must be positive")
    fig, ax = plt.subplots(figsize=(7, 7))
    fig.subplots_adjust(bottom=0.20, top=0.90)
    fig.suptitle("Infinite Minesweeper — Agent Play")
    flag_guide_text = None
    if session.agent == "human":
        flag_guide_text = fig.text(0.5, 0.93,
                                   "Left-click reveal · Right-click flag · hold both on a clue to chord · drag to inspect",
                                   ha="center", va="center", fontsize=10)
    elif session.flag_actions:
        flag_guide_text = fig.text(0.5, 0.93, "Solver auto-flags certain mines and chords matching clues",
                                   ha="center", va="center", fontsize=10)
    hw = session.window // 2
    view_center = list(session.focus)
    arr = window_array(session.board, view_center, session.window)
    image = ax.imshow(arr, cmap=CELL_CMAP, norm=CELL_NORM)
    focus_marker, = ax.plot(hw, hw, "r+", ms=14, mew=2)
    ax.set_xticks(range(session.window))
    ax.set_yticks(range(session.window))
    ax.set_xticklabels(range(session.window))
    ax.set_yticklabels(range(session.window))
    ax.grid(color="white", alpha=0.25, zorder=1)
    image.set_cmap(CELL_CMAP)
    image.set_norm(CELL_NORM)
    image.set_zorder(0)
    status = fig.text(0.5, 0.16, "", ha="center", fontsize=9)
    speed_label = fig.text(0.5, 0.125, "Observed: measuring actions/s", ha="center", fontsize=9)
    footer_text = (None if session.agent == "human" else
                   fig.text(0.5, 0.09,
                            f"{PLAYBACK_SPEED:g}x playback · drag to inspect · Continue recenters",
                            ha="center", fontsize=9))
    flag_patches, mine_patches, number_labels = [], [], []
    mine_artist = ax.scatter([], [], marker="$✹$", s=120, color="#111111", zorder=4)
    speed_samples = []

    def draw():
        arr = window_array(session.board, view_center, session.window)
        if session.last_hit and session.last_action is not None:
            x = session.last_action[0] - view_center[0] + hw
            y = session.last_action[1] - view_center[1] + hw
            if 0 <= x < session.window and 0 <= y < session.window:
                arr[y, x] = -3
        image.set_data(arr)
        for artist in flag_patches + mine_patches + number_labels:
            artist.remove()
        flag_patches.clear()
        mine_patches.clear()
        number_labels.clear()
        mine_x, mine_y = [], []
        number_labels.extend(ax.texts)
        for label in number_labels:
            label.remove()
        number_labels.clear()
        for iy in range(session.window):
            for ix in range(session.window):
                value = arr[iy, ix]
                x, y = view_center[0] + ix - hw, view_center[1] + iy - hw
                if value == -1 and (session.agent == "human" or session.flag_actions):
                    flag_patches.append(ax.add_patch(Rectangle(
                        (ix - .5, iy - .5), 1, 1,
                        facecolor=CELL_COLORS_BY_STATE["flag"], edgecolor="#777777",
                        linewidth=0.7, zorder=5)))
                    pole = ax.plot([ix - 0.12, ix - 0.12], [iy + 0.27, iy - 0.27],
                                   color="#252525", linewidth=2.8, zorder=6)[0]
                    flag = Polygon([(ix - 0.10, iy - 0.25), (ix + 0.38, iy - 0.08),
                                    (ix - 0.10, iy + 0.09)], closed=True,
                                   facecolor="#e31b23", edgecolor="#8b0000", linewidth=0.9,
                                   zorder=6)
                    ax.add_patch(flag)
                    flag_patches.extend((pole, flag))
                elif value == -1:
                    pass  # a reveal-only policy should never create flag state
                elif value == -3 or (session.board.dead and session.board.mine_at(x, y)
                                     and (x, y) not in session.board.flags):
                    if value == -3:
                        mine_patches.append(ax.add_patch(Rectangle(
                            (ix - .5, iy - .5), 1, 1,
                            facecolor="#d71920", edgecolor="none", zorder=2)))
                    mine_x.append(ix)
                    mine_y.append(iy)
                elif value >= 0:  # classic Minesweeper clue colors
                    number_labels.append(ax.text(ix, iy, str(int(value)), ha="center", va="center",
                                                 color=NUMBER_COLORS[int(value)], fontweight="bold",
                                                 zorder=4))
        mine_artist.set_offsets(np.column_stack((mine_x, mine_y)) if mine_x else np.empty((0, 2)))
        fx, fy = session.focus[0] - view_center[0] + hw, session.focus[1] - view_center[1] + hw
        focus_marker.set_data(([fx], [fy]) if 0 <= fx < session.window and 0 <= fy < session.window else ([], []))
        last = "—" if session.last_action is None else f"{session.last_action} (+{session.last_new})"
        status.set_text(
            f"{session.agent} · seed {session.seed} · points {session.points:.1f} · "
            f"revealed {len(session.board.revealed)} · actions {session.moves} · {session.message}"
        )
        rate = observed_action_rate(speed_samples)
        speed_label.set_text(
            f"Observed: {rate:.1f} actions/s (wall-clock)" if rate is not None
            else "Observed: measuring actions/s"
        )
        fig.canvas.draw_idle()

    running = [False]
    play_ax = fig.add_axes([0.25, 0.018, 0.14, 0.04])
    replay_ax = fig.add_axes([0.43, 0.018, 0.14, 0.04])
    new_game_ax = fig.add_axes([0.61, 0.018, 0.14, 0.04])
    play_button = Button(play_ax, "Continue")
    replay_button = Button(replay_ax, "Replay")
    new_game_button = Button(new_game_ax, "New Game")
    if session.agent == "human":
        play_button.ax.set_visible(False)
    if PLAYBACK_SPEED <= 0:
        raise ValueError("PLAYBACK_SPEED must be positive")
    # Poll for input at 100 Hz; action rate is measured from completed GUI ticks.
    timer = fig.canvas.new_timer(interval=PLAYBACK_FRAME_MS)
    next_step_at = [0.0]

    last_action_finished_at = [None]

    def tick():
        if not running[0] or time.perf_counter() < next_step_at[0]:
            return
        started = time.perf_counter()
        previous_moves = session.moves
        session.advance()
        view_center[:] = session.focus
        draw()
        finished_at = time.perf_counter()
        if session.waiting_for_safe_move:
            running[0] = False
            timer.stop()
            play_button.label.set_text("Continue")
        elapsed = max(finished_at - started, 1e-6)
        if session.moves > previous_moves:
            previous = last_action_finished_at[0]
            speed_samples.append(max(finished_at - (previous or started), 1e-6))
            last_action_finished_at[0] = finished_at
            del speed_samples[:-30]
            draw()  # refresh the measured rate with this completed action included
        next_step_at[0] = time.perf_counter() + elapsed * max(0.0, 1.0 / PLAYBACK_SPEED - 1.0)
        if session.done:
            running[0] = False
            timer.stop()
            play_button.label.set_text("Continue")

    def toggle(_event):
        if running[0]:
            running[0] = False
            timer.stop()
            last_action_finished_at[0] = None  # don't count time spent paused
            play_button.label.set_text("Continue")
            draw()
            return
        if session.waiting_for_safe_move:
            session.safe_only = False  # Continue explicitly accepts the solver's best-risk guess.
            session.waiting_for_safe_move = False
        view_center[:] = session.focus
        if session.done or session.agent == "human":
            draw()
            return
        running[0] = True
        play_button.label.set_text("Pause" if running[0] else "Continue")
        if running[0] and not session.done:
            timer.start()
        else:
            running[0] = False
            timer.stop()
        draw()

    def reset(_event, new_game=False):
        running[0] = False
        speed_samples.clear()
        last_action_finished_at[0] = None
        next_step_at[0] = 0.0
        timer.stop()
        session.reset(new_game=new_game)
        last_action_finished_at[0] = None
        view_center[:] = session.focus
        play_button.label.set_text("Continue")
        draw()

    timer.add_callback(tick)

    def board_click(event):
        if session.agent != "human" or drag["moved"]:
            return
        if event.inaxes is not ax or event.xdata is None or event.ydata is None:
            return
        if event.button not in (1, 3):
            return
        ix, iy = int(np.floor(event.xdata + .5)), int(np.floor(event.ydata + .5))
        if not (0 <= ix < session.window and 0 <= iy < session.window):
            return
        running[0] = False
        timer.stop()
        play_button.label.set_text("Continue")
        if session.click_cell(ix, iy, flag=(event.button == 3), center=view_center):
            view_center[:] = session.focus
            draw()

    drag = {"start": None, "center": None, "moved": False, "button": None,
            "chorded": False}
    buttons_down = {}

    def mouse_down(event):
        if event.inaxes is not ax or event.button not in (1, 3) or event.xdata is None or event.ydata is None:
            return
        ix, iy = int(np.floor(event.xdata + .5)), int(np.floor(event.ydata + .5))
        if not (0 <= ix < session.window and 0 <= iy < session.window):
            return
        buttons_down[event.button] = (ix, iy)
        drag["start"] = (event.xdata, event.ydata)
        drag["center"] = tuple(view_center)
        drag["moved"] = False
        drag["button"] = event.button
        if (set(buttons_down) == {1, 3}
                and buttons_down[1] == buttons_down[3]):
            drag["chorded"] = True
            session.click_chord(ix, iy, center=view_center)
            draw()

    def mouse_move(event):
        if running[0] or drag["start"] is None or drag["button"] != 1 or event.inaxes is not ax:
            return
        dx = int(round(drag["start"][0] - event.xdata))
        dy = int(round(drag["start"][1] - event.ydata))
        drag["moved"] |= dx != 0 or dy != 0
        view_center[0] = drag["center"][0] + dx
        view_center[1] = drag["center"][1] + dy
        draw()

    def mouse_up(event):
        buttons_down.pop(event.button, None)
        if drag["chorded"]:
            if not buttons_down:
                drag.update(start=None, center=None, moved=False, button=None, chorded=False)
            return
        was_drag = drag["moved"]
        drag["start"] = drag["center"] = None
        released_button = drag["button"]
        drag["button"] = None
        if was_drag:
            drag["moved"] = False
        elif session.agent == "human" and event.inaxes is ax and released_button in (1, 3):
            event.button = released_button
            board_click(event)

    fig.canvas.mpl_connect("button_press_event", mouse_down)
    fig.canvas.mpl_connect("motion_notify_event", mouse_move)
    fig.canvas.mpl_connect("button_release_event", mouse_up)
    play_button.on_clicked(toggle)
    replay_button.on_clicked(reset)
    new_game_button.on_clicked(lambda event: reset(event, new_game=True))

    # Button and timer objects must stay alive for Matplotlib event callbacks.
    fig._play_widgets = (play_button, replay_button, new_game_button, timer, image, ax, status,
                         mine_artist, speed_label, focus_marker)
    fig._view_center = view_center
    fig._mouse_down = mouse_down
    fig._mouse_move = mouse_move
    fig._mouse_up = mouse_up
    fig._board_click = board_click
    fig._flag_artists = flag_patches
    fig._flag_guide_text = flag_guide_text
    fig._footer_text = footer_text
    fig._draw_board = draw
    draw()
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", default="rules",
                        choices=["human", "random", "rules", "csp", "ppo", "dqn"])
    parser.add_argument("--model", help="PPO checkpoint or DQN .npz weights")
    parser.add_argument("--density", type=float, default=0.16)
    parser.add_argument("--window", type=int, default=15,
                        help="odd observation window size for non-policy agents")
    parser.add_argument("--moves", type=int, default=None,
                        help="optional maximum actions before the viewer stops (default: no action limit)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--safe-only", action="store_true",
                        help="pause instead of taking the solver's lowest-risk guess; Continue accepts it")
    args = parser.parse_args()
    if args.window < 1 or args.window % 2 == 0:
        parser.error("--window must be a positive odd number")
    if args.moves is not None and args.moves < 1:
        parser.error("--moves must be positive")
    try:
        session = PlaySession(args.agent, args.model, args.density, args.seed,
                              args.window, args.moves, safe_only=args.safe_only)
    except ValueError as exc:
        parser.error(str(exc))
    build_window(session)
    plt.show()


if __name__ == "__main__":
    main()
