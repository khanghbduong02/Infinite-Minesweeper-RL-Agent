from env.mines import InfiniteBoard, is_mine


def test_chunk_generation_deterministic():
    b1 = InfiniteBoard(seed=7, p=0.2)
    b2 = InfiniteBoard(seed=7, p=0.2)
    assert b1.chunk_mines(0, 0) == b2.chunk_mines(0, 0)
    assert b1.chunk_mines(3, -2) == b2.chunk_mines(3, -2)
    b3 = InfiniteBoard(seed=8, p=0.2)
    assert b1.chunk_mines(0, 0) != b3.chunk_mines(0, 0)


def test_safe_start():
    b = InfiniteBoard(seed=0, p=1.0)  # p=1 would mine everything except safe zone
    newly, hit = b.reveal(0, 0)
    assert not hit
    assert (0, 0) in b.revealed
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            assert not b.mine_at(dx, dy)


def test_cascade_opens_area():
    # bounded arena: ring of mines at Chebyshev distance 3 around (5,5),
    # interior all safe -> one reveal flood-fills the interior (fast, bounded).
    b = InfiniteBoard(seed=0, p=0.0)
    cx, cy = 5, 5
    ring = {(cx + dx, cy + dy)
            for dx in range(-3, 4) for dy in range(-3, 4)
            if max(abs(dx), abs(dy)) == 3}
    real_mine_at = b.mine_at
    b.mine_at = lambda x, y: True if (x, y) in ring else False
    try:
        newly, hit = b.reveal(cx, cy)
    finally:
        b.mine_at = real_mine_at
    assert not hit
    assert len(newly) > 10
    assert (cx, cy) in b.revealed


def test_number_consistency():
    b = InfiniteBoard(seed=123, p=0.2)
    newly, hit = b.reveal(0, 0)
    assert not hit
    for (x, y), num in b.revealed.items():
        expect = sum(1 for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                     if (dx, dy) != (0, 0) and b.mine_at(x + dx, y + dy))
        assert num == expect


def test_unseeded_resets_advance_the_board_seed():
    from env.gym_wrapper import MinesweeperWindowEnv

    env = MinesweeperWindowEnv(window_size=9, p=0.16, seed=7)
    env.reset(seed=123)
    first_seed = env.board.seed
    env.reset()
    second_seed = env.board.seed
    env.reset()
    third_seed = env.board.seed
    assert first_seed == 123
    assert len({first_seed, second_seed, third_seed}) == 3


def test_fully_revealed_window_episode_terminates():
    # p=0 -> one reveal floods the whole window, so no legal move remains.
    # An unmasked agent repeatedly picking a revealed cell must not loop forever.
    from env.gym_wrapper import MinesweeperWindowEnv

    env = MinesweeperWindowEnv(window_size=9, p=0.0, seed=0, max_steps=300)
    env.reset(seed=0)
    done = False
    for _ in range(1000):
        _, _, term, trunc, _ = env.step(40)  # centre cell, already revealed
        if term or trunc:
            done = True
            break
    assert done


def test_flag_action_space_rewards_and_mask():
    from env.gym_wrapper import MinesweeperWindowEnv

    n = 81
    env = MinesweeperWindowEnv(window_size=9, p=1.0, seed=0,
                               max_steps=None, max_revealed=None, flag_actions=True)
    obs, _ = env.reset(seed=0)
    assert env.action_space.n == 162
    assert obs["action_mask"].shape == (162,)
    # Safe-start cells cannot be flagged, but safe covered cells remain flaggable.
    safe_i = 40 + 1
    assert not obs["action_mask"][n + safe_i]
    safe_cell_i = 2 * 9 + 2
    cell = env._coords(n + safe_cell_i)
    assert obs["action_mask"][n + safe_cell_i]
    obs2, reward, term, trunc, info = env.step(n + safe_cell_i)
    assert not term and not trunc and info["flag_action"] and info["flagged"]
    assert info["correct"] is True and reward > 0
    assert cell in env.board.flags
    assert obs2["obs"][1, 2, 2] == 1
    assert not obs2["action_mask"][safe_cell_i]
    assert not obs2["action_mask"][n + safe_cell_i]
    # A correct flag pays once; agent actions cannot toggle it off and farm reward.
    _, reward, _, _, info = env.step(n + safe_cell_i)
    assert info["invalid"] and not info["flagged"]
    assert cell in env.board.flags and reward < 0

    false_env = MinesweeperWindowEnv(window_size=9, p=0.16, seed=0,
                                     max_steps=None, max_revealed=None, flag_actions=True)
    false_obs, _ = false_env.reset(seed=0)
    false_cell_i = next(i for i in range(n) if false_obs["action_mask"][n + i])
    false_cell = false_env._coords(n + false_cell_i)
    mine_at = false_env.board.mine_at
    false_env.board.mine_at = lambda x, y: False if (x, y) == false_cell else mine_at(x, y)
    _, reward, _, _, info = false_env.step(n + false_cell_i)
    assert info["correct"] is False and reward < 0


def test_unlimited_board_and_env_do_not_truncate_at_training_budgets():
    from env.gym_wrapper import MinesweeperWindowEnv

    board = InfiniteBoard(seed=0, p=0.16, max_steps=None, max_revealed=None)
    board.steps = 1000
    board.revealed = {(x, 0): 1 for x in range(2001)}
    assert not board.over_budget()

    env = MinesweeperWindowEnv(window_size=9, p=0.16, seed=0,
                               max_steps=None, max_revealed=None)
    env.reset(seed=0)
    assert env.board.max_steps is None
    assert env.board.max_revealed is None
    env.board.steps = 1000
    _, _, terminated, truncated, _ = env.step(41)  # safe-start neighbor
    assert not terminated and not truncated


def test_live_play_viewer_shows_controls_and_advances_agent():
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from types import SimpleNamespace

    from experiments.play import PlaySession, build_window
    from env.viewer import CELL_CMAP, CELL_NORM
    from agents.policies import load_policy

    policy, _ = load_policy("dqn", "dqn_cnn.npz")
    assert callable(policy)
    flag_policy = load_policy("dqn", "dqn_cnn_flags.npz")
    assert len(flag_policy) == 3 and flag_policy[2] is True
    flag_session = PlaySession("dqn", "dqn_cnn_flags.npz", 0.16, 0, 9, 20)
    assert flag_session.env.action_space.n == 162
    flag_session.advance()
    flag_fig = build_window(flag_session)
    assert flag_fig._flag_guide_text is not None
    plt.close(flag_fig)

    session = PlaySession("rules", None, 0.16, 0, 9, 10)
    fig = build_window(session)
    texts = [text.get_text() for text in fig.texts]
    assert any("1x agent real speed" in text for text in texts)
    assert any("drag to inspect" in text for text in texts)
    assert not any("flag" in text.lower() for text in texts)
    assert fig._flag_guide_text is None
    assert len(fig.axes[0].collections) == 1  # empty mine marker only
    assert len(fig._flag_artists) == 0
    assert fig._footer_text is not None
    assert fig._footer_text.get_position()[1] < fig.axes[0].get_position().y0
    assert len(fig._play_widgets) == 10
    assert any("points " in text for text in texts)
    assert [axis.get_title() for axis in fig.axes[1:4]] == ["", "", ""]
    assert [axis.get_children()[0].get_text() for axis in fig.axes[1:4]] == [
        "Continue", "Replay", "New Game"
    ]
    buttons = fig._play_widgets[:3]
    timer = fig._play_widgets[3]
    assert timer.interval == 10
    assert fig._play_widgets[4] is fig.axes[0].images[0]

    buttons[0]._observers.process("clicked", None)
    assert buttons[0].label.get_text() == "Pause"
    buttons[0]._observers.process("clicked", None)
    assert buttons[0].label.get_text() == "Continue"
    session.advance()
    replay_seed = session.seed
    buttons[1]._observers.process("clicked", None)
    assert session.seed == replay_seed and session.moves == 0
    session.advance()
    buttons[2]._observers.process("clicked", None)
    assert session.seed == replay_seed + 1 and session.moves == 0
    session.advance()
    assert session.moves == 1
    assert session.last_action is not None
    assert session.env is None
    center_before_pan = tuple(fig._view_center)
    fig._mouse_down(SimpleNamespace(inaxes=fig.axes[0], button=1, xdata=4, ydata=4))
    fig._mouse_move(SimpleNamespace(inaxes=fig.axes[0], xdata=2, ydata=3))
    assert tuple(fig._view_center) == (center_before_pan[0] + 2, center_before_pan[1] + 1)
    fig._mouse_up(SimpleNamespace(inaxes=fig.axes[0], button=1, xdata=2, ydata=3))
    buttons[0]._observers.process("clicked", None)
    assert tuple(fig._view_center) == session.focus
    assert buttons[0].label.get_text() == "Pause"
    buttons[0]._observers.process("clicked", None)
    assert buttons[0].label.get_text() == "Continue"
    image = fig.axes[0].images[0]
    assert image.get_zorder() < 4  # flag patches render above the board image
    assert image.get_cmap() is CELL_CMAP
    assert image.norm is CELL_NORM
    import numpy as np
    from env.viewer import CELL_COLORS_BY_STATE
    assert np.allclose(image.get_cmap().get_bad()[:3], matplotlib.colors.to_rgb(CELL_COLORS_BY_STATE["hidden"]))
    assert np.allclose(image.get_cmap()(image.norm(-3))[:3], matplotlib.colors.to_rgb(CELL_COLORS_BY_STATE["mine"]))
    assert np.allclose(image.get_cmap()(image.norm(-1))[:3], matplotlib.colors.to_rgb(CELL_COLORS_BY_STATE["flag"]))
    assert fig._play_widgets[3].interval == 10
    plt.close(fig)


def test_human_left_reveal_right_flag_and_seed_reset():
    from types import SimpleNamespace
    from experiments.play import PlaySession

    game = PlaySession("human", None, 0.16, 0, 9, None)
    assert game.click_cell(4, 4, flag=True) is False  # initial safe tile already revealed
    assert game.moves == 0

    hidden = next((x, y) for x in range(-4, 5) for y in range(-4, 5)
                  if (x, y) not in game.board.revealed and (x, y) not in game.board.flags)
    ix, iy = hidden[0] + 4, hidden[1] + 4
    assert game.click_cell(ix, iy, flag=True)
    assert hidden in game.board.flags and game.moves == 0
    from experiments.play import build_window
    import matplotlib
    matplotlib.use("Agg", force=True)
    fig = build_window(game)
    assert len(fig._flag_artists) == 3
    assert fig._flag_guide_text is not None
    assert "Right-click flag" in fig._flag_guide_text.get_text()
    assert fig._footer_text is None
    assert fig._flag_guide_text.get_position()[1] > fig.axes[0].get_position().y1
    assert all(artist.get_zorder() > fig.axes[0].images[0].get_zorder()
               for artist in fig._flag_artists)
    fig._draw_board()
    assert len(fig._flag_artists) == 3
    tile = fig.axes[0].images[0].get_cmap()(fig.axes[0].images[0].norm(-1))
    from env.viewer import CELL_COLORS_BY_STATE
    import numpy as np
    assert np.allclose(tile[:3], matplotlib.colors.to_rgb(CELL_COLORS_BY_STATE["hidden"]))
    fig._mouse_down(SimpleNamespace(inaxes=fig.axes[0], button=3, xdata=ix, ydata=iy))
    fig._mouse_up(SimpleNamespace(inaxes=fig.axes[0], button=3, xdata=ix, ydata=iy))
    assert hidden not in game.board.flags and game.moves == 0
    fig._mouse_down(SimpleNamespace(inaxes=fig.axes[0], button=3, xdata=ix, ydata=iy))
    fig._mouse_up(SimpleNamespace(inaxes=fig.axes[0], button=3, xdata=ix, ydata=iy))
    assert hidden in game.board.flags and game.moves == 0
    assert [a.get_zorder() for a in fig._flag_artists] == [5, 6, 6]
    import matplotlib.pyplot as plt
    plt.close(fig)
    assert game.click_cell(ix, iy, flag=True)
    assert hidden not in game.board.flags
    assert game.click_cell(ix, iy)
    assert game.moves == 1
    assert hidden in game.board.revealed or game.last_hit
    flag_center = (game.focus[0] + 5, game.focus[1])
    inspect_cell = (flag_center[0], flag_center[1])
    assert game.click_cell(4, 4, flag=True, center=flag_center)
    assert inspect_cell in game.board.flags


def test_mine_hit_ends():
    # find a mined cell far from safe zone for this seed
    seed, p = 1, 0.3
    target = None
    for x in range(10, 40):
        for y in range(10, 40):
            import random
            if random.Random(f"{seed}:{x},{y}").random() < p:
                target = (x, y)
                break
        if target:
            break
    assert target is not None
    b = InfiniteBoard(seed=seed, p=p)
    newly, hit = b.reveal(*target)
    assert hit and b.dead
