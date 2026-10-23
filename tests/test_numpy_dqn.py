import numpy as np
import pytest

from agents.numpy_dqn import CNN, N_CH, adam_step, double_dqn_targets, make_net, train


def test_adam_step_minimises_quadratic():
    # minimise f(x) = sum((x - 3)^2); gradient is 2 * (x - 3)
    x = np.zeros(4)
    m, v = [np.zeros_like(x)], [np.zeros_like(x)]
    for t in range(1, 2001):
        adam_step([x], [2 * (x - 3.0)], m, v, t, lr=0.05)
    assert np.allclose(x, 3.0, atol=1e-2)


@pytest.mark.parametrize("net", ["mlp", "cnn"])
def test_backward_matches_finite_differences(net):
    # loss = sum(Q * G) for fixed G, so dLoss/dQ = G and backward() must match
    # a numerical derivative for every parameter entry.
    rng = np.random.default_rng(0)
    window, B = 3, 2
    q = make_net(net, window, 2, rng)
    x = rng.normal(size=(B, N_CH * window * window))
    G = rng.normal(size=(B, window * window))

    def loss():
        Q, _ = q.forward(x)
        return float(np.sum(Q * G))

    _, cache = q.forward(x)
    grads = q.backward(cache, G)
    eps = 1e-6
    for p, g in zip(q.params(), grads):
        assert p.shape == g.shape
        for idx in np.ndindex(p.shape):
            old = p[idx]
            p[idx] = old + eps
            hi = loss()
            p[idx] = old - eps
            lo = loss()
            p[idx] = old
            assert abs((hi - lo) / (2 * eps) - g[idx]) < 1e-5


def test_mine_penalty_parameter_changes_terminal_reward():
    from env.gym_wrapper import MinesweeperWindowEnv

    env = MinesweeperWindowEnv(window_size=5, p=1.0, seed=1,
                               max_steps=None, max_revealed=None,
                               mine_penalty=37.0)
    obs, _ = env.reset(seed=1)
    action = next(i for i, legal in enumerate(obs["action_mask"])
                  if legal and env._coords(i) not in env.board.safe)
    _, reward, terminated, _, _ = env.step(action)
    assert terminated and reward == -37.0


def test_flag_action_network_outputs_spatial_reveal_and_flag_heads():
    rng = np.random.default_rng(4)
    for net in ("cnn", "mlp"):
        q = make_net(net, 3, 2, rng, flag_actions=True)
        Q, cache = q.forward(rng.normal(size=(2, N_CH * 9)))
        assert Q.shape == (2, 18)
        assert len(q.backward(cache, np.ones_like(Q))) == len(q.params())


def test_double_dqn_targets_choose_online_action_but_use_target_value():
    rewards = np.array([1.0, -2.0])
    dones = np.array([0.0, 1.0])
    online_q = np.array([[3.0, 8.0, 100.0], [9.0, 1.0, 0.0]])
    target_q = np.array([[30.0, 40.0, 500.0], [4.0, 5.0, 6.0]])
    mask = np.array([[True, True, False], [True, True, True]])
    targets = double_dqn_targets(rewards, dones, online_q, target_q, mask, gamma=0.5)
    assert np.allclose(targets, [21.0, -2.0])


def test_cnn_output_is_one_q_value_per_cell():
    rng = np.random.default_rng(1)
    q = CNN(9, 8, rng)
    Q, _ = q.forward(rng.normal(size=(5, N_CH * 81)))
    assert Q.shape == (5, 81)


def test_cnn_warm_start_supports_a_different_window_size(tmp_path):
    from agents.numpy_dqn import save_net

    source = tmp_path / "source.npz"
    target = tmp_path / "window_5.npz"
    save_net(source, make_net("cnn", 3, 2, np.random.default_rng(8),
                              flag_actions=True), 3, 0.16)

    train(episodes=1, window=5, density=0.16, seed=9, save=str(target),
          net="cnn", hid=2, batch=10000, log_every=1, ckpt_every=1,
          flag_actions=True, init_from=str(source), max_steps=1,
          max_revealed=20)

    with np.load(target) as trained:
        assert int(trained["window"]) == 5
        assert bool(trained["flag_actions"])


def test_cnn_warm_start_rejects_even_window(tmp_path):
    from agents.numpy_dqn import save_net

    source = tmp_path / "source.npz"
    save_net(source, make_net("cnn", 3, 2, np.random.default_rng(18),
                              flag_actions=True), 3, 0.16)
    with pytest.raises(AssertionError, match="window_size must be odd"):
        train(episodes=1, window=4, density=0.16, seed=9,
              save=str(tmp_path / "even.npz"), net="cnn", hid=2,
              flag_actions=True, init_from=str(source))


def test_training_can_warm_start_to_a_distinct_checkpoint(tmp_path):
    source = tmp_path / "source.npz"
    target = tmp_path / "continued.npz"
    rng = np.random.default_rng(2)
    original = make_net("cnn", 3, 2, rng, flag_actions=True)
    from agents.numpy_dqn import save_net
    save_net(source, original, 3, 0.16)
    before = {key: np.array(value) for key, value in np.load(source).items()}

    train(episodes=1, window=3, density=0.16, seed=3, save=str(target),
          net="cnn", hid=2, batch=10000, log_every=1, ckpt_every=1,
          flag_actions=True, init_from=str(source), max_steps=2,
          max_revealed=20)

    after_source = np.load(source)
    after_target = np.load(target)
    assert all(np.array_equal(before[key], after_source[key]) for key in before)
    assert bool(after_target["flag_actions"])
    assert int(after_target["window"]) == 3
