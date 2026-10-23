"""Masked DQN in pure NumPy — the working RL path on machines where
torch is blocked (like this one: WDAC policy blocks torch native DLLs).

Two Q-networks, chosen with --net:
  mlp  flattens the (11,H,W) window into an MLP (the original model).
  cnn  fully convolutional: three same-padded 3x3 convs and one Q value per
       cell. Weights are shared across positions, and the 7x7 receptive
       field covers the clue numbers that constrain a cell.

Both use masked epsilon-greedy, Adam and a target network.

Usage:
    conda activate minesweeper-rl
    python -m agents.numpy_dqn --episodes 2000 --window 9 --density 0.16 --net cnn --flag-actions --save dqn_cnn_flags.npz
    python -m agents.numpy_dqn --episodes 1000 --window 9 --density 0.16 --net cnn --flag-actions --init-from dqn_cnn_flags.npz --save dqn_cnn_flags_more.npz
"""
from __future__ import annotations

import argparse
import copy
import os
import time
from collections import deque

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from env.gym_wrapper import MinesweeperWindowEnv

N_CH = 11  # observation channels from MinesweeperWindowEnv


def relu(x):
    return np.maximum(x, 0.0)


def adam_step(params, grads, m, v, t, lr, b1=0.9, b2=0.999, eps=1e-8):
    """In-place Adam update. m, v: lists of zero-like arrays; t: step count (>=1)."""
    for i, (p, g) in enumerate(zip(params, grads)):
        m[i] = b1 * m[i] + (1 - b1) * g
        v[i] = b2 * v[i] + (1 - b2) * g * g
        mh = m[i] / (1 - b1 ** t)
        vh = v[i] / (1 - b2 ** t)
        p -= lr * mh / (np.sqrt(vh) + eps)


class _Net:
    NAMES: tuple = ()

    def params(self):
        return [getattr(self, n) for n in self.NAMES]

    def set_params(self, arrays):
        for n, a in zip(self.NAMES, arrays):
            setattr(self, n, np.array(a, dtype=np.float64))


class MLP(_Net):
    ARCH = "mlp"
    NAMES = ("W1", "b1", "W2", "b2")

    def __init__(self, inp: int, hid: int, out: int, rng: np.random.Generator,
                 flag_actions: bool = False):
        self.flag_actions = flag_actions
        self.W1 = rng.normal(0, np.sqrt(2 / inp), (inp, hid)).astype(np.float64)
        self.b1 = np.zeros(hid)
        self.W2 = rng.normal(0, np.sqrt(2 / hid), (hid, out)).astype(np.float64)
        self.b2 = np.zeros(out)

    def forward(self, x):
        h = relu(x @ self.W1 + self.b1)
        return h @ self.W2 + self.b2, (h, x)

    def backward(self, cache, dQ):
        h, x = cache
        dW2 = h.T @ dQ
        db2 = dQ.sum(0)
        dh = dQ @ self.W2.T
        dh[h <= 0] = 0
        return [x.T @ dh, dh.sum(0), dW2, db2]


def _conv(x, W, b):
    """Same-padded 3x3 cross-correlation. x: (B,Cin,H,W), W: (Cout,Cin,3,3)."""
    cols = sliding_window_view(np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1))),
                               (3, 3), axis=(2, 3))  # (B,Cin,H,W,3,3)
    out = np.einsum("bchwij,ocij->bohw", cols, W) + b[None, :, None, None]
    return out, cols


def _conv_input_grad(d, W):
    """Gradient of _conv w.r.t. its input. d: (B,Cout,H,W) -> (B,Cin,H,W)."""
    B, _, H, Wd = d.shape
    dxp = np.zeros((B, W.shape[1], H + 2, Wd + 2))
    for i in range(3):
        for j in range(3):
            dxp[:, :, i:i + H, j:j + Wd] += np.einsum("bohw,oc->bchw", d, W[:, :, i, j])
    return dxp[:, :, 1:-1, 1:-1]


def _conv_weight_grad(d, cols):
    return np.einsum("bohw,bchwij->ocij", d, cols)


class CNN(_Net):
    ARCH = "cnn"
    NAMES = ("c1W", "c1b", "c2W", "c2b", "c3W", "c3b")

    def __init__(self, window: int, hid: int, rng: np.random.Generator,
                 flag_actions: bool = False):
        self.window = window
        self.flag_actions = flag_actions

        def conv_w(cout, cin):
            return rng.normal(0, np.sqrt(2 / (cin * 9)), (cout, cin, 3, 3))

        self.c1W, self.c1b = conv_w(hid, N_CH), np.zeros(hid)
        self.c2W, self.c2b = conv_w(hid, hid), np.zeros(hid)
        n_heads = 2 if flag_actions else 1
        self.c3W, self.c3b = conv_w(n_heads, hid), np.zeros(n_heads)

    def forward(self, x):
        """x: (B, N_CH*window*window) flat window, same as the MLP input."""
        B, H = x.shape[0], self.window
        x = x.reshape(B, N_CH, H, H)
        z1, cols1 = _conv(x, self.c1W, self.c1b)
        h1 = relu(z1)
        z2, cols2 = _conv(h1, self.c2W, self.c2b)
        h2 = relu(z2)
        z3, cols3 = _conv(h2, self.c3W, self.c3b)
        return z3.reshape(B, -1), (cols1, cols2, cols3, h1 > 0, h2 > 0)

    def backward(self, cache, dQ):
        cols1, cols2, cols3, m1, m2 = cache
        B, H = dQ.shape[0], self.window
        d3 = dQ.reshape(B, self.c3W.shape[0], H, H)
        dz2 = _conv_input_grad(d3, self.c3W) * m2
        dz1 = _conv_input_grad(dz2, self.c2W) * m1
        return [_conv_weight_grad(dz1, cols1), dz1.sum((0, 2, 3)),
                _conv_weight_grad(dz2, cols2), dz2.sum((0, 2, 3)),
                _conv_weight_grad(d3, cols3), d3.sum((0, 2, 3))]


def make_net(net: str, window: int, hid: int, rng: np.random.Generator,
             flag_actions: bool = False):
    n_actions = window * window * (2 if flag_actions else 1)
    if net == "cnn":
        return CNN(window, hid, rng, flag_actions)
    return MLP(N_CH * window * window, hid, n_actions, rng, flag_actions)


def arch_of(d) -> str:
    """Architecture of a saved .npz; checkpoints from before the CNN have none (= mlp)."""
    return str(d["arch"]) if "arch" in d.files else "mlp"


def save_net(path, q, window, density):
    np.savez(path, arch=q.ARCH, window=window, density=density,
             flag_actions=getattr(q, "flag_actions", False),
             **dict(zip(q.NAMES, q.params())))


def double_dqn_targets(rewards, dones, online_q, target_q, next_masks, gamma):
    """Select legal actions online and evaluate them with target-network values."""
    next_actions = np.argmax(np.where(next_masks, online_q, -1e9), axis=1)
    next_values = target_q[np.arange(len(rewards)), next_actions]
    return rewards + (1 - dones) * gamma * next_values


def train(episodes=2000, window=9, density=0.10, seed=0, save="dqn_mines.npz",
          net="mlp", hid=None, gamma=0.99, lr=1e-3, buf_size=20_000, batch=64,
          eps0=1.0, eps1=0.05, target_sync=500, max_steps=300,
          update_every=4, log_every=25, ckpt_every=200, resume=False,
          max_revealed=10000, flag_actions=False, init_from=None,
          mine_penalty=10.0):
    if hid is None:
        hid = 32 if net == "cnn" else 256  # cnn: channels per conv layer
    t0 = time.time()
    rng = np.random.default_rng(seed)
    env = MinesweeperWindowEnv(window_size=window, p=density, seed=seed,
                               max_steps=max_steps, max_revealed=max_revealed,
                               flag_actions=flag_actions, mine_penalty=mine_penalty)
    q = make_net(net, window, hid, rng, flag_actions)
    start_ep = 0
    source = init_from if init_from is not None else (save if resume else None)
    if source is not None:
        if not os.path.exists(source):
            raise FileNotFoundError(f"initial checkpoint not found: {source}")
        d = np.load(source)
        saved_flags = bool(d["flag_actions"]) if "flag_actions" in d.files else False
        saved_window = int(d["window"])
        if (arch_of(d) != q.ARCH or saved_flags != flag_actions
                or (saved_window != window and q.ARCH != "cnn")):
            raise ValueError(f"{source} architecture/action heads/window do not match "
                             f"--net {net} --flag-actions {flag_actions} --window {window}")
        q.set_params([d[n] for n in q.NAMES])
        print(f"initialized weights from {source}", flush=True)
    qt = copy.deepcopy(q)
    buf: deque = deque(maxlen=buf_size)
    adam_m = [np.zeros_like(p) for p in q.params()]
    adam_v = [np.zeros_like(p) for p in q.params()]
    adam_t = 0
    lengths, rewards, step_counts = [], [], []
    updates = 0

    def feat(obs):
        return obs["obs"].ravel().astype(np.float64)

    for ep in range(start_ep, start_ep + episodes):
        obs, _ = env.reset(seed=seed + ep)
        eps = max(eps1, eps0 - (eps0 - eps1) * ep / max(1, episodes // 2))
        ep_r, done = 0.0, False
        ep_steps = 0
        while not done:
            s = feat(obs)
            mask = env.action_masks()
            qs, _ = q.forward(s[None, :])
            qs = qs[0]
            if rng.random() < eps:
                a = int(rng.choice(np.flatnonzero(mask)))
            else:
                masked = np.where(mask, qs, -1e9)
                a = int(np.argmax(masked))
            obs2, r, term, trunc, info = env.step(a)
            done = term or trunc
            buf.append((s, a, r, feat(obs2), done,
                        env.action_masks().astype(bool)))
            ep_r += r
            obs = obs2
            ep_steps += 1
            # one gradient update every k env steps (was: every step).
            # Cost per episode used to explode as episodes got longer.
            if len(buf) >= batch and ep_steps % update_every == 0:
                idx = rng.integers(0, len(buf), batch)
                B = [buf[i] for i in idx]
                S = np.stack([b[0] for b in B])
                A_ = np.array([b[1] for b in B])
                R = np.array([b[2] for b in B])
                S2 = np.stack([b[3] for b in B])
                D = np.array([b[4] for b in B], dtype=float)
                M2 = np.stack([b[5] for b in B])
                # Double DQN: choose next actions with the online net, then
                # evaluate them with the target net to reduce max bias.
                Q2_online, _ = q.forward(S2)
                Q2_target, _ = qt.forward(S2)
                tgt = double_dqn_targets(R, D, Q2_online, Q2_target, M2, gamma)
                Q, cache = q.forward(S)
                pred = Q[np.arange(batch), A_]
                err = pred - tgt  # MSE grad
                dQ = np.zeros_like(Q)
                dQ[np.arange(batch), A_] = 2 * err / batch  # mean over batch
                grads = q.backward(cache, dQ)
                adam_t += 1
                adam_step(q.params(), [np.clip(g, -1.0, 1.0) for g in grads],
                          adam_m, adam_v, adam_t, lr)
                updates += 1
                if updates % target_sync == 0:
                    qt = copy.deepcopy(q)
        lengths.append(len(env.board.revealed))
        rewards.append(ep_r)
        step_counts.append(ep_steps)
        n_ep = ep - start_ep + 1
        if n_ep % log_every == 0:
            dt = time.time() - t0
            w = min(len(lengths), 100)
            print(f"ep={n_ep}/{episodes} len_mean={np.mean(lengths[-w:]):.0f} "
                  f"rew_mean={np.mean(rewards[-w:]):.1f} eps={eps:.2f} "
                  f"steps_mean={np.mean(step_counts[-w:]):.0f} "
                  f"elapsed={dt:.0f}s ({dt / n_ep:.1f}s/ep)",
                  flush=True)
        if n_ep % ckpt_every == 0:
            save_net(save, q, window, density)
            print(f"  checkpoint -> {save} (ep {n_ep})", flush=True)
    save_net(save, q, window, density)
    print(f"saved -> {save}  final100_len={np.mean(lengths[-100:]):.0f}")
    return lengths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=2000)
    ap.add_argument("--window", type=int, default=9)
    ap.add_argument("--density", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", type=str, default="dqn_mines.npz")
    ap.add_argument("--net", choices=["mlp", "cnn"], default="mlp")
    ap.add_argument("--hid", type=int, default=None,
                    help="MLP hidden units (default 256) or CNN channels per layer (default 32)")
    ap.add_argument("--update-every", type=int, default=4)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--max-revealed", type=int, default=10000)
    ap.add_argument("--mine-penalty", type=float, default=10.0,
                    help="terminal reward magnitude for hitting a mine")
    ap.add_argument("--eps0", type=float, default=1.0,
                    help="initial epsilon (use a smaller value when fine-tuning)")
    ap.add_argument("--eps1", type=float, default=0.05,
                    help="final epsilon")
    ap.add_argument("--ckpt-every", type=int, default=200)
    ap.add_argument("--resume", action="store_true",
                    help="initialize from --save before training (then save back to it)")
    ap.add_argument("--init-from", type=str, default=None,
                    help="warm-start from a checkpoint while saving to a separate --save path")
    ap.add_argument("--flag-actions", action="store_true",
                    help="train separate reveal and flag actions, rewarding correct mine flags")
    a = ap.parse_args()
    train(episodes=a.episodes, window=a.window, density=a.density,
          seed=a.seed, save=a.save, net=a.net, hid=a.hid,
          update_every=a.update_every, log_every=a.log_every,
          ckpt_every=a.ckpt_every, resume=a.resume,
          max_revealed=a.max_revealed, flag_actions=a.flag_actions,
          init_from=a.init_from, mine_penalty=a.mine_penalty,
          eps0=a.eps0, eps1=a.eps1)


if __name__ == "__main__":
    main()
