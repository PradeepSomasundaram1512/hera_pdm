"""RUL Prognostics Agent (fog/cloud layer) and shared training utilities.

All models output three quantiles (tau = alpha/2, 0.5, 1-alpha/2) of the
normalized RUL y/R_MAX via a non-crossing parametrization
    q_lo = q_50 - softplus(d_lo),  q_hi = q_50 + softplus(d_hi).
The quantiles are subsequently conformalized (uncertainty.py). A point-only
variant trained with MSE is used as the "no-UQ" baseline.
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import QUANTILES, R_MAX


class QuantileHead(nn.Module):
    def __init__(self, d_in: int, d_ctx: int = 0, hidden: int = 64):
        super().__init__()
        self.d_ctx = d_ctx
        self.net = nn.Sequential(nn.Linear(d_in + d_ctx, hidden), nn.ReLU(), nn.Linear(hidden, 3))

    def forward(self, h, ctx=None):
        if self.d_ctx:
            h = torch.cat([h, ctx], -1)
        o = self.net(h)
        med = o[:, 0]
        return torch.stack([med - F.softplus(o[:, 1]), med, med + F.softplus(o[:, 2])], 1)


class RNNModel(nn.Module):
    def __init__(self, d_in, d_ctx=0, hidden=64, layers=2, cell="gru"):
        super().__init__()
        rnn = nn.GRU if cell == "gru" else nn.LSTM
        self.rnn = rnn(d_in, hidden, num_layers=layers, batch_first=True, dropout=0.1 if layers > 1 else 0.0)
        self.head = QuantileHead(hidden, d_ctx, hidden=max(32, hidden))

    def forward(self, x, ctx=None):
        out, _ = self.rnn(x)
        return self.head(out[:, -1], ctx)


class TemporalBlock(nn.Module):
    def __init__(self, c_in, c_out, k, dil):
        super().__init__()
        self.pad = (k - 1) * dil
        self.c1 = nn.Conv1d(c_in, c_out, k, dilation=dil)
        self.c2 = nn.Conv1d(c_out, c_out, k, dilation=dil)
        self.res = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        y = self.drop(F.relu(self.c1(F.pad(x, (self.pad, 0)))))
        y = self.drop(F.relu(self.c2(F.pad(y, (self.pad, 0)))))
        return F.relu(y + self.res(x))


class TCNModel(nn.Module):
    """Causal dilated TCN (Bai et al. 2018); receptive field 61 snapshots."""
    def __init__(self, d_in, d_ctx=0, ch=48, levels=4, k=3):
        super().__init__()
        self.blocks = nn.Sequential(*[TemporalBlock(d_in if i == 0 else ch, ch, k, 2 ** i) for i in range(levels)])
        self.head = QuantileHead(ch, d_ctx)

    def forward(self, x, ctx=None):
        y = self.blocks(x.transpose(1, 2))
        return self.head(y[:, :, -1], ctx)


class CNNLSTMModel(nn.Module):
    def __init__(self, d_in, d_ctx=0, ch=64, hidden=64):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(d_in, ch, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2))
        self.lstm = nn.LSTM(ch, hidden, batch_first=True)
        self.head = QuantileHead(hidden, d_ctx)

    def forward(self, x, ctx=None):
        y = self.conv(x.transpose(1, 2)).transpose(1, 2)
        out, _ = self.lstm(y)
        return self.head(out[:, -1], ctx)


class EdgeCNN(nn.Module):
    """Tiny 1D-CNN for the edge monitoring agent (~2.8k parameters)."""
    def __init__(self, d_in, d_ctx=0, ch=16):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(d_in, ch, 3, padding=1), nn.ReLU(),
                                  nn.Conv1d(ch, ch, 3, padding=2, dilation=2), nn.ReLU())
        self.head = QuantileHead(2 * ch, d_ctx, hidden=16)

    def forward(self, x, ctx=None):
        y = self.conv(x.transpose(1, 2))
        return self.head(torch.cat([y.mean(-1), y[:, :, -1]], 1), ctx)


class EdgeGRU(nn.Module):
    """Tiny single-layer GRU for the edge monitoring agent."""
    def __init__(self, d_in, d_ctx=0, hidden=16):
        super().__init__()
        self.rnn = nn.GRU(d_in, hidden, batch_first=True)
        self.head = QuantileHead(hidden, d_ctx, hidden=16)

    def forward(self, x, ctx=None):
        out, _ = self.rnn(x)
        return self.head(out[:, -1], ctx)


def build_model(name: str, d_in: int, d_ctx: int, width: int = 64) -> nn.Module:
    return {
        "LSTM": lambda: RNNModel(d_in, d_ctx, hidden=width, cell="lstm"),
        "GRU": lambda: RNNModel(d_in, d_ctx, hidden=width, cell="gru"),
        "TCN": lambda: TCNModel(d_in, d_ctx, ch=max(16, 3 * width // 4)),
        "CNN-LSTM": lambda: CNNLSTMModel(d_in, d_ctx, ch=width, hidden=width),
        "Edge-CNN": lambda: EdgeCNN(d_in, d_ctx),
        "Edge-GRU": lambda: EdgeGRU(d_in, d_ctx),
    }[name]()


def pinball(q_pred, y):
    taus = torch.tensor(QUANTILES, device=y.device)
    diff = y[:, None] - q_pred
    return torch.maximum(taus * diff, (taus - 1) * diff).mean()


class WindowData:
    """Gathers causal windows ending at every snapshot of a set of assets.
    Early windows are left-padded by repeating the first snapshot."""

    def __init__(self, assets: dict, names: list[str], W: int, ctx_stats=None):
        Zs, Cs, ys, idx, meta = [], [], [], [], []
        off = 0
        for n in names:
            a = assets[n]
            T = len(a["rul"])
            t = np.arange(T)
            idx.append(off + np.clip(t[:, None] - np.arange(W - 1, -1, -1)[None, :], 0, None))
            Zs.append(a["Z"]); Cs.append(a["C"]); ys.append(np.minimum(a["rul"], R_MAX) / R_MAX)
            meta += [(n, int(i)) for i in t]
            off += T
        self.Z = torch.from_numpy(np.concatenate(Zs))
        C = np.concatenate(Cs)
        self.ctx_stats = ctx_stats or (C.mean(0), C.std(0) + 1e-6)
        self.C = torch.from_numpy(((C - self.ctx_stats[0]) / self.ctx_stats[1]).astype(np.float32))
        self.y = torch.from_numpy(np.concatenate(ys).astype(np.float32))
        self.idx = torch.from_numpy(np.concatenate(idx))
        self.meta = meta

    def __len__(self):
        return len(self.y)

    def batch(self, ii):
        return self.Z[self.idx[ii]], self.C[ii], self.y[ii]


def train_model(model, data: WindowData, use_ctx: bool, loss: str = "quantile", epochs: int = 20,
                bs: int = 256, lr: float = 2e-3, seed: int = 0, log=None, noise: float = 0.0,
                weight_decay: float = 1e-4):
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    steps = epochs * math.ceil(len(data) / bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    model.train()
    t0 = time.time()
    for ep in range(epochs):
        perm = torch.randperm(len(data), generator=g)
        tot = 0.0
        for s in range(0, len(data), bs):
            x, c, y = data.batch(perm[s:s + bs])
            if noise > 0:  # Gaussian input jitter as regularizer (training only)
                x = x + noise * torch.randn(x.shape, generator=g)
                c = c + noise * torch.randn(c.shape, generator=g)
            q = model(x, c if use_ctx else None)
            l = pinball(q, y) if loss == "quantile" else F.mse_loss(q[:, 1], y)
            opt.zero_grad(); l.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            tot += l.item() * len(y)
        if log and (ep % 5 == 4 or ep == epochs - 1):
            log(f"    ep{ep + 1} loss={tot / len(data):.4f} ({time.time() - t0:.0f}s)")
    model.eval()
    return model


@torch.no_grad()
def predict(model, data: WindowData, use_ctx: bool, bs: int = 2048) -> np.ndarray:
    model.eval()
    out = []
    for s in range(0, len(data), bs):
        ii = torch.arange(s, min(s + bs, len(data)))
        x, c, _ = data.batch(ii)
        out.append(model(x, c if use_ctx else None))
    return (torch.cat(out).numpy() * R_MAX).astype(np.float32)  # seconds


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def count_macs(model, d_in: int, W: int, d_ctx: int) -> int:
    """Analytic multiply-accumulate count for one inference (batch size 1)."""
    macs = 0
    hooks = []

    def conv_hook(m, i, o):
        nonlocal macs
        macs += o.numel() * m.in_channels * m.kernel_size[0] // m.groups

    def lin_hook(m, i, o):
        nonlocal macs
        macs += m.in_features * m.out_features

    def rnn_hook(m, i, o):
        nonlocal macs
        gates = 3 if isinstance(m, nn.GRU) else 4
        L = i[0].shape[1]
        for layer in range(m.num_layers):
            d = m.input_size if layer == 0 else m.hidden_size
            macs += L * gates * (d * m.hidden_size + m.hidden_size ** 2)

    for m in model.modules():
        if isinstance(m, nn.Conv1d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(lin_hook))
        elif isinstance(m, (nn.GRU, nn.LSTM)):
            hooks.append(m.register_forward_hook(rnn_hook))
    with torch.no_grad():
        model(torch.zeros(1, W, d_in), torch.zeros(1, d_ctx) if d_ctx else None)
    for h in hooks:
        h.remove()
    return macs


@torch.no_grad()
def measure_latency(model, d_in: int, W: int, d_ctx: int, reps: int = 300) -> dict:
    """Wall-clock single-sample CPU inference latency with one thread."""
    torch.set_num_threads(1)
    model.eval()
    x = torch.randn(1, W, d_in)
    c = torch.randn(1, d_ctx) if d_ctx else None
    for _ in range(30):
        model(x, c)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        model(x, c)
        ts.append((time.perf_counter() - t0) * 1e3)
    ts = np.array(ts)
    return {"median_ms": float(np.median(ts)), "p95_ms": float(np.percentile(ts, 95))}
