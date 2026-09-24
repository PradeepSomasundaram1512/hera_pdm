"""Digital Twin State Agent (fog layer).

The twin is a synchronized virtual state, not a physics simulation:
    DT_t = {operating_condition, sensor_state, health_index, degradation_state,
            predicted_RUL, uncertainty, anomaly_state}
It is updated causally (only x_1..x_t are used at time t). The edge sends a
compact sync message per snapshot (health index + anomaly score); the twin
turns that stream into long-horizon degradation context c_t that the
prognostics agent consumes, while the edge only ever sees a short window.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from utils import BASELINE_N, SNAPSHOT_S

# Elapsed operating time is deliberately excluded: bearing lifetimes span
# 2.3e3-2.8e4 s, so absolute age is a non-transferable shortcut.
CONTEXT_NAMES = ["hi_fast", "hi_slow", "hi_runmax", "hi_slope", "excess_int", "hi_trend"]


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c[:2] in ("h_", "v_")]


def normalize_to_baseline(X: np.ndarray, n: int = BASELINE_N) -> np.ndarray:
    """Asset-specific z-score against the first n (healthy) snapshots, then a
    signed log transform to compress heavy-tailed impulsive features."""
    mu = X[:n].mean(0)
    sd = X[:n].std(0) + 1e-6
    z = np.clip((X - mu) / sd, -1e3, 1e3)
    return np.sign(z) * np.log1p(np.abs(z))


def health_index(Z: np.ndarray) -> np.ndarray:
    return np.log1p(np.abs(np.expm1(np.abs(Z))).mean(1))


def ewma(x: np.ndarray, span: float) -> np.ndarray:
    return pd.Series(x).ewm(span=span, adjust=False).mean().to_numpy()


def anomaly_score(hi: np.ndarray, n: int = BASELINE_N) -> np.ndarray:
    """Edge anomaly score a_t: EWMA-smoothed HI standardized by the mean and
    snapshot-level spread of the asset's own raw baseline HI (O(1) state on
    device). Standardizing by the spread of the *raw* baseline avoids the
    over-sensitivity of dividing by the (much smaller) spread of a smoothed
    baseline. The alarm threshold is calibrated on training assets."""
    f = ewma(hi, 6)
    return (f - hi[:n].mean()) / (hi[:n].std() + 1e-6)


def context_features(hi: np.ndarray) -> np.ndarray:
    """Threshold-free long-horizon degradation context maintained by the twin."""
    T = len(hi)
    fast, slow = ewma(hi, 6), ewma(hi, 60)
    runmax = np.maximum.accumulate(fast)
    slope = np.zeros(T)
    k = 30
    tt = np.arange(k) - (k - 1) / 2
    for t in range(k, T):
        slope[t] = (tt * fast[t - k + 1:t + 1]).sum() / (tt ** 2).sum()
    base = hi[:BASELINE_N].mean()
    excess = np.log1p(np.cumsum(np.maximum(0.0, fast - base)) * SNAPSHOT_S / 3600)
    return np.stack([fast, slow, runmax, slope * 100, excess, fast - slow], 1).astype(np.float32)


@dataclass
class TwinState:
    """Synchronized virtual state of one asset at snapshot t."""
    asset: str
    t: int
    operating_condition: dict
    sensor_state: dict
    health_index: float
    degradation_state: dict
    anomaly_state: dict
    predicted_rul: float | None = None
    rul_interval: tuple | None = None
    uncertainty: float | None = None
    history: list = field(default_factory=list)


def build_asset(df_b: pd.DataFrame, feats: list[str]) -> dict:
    """Per-asset arrays used by all agents (all causal / online computable)."""
    X = df_b[feats].to_numpy(dtype=np.float64)
    Z = normalize_to_baseline(X).astype(np.float32)
    hi = health_index(Z)
    a = anomaly_score(hi)
    C = context_features(hi)
    return {"Z": Z, "hi": hi.astype(np.float32), "a": a.astype(np.float32), "C": C,
            "rul": df_b["rul_s"].to_numpy(dtype=np.float32), "cond": int(df_b["condition"].iloc[0])}


def twin_state(asset: str, arr: dict, t: int, feats: list[str], cond_table: dict,
               pred=None, tau_a: float = 3.0) -> TwinState:
    rpm, load = cond_table[arr["cond"]]
    c = arr["C"][t]
    st = TwinState(
        asset=asset, t=t,
        operating_condition={"condition": arr["cond"], "rpm": rpm, "radial_load_N": load},
        sensor_state={f: float(v) for f, v in zip(feats, arr["Z"][t])},
        health_index=float(arr["hi"][t]),
        degradation_state={n: float(v) for n, v in zip(CONTEXT_NAMES, c)},
        anomaly_state={"score": float(arr["a"][t]), "alarm": bool(arr["a"][t] > tau_a)},
    )
    if pred is not None:
        lo, med, hi_ = pred
        st.predicted_rul, st.rul_interval, st.uncertainty = float(med), (float(lo), float(hi_)), float(hi_ - lo)
    return st
