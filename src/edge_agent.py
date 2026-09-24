"""Edge Monitoring Agent (machine layer).

Responsibilities (all O(1)-state or short-window operations suitable for an
MCU/SBC-class device): per-snapshot vibration feature extraction
(preprocessing.channel_features), asset-baseline normalization, health index
and anomaly score (digital_twin.anomaly_score), a tiny quantile RUL model over a
W_EDGE-snapshot window, and conformal calibration of its interval. The agent
resolves decision-sufficient cases locally and escalates the rest.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

from escalation import decide, gate_hera
from uncertainty import apply_cqr
from utils import BASELINE_N, H_CRIT


@dataclass
class EdgeOutput:
    interval: np.ndarray       # (N,3) [L, median, U] in seconds
    anomaly: np.ndarray        # (N,) anomaly score a_t
    local_action: np.ndarray   # (N,) action if resolved at the edge
    escalate: np.ndarray       # (N,) bool


class EdgeMonitoringAgent:
    def __init__(self, cqr_correction: float, tau_a: float = 3.0, use_anomaly: bool = True):
        self.corr, self.tau_a, self.use_anomaly = cqr_correction, tau_a, use_anomaly

    def step(self, q_raw: np.ndarray, a: np.ndarray) -> EdgeOutput:
        I = apply_cqr(q_raw, self.corr)
        return EdgeOutput(I, a, decide(I, a, self.tau_a), gate_hera(I, a, self.tau_a, self.use_anomaly))


def isolation_forest_scores(train_Z: list[np.ndarray], test_Z: np.ndarray, seed: int = 0) -> np.ndarray:
    """Baseline anomaly detector fitted on healthy baseline snapshots of the
    training assets (first BASELINE_N snapshots each)."""
    Xh = np.concatenate([z[:BASELINE_N] for z in train_Z])
    iso = IsolationForest(n_estimators=200, random_state=seed).fit(Xh)
    return -iso.score_samples(test_Z)


def detection_metrics(score: np.ndarray, rul: np.ndarray, thr: float) -> dict:
    """Imminent-failure detection (positive: RUL <= H_CRIT)."""
    y = (rul <= H_CRIT).astype(int)
    p = (score > thr).astype(int)
    return {"precision": precision_score(y, p, zero_division=0), "recall": recall_score(y, p, zero_division=0),
            "f1": f1_score(y, p, zero_division=0), "auroc": roc_auc_score(y, score),
            "auprc": average_precision_score(y, score)}
