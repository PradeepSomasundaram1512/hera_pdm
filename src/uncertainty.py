"""Split conformalized quantile regression (CQR; Romano et al., NeurIPS 2019).

Given quantile predictions (q_lo, q_50, q_hi) on held-out calibration bearings,
the conformity score is s_i = max(q_lo_i - y_i, y_i - q_hi_i). For target
miscoverage alpha the correction is the ceil((n+1)(1-alpha))/n empirical
quantile of s, and the interval is [q_lo - Q, q_hi + Q], clipped to [0, R_MAX].
Coverage >= 1-alpha holds marginally if calibration and test assets are
exchangeable; snapshots within one bearing are dependent, so we report
coverage per bearing as well as pooled.
"""
from __future__ import annotations

import numpy as np

from utils import R_MAX


def cqr_scores(q: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.maximum(q[:, 0] - y, y - q[:, 2])


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    n = len(scores)
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, level, method="higher"))


def apply_cqr(q: np.ndarray, corr: float) -> np.ndarray:
    lo = np.clip(q[:, 0] - corr, 0, R_MAX)
    hi = np.clip(q[:, 2] + corr, 0, R_MAX)
    med = np.clip(q[:, 1], lo, hi)
    return np.stack([lo, med, hi], 1)


def asymmetric_cqr(q_cal, y_cal, q_test, alpha) -> np.ndarray:
    """Asymmetric CQR (Romano et al., 2019): the lower and upper bounds are
    calibrated separately, each at level alpha/2, so that errors on one side
    (e.g. early false alarms on capped labels) do not widen the other bound.
    Coverage >= 1 - alpha by a union bound. The maintenance decision uses the
    lower bound, which then depends only on lower-side (late-warning) errors."""
    c_lo = conformal_quantile(q_cal[:, 0] - y_cal, alpha / 2)
    c_hi = conformal_quantile(y_cal - q_cal[:, 2], alpha / 2)
    lo = np.clip(q_test[:, 0] - c_lo, 0, R_MAX)
    hi = np.clip(np.maximum(q_test[:, 2] + c_hi, lo), 0, R_MAX)
    return np.stack([lo, np.clip(q_test[:, 1], lo, hi), hi], 1)


def calibrate(q_cal, y_cal, q_test, alpha, mode: str = "asym") -> np.ndarray:
    if mode == "asym":
        return asymmetric_cqr(q_cal, y_cal, q_test, alpha)
    return apply_cqr(q_test, conformal_quantile(cqr_scores(q_cal, y_cal), alpha))


def interval_metrics(I: np.ndarray, y: np.ndarray, alpha: float) -> dict:
    lo, hi = I[:, 0], I[:, 2]
    cov = (y >= lo) & (y <= hi)
    width = hi - lo
    # Winkler / interval score (lower is better)
    iscore = width + (2 / alpha) * (lo - y) * (y < lo) + (2 / alpha) * (y - hi) * (y > hi)
    return {"picp": float(cov.mean()), "mpiw": float(width.mean()), "nmpiw": float(width.mean() / R_MAX),
            "interval_score": float(iscore.mean())}


def calibration_curve(q_cal, y_cal, q_test, y_test, alphas=np.linspace(0.05, 0.5, 10), mode="asym") -> list:
    """Nominal vs empirical coverage over a range of alpha (for ECE-style error)."""
    out = []
    for a in alphas:
        I = calibrate(q_cal, y_cal, q_test, a, mode)
        out.append((1 - a, float(((y_test >= I[:, 0]) & (y_test <= I[:, 2])).mean())))
    return out


def coverage_calibration_error(curve) -> float:
    return float(np.mean([abs(n - e) for n, e in curve]))
