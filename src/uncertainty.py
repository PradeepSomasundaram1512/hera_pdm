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


def _kth_union(base: np.ndarray, sorted_scores: list, k_target: int, sign: int, iters: int = 60) -> np.ndarray:
    """k-th smallest element of the union over groups g of {base[:, g] + sign * s : s in group g},
    computed for every test row by bisection on the value (monotone count)."""
    n_test = base.shape[0]
    smax = max(np.abs(sc).max() for sc in sorted_scores)
    lo_v = base.min(1) - smax - 1.0
    hi_v = base.max(1) + smax + 1.0
    for _ in range(iters):
        mid = (lo_v + hi_v) / 2
        cnt = np.zeros(n_test)
        for g, s in enumerate(sorted_scores):
            if sign > 0:   # elements base + s <= mid  <=>  s <= mid - base
                cnt += np.searchsorted(s, mid - base[:, g], side="right")
            else:          # elements base - s <= mid  <=>  s >= base - mid
                cnt += len(s) - np.searchsorted(s, base[:, g] - mid, side="left")
        ok = cnt >= k_target
        hi_v = np.where(ok, mid, hi_v)
        lo_v = np.where(ok, lo_v, mid)
    return hi_v


def cvplus_cqr(q_oof, g_oof, y_oof, q_test_k, alpha) -> np.ndarray:
    """CV+ for conformalized quantile regression (Barber et al., Ann. Stat. 2021;
    CQR scores of Romano et al., 2019). q_oof: out-of-fold quantiles of the
    calibration points (from the inner model that did not see them); g_oof:
    inner-group index; q_test_k: (K, n_test, 3) test quantiles of each inner
    model. Interval: L = floor(alpha(n+1))-th smallest of q_lo^{-k(i)}(x) - s_i,
    U = ceil((1-alpha)(n+1))-th smallest of q_hi^{-k(i)}(x) + s_i.
    Guarantee: coverage >= 1 - 2 alpha (typically close to 1 - alpha) under
    exchangeability of assets. Point estimate: mean of inner-model medians."""
    s = cqr_scores(q_oof, y_oof)
    K = q_test_k.shape[0]
    groups = [np.sort(s[g_oof == k]) for k in range(K)]
    n = len(s)
    k_lo = int(np.floor(alpha * (n + 1)))
    k_hi = int(np.ceil((1 - alpha) * (n + 1)))
    base_lo = q_test_k[:, :, 0].T
    base_hi = q_test_k[:, :, 2].T
    L = _kth_union(base_lo, groups, max(k_lo, 1), -1) if k_lo >= 1 else np.full(base_lo.shape[0], -np.inf)
    U = _kth_union(base_hi, groups, min(k_hi, n), +1)
    L = np.clip(L, 0, R_MAX)
    U = np.clip(np.maximum(U, L), 0, R_MAX)
    med = np.clip(q_test_k[:, :, 1].mean(0), L, U)
    return np.stack([L, med, U], 1)


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
