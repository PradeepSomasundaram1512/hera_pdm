"""HERA-PdM v3: onset-gated escalation with lead-time-calibrated maintenance alarms.

Two-stage policy with a single threshold offset lambda:
  Stage 1 (edge, onset gate): the edge escalates to the fog/cloud tier from the
      first time its median RUL m^e_t <= H_PLAN + lambda holds for SUSTAIN
      consecutive snapshots; afterwards it stays escalated (latched).
  Stage 2 (fog/cloud, alarm): after escalation, a maintenance alarm is raised at
      the first time the fog/cloud median m^c_t <= H_CRIT + lambda holds for
      SUSTAIN consecutive snapshots.
Bearing-level loss: late(lambda) = 1[no alarm, or alarm raised when the true RUL
is already < H_CRIT]. Because a larger lambda can only make escalation and the
alarm earlier, late(lambda) is non-increasing in lambda.

Conformal risk control (Angelopoulos et al., ICLR 2024) on n exchangeable
calibration bearings (CV+ out-of-fold predictions of the non-test bearings):
    lambda_hat = inf{ lambda : (n / (n+1)) * Rhat_n(lambda) + 1/(n+1) <= eps }
guarantees E[late_{n+1}(lambda_hat)] <= eps for a new bearing, i.e. the
probability that a bearing's maintenance alarm is late or missing is at most eps.
lambda_hat is the smallest such offset, so it limits premature alarms.
"""
from __future__ import annotations

import numpy as np

SUSTAIN = 3


def first_sustained(cond: np.ndarray, k: int = SUSTAIN) -> int | None:
    run = np.convolve(cond.astype(int), np.ones(k, int), "full")[:len(cond)] >= k
    return int(np.argmax(run)) if run.any() else None


def two_stage_alarm(m_edge, m_cloud, lam, h_crit, h_plan, gate=True):
    """Return (escalation index, alarm index) for one bearing (time-ordered arrays)."""
    esc = first_sustained(m_edge <= h_plan + lam) if gate else 0
    if esc is None:
        return None, None
    cond = np.zeros(len(m_cloud), bool)
    cond[esc:] = m_cloud[esc:] <= h_crit + lam
    return esc, first_sustained(cond)


def gated_alarm(m_edge, m_cloud, a, tau, lam, h_crit, h_plan, burn_in):
    """Variant v3b: onset gate fixed from training data (sustained anomaly a_t > tau
    or edge median <= H_PLAN), no alarms during the healthy-baseline burn-in, and
    CRC calibrates only the fog/cloud alarm offset lambda."""
    onset = (a > tau) | (m_edge <= h_plan)
    onset[:burn_in] = False
    esc = first_sustained(onset)
    if esc is None:
        return None, None
    cond = np.zeros(len(m_cloud), bool)
    cond[max(esc, burn_in):] = m_cloud[max(esc, burn_in):] <= h_crit + lam
    return esc, first_sustained(cond)


def single_stage_alarm(stat, lam, thr, burn_in=0):
    if burn_in:
        stat = stat.copy()
        stat[:burn_in] = np.inf
        return first_sustained(stat <= thr + lam)
    return first_sustained(stat <= thr + lam)


def bearing_outcome(alarm, rul_raw, h_crit, r_max):
    if alarm is None:
        return "late", np.nan
    r = float(rul_raw[alarm])
    return ("late" if r < h_crit else "premature" if r > r_max else "timely"), r


def crc_lambda(late_matrix: np.ndarray, grid: np.ndarray, eps: float) -> float:
    """late_matrix: (n_bearings, len(grid)) losses in {0,1}, non-increasing along grid."""
    n = late_matrix.shape[0]
    risk = late_matrix.mean(0)
    ok = (n / (n + 1)) * risk + 1.0 / (n + 1) <= eps
    return float(grid[np.argmax(ok)]) if ok.any() else float(grid[-1])


def split_by_bearing(bearing, t, *arrays):
    out = {}
    for b in np.unique(bearing):
        ii = np.where(bearing == b)[0]
        ii = ii[np.argsort(t[ii])]
        out[b] = [a[ii] for a in arrays]
    return out
