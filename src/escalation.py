"""Uncertainty-aware hierarchical escalation and maintenance decision logic.

Action regions on the RUL axis (seconds):
    r(v) = URGENT if v <= H_CRIT, SCHEDULE if v <= H_PLAN, CONTINUE otherwise.
A calibrated interval I_t = [L_t, U_t] is *decision-sufficient* when it lies in
a single action region, i.e. it does not straddle any decision boundary
h in B = {H_CRIT, H_PLAN}. The HERA gate escalates exactly the snapshots whose
edge interval is not decision-sufficient (plus, optionally, snapshots whose
anomaly evidence contradicts a CONTINUE verdict):
    e_t = 1[ exists h in B : L_t <= h < U_t ]  OR  1[ a_t > tau_a  AND  r(L_t) = CONTINUE ].
Because a locally resolved CONTINUE implies L_t > H_PLAN >= H_CRIT, the joint
probability of a local CONTINUE on a critical asset (y <= H_CRIT) is at most
P(y not in I_t) <= alpha_edge; likewise a local URGENT on a healthy asset
(y > H_PLAN) requires y > U_t. Edge-side unsafe decisions are therefore
bounded by the edge miscoverage level, which becomes the single control knob.
"""
from __future__ import annotations

import numpy as np

from utils import H_CRIT, H_PLAN, R_MAX

CONTINUE, INSPECT, SCHEDULE, URGENT = 0, 1, 2, 3
ACTION_NAMES = {CONTINUE: "continue operation", INSPECT: "inspect", SCHEDULE: "schedule maintenance",
                URGENT: "urgent maintenance"}
HIGH_IMPACT = {SCHEDULE, URGENT}   # require human authorization
_H = {"c": H_CRIT, "p": H_PLAN}


def set_horizons(h_crit: float, h_plan: float) -> None:
    """Override the decision horizons (used only by the sensitivity analysis)."""
    _H["c"], _H["p"] = float(h_crit), float(h_plan)


def boundaries():
    return (_H["c"], _H["p"])


def region(v: np.ndarray) -> np.ndarray:
    return np.where(v <= _H["c"], URGENT, np.where(v <= _H["p"], SCHEDULE, CONTINUE))


def decide(I: np.ndarray, a: np.ndarray | None = None, tau_a: float = 3.0) -> np.ndarray:
    """Maintenance Decision Agent: act on the conservative lower bound; an
    anomaly alarm on an otherwise healthy verdict yields INSPECT."""
    act = region(I[:, 0])
    if a is not None:
        act = np.where((act == CONTINUE) & (a > tau_a), INSPECT, act)
    return act


def gate_hera(I_edge, a, tau_a=3.0, use_anomaly=True):
    L, U = I_edge[:, 0], I_edge[:, 2]
    ambiguous = np.zeros(len(L), bool)
    for h in boundaries():
        ambiguous |= (L <= h) & (U > h)
    if use_anomaly:
        ambiguous |= (a > tau_a) & (region(L) == CONTINUE)
    return ambiguous


def gate_width(I_edge, tau_u):
    return (I_edge[:, 2] - I_edge[:, 0]) / R_MAX > tau_u


def gate_score(I_edge, a, tau, lam=(1 / 3, 1 / 3, 1 / 3), a_scale=10.0):
    """Linear escalation score E_t = l1*a~ + l2*u~ + l3*r~ > tau (all in [0,1])."""
    a_n = np.clip(a / a_scale, 0, 1)
    u_n = (I_edge[:, 2] - I_edge[:, 0]) / R_MAX
    r_n = 1 - I_edge[:, 1] / R_MAX
    return lam[0] * a_n + lam[1] * u_n + lam[2] * r_n > tau


def gate_anomaly(a, tau_a):
    return a > tau_a


def gate_point_margin(m_edge, delta):
    esc = np.zeros(len(m_edge), bool)
    for h in boundaries():
        esc |= np.abs(m_edge - h) < delta
    return esc


def gate_random(n, p, rng):
    return rng.random(n) < p


def hierarchical_decision(esc, act_edge, act_cloud):
    return np.where(esc, act_cloud, act_edge)


def decision_metrics(act: np.ndarray, y: np.ndarray, esc: np.ndarray | None = None) -> dict:
    crit = y <= _H["c"]
    healthy = y > _H["p"]
    true_r = region(y)
    no_maint = (act == CONTINUE) | (act == INSPECT)
    out = {
        "unsafe_rate": float(no_maint[crit].mean()) if crit.any() else np.nan,     # critical asset left running
        "missed_urgent": float((act != URGENT)[crit].mean()) if crit.any() else np.nan,
        "false_urgent": float((act == URGENT)[healthy].mean()) if healthy.any() else np.nan,
        "false_maint": float(np.isin(act, list(HIGH_IMPACT))[healthy].mean()) if healthy.any() else np.nan,
        "action_acc": float((np.where(act == INSPECT, CONTINUE, act) == true_r).mean()),
        "approval_load": float(np.isin(act, list(HIGH_IMPACT)).mean()),
    }
    if esc is not None:
        out["escalation_rate"] = float(esc.mean())
    return out
