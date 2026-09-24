"""Bearing-level evaluation of lead-time-calibrated alarms (HERA-PdM v3).

    HERA_DATASET=femto python src/evaluate_alarms.py

For every fold and seed, lambda is calibrated by conformal risk control on the
fold's non-test bearings (CV+ out-of-fold predictions, never seen by the models
that produced them) and applied to the held-out test bearings.
Outputs (results/<dataset>/): alarm_bearings.csv (one row per policy/eps/seed/bearing),
alarm_summary.csv, alarm_eps_sweep.csv, alarm_lambda.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from alarm_crc import bearing_outcome, crc_lambda, gated_alarm, single_stage_alarm, split_by_bearing, two_stage_alarm  # noqa: E402
from utils import BASELINE_N, DATASET, FOLDS, H_CRIT, H_PLAN, R_MAX, RESULTS, ROOT, SEEDS, W_CLOUD  # noqa: E402

SEL = json.loads((ROOT / "results" / DATASET / "selected_models.json").read_text())
EDGE, CLOUD = SEL["edge_model"], SEL["cloud_model"] + "+DT"
GRID = np.linspace(-H_CRIT, R_MAX, 321)
EPS_MAIN = (0.1, 0.2)
EPS_SWEEP = (0.1, 0.15, 0.2, 0.25, 0.3, 0.4)
from utils import n_features  # noqa: E402
FEAT_B, SYNC_B = n_features() * 4, 3 * 4


def tau_anomaly(fold):
    info = json.loads((RESULTS / "run_info.json").read_text())
    return info["tau_a_per_fold"][fold]


def load_fold(fold, seed):
    e = np.load(RESULTS / "cvplus" / f"fold{fold}_seed{seed}_{EDGE}.npz", allow_pickle=True)
    c = np.load(RESULTS / "cvplus" / f"fold{fold}_seed{seed}_{CLOUD}.npz", allow_pickle=True)
    m = np.load(RESULTS / "preds" / f"fold{fold}_meta.npz", allow_pickle=True)
    assert (e["oof_bearing"] == c["oof_bearing"]).all() and (e["oof_t"] == c["oof_t"]).all()
    cal = split_by_bearing(c["oof_bearing"], c["oof_t"], e["q_oof"][:, 1], c["q_oof"][:, 1], c["y_oof"], c["a_oof"])
    test = split_by_bearing(m["test_bearing"], m["test_t"], e["q_test"][:, :, 1].mean(0), c["q_test"][:, :, 1].mean(0),
                            m["test_rul"], m["test_a"])
    return cal, test


POLICIES = {
    # name: (calibrated?, function(me, mc, lam, a, tau) -> (esc_idx, alarm_idx))
    "HERA-v3 (onset-gated, CRC)": (True, lambda me, mc, lam, a, tau: two_stage_alarm(me, mc, lam, H_CRIT, H_PLAN)),
    "HERA-v3b (onset gate + burn-in, CRC)": (True, lambda me, mc, lam, a, tau: gated_alarm(me, mc, a, tau, lam, H_CRIT, H_PLAN, BASELINE_N)),
    "Always-cloud, CRC + burn-in": (True, lambda me, mc, lam, a, tau: (0, single_stage_alarm(mc, lam, H_CRIT, BASELINE_N))),
    "Always-cloud, CRC": (True, lambda me, mc, lam, a, tau: (0, single_stage_alarm(mc, lam, H_CRIT))),
    "Always-edge, CRC": (True, lambda me, mc, lam, a, tau: (None, single_stage_alarm(me, lam, H_CRIT))),
    "Cloud point, fixed thr. H_p": (False, lambda me, mc, lam, a, tau: (0, single_stage_alarm(mc, 0.0, H_PLAN))),
    "Cloud point, fixed thr. H_c": (False, lambda me, mc, lam, a, tau: (0, single_stage_alarm(mc, 0.0, H_CRIT))),
    "Anomaly threshold": (False, lambda me, mc, lam, a, tau: (None, single_stage_alarm(-a, 0.0, -tau))),
}


def late_matrix(cal, fn, tau):
    L = np.zeros((len(cal), len(GRID)))
    for i, (b, (me, mc, yr, a)) in enumerate(sorted(cal.items())):
        for j, lam in enumerate(GRID):
            _, al = fn(me, mc, lam, a, tau)
            L[i, j] = bearing_outcome(al, yr, H_CRIT, R_MAX)[0] == "late"
    return L


def comm(esc, T):
    """Payload per snapshot for a latched escalation starting at index esc (None: never)."""
    if esc is None:
        return SYNC_B, 0.0
    after = T - esc
    return (SYNC_B * T + FEAT_B * (after + min(W_CLOUD - 1, esc))) / T, after / T


def main():
    rows, lam_rows = [], []
    for seed in SEEDS:
        for f in range(len(FOLDS)):
            cal, test = load_fold(f, seed)
            tau = tau_anomaly(f)
            for name, (calib, fn) in POLICIES.items():
                if calib:
                    L = late_matrix(cal, fn, tau)
                    lams = {eps: crc_lambda(L, GRID, eps) for eps in EPS_SWEEP}
                else:
                    lams = {None: 0.0}
                for eps, lam in lams.items():
                    lam_rows.append({"policy": name, "eps": eps, "seed": seed, "fold": f, "lambda": lam,
                                     "n_cal": len(cal)})
                    for b, (me, mc, yr, a) in test.items():
                        esc, al = fn(me, mc, lam, a, tau)
                        out, rul = bearing_outcome(al, yr, H_CRIT, R_MAX)
                        if name.startswith("HERA"):  # latched escalation
                            by, frac = comm(esc, len(yr))
                        elif "cloud" in name.lower():
                            by, frac = float(FEAT_B), 1.0          # always-cloud feature streaming
                        else:
                            by, frac = 0.0, 0.0                    # edge only
                        rows.append({"policy": name, "eps": eps, "seed": seed, "fold": f, "bearing": b, "outcome": out,
                                     "rul_at_alarm_s": rul, "lambda": lam, "esc_frac": frac, "bytes_per_snapshot": by,
                                     "life_s": float(yr.max())})
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "alarm_bearings.csv", index=False)
    pd.DataFrame(lam_rows).to_csv(RESULTS / "alarm_lambda.csv", index=False)

    def summarize(d):
        g = d.groupby(["policy", "eps", "seed"], dropna=False)
        s = g.apply(lambda x: pd.Series({
            "late_rate": (x.outcome == "late").mean(), "premature_rate": (x.outcome == "premature").mean(),
            "timely_rate": (x.outcome == "timely").mean(), "n_bearings": len(x),
            "median_lead_min": x.loc[x.outcome != "late", "rul_at_alarm_s"].median() / 60,
            "esc_frac": x.esc_frac.mean(), "bytes": x.bytes_per_snapshot.mean()}), include_groups=False)
        s = s.reset_index()
        num = s.drop(columns="seed").groupby(["policy", "eps"], dropna=False)
        return num.mean().add_suffix("_mean").join(num.std(ddof=1).add_suffix("_std")).reset_index()

    summ = summarize(df)
    summ.to_csv(RESULTS / "alarm_eps_sweep.csv", index=False)
    main_rows = summ[summ.eps.isna() | summ.eps.isin(EPS_MAIN)]
    main_rows.to_csv(RESULTS / "alarm_summary.csv", index=False)
    pd.set_option("display.width", 220)
    print(main_rows[["policy", "eps", "late_rate_mean", "premature_rate_mean", "timely_rate_mean", "median_lead_min_mean",
                     "esc_frac_mean", "bytes_mean"]].round(3).to_string(index=False))
    print(pd.DataFrame(lam_rows).groupby(["policy", "eps"]).agg(lam=("lambda", "mean"), n_cal=("n_cal", "mean")).round(1))


if __name__ == "__main__":
    main()
