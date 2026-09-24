"""Robustness analyses added in v4 (no retraining).

1. Bearing-level bootstrap 95% CIs (10,000 resamples of bearings, snapshot-weighted
   pooled rates, all three seeds pooled per bearing) for the key policy comparisons.
2. Required alarm offset per bearing: the smallest lambda for which the always-cloud
   CRC alarm of that bearing is not late. CRC's lambda_hat is (approximately) an upper
   quantile of these offsets over calibration bearings, which explains early alarms.
   Also the finite-sample CRC feasibility bound for a lateness target eps when a
   fraction pi of units gives no usable warning: n >= (1 - eps) / (eps - pi).
Outputs: results/<ds>/bootstrap_ci.json, results/<ds>/required_offset.csv, results/crc_sample_size.csv
Run: python experiments/bootstrap_offsets.py
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PAIRS = [("unsafe", "HERA-full", "Cloud point+threshold"), ("fm", "HERA-full", "Cloud point+threshold"),
         ("unsafe", "HERA-full", "Always-cloud"), ("fm", "HERA-full", "Always-cloud"),
         ("unsafe", "HERA-full", "Width gate (matched)"), ("fm", "HERA-full", "Width gate (matched)")]


def per_dataset():
    sys.path.insert(0, str(ROOT / "src"))
    import evaluation as E
    from alarm_crc import bearing_outcome, single_stage_alarm
    from escalation import CONTINUE, INSPECT, SCHEDULE, URGENT
    from utils import H_CRIT, H_PLAN, R_MAX, RESULTS, SEEDS
    lat = json.loads((RESULTS / "system_metrics.json").read_text())
    lat = {"edge_ms": lat[E.EDGE_TAG]["median_ms"], "cloud_ms": lat[E.CLOUD + "+DT"]["median_ms"],
           "edge_macs": lat[E.EDGE_TAG]["macs"], "cloud_macs": lat[E.CLOUD + "+DT"]["macs"]}
    rng = np.random.default_rng(0)
    counts = {}   # (policy, bearing) -> [unsafe_num, crit_den, fm_num, healthy_den]
    for s in SEEDS:
        _, D, acts = E.run_policies(s, lat, rng)
        crit, healthy = D["y"] <= H_CRIT, D["y"] > H_PLAN
        for name, (act, esc, I) in acts.items():
            nom = (act == CONTINUE) | (act == INSPECT)
            fm = np.isin(act, [SCHEDULE, URGENT])
            for b in np.unique(D["b"]):
                m = D["b"] == b
                c = counts.setdefault((name, b), np.zeros(4))
                c += [(nom & crit & m).sum(), (crit & m).sum(), (fm & healthy & m).sum(), (healthy & m).sum()]
    bearings = sorted({b for _, b in counts})
    boot = np.random.default_rng(1).integers(0, len(bearings), (10000, len(bearings)))
    out = {}
    for metric, a, bpol in PAIRS:
        k = (0, 1) if metric == "unsafe" else (2, 3)
        A = np.array([counts[(a, b)] for b in bearings])
        B = np.array([counts[(bpol, b)] for b in bearings])
        ra = A[boot][:, :, k[0]].sum(1) / np.maximum(A[boot][:, :, k[1]].sum(1), 1)
        rb = B[boot][:, :, k[0]].sum(1) / np.maximum(B[boot][:, :, k[1]].sum(1), 1)
        d = 100 * (ra - rb)
        point = 100 * (A[:, k[0]].sum() / A[:, k[1]].sum() - B[:, k[0]].sum() / B[:, k[1]].sum())
        out[f"{metric}|{a}|{bpol}"] = {"diff_pp": float(point), "ci_lo": float(np.percentile(d, 2.5)),
                                       "ci_hi": float(np.percentile(d, 97.5))}
    (RESULTS / "bootstrap_ci.json").write_text(json.dumps(out, indent=1))

    # required offset per test bearing (always-cloud alarm), pooled over folds and seeds
    import evaluate_alarms as EA
    grid = EA.GRID
    rows = []
    for s in SEEDS:
        for f in range(len(E.FOLDS)):
            _, test = EA.load_fold(f, s)
            for b, (me, mc, yr, a) in test.items():
                req = next((lam for lam in grid
                            if bearing_outcome(single_stage_alarm(mc, lam, H_CRIT), yr, H_CRIT, R_MAX)[0] != "late"),
                           np.nan)
                rows.append({"bearing": b, "seed": s, "required_offset_s": req, "r_max": R_MAX})
    df = pd.DataFrame(rows).groupby("bearing").agg(required_offset_s=("required_offset_s", "median"),
                                                  r_max=("r_max", "first")).reset_index()
    df["required_offset_frac"] = df.required_offset_s / df.r_max
    df.to_csv(RESULTS / "required_offset.csv", index=False)
    print(json.dumps(out, indent=1))
    print(df.sort_values("required_offset_frac").to_string(index=False))


def sample_size():
    rows = []
    for eps in (0.05, 0.1, 0.2):
        for pi in (0.0, 0.02, 0.05, 0.1, 0.15):
            n = np.inf if pi >= eps else int(np.ceil((1 - eps) / (eps - pi)))
            rows.append({"eps": eps, "pi_no_warning": pi, "n_min": n})
    pd.DataFrame(rows).to_csv(ROOT / "results" / "crc_sample_size.csv", index=False)
    print(pd.DataFrame(rows).pivot(index="pi_no_warning", columns="eps", values="n_min"))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--one":
        per_dataset()
    else:
        for ds in ("femto", "xjtu"):
            subprocess.run([sys.executable, __file__, "--one"], env=dict(os.environ, HERA_DATASET=ds), check=True)
        sample_size()
