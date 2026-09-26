"""Why does per-bearing coverage vary? (reviewer question, ICAIET 2027)

For the selected fog/cloud (+DT) and edge models with CQR-CV+ intervals (seeds 0-2, averaged per bearing),
reports each bearing's coverage, the share of its snapshots whose true RUL lies below / above the interval,
and its lifetime relative to the median lifetime of bearings run under the same operating condition.
Outputs results/<dataset>/coverage_diagnosis.json (summary) and coverage_per_bearing.csv.

    HERA_DATASET=femto python experiments/coverage_diagnosis.py
    HERA_DATASET=xjtu  python experiments/coverage_diagnosis.py
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import evaluation as ev  # noqa: E402

DS = os.environ.get("HERA_DATASET", "femto")
SUMMARY = ROOT / "data" / "processed" / ("dataset_summary.csv" if DS == "femto" else "xjtu_dataset_summary.csv")


def main():
    summ = pd.read_csv(SUMMARY)
    summ["life_rel"] = summ.life_s / summ.groupby("condition").life_s.transform("median")
    rows = []
    for s in (0, 1, 2):
        for tag, name in ((ev.CLOUD + "+DT", "cloud"), (ev.EDGE_TAG, "edge")):
            R = ev.load(tag, s)
            I = ev.calibrated(R, ev.ALPHA)
            y, b = ev.cat(R, "y"), ev.cat(R, "bearing")
            for bn in np.unique(b):
                m = b == bn
                rows.append({"bearing": bn, "model": name, "seed": s,
                             "cov": ((y[m] >= I[m, 0]) & (y[m] <= I[m, 2])).mean(),
                             "below": (y[m] < I[m, 0]).mean(), "above": (y[m] > I[m, 2]).mean()})
    d = (pd.DataFrame(rows).groupby(["bearing", "model"]).mean(numeric_only=True).drop(columns="seed").reset_index()
         .merge(summ[["bearing", "condition", "life_s", "life_rel"]], on="bearing"))
    d.to_csv(ev.RESULTS / "coverage_per_bearing.csv", index=False)
    c = d[d.model == "cloud"].sort_values("cov")
    low = c[c["cov"] < 1 - ev.ALPHA]
    rho = spearmanr(c["cov"], np.abs(np.log(c.life_rel)))
    worst = c.iloc[0]
    out = {"n_bearings": int(len(c)), "n_low": int(len(low)),
           "below_share_low": float(low.below.sum() / (low.below.sum() + low.above.sum())),
           "spearman_cov_vs_abs_log_life_rel": float(rho.statistic), "spearman_p": float(rho.pvalue),
           "worst_bearing": worst.bearing, "worst_cov": float(worst["cov"]), "worst_life_rel": float(worst.life_rel),
           "edge_n_low": int((d[d.model == "edge"]["cov"] < 1 - ev.ALPHA).sum())}
    (ev.RESULTS / "coverage_diagnosis.json").write_text(json.dumps(out, indent=1))
    print(DS, json.dumps(out))


if __name__ == "__main__":
    main()
