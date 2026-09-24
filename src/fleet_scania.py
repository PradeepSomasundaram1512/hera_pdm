"""Fleet-scale study on SCANIA Component X (Kharazian et al., Sci. Data 2025; CC BY 4.0).

Units are trucks. For a truck with an in-study repair, the remaining life at readout t
is RUL_t = length_of_study - t; trucks without repair are right-censored.
Per-readout risk model: p_t = P(RUL_t <= H_P) (H_P = 48 time units, Scania's longest label window).
  edge  : logistic regression on the current readout (standardized counters)
  cloud : histogram gradient boosting on counters, per-time-unit rates since the previous readout,
          time step and truck specifications
Both are trained with vehicle-grouped 5-fold cross-fitting, so every score is out-of-fold.

Alarm (sustained over K=2 consecutive readouts): single-stage alarm when p_t >= theta; HERA two-stage:
the edge opens a (latched) gate when p^e_t >= theta/2, afterwards the cloud alarms when p^c_t >= theta.
Loss for a failing truck: late = no alarm before RUL < H_C (H_C = 12). Conformal risk control over n
calibration failing trucks picks the largest theta with (n Rhat(theta) + 1)/(n+1) <= eps, which bounds
P(late | failing truck) <= eps. False alarm on a non-failing truck: an alarm while more than H_P time
units of observed, repair-free operation remain. Premature: alarm with RUL > R_MAX = 96.

    python src/fleet_scania.py        -> results/scania/*.csv|json
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "SCANIA"
OUT = ROOT / "results" / "scania"
H_C, H_P, R_MAX, K = 12.0, 48.0, 96.0, 2
EPS = (0.05, 0.1, 0.2)
N_CAL = (12, 25, 50, 100, 200, 500, 1000)
REPS = 50
SYNC_B, FEAT_B = 12, None


def load():
    ops = pd.read_csv(RAW / "train_operational_readouts.csv").sort_values(["vehicle_id", "time_step"])
    tte = pd.read_csv(RAW / "train_tte.csv")
    spec = pd.read_csv(RAW / "train_specifications.csv")
    counters = [c for c in ops.columns if c not in ("vehicle_id", "time_step")]
    ops[counters] = ops.groupby("vehicle_id")[counters].ffill().fillna(0.0)
    d = ops.merge(tte, on="vehicle_id")
    d["remain"] = d.length_of_study_time_step - d.time_step
    fail = d.in_study_repair == 1
    d["y"] = np.where(fail, (d.remain <= H_P).astype(float), np.where(d.remain > H_P, 0.0, np.nan))
    dt = d.groupby("vehicle_id").time_step.diff()
    rates = d.groupby("vehicle_id")[counters].diff().div(dt, axis=0).fillna(0.0)
    rates.columns = [c + "_rate" for c in counters]
    for c in spec.columns[1:]:
        spec[c] = spec[c].astype("category").cat.codes
    d = pd.concat([d, rates], axis=1).merge(spec, on="vehicle_id")
    return d, counters, list(rates.columns), [c for c in spec.columns[1:]]


def crossfit(d, counters, rates, specs):
    groups = d.vehicle_id.values
    lab = ~d.y.isna().values
    pe = np.zeros(len(d))
    pc = np.zeros(len(d))
    Xe = np.log1p(np.clip(d[counters].values, 0, None))
    Xc = np.column_stack([Xe, np.sign(d[rates].values) * np.log1p(np.abs(d[rates].values)),
                          d[["time_step"]].values, d[specs].values])
    for k, (tr, te) in enumerate(GroupKFold(5).split(Xe, groups=groups)):
        t0 = time.time()
        trl = tr[lab[tr]]
        sc = StandardScaler().fit(Xe[trl])
        lr = LogisticRegression(max_iter=300, C=0.1, class_weight="balanced").fit(sc.transform(Xe[trl]), d.y.values[trl])
        pe[te] = lr.predict_proba(sc.transform(Xe[te]))[:, 1]
        gb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_leaf_nodes=31,
                                            class_weight="balanced", random_state=k).fit(Xc[trl], d.y.values[trl])
        pc[te] = gb.predict_proba(Xc[te])[:, 1]
        print(f"fold {k}: {time.time() - t0:.0f}s", flush=True)
    return pe, pc, Xe.shape[1], Xc.shape[1]


def sustained(p):
    s = p.copy()
    s[1:] = np.minimum(p[1:], p[:-1])      # K = 2 consecutive readouts
    s[0] = -1.0
    return s


def vehicles(d, pe, pc):
    V = []
    for vid, g in d.assign(pe=pe, pc=pc).groupby("vehicle_id", sort=False):
        V.append({"id": vid, "fail": bool(g.in_study_repair.iloc[0]), "remain": g.remain.values,
                  "se": sustained(g.pe.values), "sc": sustained(g.pc.values), "n": len(g)})
    return V


def first(mask):
    return int(np.argmax(mask)) if mask.any() else None


def alarm(v, theta, policy):
    if policy == "cloud":
        return None, first(v["sc"] >= theta)
    if policy == "edge":
        return None, first(v["se"] >= theta)
    g = first(v["se"] >= theta / 2)                     # HERA: latched edge gate, cloud alarm
    if g is None:
        return None, None
    m = v["sc"] >= theta
    m[:g] = False
    return g, first(m)


def outcome(v, a):
    r = v["remain"]
    if v["fail"]:
        if a is None or r[a] < H_C:
            return "late", np.nan
        return ("premature" if r[a] > R_MAX else "timely"), float(r[a])
    return ("false_alarm" if a is not None and r[a] > H_P else "quiet"), np.nan


GRID = np.round(np.linspace(0.01, 0.99, 197), 4)
CODES = {"late": 0, "premature": 1, "timely": 2, "false_alarm": 3, "quiet": 4}


def tables(vs, policy):
    """Outcome code, lead and escalation fraction of every truck for every theta in GRID
    (vectorized over GRID; identical to evaluating alarm()/outcome() per theta)."""
    G = len(GRID)
    O = np.zeros((len(vs), G), np.int8)
    LEAD = np.full((len(vs), G), np.nan)
    ESC = np.zeros((len(vs), G))
    for i, v in enumerate(vs):
        n, r = v["n"], v["remain"]
        mc = v["sc"][None, :] >= GRID[:, None]            # (G, n)
        me = v["se"][None, :] >= GRID[:, None]
        if policy == "edge":
            m = me
        elif policy == "cloud":
            m = mc
        else:
            mg = v["se"][None, :] >= (GRID / 2)[:, None]
            has_g = mg.any(1)
            g = np.where(has_g, mg.argmax(1), n)
            m = mc & (np.arange(n)[None, :] >= g[:, None])
            ESC[i] = np.where(has_g, (n - g) / n, 0.0)
        has = m.any(1)
        a = m.argmax(1)
        ra = np.where(has, r[np.minimum(a, n - 1)], np.nan)
        if v["fail"]:
            late = ~has | (ra < H_C)
            O[i] = np.where(late, 0, np.where(ra > R_MAX, 1, 2))
            LEAD[i] = np.where(late, np.nan, ra)
        else:
            O[i] = np.where(has & (ra > H_P), 3, 4)
        if policy == "cloud":
            ESC[i] = 1.0
    return O, LEAD, ESC


def crc_theta(late, eps):
    n = late.shape[0]
    ok = (n * late.mean(0) + 1) / (n + 1) <= eps
    return int(np.where(ok)[0].max()) if ok.any() else 0


def evaluate(T, idx_f, idx_h, j):
    O, LEAD, ESC = T
    of, oh = O[idx_f, j], O[idx_h, j]
    ld = LEAD[idx_f, j]
    ld = ld[(of == 1) | (of == 2)]
    return {"late_rate": float((of == 0).mean()), "premature_rate": float((of == 1).mean()),
            "timely_rate": float((of == 2).mean()), "false_alarm_rate": float((oh == 3).mean()),
            "median_lead": float(np.median(ld)) if len(ld) else np.nan,
            "esc_frac": float(np.concatenate([ESC[idx_f, j], ESC[idx_h, j]]).mean())}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    d, counters, rates, specs = load()
    print("loaded", d.shape, f"{time.time() - t0:.0f}s", flush=True)
    pe, pc, n_edge_feat, n_cloud_feat = crossfit(d, counters, rates, specs)
    from sklearn.metrics import roc_auc_score
    lab = ~d.y.isna().values
    auc = {"edge": float(roc_auc_score(d.y.values[lab], pe[lab])), "cloud": float(roc_auc_score(d.y.values[lab], pc[lab]))}
    V = vehicles(d, pe, pc)
    fail = [v for v in V if v["fail"]]
    ok = [v for v in V if not v["fail"]]
    print("vehicles", len(V), "failing", len(fail), "AUROC", auc, flush=True)
    rng = np.random.default_rng(0)
    order = fail + ok
    nf = len(fail)
    T = {}
    for p in ("edge", "cloud", "hera"):
        t1 = time.time()
        T[p] = tables(order, p)
        print(f"table {p}: {time.time() - t1:.0f}s", flush=True)
    j05 = int(np.argmin(np.abs(GRID - 0.5)))
    rows = []
    for rep in range(REPS):
        pf = rng.permutation(nf)
        ph = nf + rng.permutation(len(ok))
        cal_pool, test_f, test_h = pf[: nf // 2], pf[nf // 2:], ph[len(ok) // 2:]
        for n in N_CAL:
            idx = cal_pool[:n]
            for p in ("edge", "cloud", "hera"):
                late = T[p][0][idx] == 0
                for eps in EPS:
                    jj = crc_theta(late, eps)
                    rows.append({"rep": rep, "n_cal": n, "policy": p, "eps": eps, "theta": float(GRID[jj]),
                                 **evaluate(T[p], test_f, test_h, jj)})
        for p in ("edge", "cloud"):
            rows.append({"rep": rep, "n_cal": 0, "policy": p + "_fixed0.5", "eps": np.nan, "theta": 0.5,
                         **evaluate(T[p], test_f, test_h, j05)})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "fleet_runs.csv", index=False)
    summ = df.groupby(["policy", "eps", "n_cal"], dropna=False).agg(
        late=("late_rate", "mean"), late_q95=("late_rate", lambda x: np.quantile(x, 0.95)),
        premature=("premature_rate", "mean"), timely=("timely_rate", "mean"), false_alarm=("false_alarm_rate", "mean"),
        lead=("median_lead", "mean"), esc=("esc_frac", "mean"), theta=("theta", "mean")).reset_index()
    summ.to_csv(OUT / "fleet_summary.csv", index=False)
    info = {"n_vehicles": len(V), "n_failing": len(fail), "n_readouts": int(len(d)), "auroc": auc,
            "n_edge_features": n_edge_feat, "n_cloud_features": n_cloud_feat, "H_C": H_C, "H_P": H_P, "R_MAX": R_MAX,
            "K": K, "reps": REPS}
    (OUT / "fleet_info.json").write_text(json.dumps(info, indent=2))
    pd.set_option("display.width", 200)
    print(summ.round(3).to_string(index=False))
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
