"""Evaluation of HERA-PdM from saved predictions (v2: CV+ calibration, two datasets).

    HERA_DATASET=femto python src/evaluation.py          # full evaluation
    HERA_DATASET=femto HERA_RMAX=1500 python src/evaluation.py --light   # R_MAX sensitivity run

Produces (in results/<dataset>/):
  rul_metrics.csv          accuracy + calibration per model (split CQR for all, CV+ for the selected)
  policy_metrics.csv       hierarchical policies & ablations (mean/std over seeds)
  tradeoff_curves.csv      escalation-budget sweeps and decision Pareto sweeps
  alarm_timing.csv         per-bearing first sustained maintenance alarm (timely / late / premature)
  horizon_sensitivity.csv  decision metrics for alternative (H_c, H_p)
  per_bearing.csv, stat_tests.json, detection_metrics.csv, system_metrics.json, trajectories.csv.gz
Protocol: metrics are pooled over the held-out bearings of all folds (each bearing
is tested exactly once per seed) and summarized as mean +- std over three seeds.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).parent))
from digital_twin import CONTEXT_NAMES  # noqa: E402
from edge_agent import detection_metrics, isolation_forest_scores  # noqa: E402
from escalation import (CONTINUE, INSPECT, SCHEDULE, URGENT, decide, decision_metrics, gate_anomaly,  # noqa: E402
                        gate_hera, gate_point_margin, gate_random, gate_score, gate_width, hierarchical_decision,
                        set_horizons)
from uncertainty import (apply_cqr, calibrate, conformal_quantile, coverage_calibration_error,  # noqa: E402
                         cqr_scores, cvplus_cqr, interval_metrics)
from utils import (ALPHA, CONFIGS, DATASET, FEATURES, FOLDS, H_CRIT, H_PLAN, QUANTILES, R_MAX,  # noqa: E402
                   RESULTS, ROOT, SEEDS, W_CLOUD, W_EDGE, save_json)

PRED, CVP = RESULTS / "preds", RESULTS / "cvplus"
BASE = ROOT / "results" / DATASET
N_CTX = len(CONTEXT_NAMES)
CLOUD_ARCHS = ["LSTM", "GRU", "TCN", "CNN-LSTM"]
EDGE_ARCHS = ["Edge-CNN", "Edge-GRU"]
TAU_Q = 0.99               # alarm threshold = 99th pct of a_t on healthy (y >= R_MAX) training snapshots
RTT_MS = 50.0              # assumed edge<->fog/cloud round-trip time (not measured)
FEAT_BYTES = 28 * 4        # one feature vector, float32
SYNC_BYTES = 3 * 4         # DT sync message: (t, HI, a_t)
RAW_BYTES = CONFIGS[DATASET]["snapshot_samples"] * 2 * 2   # one raw 2-channel snapshot at 16 bit
SUSTAIN = 3                # consecutive snapshots for a sustained alarm (per-bearing timing)

SEL = json.loads((BASE / "selected_models.json").read_text())
EDGE_TAG, CLOUD = SEL["edge_model"], SEL["cloud_model"]
CV_TAGS = {EDGE_TAG, CLOUD + "+DT", CLOUD}

_ASSETS = {}


def assets():
    if not _ASSETS:
        from digital_twin import build_asset, feature_columns
        df = pd.read_csv(FEATURES)
        feats = feature_columns(df)
        _ASSETS.update({b: build_asset(g.reset_index(drop=True), feats) for b, g in df.groupby("bearing")})
    return _ASSETS


def tau_anomaly(fold):
    from utils import fold_split
    A = assets()
    train, _, _ = fold_split(fold, sorted(A))
    a = np.concatenate([A[n]["a"][A[n]["rul"] >= R_MAX] for n in train])
    return float(np.quantile(a, TAU_Q)) if len(a) else float(np.quantile(np.concatenate([A[n]["a"] for n in train]), 0.5))


def load(tag, seed, calib="auto"):
    """Rows per fold. For CV+ tags the test quantiles are the mean of the inner
    models (point estimate) and the CV+ ingredients are kept for calibration."""
    use_cv = calib == "cvplus" or (calib == "auto" and tag in CV_TAGS
                                   and (CVP / f"fold0_seed{seed}_{tag}.npz").exists())
    rows = []
    for f in range(len(FOLDS)):
        m = np.load(PRED / f"fold{f}_meta.npz", allow_pickle=True)
        tau = tau_anomaly(f)
        r = {"fold": f, "y": np.minimum(m["test_rul"], R_MAX), "a": m["test_a"], "bearing": m["test_bearing"],
             "t": m["test_t"], "rul_raw": m["test_rul"], "tau_a": np.full(len(m["test_rul"]), tau)}
        if use_cv:
            c = np.load(CVP / f"fold{f}_seed{seed}_{tag}.npz", allow_pickle=True)
            r |= {"cv": c, "q_test": c["q_test"].mean(0), "q_cal": c["q_oof"], "y_cal": np.minimum(c["y_oof"], R_MAX),
                  "a_cal": c["a_oof"], "tau_a_cal": np.full(len(c["y_oof"]), tau)}
        else:
            p = np.load(PRED / f"fold{f}_seed{seed}_{tag}.npz")
            r |= {"q_test": p["q_test"], "q_cal": p["q_cal"], "y_cal": np.minimum(m["cal_rul"], R_MAX),
                  "a_cal": m["cal_a"], "tau_a_cal": np.full(len(m["cal_rul"]), tau)}
        rows.append(r)
    return rows


def calibrated(rows, alpha, mode=None):
    out = []
    for r in rows:
        if "cv" in r and mode in (None, "cvplus"):
            c = r["cv"]
            out.append(cvplus_cqr(c["q_oof"], c["g_oof"], np.minimum(c["y_oof"], R_MAX), c["q_test"], alpha))
        else:
            out.append(calibrate(r["q_cal"], r["y_cal"], r["q_test"], alpha, mode or "sym"))
    return np.concatenate(out)


def calibrated_cal(rows, alpha):
    """Marginal CQR intervals on the calibration (or CV+ out-of-fold) points themselves;
    used only to set baseline thresholds to a matched escalation budget without test data."""
    return np.concatenate([apply_cqr(r["q_cal"], conformal_quantile(cqr_scores(r["q_cal"], r["y_cal"]), alpha))
                           for r in rows])


def cat(rows, k):
    return np.concatenate([r[k] for r in rows])


def rul_block(rows, modes, point_only=False):
    y, q = cat(rows, "y"), cat(rows, "q_test")
    err = q[:, 1] - y
    out = {"rmse": float(np.sqrt(np.mean(err ** 2))), "mae": float(np.mean(np.abs(err))),
           "rmse_deg": float(np.sqrt(np.mean(err[y < R_MAX] ** 2)))}
    if point_only:
        return out
    out["raw_picp"] = float(((y >= q[:, 0]) & (y <= q[:, 2])).mean())
    for i, mode in enumerate(modes):
        sfx = "" if i == 0 else "_" + mode
        I = calibrated(rows, ALPHA, mode)
        out |= {k + sfx: v for k, v in interval_metrics(I, y, ALPHA).items()}
        out["lower_miss_crit" + sfx] = float((y < I[:, 0])[y <= H_CRIT].mean())
        curve = []
        for a in np.linspace(0.05, 0.5, 10):
            Ia = calibrated(rows, a, mode)
            curve.append((1 - a, float(((y >= Ia[:, 0]) & (y <= Ia[:, 2])).mean())))
        out["cce" + sfx] = coverage_calibration_error(curve)
        cov_b = pd.Series((y >= I[:, 0]) & (y <= I[:, 2])).groupby(cat(rows, "bearing")).mean()
        out["picp_bearing_min" + sfx] = float(cov_b.min())
        out["n_bearing_cov_below" + sfx] = int((cov_b < 1 - ALPHA).sum())
    return out


def mean_std(dicts):
    df = pd.DataFrame(dicts)
    return {f"{k}_mean": df[k].mean() for k in df} | {f"{k}_std": df[k].std(ddof=1) for k in df}


# ----------------------------------------------------------------------------- policies
def comm_bytes(esc, bearing, t, mode):
    """Analytic communication payload per snapshot (bytes); header overhead excluded."""
    if mode == "cloud_features":
        return np.full(len(esc), FEAT_BYTES, float)
    if mode == "edge":
        return np.zeros(len(esc))
    out = np.full(len(esc), SYNC_BYTES, float)
    for b in np.unique(bearing):
        ii = np.where(bearing == b)[0]
        ii = ii[np.argsort(t[ii])]
        sent_upto = -1
        for j in ii:
            if esc[j]:
                start = max(t[j] - W_CLOUD + 1, sent_upto + 1)
                out[j] += FEAT_BYTES * (t[j] - start + 1)
                sent_upto = t[j]
    return out


def policy_eval(name, act, esc, I_final, y, bearing, t, lat):
    d = decision_metrics(act, y, esc)
    comm = comm_bytes(esc, bearing, t, lat["comm_mode"])
    if lat["comm_mode"] == "edge":
        e2e = np.full(len(esc), lat["edge_ms"])
    elif lat["comm_mode"] == "cloud_features":
        e2e = np.full(len(esc), RTT_MS + lat["cloud_ms"])
    else:
        e2e = lat["edge_ms"] + esc * (RTT_MS + lat["cloud_ms"])
    macs = {"edge": lat["edge_macs"], "cloud_features": lat["cloud_macs"]}.get(
        lat["comm_mode"], lat["edge_macs"] + esc * lat["cloud_macs"])
    err = I_final[:, 1] - y
    out = {"policy": name, **d, "rmse": float(np.sqrt(np.mean(err ** 2))), "mae": float(np.abs(err).mean()),
           "bytes_per_snapshot": float(comm.mean()), "latency_mean_ms": float(np.mean(e2e)),
           "kmacs_per_snapshot": float(np.mean(macs) / 1e3)}
    if not np.allclose(I_final[:, 0], I_final[:, 2]):
        out |= {"picp": float(((y >= I_final[:, 0]) & (y <= I_final[:, 2])).mean()),
                "mpiw": float((I_final[:, 2] - I_final[:, 0]).mean())}
    return out


def match_threshold(scores_cal, target_rate):
    return float(np.quantile(scores_cal, 1 - target_rate))


def point(m):
    return np.stack([m, m, m], 1)


def policy_actions(seed, alpha_edge=ALPHA, rng=None):
    """All decision arrays needed by the policy table, timing and sensitivity analyses."""
    E, C = load(EDGE_TAG, seed), load(CLOUD + "+DT", seed)
    Cn, Cp = load(CLOUD, seed), load(CLOUD + "+DT-point", seed)
    y, a, b, t, yr = cat(E, "y"), cat(E, "a"), cat(E, "bearing"), cat(E, "t"), cat(E, "rul_raw")
    tau, tau_cal = cat(E, "tau_a"), cat(E, "tau_a_cal")
    Ie, Ic, Icn = calibrated(E, alpha_edge), calibrated(C, ALPHA), calibrated(Cn, ALPHA)
    pe, pc = cat(E, "q_test")[:, 1], cat(Cp, "q_test")[:, 1]
    Ie_cal, a_cal = calibrated_cal(E, alpha_edge), cat(E, "a_cal")
    act_e, act_c, act_cn = decide(Ie, a, tau), decide(Ic, a, tau), decide(Icn, a, tau)
    act_pe, act_pc = decide(point(pe), a, tau), decide(point(pc), a, tau)
    n = len(y)
    esc = {"HERA": gate_hera(Ie, a, tau, True), "HERA-noA": gate_hera(Ie, a, tau, False)}
    target = float(gate_hera(Ie_cal, a_cal, tau_cal, True).mean())
    w_cal = (Ie_cal[:, 2] - Ie_cal[:, 0]) / R_MAX
    esc["Width"] = gate_width(Ie, match_threshold(w_cal, target))
    s_cal = (np.clip(a_cal / (2 * tau_cal), 0, 1) + w_cal + 1 - Ie_cal[:, 1] / R_MAX) / 3
    esc["Score"] = gate_score(Ie, a, match_threshold(s_cal, target), a_scale=2 * tau)
    esc["Random"] = gate_random(n, target, rng or np.random.default_rng(seed))
    esc["Anomaly"] = gate_anomaly(a, tau)
    delta = float(np.sqrt(np.mean((cat(E, "q_cal")[:, 1] - cat(E, "y_cal")) ** 2)))
    esc["Point"] = gate_point_margin(pe, delta)
    return dict(y=y, a=a, b=b, t=t, yr=yr, Ie=Ie, Ic=Ic, Icn=Icn, pe=pe, pc=pc, act_e=act_e, act_c=act_c,
                act_cn=act_cn, act_pe=act_pe, act_pc=act_pc, esc=esc, target=target)


POLICIES = [  # name, escalation key (None: fixed), edge action, cloud action, edge interval, cloud interval, comm mode
    ("Always-edge", "none", "act_e", "act_e", "Ie", "Ie", "edge"),
    ("Always-cloud", "all", "act_c", "act_c", "Ic", "Ic", "cloud_features"),
    ("Cloud point+threshold", "all", "act_pc", "act_pc", "Pc", "Pc", "cloud_features"),
    ("HERA-full", "HERA", "act_e", "act_c", "Ie", "Ic", "hier"),
    ("HERA w/o anomaly term", "HERA-noA", "act_e", "act_c", "Ie", "Ic", "hier"),
    ("HERA-no-DT", "HERA", "act_e", "act_cn", "Ie", "Icn", "hier"),
    ("HERA-no-UQ", "Point", "act_pe", "act_pc", "Pe", "Pc", "hier"),
    ("HERA-fixed (anomaly trigger)", "Anomaly", "act_e", "act_c", "Ie", "Ic", "hier"),
    ("Width gate (matched)", "Width", "act_e", "act_c", "Ie", "Ic", "hier"),
    ("Linear score gate (matched)", "Score", "act_e", "act_c", "Ie", "Ic", "hier"),
    ("Random gate (matched)", "Random", "act_e", "act_c", "Ie", "Ic", "hier"),
]


def resolve(D, name, esc_key, ae, ac, ie, ic):
    n = len(D["y"])
    esc = np.zeros(n, bool) if esc_key == "none" else np.ones(n, bool) if esc_key == "all" else D["esc"][esc_key]
    iv = {"Ie": D["Ie"], "Ic": D["Ic"], "Icn": D["Icn"], "Pe": point(D["pe"]), "Pc": point(D["pc"])}
    act = hierarchical_decision(esc, D[ae], D[ac])
    I = np.where(esc[:, None], iv[ic], iv[ie])
    return act, esc, I


def run_policies(seed, lat, rng):
    D = policy_actions(seed, rng=rng)
    res, acts = [], {}
    for name, ek, ae, ac, ie, ic, mode in POLICIES:
        act, esc, I = resolve(D, name, ek, ae, ac, ie, ic)
        acts[name] = (act, esc, I)
        res.append(policy_eval(name, act, esc, I, D["y"], D["b"], D["t"], dict(lat, comm_mode=mode)) | {"seed": seed})
    return res, D, acts


def alarm_timing(act, yr, b, t):
    """First sustained (SUSTAIN consecutive snapshots) SCHEDULE/URGENT recommendation per bearing.
    late: none or raised with true RUL < H_CRIT; premature: raised while true RUL > R_MAX; else timely."""
    out = []
    for bn in np.unique(b):
        ii = np.where(b == bn)[0]
        ii = ii[np.argsort(t[ii])]
        maint = np.isin(act[ii], [SCHEDULE, URGENT]).astype(int)
        run = np.convolve(maint, np.ones(SUSTAIN, int), "full")[:len(ii)] >= SUSTAIN
        if run.any():
            k = int(np.argmax(run))
            rul = float(yr[ii][k])
            cat_ = "late" if rul < H_CRIT else "premature" if rul > R_MAX else "timely"
        else:
            rul, cat_ = np.nan, "late"
        out.append({"bearing": bn, "rul_at_alarm_s": rul, "outcome": cat_})
    return out


def tradeoff(seed, D, rng):
    E, C = load(EDGE_TAG, seed), load(CLOUD + "+DT", seed)
    y, a = D["y"], D["a"]
    tau = cat(E, "tau_a")
    rows = []

    def add(fam, par, esc, act_edge, act_cloud, I_edge, I_cloud):
        act = hierarchical_decision(esc, act_edge, act_cloud)
        med = np.where(esc, I_cloud[:, 1], I_edge[:, 1])
        rows.append({"family": fam, "param": par, "seed": seed, **decision_metrics(act, y, esc),
                     "rmse": float(np.sqrt(np.mean((med - y) ** 2)))})

    Ic, act_c, Ie, act_e = D["Ic"], D["act_c"], D["Ie"], D["act_e"]
    for ae in [0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]:
        Iea = calibrated(E, ae)
        add("HERA (decision-sufficiency)", ae, gate_hera(Iea, a, tau, True), decide(Iea, a, tau), act_c, Iea, Ic)
    w = (Ie[:, 2] - Ie[:, 0]) / R_MAX
    s = (np.clip(a / (2 * tau), 0, 1) + w + 1 - Ie[:, 1] / R_MAX) / 3
    for q in np.linspace(0.02, 0.98, 25):
        add("Width gate", q, gate_width(Ie, np.quantile(w, 1 - q)), act_e, act_c, Ie, Ic)
        add("Linear score gate", q, s > np.quantile(s, 1 - q), act_e, act_c, Ie, Ic)
        add("Anomaly trigger", q, a > np.quantile(a, 1 - q), act_e, act_c, Ie, Ic)
        add("Random gate", q, gate_random(len(y), q, rng), act_e, act_c, Ie, Ic)
    none, allc = np.zeros(len(y), bool), np.ones(len(y), bool)
    for al in [0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        Iea, Ica = calibrated(E, al), calibrated(C, al)
        ae_, ac_ = decide(Iea, a, tau), decide(Ica, a, tau)
        add("P: Always-edge (CQR)", al, none, ae_, ae_, Iea, Iea)
        add("P: Always-cloud (CQR)", al, allc, ac_, ac_, Ica, Ica)
        add("P: HERA", al, gate_hera(Iea, a, tau, True), ae_, ac_, Iea, Ica)
    for frac in np.linspace(0, 0.8, 11):
        dl = frac * R_MAX
        ape = decide(point(np.clip(D["pe"] - dl, 0, None)), a, tau)
        apc = decide(point(np.clip(D["pc"] - dl, 0, None)), a, tau)
        add("P: Edge point - margin", dl, none, ape, ape, point(D["pe"]), point(D["pe"]))
        add("P: Cloud point - margin", dl, allc, apc, apc, point(D["pc"]), point(D["pc"]))
    return rows


def horizon_sensitivity(seed, rng):
    rows = []
    for fc, fp in [(0.1, 0.4), (0.1, 0.6), (0.2, 0.6), (0.2, 0.8), (0.3, 0.8)]:
        set_horizons(fc * R_MAX, fp * R_MAX)
        D = policy_actions(seed, rng=rng)
        for name, ek, ae, ac, ie, ic, _ in POLICIES:
            if name not in ("Always-edge", "Always-cloud", "Cloud point+threshold", "HERA-full", "Width gate (matched)"):
                continue
            act, esc, _ = resolve(D, name, ek, ae, ac, ie, ic)
            rows.append({"hc_frac": fc, "hp_frac": fp, "policy": name, "seed": seed,
                         **decision_metrics(act, D["y"], esc)})
    set_horizons(H_CRIT, H_PLAN)
    return rows


def system_metrics():
    import torch

    from prognostics import build_model, count_macs, count_params, measure_latency
    out = {}
    for tag, arch, W, dctx in [(e, e, W_EDGE, 0) for e in EDGE_ARCHS] + [
            (f"{a}+DT", a, W_CLOUD, N_CTX) for a in CLOUD_ARCHS] + [(a, a, W_CLOUD, 0) for a in CLOUD_ARCHS]:
        torch.manual_seed(0)
        ck = torch.load(RESULTS / "checkpoints" / f"fold0_seed0_{tag}.pt", weights_only=False)
        m = build_model(arch, 28, dctx, width=ck["hp"]["width"])
        m.load_state_dict(ck["state"])
        out[tag] = {"params": count_params(m), "macs": count_macs(m, 28, W, dctx), **measure_latency(m, 28, W, dctx)}
    out["assumptions"] = {"rtt_ms": RTT_MS, "feature_bytes": FEAT_BYTES, "sync_bytes": SYNC_BYTES,
                          "raw_snapshot_bytes": RAW_BYTES,
                          "latency_note": "CPU wall-clock, 1 thread, batch 1, Apple M2 Pro (host, not an MCU)"}
    return out


def detection():
    from utils import fold_split
    A = assets()
    rows = []
    for seed in SEEDS:
        E, C = load(EDGE_TAG, seed), load(CLOUD + "+DT", seed)
        y_raw, a = cat(E, "rul_raw"), cat(E, "a")
        iso, thr_iso = [], []
        for f in range(len(FOLDS)):
            tr, _, te = fold_split(f, sorted(A))
            both = isolation_forest_scores([A[n]["Z"] for n in tr], np.concatenate([A[n]["Z"] for n in te + tr]), seed)
            n_te = sum(len(A[n]["rul"]) for n in te)
            iso.append(both[:n_te])
            healthy_tr = np.concatenate([A[n]["rul"] >= R_MAX for n in tr])
            ref = both[n_te:][healthy_tr] if healthy_tr.any() else both[n_te:]
            thr_iso.append(np.full(n_te, np.quantile(ref, TAU_Q)))
        iso, thr_iso = np.concatenate(iso), np.concatenate(thr_iso)
        Ie, Ic = calibrated(E, ALPHA), calibrated(C, ALPHA)
        for name, score, thr in [("Anomaly score $a_t$ (edge)", a - cat(E, "tau_a"), 0.0),
                                 ("Isolation Forest", iso - thr_iso, 0.0),
                                 ("Edge CQR lower bound", R_MAX - Ie[:, 0], R_MAX - H_CRIT),
                                 ("Cloud CQR lower bound (+DT)", R_MAX - Ic[:, 0], R_MAX - H_CRIT)]:
            rows.append({"detector": name, "seed": seed, **detection_metrics(score, y_raw, thr - 1e-9)})
    df = pd.DataFrame(rows)
    return df.groupby("detector", sort=False).agg(["mean", "std"])


def light(seeds):
    """Reduced evaluation for the R_MAX sensitivity runs (selected models only)."""
    rng = np.random.default_rng(0)
    rows, rul = [], []
    for s in seeds:
        D = policy_actions(s, rng=rng)
        for name, ek, ae, ac, ie, ic, _ in POLICIES[:4] + [POLICIES[8]]:
            act, esc, I = resolve(D, name, ek, ae, ac, ie, ic)
            rows.append({"policy": name, "seed": s, **decision_metrics(act, D["y"], esc),
                         "rmse": float(np.sqrt(np.mean((I[:, 1] - D["y"]) ** 2)))})
        for tag in (EDGE_TAG, CLOUD + "+DT"):
            rul.append({"model": tag, "seed": s, **rul_block(load(tag, s), ["cvplus"])})
    out = {"r_max": R_MAX, "h_crit": H_CRIT, "h_plan": H_PLAN,
           "policies": pd.DataFrame(rows).groupby("policy", sort=False).mean(numeric_only=True).drop(columns="seed")
           .reset_index().to_dict("records"),
           "rul": pd.DataFrame(rul).groupby("model").mean(numeric_only=True).drop(columns="seed").reset_index()
           .to_dict("records")}
    save_json(out, RESULTS / "light_summary.json")
    print(json.dumps(out, indent=1, default=float)[:3000])


def main():
    RESULTS.mkdir(exist_ok=True)
    # ---------------- RUL accuracy + calibration
    rul_rows = []
    split_tags = EDGE_ARCHS + [f"{a}{s}" for a in CLOUD_ARCHS for s in ("+DT", "")]
    for tag in split_tags + [f"{a}+DT-point" for a in CLOUD_ARCHS]:
        per = [rul_block(load(tag, s, "split"), ["sym", "asym"], tag.endswith("point")) for s in SEEDS]
        rul_rows.append({"model": tag, "calib": "split", **mean_std(per)})
    for tag in sorted(CV_TAGS):
        per = [rul_block(load(tag, s, "cvplus"), ["cvplus"]) for s in SEEDS]
        rul_rows.append({"model": tag, "calib": "cvplus", **mean_std(per)})
    rul = pd.DataFrame(rul_rows)
    rul.to_csv(RESULTS / "rul_metrics.csv", index=False)
    print(rul[["model", "calib", "rmse_mean", "mae_mean", "raw_picp_mean", "picp_mean", "mpiw_mean", "cce_mean"]]
          .round(3).to_string(index=False))

    sysm = system_metrics()
    save_json(sysm, RESULTS / "system_metrics.json")
    lat = {"edge_ms": sysm[EDGE_TAG]["median_ms"], "cloud_ms": sysm[CLOUD + "+DT"]["median_ms"],
           "edge_macs": sysm[EDGE_TAG]["macs"], "cloud_macs": sysm[CLOUD + "+DT"]["macs"]}

    # ---------------- policies, timing, trade-offs, horizon sensitivity
    rng = np.random.default_rng(0)
    pol, curves, timing, hs, arrays = [], [], [], [], {}
    for s in SEEDS:
        r, D, acts = run_policies(s, lat, rng)
        pol += r
        arrays[s] = (D, acts)
        curves += tradeoff(s, D, rng)
        for name in ("Always-edge", "Always-cloud", "Cloud point+threshold", "HERA-full", "Width gate (matched)"):
            timing += [x | {"policy": name, "seed": s} for x in alarm_timing(acts[name][0], D["yr"], D["b"], D["t"])]
        hs += horizon_sensitivity(s, rng)
    pol = pd.DataFrame(pol)
    num = pol.drop(columns=["seed"]).groupby("policy", sort=False)
    num.mean().add_suffix("_mean").join(num.std(ddof=1).add_suffix("_std")).reset_index() \
        .to_csv(RESULTS / "policy_metrics.csv", index=False)
    pol.to_csv(RESULTS / "policy_metrics_per_seed.csv", index=False)
    pd.DataFrame(curves).to_csv(RESULTS / "tradeoff_curves.csv", index=False)
    tim = pd.DataFrame(timing)
    tim.to_csv(RESULTS / "alarm_timing_per_bearing.csv", index=False)
    summ = tim.groupby(["policy", "seed"]).apply(lambda g: pd.Series({
        "timely": (g.outcome == "timely").sum(), "late": (g.outcome == "late").sum(),
        "premature": (g.outcome == "premature").sum(),
        "median_lead_timely_s": g.loc[g.outcome == "timely", "rul_at_alarm_s"].median()}), include_groups=False)
    summ.groupby("policy", sort=False).agg(["mean", "std"]).pipe(
        lambda d: d.set_axis([f"{a}_{b}" for a, b in d.columns], axis=1)).reset_index() \
        .to_csv(RESULTS / "alarm_timing.csv", index=False)
    hsd = pd.DataFrame(hs).groupby(["hc_frac", "hp_frac", "policy"], sort=False).mean(numeric_only=True) \
        .drop(columns="seed").reset_index()
    hsd.to_csv(RESULTS / "horizon_sensitivity.csv", index=False)
    print(num.mean()[["escalation_rate", "unsafe_rate", "false_maint", "rmse", "bytes_per_snapshot"]].round(3))
    print(pd.read_csv(RESULTS / "alarm_timing.csv").round(2).to_string(index=False))

    det = detection()
    det.columns = [f"{a}_{b}" for a, b in det.columns]
    det.reset_index().to_csv(RESULTS / "detection_metrics.csv", index=False)

    # ---------------- per-bearing statistics (unit = bearing, seeds averaged)
    pb = []
    for s, (D, acts) in arrays.items():
        crit, healthy = D["y"] <= H_CRIT, D["y"] > H_PLAN
        nomaint = lambda act: (act == CONTINUE) | (act == INSPECT)  # noqa: E731
        for bn in np.unique(D["b"]):
            m = D["b"] == bn
            row = {"bearing": bn, "seed": s}
            for key, name in [("hera", "HERA-full"), ("cloud", "Always-cloud"), ("edge", "Always-edge"),
                              ("point", "Cloud point+threshold"), ("width", "Width gate (matched)")]:
                act, esc, I = acts[name]
                row[f"unsafe_{key}"] = nomaint(act)[m & crit].mean() if (m & crit).any() else np.nan
                row[f"fm_{key}"] = np.isin(act, [SCHEDULE, URGENT])[m & healthy].mean() if (m & healthy).any() else np.nan
                row[f"cov_{key}"] = ((D["y"] >= I[:, 0]) & (D["y"] <= I[:, 2]))[m].mean()
            row["escalation"] = acts["HERA-full"][1][m].mean()
            pb.append(row)
    pb = pd.DataFrame(pb).groupby("bearing").mean(numeric_only=True).drop(columns="seed")
    pb.to_csv(RESULTS / "per_bearing.csv")
    tests = {}
    for x, yv in [("unsafe_hera", "unsafe_point"), ("unsafe_hera", "unsafe_cloud"), ("unsafe_hera", "unsafe_width"),
                  ("unsafe_hera", "unsafe_edge"), ("fm_hera", "fm_point"), ("fm_hera", "fm_cloud"),
                  ("fm_hera", "fm_edge")]:
        d = (pb[x] - pb[yv]).dropna()
        nz = d[d.abs() > 1e-12]
        tests[f"{x}_vs_{yv}"] = {"mean_diff": float(d.mean()), "median_diff": float(d.median()),
                                 "n_bearings": int(len(d)), "n_nonzero": int(len(nz)),
                                 "p_wilcoxon": float(wilcoxon(nz).pvalue) if len(nz) >= 5 else None}
    save_json(tests, RESULTS / "stat_tests.json")
    print(json.dumps(tests, indent=1))

    D, acts = arrays[0]
    act, esc, _ = acts["HERA-full"]
    pd.DataFrame({"bearing": D["b"], "t": D["t"], "y": D["y"], "a": D["a"], "esc": esc, "act": act,
                  "e_lo": D["Ie"][:, 0], "e_med": D["Ie"][:, 1], "e_hi": D["Ie"][:, 2],
                  "c_lo": D["Ic"][:, 0], "c_med": D["Ic"][:, 1], "c_hi": D["Ic"][:, 2]}) \
        .to_csv(RESULTS / "trajectories.csv.gz", index=False)
    A = assets()
    save_json(SEL | {"alpha": ALPHA, "tau_quantile": TAU_Q, "tau_a_per_fold": [tau_anomaly(f) for f in range(len(FOLDS))],
                     "dataset": DATASET, "n_bearings": len(A), "n_snapshots": int(sum(len(x["rul"]) for x in A.values())),
                     "r_max": R_MAX, "h_crit": H_CRIT, "h_plan": H_PLAN, "n_folds": len(FOLDS)},
              RESULTS / "run_info.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--light", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="*", default=list(SEEDS))
    a = ap.parse_args()
    light(a.seeds) if a.light else main()
