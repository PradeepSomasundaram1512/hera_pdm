"""Evaluation of HERA-PdM from saved predictions (results/preds).

Produces every number used in the paper:
  results/rul_metrics.csv          accuracy + calibration per model (mean/std over seeds)
  results/model_selection.csv      calibration-set pinball loss used for architecture choice
  results/detection_metrics.csv    imminent-failure detection (P/R/F1/AUROC)
  results/policy_metrics.csv       hierarchical policies & ablations (mean/std over seeds)
  results/tradeoff_curves.csv      escalation-budget sweeps for every gating family
  results/per_bearing.csv          per-bearing coverage / unsafe rates (HERA-full)
  results/system_metrics.json      params, MACs, measured CPU latency, comm. model
  results/stat_tests.json          paired Wilcoxon tests across bearings
  results/trajectories.csv         trajectories for Fig. 3
Protocol: metrics are computed on the pooled held-out predictions of all four
bearing-grouped folds (every bearing is tested exactly once per seed) and then
summarized as mean +- std over the three training seeds.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).parent))
from digital_twin import CONTEXT_NAMES  # noqa: E402
from edge_agent import detection_metrics, isolation_forest_scores  # noqa: E402
from escalation import (CONTINUE, decide, decision_metrics, gate_anomaly, gate_hera, gate_point_margin,  # noqa: E402
                        gate_random, gate_score, gate_width, hierarchical_decision, region)
from uncertainty import calibrate, coverage_calibration_error, interval_metrics  # noqa: E402
from utils import ALPHA, FOLDS, H_CRIT, PROC, QUANTILES, R_MAX, RESULTS, SEEDS, W_CLOUD, W_EDGE, save_json  # noqa: E402

PRED = RESULTS / "preds"
N_CTX = len(CONTEXT_NAMES)
CLOUD_ARCHS = ["LSTM", "GRU", "TCN", "CNN-LSTM"]
EDGE_ARCHS = ["Edge-CNN", "Edge-GRU"]
TAU_Q = 0.99               # alarm threshold = 99th pct of a_t on healthy (y >= R_MAX) training snapshots
RTT_MS = 50.0              # assumed edge<->fog/cloud round-trip time (not measured)
FEAT_BYTES = 28 * 4        # one feature vector, float32
SYNC_BYTES = 3 * 4         # DT sync message: (t, HI, a_t)
RAW_BYTES = 2560 * 2 * 2   # one raw 2-channel snapshot at 16 bit


_ASSETS = {}


def assets():
    if not _ASSETS:
        from digital_twin import build_asset, feature_columns
        df = pd.read_csv(PROC / "femto_features.csv.gz")
        feats = feature_columns(df)
        _ASSETS.update({b: build_asset(g.reset_index(drop=True), feats) for b, g in df.groupby("bearing")})
    return _ASSETS


def tau_anomaly(fold):
    from utils import fold_split
    A = assets()
    train, _, _ = fold_split(fold, sorted(A))
    a = np.concatenate([A[n]["a"][A[n]["rul"] >= R_MAX] for n in train])
    return float(np.quantile(a, TAU_Q))


def load(tag, seed):
    """Pool calibration-corrected predictions across folds for one model/seed.
    Conformal calibration is done per fold on that fold's calibration bearings."""
    rows = []
    for f in range(len(FOLDS)):
        p = np.load(PRED / f"fold{f}_seed{seed}_{tag}.npz")
        m = np.load(PRED / f"fold{f}_meta.npz", allow_pickle=True)
        rows.append({"fold": f, "q_cal": p["q_cal"], "q_test": p["q_test"],
                     "y_cal": np.minimum(m["cal_rul"], R_MAX), "y": np.minimum(m["test_rul"], R_MAX),
                     "a": m["test_a"], "a_cal": m["cal_a"], "bearing": m["test_bearing"], "t": m["test_t"],
                     "rul_raw": m["test_rul"], "tau_a": np.full(len(m["test_rul"]), tau_anomaly(f)),
                     "tau_a_cal": np.full(len(m["cal_rul"]), tau_anomaly(f))})
    return rows


CAL_MODE = "sym"    # primary: standard (symmetric) CQR, fixed before experiments; "asym" reported as variant


def calibrated(rows, alpha, mode=None):
    """Per-fold conformal calibration on that fold's calibration bearings."""
    return np.concatenate([calibrate(r["q_cal"], r["y_cal"], r["q_test"], alpha, mode or CAL_MODE) for r in rows])


def calibrated_cal(rows, alpha, mode=None):
    """Calibrated intervals on the calibration bearings themselves (used only to
    tune baseline thresholds to a matched escalation budget without test data)."""
    return np.concatenate([calibrate(r["q_cal"], r["y_cal"], r["q_cal"], alpha, mode or CAL_MODE) for r in rows])


def cat(rows, k):
    return np.concatenate([r[k] for r in rows])


def pinball_np(q, y):
    taus = np.array(QUANTILES)
    d = y[:, None] - q
    return float(np.maximum(taus * d, (taus - 1) * d).mean())


def rul_block(rows, alpha=ALPHA, point_only=False):
    y = cat(rows, "y")
    q = cat(rows, "q_test")
    med = q[:, 1]
    err = med - y
    deg = y < R_MAX
    out = {"rmse": float(np.sqrt(np.mean(err ** 2))), "mae": float(np.mean(np.abs(err))),
           "rmse_deg": float(np.sqrt(np.mean(err[deg] ** 2))),
           "overest_crit": float((med[y <= H_CRIT] > H_CRIT).mean())}
    if not point_only:
        out["raw_picp"] = float(((y >= q[:, 0]) & (y <= q[:, 2])).mean())
        for mode in ("sym", "asym"):
            I = calibrated(rows, alpha, mode)
            m = interval_metrics(I, y, alpha)
            sfx = "" if mode == CAL_MODE else "_" + mode
            out |= {k + sfx: v for k, v in m.items()}
            out["lower_miss" + sfx] = float((y < I[:, 0]).mean())
            out["lower_miss_crit" + sfx] = float((y < I[:, 0])[y <= H_CRIT].mean())
            curve = []
            for a in np.linspace(0.05, 0.5, 10):
                Ia = calibrated(rows, a, mode)
                curve.append((1 - a, float(((y >= Ia[:, 0]) & (y <= Ia[:, 2])).mean())))
            out["cce" + sfx] = coverage_calibration_error(curve)
            cov_b = pd.Series((y >= I[:, 0]) & (y <= I[:, 2])).groupby(cat(rows, "bearing")).mean()
            out["picp_bearing_min" + sfx] = float(cov_b.min())
            out["picp_bearing_median" + sfx] = float(cov_b.median())
    return out


def mean_std(dicts):
    df = pd.DataFrame(dicts)
    return {f"{k}_mean": df[k].mean() for k in df} | {f"{k}_std": df[k].std(ddof=1) for k in df}


# ----------------------------------------------------------------------------- policies
def comm_bytes(esc, bearing, t, mode):
    """Analytic communication payload per snapshot (bytes); header overhead excluded."""
    if mode == "cloud_features":
        return np.full(len(esc), FEAT_BYTES, float)
    if mode == "cloud_raw":
        return np.full(len(esc), RAW_BYTES, float)
    if mode == "edge":
        return np.zeros(len(esc))
    # hierarchical: DT sync every snapshot + the not-yet-transmitted part of the W_CLOUD window on escalation
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
    edge_ms, cloud_ms = lat["edge_ms"], lat["cloud_ms"]
    if lat["comm_mode"] == "edge":
        e2e = np.full(len(esc), edge_ms)
    elif lat["comm_mode"].startswith("cloud"):
        e2e = np.full(len(esc), RTT_MS + cloud_ms)
    else:
        e2e = edge_ms + esc * (RTT_MS + cloud_ms)
    macs = lat["edge_macs"] * (lat["comm_mode"] != "cloud_features") + lat["cloud_macs"] * (
        esc if lat["comm_mode"] == "hier" else float(lat["comm_mode"].startswith("cloud")))
    err = I_final[:, 1] - y
    out = {"policy": name, **d, "rmse": float(np.sqrt(np.mean(err ** 2))), "mae": float(np.abs(err).mean()),
           "bytes_per_snapshot": float(comm.mean()), "latency_mean_ms": float(np.mean(e2e)),
           "latency_p95_ms": float(np.percentile(e2e, 95)), "kmacs_per_snapshot": float(np.mean(macs) / 1e3)}
    if I_final.shape[1] == 3 and not np.allclose(I_final[:, 0], I_final[:, 2]):
        out |= {"picp": float(((y >= I_final[:, 0]) & (y <= I_final[:, 2])).mean()),
                "mpiw": float((I_final[:, 2] - I_final[:, 0]).mean())}
    return out


def match_threshold(scores_cal, target_rate):
    """Threshold on a calibration-set score giving the target escalation rate."""
    return float(np.quantile(scores_cal, 1 - target_rate))


def run_policies(seed, edge_tag, cloud_tag, lat, alpha_edge=ALPHA, rng=None):
    E, C = load(edge_tag, seed), load(f"{cloud_tag}+DT", seed)
    Cn, Cp = load(cloud_tag, seed), load(f"{cloud_tag}+DT-point", seed)
    y, a, b, t = cat(E, "y"), cat(E, "a"), cat(E, "bearing"), cat(E, "t")
    TAU_A, tau_cal = cat(E, "tau_a"), cat(E, "tau_a_cal")
    Ie, Ic, Icn = calibrated(E, alpha_edge), calibrated(C, ALPHA), calibrated(Cn, ALPHA)
    pe, pc = cat(E, "q_test")[:, 1], cat(Cp, "q_test")[:, 1]
    Ie_cal, a_cal = calibrated_cal(E, alpha_edge), cat(E, "a_cal")
    act_e, act_c, act_cn = decide(Ie, a, TAU_A), decide(Ic, a, TAU_A), decide(Icn, a, TAU_A)
    point = lambda m: np.stack([m, m, m], 1)  # noqa: E731
    act_pe = decide(point(pe), a, TAU_A)
    act_pc = decide(point(pc), a, TAU_A)
    hier = dict(lat, comm_mode="hier")
    res = []
    n = len(y)
    esc_h = gate_hera(Ie, a, TAU_A, True)
    target = float(gate_hera(Ie_cal, a_cal, tau_cal, True).mean())   # HERA budget measured on calibration bearings
    res.append(policy_eval("Always-edge", act_e, np.zeros(n, bool), Ie, y, b, t, dict(lat, comm_mode="edge")))
    res.append(policy_eval("Always-cloud", act_c, np.ones(n, bool), Ic, y, b, t, dict(lat, comm_mode="cloud_features")))
    res.append(policy_eval("Cloud point+threshold", act_pc, np.ones(n, bool), point(pc), y, b, t,
                           dict(lat, comm_mode="cloud_features")))
    res.append(policy_eval("HERA-full", hierarchical_decision(esc_h, act_e, act_c), esc_h,
                           np.where(esc_h[:, None], Ic, Ie), y, b, t, hier))
    esc_na = gate_hera(Ie, a, TAU_A, False)
    res.append(policy_eval("HERA w/o anomaly term", hierarchical_decision(esc_na, act_e, act_c), esc_na,
                           np.where(esc_na[:, None], Ic, Ie), y, b, t, hier))
    res.append(policy_eval("HERA-no-DT", hierarchical_decision(esc_h, act_e, act_cn), esc_h,
                           np.where(esc_h[:, None], Icn, Ie), y, b, t, hier))
    # no-UQ: point estimates everywhere; margin = edge calibration RMSE (fixed a priori)
    delta = float(np.sqrt(np.mean((cat(E, "q_cal")[:, 1] - cat(E, "y_cal")) ** 2)))
    esc_p = gate_point_margin(pe, delta)
    res.append(policy_eval("HERA-no-UQ", hierarchical_decision(esc_p, act_pe, act_pc), esc_p,
                           np.where(esc_p[:, None], point(pc), point(pe)), y, b, t, hier))
    esc_f = gate_anomaly(a, TAU_A)
    res.append(policy_eval("HERA-fixed (anomaly trigger)", hierarchical_decision(esc_f, act_e, act_c), esc_f,
                           np.where(esc_f[:, None], Ic, Ie), y, b, t, hier))
    # budget-matched alternatives (thresholds set on calibration bearings)
    w_cal = (Ie_cal[:, 2] - Ie_cal[:, 0]) / R_MAX
    esc_w = gate_width(Ie, match_threshold(w_cal, target))
    res.append(policy_eval("Width gate (matched)", hierarchical_decision(esc_w, act_e, act_c), esc_w,
                           np.where(esc_w[:, None], Ic, Ie), y, b, t, hier))
    s_cal = (np.clip(a_cal / (2 * tau_cal), 0, 1) + w_cal + 1 - Ie_cal[:, 1] / R_MAX) / 3
    esc_s = gate_score(Ie, a, match_threshold(s_cal, target), a_scale=2 * TAU_A)
    res.append(policy_eval("Linear score gate (matched)", hierarchical_decision(esc_s, act_e, act_c), esc_s,
                           np.where(esc_s[:, None], Ic, Ie), y, b, t, hier))
    esc_r = gate_random(n, target, rng)
    res.append(policy_eval("Random gate (matched)", hierarchical_decision(esc_r, act_e, act_c), esc_r,
                           np.where(esc_r[:, None], Ic, Ie), y, b, t, hier))
    for r in res:
        r["seed"] = seed
    arrays = {"y": y, "b": b, "t": t, "a": a, "Ie": Ie, "Ic": Ic, "esc": esc_h,
              "act": hierarchical_decision(esc_h, act_e, act_c), "act_cloud": act_c, "act_pc": act_pc,
              "esc_w": esc_w, "act_w": hierarchical_decision(esc_w, act_e, act_c)}
    return res, arrays


def tradeoff(seed, edge_tag, cloud_tag, rng):
    E, C, Cp = load(edge_tag, seed), load(f"{cloud_tag}+DT", seed), load(f"{cloud_tag}+DT-point", seed)
    y, a = cat(E, "y"), cat(E, "a")
    TAU_A = cat(E, "tau_a")
    Ic = calibrated(C, ALPHA)
    act_c = decide(Ic, a, TAU_A)
    pe, pc = cat(E, "q_test")[:, 1], cat(Cp, "q_test")[:, 1]
    rows = []

    def add(fam, par, esc, act_edge, act_cloud, I_edge, I_cloud):
        act = hierarchical_decision(esc, act_edge, act_cloud)
        med = np.where(esc, I_cloud[:, 1], I_edge[:, 1])
        d = decision_metrics(act, y, esc)
        rows.append({"family": fam, "param": par, "seed": seed, **d,
                     "rmse": float(np.sqrt(np.mean((med - y) ** 2)))})

    for ae in [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]:
        Ie = calibrated(E, ae)
        act_e = decide(Ie, a, TAU_A)
        add("HERA (decision-sufficiency)", ae, gate_hera(Ie, a, TAU_A, True), act_e, act_c, Ie, Ic)
    Ie = calibrated(E, ALPHA)
    act_e = decide(Ie, a, TAU_A)
    w = (Ie[:, 2] - Ie[:, 0]) / R_MAX
    s = (np.clip(a / (2 * TAU_A), 0, 1) + w + 1 - Ie[:, 1] / R_MAX) / 3
    for q in np.linspace(0.02, 0.98, 25):
        add("Width gate", q, gate_width(Ie, np.quantile(w, 1 - q)), act_e, act_c, Ie, Ic)
        add("Linear score gate", q, s > np.quantile(s, 1 - q), act_e, act_c, Ie, Ic)
        add("Anomaly trigger", q, a > np.quantile(a, 1 - q), act_e, act_c, Ie, Ic)
        add("Random gate", q, gate_random(len(y), q, rng), act_e, act_c, Ie, Ic)
    point = lambda m: np.stack([m, m, m], 1)  # noqa: E731
    for dl in [0, 100, 200, 300, 400, 600, 800, 1000, 1500]:
        add("Point margin (no UQ)", dl, gate_point_margin(pe, dl), decide(point(pe), a, TAU_A),
            decide(point(pc), a, TAU_A), point(pe), point(pc))
    # ---- decision Pareto: shared miscoverage alpha for both tiers vs. point + safety margin
    none, allc = np.zeros(len(y), bool), np.ones(len(y), bool)
    for al in [0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        Ie_a, Ic_a = calibrated(E, al), calibrated(C, al)
        ae, ac = decide(Ie_a, a, TAU_A), decide(Ic_a, a, TAU_A)
        add("P: Always-edge (CQR)", al, none, ae, ae, Ie_a, Ie_a)
        add("P: Always-cloud (CQR)", al, allc, ac, ac, Ic_a, Ic_a)
        add("P: HERA", al, gate_hera(Ie_a, a, TAU_A, True), ae, ac, Ie_a, Ic_a)
    for dl in [0, 150, 300, 450, 600, 900, 1200, 1500, 1800, 2100, 2400]:
        ape = decide(point(np.clip(pe - dl, 0, None)), a, TAU_A)
        apc = decide(point(np.clip(pc - dl, 0, None)), a, TAU_A)
        add("P: Edge point - margin", dl, none, ape, ape, point(pe), point(pe))
        add("P: Cloud point - margin", dl, allc, apc, apc, point(pc), point(pc))
    return rows


def system_metrics(edge_tag, cloud_tag):
    import torch

    from prognostics import build_model, count_macs, count_params, measure_latency
    out = {}
    for tag, arch, W, dctx in [(e, e, W_EDGE, 0) for e in EDGE_ARCHS] + [
            (f"{a}+DT", a, W_CLOUD, N_CTX) for a in CLOUD_ARCHS] + [(a, a, W_CLOUD, 0) for a in CLOUD_ARCHS]:
        torch.manual_seed(0)
        ck = torch.load(RESULTS / "checkpoints" / f"fold0_seed0_{tag}.pt", weights_only=False)
        m = build_model(arch, 28, dctx, width=ck["hp"]["width"])
        m.load_state_dict(ck["state"])
        out[tag] = {"params": count_params(m), "macs": count_macs(m, 28, W, dctx),
                    **measure_latency(m, 28, W, dctx)}
    out["assumptions"] = {"rtt_ms": RTT_MS, "feature_bytes": FEAT_BYTES, "sync_bytes": SYNC_BYTES,
                          "raw_snapshot_bytes": RAW_BYTES,
                          "latency_note": "CPU wall-clock, 1 thread, batch 1, Apple M2 Pro (host, not an MCU)"}
    return out


def detection(edge_tag, cloud_tag):
    """Imminent failure detection (RUL <= H_CRIT) for the anomaly score, an
    Isolation Forest baseline, and the calibrated edge / cloud RUL bounds.
    All alarm thresholds are set on training bearings only."""
    from utils import fold_split
    A = assets()
    rows = []
    for seed in SEEDS:
        E, C = load(edge_tag, seed), load(f"{cloud_tag}+DT", seed)
        y_raw, a = cat(E, "rul_raw"), cat(E, "a")
        iso, thr_iso = [], []
        for f in range(len(FOLDS)):
            tr, _, te = fold_split(f, sorted(A))
            both = isolation_forest_scores([A[n]["Z"] for n in tr],
                                           np.concatenate([A[n]["Z"] for n in te + tr]), seed)
            n_te = sum(len(A[n]["rul"]) for n in te)
            iso.append(both[:n_te])
            healthy_tr = np.concatenate([A[n]["rul"] >= R_MAX for n in tr])
            thr_iso.append(np.full(n_te, np.quantile(both[n_te:][healthy_tr], TAU_Q)))
        iso, thr_iso = np.concatenate(iso), np.concatenate(thr_iso)
        Ie, Ic = calibrated(E, ALPHA), calibrated(C, ALPHA)
        for name, score, thr in [("Anomaly score $a_t$ (edge)", a - cat(E, "tau_a"), 0.0),
                                 ("Isolation Forest", iso - thr_iso, 0.0),
                                 ("Edge CQR lower bound", R_MAX - Ie[:, 0], R_MAX - H_CRIT),
                                 ("Cloud CQR lower bound (+DT)", R_MAX - Ic[:, 0], R_MAX - H_CRIT)]:
            rows.append({"detector": name, "seed": seed, **detection_metrics(score, y_raw, thr - 1e-9)})
    df = pd.DataFrame(rows)
    return df.groupby("detector", sort=False).agg(["mean", "std"])


def main():
    RESULTS.mkdir(exist_ok=True)
    # ---------------- RUL accuracy + calibration for every trained configuration
    tags = EDGE_ARCHS + [f"{a}{s}" for a in CLOUD_ARCHS for s in ("+DT", "")]
    rul_rows, sel_rows = [], []
    for tag in tags + [f"{a}+DT-point" for a in CLOUD_ARCHS]:
        per_seed = []
        for s in SEEDS:
            rows = load(tag, s)
            per_seed.append(rul_block(rows, point_only=tag.endswith("point")))
            if not tag.endswith("point"):
                sel_rows.append({"model": tag, "seed": s,
                                 "cal_pinball": pinball_np(cat(rows, "q_cal") / R_MAX, cat(rows, "y_cal") / R_MAX)})
        rul_rows.append({"model": tag, **mean_std(per_seed)})
    rul = pd.DataFrame(rul_rows)
    rul.to_csv(RESULTS / "rul_metrics.csv", index=False)
    sel = pd.DataFrame(sel_rows).groupby("model").cal_pinball.mean().reset_index()
    sel.to_csv(RESULTS / "model_selection.csv", index=False)
    # architecture selection uses calibration bearings only (never test bearings)
    edge_tag = sel[sel.model.isin(EDGE_ARCHS)].sort_values("cal_pinball").model.iloc[0]
    cloud_tag = sel[sel.model.isin([f"{a}+DT" for a in CLOUD_ARCHS])].sort_values(
        "cal_pinball").model.iloc[0].replace("+DT", "")
    print("selected edge:", edge_tag, "cloud:", cloud_tag)
    print(rul[["model", "rmse_mean", "rmse_std", "mae_mean", "picp_mean", "mpiw_mean", "raw_picp_mean", "cce_mean"]]
          .round(3).to_string(index=False))

    # ---------------- system metrics
    sysm = system_metrics(edge_tag, cloud_tag)
    save_json(sysm, RESULTS / "system_metrics.json")
    lat = {"edge_ms": sysm[edge_tag]["median_ms"], "cloud_ms": sysm[f"{cloud_tag}+DT"]["median_ms"],
           "edge_macs": sysm[edge_tag]["macs"], "cloud_macs": sysm[f"{cloud_tag}+DT"]["macs"]}

    # ---------------- policies + trade-off curves
    rng = np.random.default_rng(0)
    pol, curves, arrays = [], [], {}
    for s in SEEDS:
        r, arr = run_policies(s, edge_tag, cloud_tag, lat, rng=rng)
        pol += r
        arrays[s] = arr
        curves += tradeoff(s, edge_tag, cloud_tag, rng)
    pol = pd.DataFrame(pol)
    num = pol.drop(columns=["seed"]).groupby("policy", sort=False)
    agg = num.mean().add_suffix("_mean").join(num.std(ddof=1).add_suffix("_std")).reset_index()
    agg.to_csv(RESULTS / "policy_metrics.csv", index=False)
    pol.to_csv(RESULTS / "policy_metrics_per_seed.csv", index=False)
    pd.DataFrame(curves).to_csv(RESULTS / "tradeoff_curves.csv", index=False)
    print(agg[["policy", "escalation_rate_mean", "unsafe_rate_mean", "missed_urgent_mean", "false_urgent_mean",
               "false_maint_mean", "rmse_mean", "bytes_per_snapshot_mean", "latency_mean_ms_mean"]]
          .round(4).to_string(index=False))

    # ---------------- detection
    det = detection(edge_tag, cloud_tag)
    det.columns = [f"{a}_{b}" for a, b in det.columns]
    det.reset_index().to_csv(RESULTS / "detection_metrics.csv", index=False)
    print(det.round(3).to_string())

    # ---------------- per-bearing analysis and paired tests (unit = bearing, seeds averaged)
    pb = []
    for s, A in arrays.items():
        crit = A["y"] <= H_CRIT
        healthy_like = A["y"] > 1800
        cov = (A["y"] >= np.where(A["esc"], A["Ic"][:, 0], A["Ie"][:, 0])) & (
            A["y"] <= np.where(A["esc"], A["Ic"][:, 2], A["Ie"][:, 2]))
        nomaint = lambda act: (act == CONTINUE) | (act == 1)  # noqa: E731
        for bname in np.unique(A["b"]):
            m = A["b"] == bname
            mc = m & crit
            pb.append({"bearing": bname, "seed": s, "coverage": cov[m].mean(), "escalation": A["esc"][m].mean(),
                       "unsafe_hera": nomaint(A["act"])[mc].mean(), "unsafe_cloud": nomaint(A["act_cloud"])[mc].mean(),
                       "unsafe_point": nomaint(A["act_pc"])[mc].mean(), "unsafe_width": nomaint(A["act_w"])[mc].mean(),
                       "false_maint_hera": np.isin(A["act"], [2, 3])[m & healthy_like].mean(),
                       "false_maint_point": np.isin(A["act_pc"], [2, 3])[m & healthy_like].mean()})
    pb = pd.DataFrame(pb).groupby("bearing").mean(numeric_only=True).drop(columns="seed")
    pb.to_csv(RESULTS / "per_bearing.csv")
    tests = {}
    for x, yv in [("unsafe_hera", "unsafe_point"), ("unsafe_hera", "unsafe_cloud"), ("unsafe_hera", "unsafe_width"),
                  ("false_maint_hera", "false_maint_point")]:
        d = pb[x] - pb[yv]
        nz = d[d.abs() > 1e-12]
        tests[f"{x}_vs_{yv}"] = {"median_diff": float(d.median()), "mean_diff": float(d.mean()),
                                 "n_bearings": int(len(d)), "n_nonzero": int(len(nz)),
                                 "p_wilcoxon": float(wilcoxon(nz).pvalue) if len(nz) >= 5 else None}
    save_json(tests, RESULTS / "stat_tests.json")
    print(json.dumps(tests, indent=1))

    # ---------------- trajectories for Fig. 3 (seed 0)
    A = arrays[0]
    traj = pd.DataFrame({"bearing": A["b"], "t": A["t"], "y": A["y"], "a": A["a"], "esc": A["esc"],
                         "act": A["act"], "e_lo": A["Ie"][:, 0], "e_med": A["Ie"][:, 1], "e_hi": A["Ie"][:, 2],
                         "c_lo": A["Ic"][:, 0], "c_med": A["Ic"][:, 1], "c_hi": A["Ic"][:, 2]})
    traj.to_csv(RESULTS / "trajectories.csv.gz", index=False)
    save_json({"edge_model": edge_tag, "cloud_model": cloud_tag, "alpha": ALPHA, "tau_quantile": TAU_Q,
               "tau_a_per_fold": [tau_anomaly(f) for f in range(len(FOLDS))]},
              RESULTS / "selected_models.json")


if __name__ == "__main__":
    main()
