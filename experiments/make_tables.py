"""Generate LaTeX tables and numeric macros for the paper from results/.

Outputs
  paper/tables/table2_rul.tex       RUL accuracy + calibration (mean ± std over 3 seeds)
  paper/tables/table3_policies.tex  hierarchical policies, ablations, system cost
  paper/numbers.tex                 \\newcommand macros for every number quoted in the text
No number in these files is typed by hand.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES, OUT = ROOT / "results", ROOT / "paper"
(OUT / "tables").mkdir(parents=True, exist_ok=True)
MACROS = {}


def pm(mean, std, fmt="{:.0f}"):
    if std is None or (isinstance(std, float) and np.isnan(std)):
        return fmt.format(mean)
    return f"{fmt.format(mean)}$\\pm${fmt.format(std)}"


def macro(name, value):
    MACROS[name] = value


def table2():
    r = pd.read_csv(RES / "rul_metrics.csv").set_index("model")
    sysm = json.loads((RES / "system_metrics.json").read_text())
    sel = json.loads((RES / "selected_models.json").read_text())
    order = ["Edge-CNN", "Edge-GRU"] + [f"{a}{s}" for a in ["LSTM", "GRU", "TCN", "CNN-LSTM"] for s in ("", "+DT")]
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{RUL accuracy and calibration on 17 held-out FEMTO bearings (4 bearing-grouped folds; "
             r"mean$\pm$std over 3 seeds; RUL capped at 3000\,s; nominal coverage 90\%). "
             r"Raw: uncalibrated quantiles; CQR: conformalized. $^\dagger$selected on calibration bearings.}",
             r"\label{tab:rul}", r"\setlength{\tabcolsep}{2.6pt}\footnotesize", r"\resizebox{\columnwidth}{!}{%",
             r"\begin{tabular}{lrcccccc}", r"\toprule",
             r"Model & Params & RMSE (s) & MAE (s) & Raw cov. & CQR cov. & MPIW (s) & CCE \\", r"\midrule"]
    for m in order:
        if m not in r.index:
            continue
        x = r.loc[m]
        tag = m + (r"$^\dagger$" if m in (sel["edge_model"], sel["cloud_model"] + "+DT") else "")
        params = sysm.get(m, {}).get("params", float("nan"))
        lines.append(f"{tag} & {params/1e3:.1f}k & {pm(x.rmse_mean, x.rmse_std)} & {pm(x.mae_mean, x.mae_std)} & "
                     f"{x.raw_picp_mean:.2f} & {pm(x.picp_mean, x.picp_std, '{:.2f}')} & "
                     f"{pm(x.mpiw_mean, x.mpiw_std)} & {x.cce_mean:.3f} \\\\")
        if m == "Edge-GRU":
            lines.append(r"\midrule")
    lines.append(r"\midrule")
    for m in [f"{a}+DT-point" for a in ["LSTM", "GRU", "TCN", "CNN-LSTM"]]:
        if m in r.index:
            x = r.loc[m]
            lines.append(f"{m.replace('-point', ' (MSE)')} & -- & {pm(x.rmse_mean, x.rmse_std)} & "
                         f"{pm(x.mae_mean, x.mae_std)} & -- & -- & -- & -- \\\\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    (OUT / "tables" / "table2_rul.tex").write_text("\n".join(lines) + "\n")
    # macros
    e, c = sel["edge_model"], sel["cloud_model"]
    for key, m in [("Edge", e), ("Cloud", c + "+DT"), ("CloudNoDT", c)]:
        x = r.loc[m]
        macro(f"rmse{key}", f"{x.rmse_mean:.0f}")
        macro(f"mae{key}", f"{x.mae_mean:.0f}")
        macro(f"picp{key}", f"{x.picp_mean:.2f}")
        macro(f"rawpicp{key}", f"{x.raw_picp_mean:.2f}")
        macro(f"mpiw{key}", f"{x.mpiw_mean:.0f}")
        macro(f"picpAsym{key}", f"{x.picp_asym_mean:.2f}")
        macro(f"mpiwAsym{key}", f"{x.mpiw_asym_mean:.0f}")
        macro(f"cce{key}", f"{x.cce_mean:.3f}")
        macro(f"covMin{key}", f"{x.picp_bearing_min_mean:.2f}")
        macro(f"lowMissCrit{key}", f"{100 * x.lower_miss_crit_mean:.0f}")
    macro("edgeModel", e)
    macro("cloudModel", c)
    rr = r.loc[[m for m in order if m in r.index and not m.startswith("Edge")]]
    best = rr.rmse_mean.idxmin()
    macro("bestRmseModel", best)
    macro("bestRmse", f"{rr.rmse_mean.min():.0f}")
    raw = r.loc[[m for m in order if m in r.index], "raw_picp_mean"]
    macro("rawCovMin", f"{raw.min():.2f}")
    macro("rawCovMax", f"{raw.max():.2f}")
    cq = r.loc[[m for m in order if m in r.index], "picp_mean"]
    macro("cqrCovMin", f"{cq.min():.2f}")
    macro("cqrCovMax", f"{cq.max():.2f}")


POLICY_ORDER = [
    ("Always-edge", "Always-edge"), ("Always-cloud", "Always-cloud"),
    ("Cloud point+threshold", "Cloud point + fixed thr."),
    ("HERA-full", r"\textbf{HERA-full}"),
    ("HERA w/o anomaly term", "HERA w/o anomaly term"), ("HERA-no-DT", "HERA-no-DT"),
    ("HERA-no-UQ", "HERA-no-UQ (point margin)"), ("HERA-fixed (anomaly trigger)", "HERA-fixed (anomaly trig.)"),
    ("Width gate (matched)", "Width gate$^\\ddagger$"), ("Linear score gate (matched)", "Linear score gate$^\\ddagger$"),
    ("Random gate (matched)", "Random gate$^\\ddagger$"),
]


def table3():
    p = pd.read_csv(RES / "policy_metrics.csv").set_index("policy")
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Hierarchical policies, ablations and system cost (all 17 bearings, mean$\pm$std over 3 seeds). "
             r"Esc.: escalated snapshots; Unsafe: critical snapshots ($y{\le}H_c$) left without maintenance action; "
             r"FM: false maintenance on healthy snapshots ($y{>}H_p$); B/s: payload bytes per snapshot; "
             r"Lat.: mean decision latency with an assumed 50\,ms RTT. $^\ddagger$escalation budget matched to HERA "
             r"on calibration bearings.}",
             r"\label{tab:policies}", r"\setlength{\tabcolsep}{2.2pt}\footnotesize", r"\resizebox{\columnwidth}{!}{%",
             r"\begin{tabular}{lcccccc}", r"\toprule",
             r"Policy & Esc.\,(\%) & Unsafe\,(\%) & FM\,(\%) & RMSE\,(s) & B/s & Lat.\,(ms) \\", r"\midrule"]
    for key, lab in POLICY_ORDER:
        if key not in p.index:
            continue
        x = p.loc[key]
        lines.append(f"{lab} & {100 * x.escalation_rate_mean:.1f} & {pm(100 * x.unsafe_rate_mean, 100 * x.unsafe_rate_std, '{:.1f}')} & "
                     f"{pm(100 * x.false_maint_mean, 100 * x.false_maint_std, '{:.1f}')} & {x.rmse_mean:.0f} & "
                     f"{x.bytes_per_snapshot_mean:.0f} & {x.latency_mean_ms_mean:.1f} \\\\")
        if key in ("Cloud point+threshold", "HERA-no-DT", "HERA-fixed (anomaly trigger)"):
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    (OUT / "tables" / "table3_policies.tex").write_text("\n".join(lines) + "\n")
    names = {"Always-edge": "Edge", "Always-cloud": "Cloud", "Cloud point+threshold": "Point", "HERA-full": "Hera",
             "HERA w/o anomaly term": "HeraNoA", "HERA-no-DT": "HeraNoDT", "HERA-no-UQ": "HeraNoUQ",
             "HERA-fixed (anomaly trigger)": "HeraFixed", "Width gate (matched)": "Width",
             "Linear score gate (matched)": "Score", "Random gate (matched)": "Random"}
    for key, n in names.items():
        if key not in p.index:
            continue
        x = p.loc[key]
        macro(f"esc{n}", f"{100 * x.escalation_rate_mean:.1f}")
        macro(f"unsafe{n}", f"{100 * x.unsafe_rate_mean:.1f}")
        macro(f"missUrg{n}", f"{100 * x.missed_urgent_mean:.1f}")
        macro(f"fm{n}", f"{100 * x.false_maint_mean:.1f}")
        macro(f"furg{n}", f"{100 * x.false_urgent_mean:.1f}")
        macro(f"rmsePol{n}", f"{x.rmse_mean:.0f}")
        macro(f"bytes{n}", f"{x.bytes_per_snapshot_mean:.1f}")
        macro(f"lat{n}", f"{x.latency_mean_ms_mean:.1f}")
        macro(f"kmacs{n}", f"{x.kmacs_per_snapshot_mean:.0f}")
        macro(f"appr{n}", f"{100 * x.approval_load_mean:.1f}")
    cb = p.loc["Always-cloud", "bytes_per_snapshot_mean"]
    macro("commReduction", f"{100 * (1 - p.loc['HERA-full', 'bytes_per_snapshot_mean'] / cb):.0f}")
    macro("macsReduction", f"{100 * (1 - p.loc['HERA-full', 'kmacs_per_snapshot_mean'] / p.loc['Always-cloud', 'kmacs_per_snapshot_mean']):.0f}")


def other_macros():
    sysm = json.loads((RES / "system_metrics.json").read_text())
    sel = json.loads((RES / "selected_models.json").read_text())
    e, c = sel["edge_model"], sel["cloud_model"] + "+DT"
    macro("edgeParams", f"{sysm[e]['params'] / 1e3:.1f}k")
    macro("cloudParams", f"{sysm[c]['params'] / 1e3:.1f}k")
    macro("edgeMacs", f"{sysm[e]['macs'] / 1e3:.0f}k")
    macro("cloudMacs", f"{sysm[c]['macs'] / 1e6:.2f}M")
    macro("edgeLat", f"{sysm[e]['median_ms']:.2f}")
    macro("cloudLat", f"{sysm[c]['median_ms']:.2f}")
    macro("rttMs", f"{sysm['assumptions']['rtt_ms']:.0f}")
    macro("tauAmin", f"{min(sel['tau_a_per_fold']):.1f}")
    macro("tauAmax", f"{max(sel['tau_a_per_fold']):.1f}")
    det = pd.read_csv(RES / "detection_metrics.csv").set_index("detector")
    for key, n in [("Anomaly score $a_t$ (edge)", "Anom"), ("Isolation Forest", "Iso"),
                   ("Edge CQR lower bound", "EdgeLB"), ("Cloud CQR lower bound (+DT)", "CloudLB")]:
        x = det.loc[key]
        for mtr in ["precision", "recall", "f1", "auroc"]:
            macro(f"det{n}{mtr.capitalize().replace("1", "one")}", f"{x[mtr + '_mean']:.2f}")
    st = json.loads((RES / "stat_tests.json").read_text())
    for k, n in [("unsafe_hera_vs_unsafe_point", "HeraPoint"), ("unsafe_hera_vs_unsafe_cloud", "HeraCloud"),
                 ("unsafe_hera_vs_unsafe_width", "HeraWidth"), ("false_maint_hera_vs_false_maint_point", "FmHeraPoint")]:
        pv = st[k]["p_wilcoxon"]
        macro(f"p{n}", "n/a" if pv is None else (f"{pv:.3f}" if pv >= 0.001 else "$<$0.001"))
        macro(f"nz{n}", str(st[k]["n_nonzero"]))
    pb = pd.read_csv(RES / "per_bearing.csv")
    macro("nBearingsCovLow", str(int((pb.coverage < 0.9).sum())))
    ex = RES / "explanation_cards.json"
    if ex.exists():
        macro("shapAdditivity", f"{json.loads(ex.read_text())['additivity_mae_s']:.1f}")
    cv = pd.read_csv(RES / "tradeoff_curves.csv")
    g = cv.groupby(["family", "param"]).mean(numeric_only=True).reset_index()
    h = g[(g.family == "HERA (decision-sufficiency)")].sort_values("param")
    for al, n in [(0.05, "Five"), (0.3, "Thirty"), (0.5, "Fifty")]:
        row = h[np.isclose(h.param, al)].iloc[0]
        macro(f"escAlpha{n}", f"{100 * row.escalation_rate:.1f}")
        macro(f"unsafeAlpha{n}", f"{100 * row.unsafe_rate:.1f}")
        macro(f"fmAlpha{n}", f"{100 * row.false_maint:.1f}")


def write_macros():
    lines = ["% Auto-generated by experiments/make_tables.py -- do not edit"]
    for k, v in sorted(MACROS.items()):
        assert k.isalpha(), k
        lines.append(f"\\newcommand{{\\{k}}}{{{v}}}")
    (OUT / "numbers.tex").write_text("\n".join(lines) + "\n")
    (RES / "paper_numbers.json").write_text(json.dumps(MACROS, indent=1))


def card():
    """Explanation card of the most critical explained snapshot (explainability.py)."""
    c = json.loads((RES / "explanation_cards.json").read_text())["cards"][0]
    top = ", ".join(f"{t['feature'].replace('_', chr(92) + '_')} ({t['shap_s']:+.0f}\\,s)" for t in c["top_contributors"])
    lo, hi = c["interval_90pct_s"]
    body = (f"\\textbf{{Asset}} {c['asset'].replace('_', chr(92) + '_')}, snapshot {c['snapshot']}. "
            f"\\textbf{{RUL}} {c['predicted_RUL_s']}\\,s, 90\\% interval [{lo}, {hi}]\\,s. "
            f"\\textbf{{Anomaly}} $a_t$={c['anomaly_evidence']['score']} (threshold {c['anomaly_evidence']['alarm_threshold']}, "
            f"{'alarm' if c['anomaly_evidence']['alarm'] else 'no alarm'}). \\textbf{{Top SHAP}} {top}. "
            f"\\textbf{{Recommendation}} {c['recommended_action']} --- \\emph{{requires human approval}}. "
            f"\\textbf{{Rationale}} lower bound {lo}\\,s $\\le H_{{\\mathrm{{c}}}}$.")
    (OUT / "tables" / "card.tex").write_text("\\fbox{\\parbox{0.96\\columnwidth}{\\footnotesize " + body + "}}\n")


if __name__ == "__main__":
    table2(); table3(); other_macros(); write_macros(); card()
    print(f"{len(MACROS)} macros; tables in paper/tables/")
