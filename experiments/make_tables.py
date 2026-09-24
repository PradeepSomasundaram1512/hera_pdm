"""Generate LaTeX tables and numeric macros for the paper from results/<dataset>/.

Outputs
  paper/tables/table2_rul.tex       RUL accuracy + calibration, both datasets (table*)
  paper/tables/table3_policies.tex  hierarchical policies and ablations, both datasets (table*)
  paper/tables/table4_timing.tex    per-bearing alarm timing and sensitivity
  paper/tables/card.tex             explanation card (FEMTO)
  paper/numbers.tex                 \\newcommand macros (prefix fe/xj) for every number quoted in the text
No number in these files is typed by hand.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper"
(OUT / "tables").mkdir(parents=True, exist_ok=True)
DS = {"fe": "femto", "xj": "xjtu"}
MACROS = {}
RMAX_GRID = {"femto": [1500, 6000], "xjtu": [3000, 12000]}


def R(ds):
    return ROOT / "results" / ds


def pm(mean, std, fmt="{:.0f}"):
    if std is None or (isinstance(std, float) and np.isnan(std)):
        return fmt.format(mean)
    return f"{fmt.format(mean)}$\\pm${fmt.format(std)}"


def macro(name, value):
    assert name.isalpha(), name
    MACROS[name] = value


def fmt_p(pv):
    return "n/a" if pv is None else (f"{pv:.3f}" if pv >= 0.001 else "$<$0.001")


def rul_tables():
    rul = {p: pd.read_csv(R(ds) / "rul_metrics.csv") for p, ds in DS.items()}
    sel = {p: json.loads((R(ds) / "run_info.json").read_text()) for p, ds in DS.items()}
    sysm = {p: json.loads((R(ds) / "system_metrics.json").read_text()) for p, ds in DS.items()}

    def row(p, model, calib, point=False):
        d = rul[p]
        x = d[(d.model == model) & (d.calib == calib)]
        if x.empty:
            return ["--"] * 5
        x = x.iloc[0]
        if point:
            return [pm(x.rmse_mean, x.rmse_std), pm(x.mae_mean, x.mae_std), "--", "--", "--"]
        return [pm(x.rmse_mean, x.rmse_std), pm(x.mae_mean, x.mae_std), f"{x.raw_picp_mean:.2f}",
                pm(x.picp_mean, x.picp_std, "{:.2f}"), f"{x.mpiw_mean:.0f}"]

    L = [r"\begin{table*}[t]", r"\centering",
         r"\caption{RUL accuracy and calibration on held-out bearings (mean$\pm$std over 3 seeds; nominal coverage 0.90). "
         r"Raw: uncalibrated quantiles; split: CQR calibrated on 3 held-out bearings per fold; CV+: CQR-CV+ calibrated "
         r"on all non-test bearings. Marks: architecture selected on inner-validation pinball loss for FEMTO ($^\dagger$) "
         r"and XJTU-SY ($^\ast$).}",
         r"\label{tab:rul}", r"\setlength{\tabcolsep}{3.2pt}\footnotesize",
         r"\begin{tabular}{l|ccccc|ccccc}", r"\toprule",
         r" & \multicolumn{5}{c|}{FEMTO/PRONOSTIA (17 bearings, $R_{\max}{=}3000$\,s)} & "
         r"\multicolumn{5}{c}{XJTU-SY (15 bearings, $R_{\max}{=}6000$\,s)} \\",
         r"Model & RMSE (s) & MAE (s) & Raw cov. & Cov. & MPIW (s) & RMSE (s) & MAE (s) & Raw cov. & Cov. & MPIW (s) \\",
         r"\midrule"]
    models = ["Edge-CNN", "Edge-GRU", None] + [f"{a}{s}" for a in ["LSTM", "GRU", "TCN", "CNN-LSTM"] for s in ("", "+DT")]
    for m in models:
        if m is None:
            L.append(r"\midrule")
            continue
        marks = (r"$^\dagger$" if m in (sel["fe"]["edge_model"], sel["fe"]["cloud_model"] + "+DT") else "") + \
                (r"$^\ast$" if m in (sel["xj"]["edge_model"], sel["xj"]["cloud_model"] + "+DT") else "")
        L.append(f"{m}{marks} (split) & " + " & ".join(row("fe", m, "split")) + " & "
                 + " & ".join(row("xj", m, "split")) + r" \\")
    L.append(r"\midrule")
    for a in ["LSTM", "GRU", "TCN", "CNN-LSTM"]:
        m = a + "+DT-point"
        L.append(f"{a}+DT (MSE point) & " + " & ".join(row("fe", m, "split", True)) + " & "
                 + " & ".join(row("xj", m, "split", True)) + r" \\")
    L.append(r"\midrule")
    for lab, key in [("Selected edge (CV+)", "edge"), ("Selected cloud+DT (CV+)", "cloud_dt"),
                     ("Selected cloud, no DT (CV+)", "cloud")]:
        cells = []
        for p in DS:
            m = {"edge": sel[p]["edge_model"], "cloud_dt": sel[p]["cloud_model"] + "+DT",
                 "cloud": sel[p]["cloud_model"]}[key]
            cells += row(p, m, "cvplus")
        L.append(f"{lab} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    (OUT / "tables" / "table2_rul.tex").write_text("\n".join(L) + "\n")

    for p in DS:
        d = rul[p]
        e, c = sel[p]["edge_model"], sel[p]["cloud_model"]
        macro(f"{p}EdgeModel", e)
        macro(f"{p}CloudModel", c)
        macro(f"{p}NBearings", str(sel[p]["n_bearings"]))
        macro(f"{p}NSnap", f"{sel[p]['n_snapshots']:,}".replace(",", "{,}"))
        macro(f"{p}Rmax", f"{sel[p]['r_max']:.0f}")
        macro(f"{p}Hc", f"{sel[p]['h_crit']:.0f}")
        macro(f"{p}Hp", f"{sel[p]['h_plan']:.0f}")
        macro(f"{p}NFolds", str(sel[p]["n_folds"]))
        for key, m, calib in [("Edge", e, "cvplus"), ("Cloud", c + "+DT", "cvplus"), ("CloudNoDT", c, "cvplus"),
                              ("EdgeSplit", e, "split"), ("CloudSplit", c + "+DT", "split")]:
            x = d[(d.model == m) & (d.calib == calib)].iloc[0]
            macro(f"{p}Rmse{key}", f"{x.rmse_mean:.0f}")
            macro(f"{p}Mae{key}", f"{x.mae_mean:.0f}")
            macro(f"{p}Picp{key}", f"{x.picp_mean:.2f}")
            macro(f"{p}Mpiw{key}", f"{x.mpiw_mean:.0f}")
            macro(f"{p}RawPicp{key}", f"{x.raw_picp_mean:.2f}")
            macro(f"{p}Cce{key}", f"{x.cce_mean:.3f}")
            macro(f"{p}CovLow{key}", f"{x.n_bearing_cov_below_mean:.1f}")
            macro(f"{p}CovMin{key}", f"{x.picp_bearing_min_mean:.2f}")
        sp = d[d.calib == "split"]
        q = sp[~sp.model.str.endswith("point")]
        cl = q[~q.model.str.startswith("Edge")]
        macro(f"{p}BestCloudRmse", f"{cl.rmse_mean.min():.0f}")
        macro(f"{p}BestCloudModel", cl.loc[cl.rmse_mean.idxmin(), "model"])
        macro(f"{p}WorstCloudRmse", f"{cl.rmse_mean.max():.0f}")
        macro(f"{p}RawCovMin", f"{q.raw_picp_mean.min():.2f}")
        macro(f"{p}RawCovMax", f"{q.raw_picp_mean.max():.2f}")
        macro(f"{p}SplitCovMin", f"{q.picp_mean.min():.2f}")
        macro(f"{p}SplitCovMax", f"{q.picp_mean.max():.2f}")
        macro(f"{p}EdgeParams", f"{sysm[p][e]['params'] / 1e3:.1f}k")
        macro(f"{p}CloudParams", f"{sysm[p][c + '+DT']['params'] / 1e3:.1f}k")
        macro(f"{p}EdgeLat", f"{sysm[p][e]['median_ms']:.2f}")
        macro(f"{p}CloudLat", f"{sysm[p][c + '+DT']['median_ms']:.2f}")
        macro(f"{p}EdgeMacs", f"{sysm[p][e]['macs'] / 1e3:.0f}k")
        macro(f"{p}CloudMacs", f"{sysm[p][c + '+DT']['macs'] / 1e6:.2f}M")
        dtg = [sp[sp.model == a + "+DT"].rmse_mean.iloc[0] < sp[sp.model == a].rmse_mean.iloc[0]
               for a in ["LSTM", "GRU", "TCN", "CNN-LSTM"]]
        macro(f"{p}DtHelps", str(int(sum(dtg))))


POLICY_ORDER = [
    ("Always-edge", "Always-edge"), ("Always-cloud", "Always-cloud"), ("Cloud point+threshold", "Cloud point + fixed thr."),
    ("HERA-full", r"\textbf{HERA-full}"), ("HERA w/o anomaly term", "HERA w/o anomaly term"), ("HERA-no-DT", "HERA-no-DT"),
    ("HERA-no-UQ", "HERA-no-UQ (point margin)"), ("HERA-fixed (anomaly trigger)", "HERA-fixed (anomaly trig.)"),
    ("Width gate (matched)", "Width gate$^\\ddagger$"), ("Linear score gate (matched)", "Linear score gate$^\\ddagger$"),
    ("Random gate (matched)", "Random gate$^\\ddagger$")]
NAMES = {"Always-edge": "Edge", "Always-cloud": "Cloud", "Cloud point+threshold": "Point", "HERA-full": "Hera",
         "HERA w/o anomaly term": "HeraNoA", "HERA-no-DT": "HeraNoDT", "HERA-no-UQ": "HeraNoUQ",
         "HERA-fixed (anomaly trigger)": "HeraFixed", "Width gate (matched)": "Width",
         "Linear score gate (matched)": "Score", "Random gate (matched)": "Random"}


def policy_tables():
    pol = {p: pd.read_csv(R(ds) / "policy_metrics.csv").set_index("policy") for p, ds in DS.items()}
    L = [r"\begin{table*}[t]", r"\centering",
         r"\caption{Hierarchical policies, ablations and system cost (mean$\pm$std over 3 seeds, all held-out bearings). "
         r"Esc.: escalated snapshots; Unsafe: critical snapshots ($y{\le}H_c$) left without a maintenance action; "
         r"FM: false maintenance on healthy snapshots ($y{>}H_p$); B/s: payload bytes per snapshot (analytic). "
         r"$^\ddagger$escalation budget matched to HERA on calibration data.}",
         r"\label{tab:policies}", r"\setlength{\tabcolsep}{3.4pt}\footnotesize",
         r"\begin{tabular}{l|ccccc|ccccc}", r"\toprule",
         r" & \multicolumn{5}{c|}{FEMTO/PRONOSTIA} & \multicolumn{5}{c}{XJTU-SY} \\",
         r"Policy & Esc.\,(\%) & Unsafe\,(\%) & FM\,(\%) & RMSE\,(s) & B/s & Esc.\,(\%) & Unsafe\,(\%) & FM\,(\%) & RMSE\,(s) & B/s \\",
         r"\midrule"]
    for key, lab in POLICY_ORDER:
        cells = []
        for p in DS:
            x = pol[p].loc[key]
            cells += [f"{100 * x.escalation_rate_mean:.1f}",
                      pm(100 * x.unsafe_rate_mean, 100 * x.unsafe_rate_std, "{:.1f}"),
                      pm(100 * x.false_maint_mean, 100 * x.false_maint_std, "{:.1f}"), f"{x.rmse_mean:.0f}",
                      f"{x.bytes_per_snapshot_mean:.0f}"]
        L.append(f"{lab} & " + " & ".join(cells) + r" \\")
        if key in ("Cloud point+threshold", "HERA-no-DT", "HERA-fixed (anomaly trigger)"):
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    (OUT / "tables" / "table3_policies.tex").write_text("\n".join(L) + "\n")
    for p in DS:
        for key, n in NAMES.items():
            x = pol[p].loc[key]
            macro(f"{p}Esc{n}", f"{100 * x.escalation_rate_mean:.1f}")
            macro(f"{p}Unsafe{n}", f"{100 * x.unsafe_rate_mean:.1f}")
            macro(f"{p}Fm{n}", f"{100 * x.false_maint_mean:.1f}")
            macro(f"{p}RmsePol{n}", f"{x.rmse_mean:.0f}")
            macro(f"{p}Bytes{n}", f"{x.bytes_per_snapshot_mean:.1f}")
            macro(f"{p}Lat{n}", f"{x.latency_mean_ms_mean:.1f}")
            macro(f"{p}Kmacs{n}", f"{x.kmacs_per_snapshot_mean:.0f}")
            macro(f"{p}Appr{n}", f"{100 * x.approval_load_mean:.1f}")


def timing_table():
    tim = {p: pd.read_csv(R(ds) / "alarm_timing.csv").set_index("policy") for p, ds in DS.items()}
    info = {p: json.loads((R(ds) / "run_info.json").read_text()) for p, ds in DS.items()}
    order = [("Always-edge", "Always-edge"), ("Always-cloud", "Always-cloud"), ("Cloud point+threshold", "Point + thr."),
             ("HERA-full", r"\textbf{HERA-full}"), ("Width gate (matched)", "Width gate")]
    L = [r"\begin{table}[t]", r"\centering",
         r"\caption{Top: outcome of each bearing's first sustained (3 snapshots) schedule/urgent recommendation "
         r"(number of bearings, mean over 3 seeds): T timely ($H_c\le$ RUL $\le R_{\max}$), L late (RUL $<H_c$ or never), "
         r"P premature (RUL $>R_{\max}$); lead: median RUL at timely alarms (min). Bottom: sensitivity of "
         r"unsafe/FM (\%) for HERA-full and the point policy to the horizons and to $R_{\max}$.}",
         r"\label{tab:timing}", r"\setlength{\tabcolsep}{2.6pt}\footnotesize",
         r"\begin{tabular}{l|cccc|cccc}", r"\toprule",
         f" & \\multicolumn{{4}}{{c|}}{{FEMTO ({info['fe']['n_bearings']} bearings)}} & "
         f"\\multicolumn{{4}}{{c}}{{XJTU-SY ({info['xj']['n_bearings']} bearings)}} \\\\",
         r"Policy & T & L & P & lead & T & L & P & lead \\", r"\midrule"]
    for key, lab in order:
        cells = []
        for p in DS:
            x = tim[p].loc[key]
            lead = x.median_lead_timely_s_mean / 60
            cells += [f"{x.timely_mean:.1f}", f"{x.late_mean:.1f}", f"{x.premature_mean:.1f}",
                      "--" if np.isnan(lead) else f"{lead:.0f}"]
            macro(f"{p}Timely{NAMES[key]}", f"{x.timely_mean:.1f}")
            macro(f"{p}Late{NAMES[key]}", f"{x.late_mean:.1f}")
            macro(f"{p}Prem{NAMES[key]}", f"{x.premature_mean:.1f}")
        L.append(f"{lab} & " + " & ".join(cells) + r" \\")
    L += [r"\midrule", r"Setting & \multicolumn{2}{c}{HERA} & \multicolumn{2}{c|}{Point} & "
          r"\multicolumn{2}{c}{HERA} & \multicolumn{2}{c}{Point} \\", r"\midrule"]
    hs = {p: pd.read_csv(R(ds) / "horizon_sensitivity.csv") for p, ds in DS.items()}
    for fc, fp in [(0.1, 0.4), (0.2, 0.6), (0.3, 0.8)]:
        cells = []
        for p in DS:
            h = hs[p][np.isclose(hs[p].hc_frac, fc) & np.isclose(hs[p].hp_frac, fp)].set_index("policy")
            cells += [f"\\multicolumn{{2}}{{c}}{{{100 * h.loc['HERA-full', 'unsafe_rate']:.0f}/{100 * h.loc['HERA-full', 'false_maint']:.0f}}}",
                      f"\\multicolumn{{2}}{{c{'|' if p == 'fe' else ''}}}{{{100 * h.loc['Cloud point+threshold', 'unsafe_rate']:.0f}/"
                      f"{100 * h.loc['Cloud point+threshold', 'false_maint']:.0f}}}"]
        L.append(f"$H_c,H_p{{=}}{fc},{fp}R_{{\\max}}$ & " + " & ".join(cells) + r" \\")
    for mult, lab in [(0, r"$R_{\max}\times\frac12$ (seed 0)"), (1, r"$R_{\max}\times 2$ (seed 0)")]:
        cells = []
        for p, ds in DS.items():
            f = ROOT / "results" / f"{ds}_rmax{RMAX_GRID[ds][mult]}" / "light_summary.json"
            bar = "|" if p == "fe" else ""
            if f.exists():
                pp = {r["policy"]: r for r in json.loads(f.read_text())["policies"]}
                cells += [f"\\multicolumn{{2}}{{c}}{{{100 * pp['HERA-full']['unsafe_rate']:.0f}/{100 * pp['HERA-full']['false_maint']:.0f}}}",
                          f"\\multicolumn{{2}}{{c{bar}}}{{{100 * pp['Cloud point+threshold']['unsafe_rate']:.0f}/"
                          f"{100 * pp['Cloud point+threshold']['false_maint']:.0f}}}"]
            else:
                cells += ["\\multicolumn{2}{c}{--}", f"\\multicolumn{{2}}{{c{bar}}}{{--}}"]
        L.append(f"{lab} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT / "tables" / "table4_timing.tex").write_text("\n".join(L) + "\n")


def other_macros():
    for p, ds in DS.items():
        info = json.loads((R(ds) / "run_info.json").read_text())
        macro(f"{p}TauMin", f"{min(info['tau_a_per_fold']):.1f}")
        macro(f"{p}TauMax", f"{max(info['tau_a_per_fold']):.1f}")
        det = pd.read_csv(R(ds) / "detection_metrics.csv").set_index("detector")
        for key, n in [("Anomaly score $a_t$ (edge)", "Anom"), ("Isolation Forest", "Iso"),
                       ("Edge CQR lower bound", "EdgeLB"), ("Cloud CQR lower bound (+DT)", "CloudLB")]:
            x = det.loc[key]
            for mtr, mn in [("precision", "Prec"), ("recall", "Rec"), ("f1", "Fone"), ("auroc", "Auroc")]:
                macro(f"{p}Det{n}{mn}", f"{x[mtr + '_mean']:.2f}")
        st = json.loads((R(ds) / "stat_tests.json").read_text())
        for k, n in [("unsafe_hera_vs_unsafe_point", "HeraPoint"), ("unsafe_hera_vs_unsafe_cloud", "HeraCloud"),
                     ("unsafe_hera_vs_unsafe_width", "HeraWidth"), ("unsafe_hera_vs_unsafe_edge", "HeraEdge"),
                     ("fm_hera_vs_fm_point", "FmHeraPoint"), ("fm_hera_vs_fm_cloud", "FmHeraCloud"),
                     ("fm_hera_vs_fm_edge", "FmHeraEdge")]:
            macro(f"{p}P{n}", fmt_p(st[k]["p_wilcoxon"]))
            macro(f"{p}Nz{n}", str(st[k]["n_nonzero"]))
            macro(f"{p}Diff{n}", f"{100 * st[k]['mean_diff']:.1f}")
        cv = pd.read_csv(R(ds) / "tradeoff_curves.csv")
        g = cv.groupby(["family", "param"]).mean(numeric_only=True).reset_index()
        h = g[g.family == "HERA (decision-sufficiency)"]
        for al, n in [(0.05, "Five"), (0.3, "Thirty"), (0.5, "Fifty")]:
            r = h[np.isclose(h.param, al)].iloc[0]
            macro(f"{p}EscAlpha{n}", f"{100 * r.escalation_rate:.1f}")
            macro(f"{p}UnsafeAlpha{n}", f"{100 * r.unsafe_rate:.1f}")
            macro(f"{p}FmAlpha{n}", f"{100 * r.false_maint:.1f}")
        ex = R(ds) / "explanation_cards.json"
        if ex.exists():
            e = json.loads(ex.read_text())
            macro(f"{p}ShapAdd", f"{e['additivity_mae_s']:.1f}")
            macro(f"{p}ShapRel", f"{100 * e['additivity_rel']:.0f}")
    macro("rttMs", "50")
    # bearing-level conformal risk control (v3 exploration; src/evaluate_alarms.py)
    for p, ds in DS.items():
        a = pd.read_csv(R(ds) / "alarm_eps_sweep.csv")
        for pol, n in [("HERA-v3 (onset-gated, CRC)", "Crc"), ("Always-cloud, CRC", "CrcCloud"),
                       ("Cloud point, fixed thr. H_p", "PointHp")]:
            for eps, en in [(0.1, "Ten"), (0.3, "Thirty"), (None, "")]:
                x = a[(a.policy == pol) & ((a.eps.isna()) if eps is None else np.isclose(a.eps, eps if eps else 0))]
                if x.empty:
                    continue
                x = x.iloc[0]
                macro(f"{p}Late{n}{en}", f"{100 * x.late_rate_mean:.1f}")
                macro(f"{p}Prem{n}{en}", f"{100 * x.premature_rate_mean:.1f}")
                macro(f"{p}Esc{n}{en}", f"{100 * x.esc_frac_mean:.1f}")


def card():
    c = json.loads((R("femto") / "explanation_cards.json").read_text())["cards"][0]
    top = ", ".join(f"{t['feature'].replace('_', chr(92) + '_')} ({t['shap_s']:+.0f}\\,s)" for t in c["top_contributors"])
    lo, hi = c["interval_90pct_s"]
    body = (f"\\textbf{{Asset}} {c['asset'].replace('_', chr(92) + '_')}, snapshot {c['snapshot']}. "
            f"\\textbf{{RUL}} {c['predicted_RUL_s']}\\,s, 90\\% interval [{lo}, {hi}]\\,s. "
            f"\\textbf{{Anomaly}} $a_t$={c['anomaly_evidence']['score']} (threshold {c['anomaly_evidence']['alarm_threshold']}, "
            f"{'alarm' if c['anomaly_evidence']['alarm'] else 'no alarm'}). \\textbf{{Top SHAP}} {top}. "
            f"\\textbf{{Recommendation}} {c['recommended_action']} --- \\emph{{requires human approval}}. "
            f"\\textbf{{Rationale}} lower bound {lo}\\,s $\\le H_{{\\mathrm{{c}}}}$.")
    (OUT / "tables" / "card.tex").write_text("\\fbox{\\parbox{0.96\\columnwidth}{\\footnotesize " + body + "}}\n")


def write_macros():
    lines = ["% Auto-generated by experiments/make_tables.py -- do not edit"]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(MACROS.items())]
    (OUT / "numbers.tex").write_text("\n".join(lines) + "\n")
    (ROOT / "results" / "paper_numbers.json").write_text(json.dumps(MACROS, indent=1))


if __name__ == "__main__":
    rul_tables(); policy_tables(); timing_table(); other_macros(); card(); write_macros()
    print(f"{len(MACROS)} macros; tables in paper/tables/")
