"""Publication figures (vector PDF, IEEE single-column width 3.5 in).

Fig. 1  architecture (drawn programmatically)          figures/fig1_architecture.pdf
Fig. 2  decision-sufficiency escalation mechanism     figures/fig2_escalation.pdf
Fig. 3  RUL trajectories with calibrated intervals     figures/fig3_trajectories.pdf
Fig. 4  unsafe-decision vs. escalation trade-off       figures/fig4_tradeoff.pdf
All data are read from results/; nothing is hard-coded.
"""
import json
import shutil
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

matplotlib.use("pdf")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from utils import H_CRIT, H_PLAN, R_MAX  # noqa: E402

RES, FIG = ROOT / "results", ROOT / "figures"
FIG.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.size": 7.5, "axes.labelsize": 7.5, "legend.fontsize": 6.5,
                     "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "axes.linewidth": 0.6,
                     "lines.linewidth": 1.0, "pdf.fonttype": 42})
C = {"edge": "#2a6f97", "cloud": "#c05621", "true": "#111111", "hera": "#2f855a", "grey": "#718096",
     "red": "#c53030", "amber": "#d69e2e"}
W1 = 3.5


def box(ax, x, y, w, h, text, fc, ec="#333", fs=6.6, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02", fc=fc, ec=ec, lw=0.6))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, weight="bold" if bold else "normal",
            linespacing=1.15)


def arrow(ax, p, q, style="-|>", color="#333", ls="-", lw=0.7, text=None, tx=0, ty=0, fs=5.8):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=7, color=color, lw=lw, linestyle=ls))
    if text:
        ax.text((p[0] + q[0]) / 2 + tx, (p[1] + q[1]) / 2 + ty, text, fontsize=fs, ha="center", va="center",
                color=color)


def fig1():
    fig, ax = plt.subplots(figsize=(W1, 2.35))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    for y0, h, lab, col in [(0.0, 0.30, "L1 edge", "#ebf4fa"), (0.34, 0.30, "L2-L3 fog/cloud", "#fdf1e7"),
                            (0.68, 0.32, "L4 decision", "#edf7ef")]:
        ax.add_patch(FancyBboxPatch((0.0, y0), 1.0, h, boxstyle="square,pad=0", fc=col, ec="none"))
        ax.text(0.005, y0 + h - 0.015, lab, fontsize=5.5, va="top", color="#4a5568", style="italic")
    box(ax, 0.02, 0.04, 0.17, 0.17, "Vibration\nsensors", "white")
    box(ax, 0.25, 0.04, 0.34, 0.17, "Edge Monitoring Agent\nHI, $a_t$, tiny GRU\n$I^e_t=[L^e_t,U^e_t]$", "white")
    box(ax, 0.66, 0.04, 0.32, 0.17, "Escalation gate $e_t$\n$I^e_t$ ambiguous\nw.r.t. $H_c,H_p$?", "white", ec=C["hera"])
    box(ax, 0.02, 0.38, 0.42, 0.18, "Digital Twin State Agent\n$DT_t$: HI, context $c_t$,\nanomaly, RUL, $u_t$", "white")
    box(ax, 0.54, 0.38, 0.44, 0.18, "RUL Prognostics Agent\nsequence model + $c_t$\nCQR $I^c_t=[L^c_t,U^c_t]$", "white")
    box(ax, 0.02, 0.73, 0.44, 0.19, "Maintenance Decision\nAgent: action $=r(L_t)$\n+ SHAP evidence card", "white")
    box(ax, 0.56, 0.73, 0.42, 0.19, "Human approval\n(schedule / urgent are\nrecommendations)", "white", ec=C["red"])
    arrow(ax, (0.19, 0.125), (0.25, 0.125))
    arrow(ax, (0.59, 0.125), (0.66, 0.125))
    arrow(ax, (0.40, 0.21), (0.25, 0.38), color=C["grey"], ls="--", text="sync", tx=-0.04)
    arrow(ax, (0.80, 0.21), (0.76, 0.38), color=C["cloud"], text="$e_t{=}1$", tx=-0.06)
    arrow(ax, (0.44, 0.47), (0.54, 0.47))
    arrow(ax, (0.70, 0.56), (0.30, 0.73), color=C["cloud"])
    arrow(ax, (0.975, 0.21), (0.46, 0.80), color=C["edge"], ls=":", text="$e_t{=}0$", tx=0.34, ty=0.06)
    arrow(ax, (0.46, 0.825), (0.56, 0.825))
    fig.savefig(FIG / "fig1_architecture.pdf", bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


def fig2():
    """Illustrates the gate using real edge intervals from the saved trajectories."""
    tr = pd.read_csv(RES / "trajectories.csv.gz")
    b = json.loads((RES / "figure_choices.json").read_text())["fig2_bearing"]
    d = tr[tr.bearing == b].sort_values("t")
    t_h = d.t * 10 / 3600
    fig, ax = plt.subplots(figsize=(W1, 1.45))
    for lo, hi, col, lab in [(0, H_CRIT, "#fed7d7", "urgent"), (H_CRIT, H_PLAN, "#fefcbf", "schedule"),
                             (H_PLAN, R_MAX, "#e6fffa", "continue")]:
        ax.axhspan(lo, hi, color=col, lw=0, zorder=0)
        ax.text(t_h.max() * 0.5, (lo + hi) / 2, lab, fontsize=5.5, va="center", ha="center", color="#2d3748",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8), zorder=5)
    ax.fill_between(t_h, d.e_lo, d.e_hi, color=C["edge"], alpha=0.30, lw=0, label="edge 90% CQR interval")
    ax.plot(t_h, d.y, color=C["true"], lw=0.9, label="true RUL (capped)")
    esc = d.esc.astype(bool).to_numpy()
    ax.fill_between(t_h, R_MAX * 1.02, R_MAX * 1.07, where=esc, color=C["cloud"], lw=0, step="mid",
                    label="escalated ($e_t{=}1$)")
    ax.set_ylim(0, R_MAX * 1.08); ax.set_xlabel("operating time (h)"); ax.set_ylabel("RUL (s)")
    ax.legend(loc="lower left", frameon=False, ncol=3, bbox_to_anchor=(-0.02, 1.0), handlelength=1.2,
              columnspacing=0.8)
    fig.savefig(FIG / "fig2_escalation.pdf", bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


def fig3():
    tr = pd.read_csv(RES / "trajectories.csv.gz")
    bs = json.loads((RES / "figure_choices.json").read_text())["fig3_bearings"]
    fig, axs = plt.subplots(1, len(bs), figsize=(W1, 1.45), sharey=True)
    for ax, b in zip(axs, bs):
        d = tr[tr.bearing == b].sort_values("t")
        d = d[d.y < R_MAX + 1]
        d = d.iloc[-min(len(d), 450):]
        x = -(d.t.max() - d.t) * 10 / 60
        ax.fill_between(x, d.c_lo, d.c_hi, color=C["cloud"], alpha=0.25, lw=0, label="90% CQR (fog/cloud)")
        ax.plot(x, d.c_med, color=C["cloud"], lw=0.8, label="predicted (median)")
        ax.plot(x, d.y, color=C["true"], lw=0.9, ls="--", label="true RUL")
        ax.axhline(H_CRIT, color=C["red"], lw=0.5, ls=":")
        ax.set_title(b.replace("Bearing", "Bearing "), fontsize=6.5, pad=2)
    axs[1].set_xlabel("time to failure (min)")
    axs[0].set_ylabel("RUL (s)")
    axs[0].legend(loc="upper center", frameon=False, ncol=3, bbox_to_anchor=(1.6, 1.42), handlelength=1.2)
    fig.subplots_adjust(wspace=0.08)
    fig.savefig(FIG / "fig3_trajectories.pdf", bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


def fig4():
    cv = pd.read_csv(RES / "tradeoff_curves.csv")
    g = cv.groupby(["family", "param"]).mean(numeric_only=True).reset_index()
    fig, axs = plt.subplots(1, 2, figsize=(W1, 1.75))
    # (a) decision Pareto: unsafe vs false maintenance
    pareto = {"P: Always-edge (CQR)": (C["edge"], "o", "-", "edge, CQR ($\\alpha$)"),
              "P: Always-cloud (CQR)": (C["cloud"], "s", "-", "cloud, CQR ($\\alpha$)"),
              "P: HERA": (C["hera"], "^", "-", "HERA ($\\alpha$)"),
              "P: Edge point - margin": (C["edge"], "o", ":", "edge, point$-\\delta$"),
              "P: Cloud point - margin": (C["cloud"], "s", ":", "cloud, point$-\\delta$")}
    ax = axs[0]
    for fam, (col, mk, ls, lab) in pareto.items():
        d = g[g.family == fam].sort_values("false_maint")
        ax.plot(d.false_maint * 100, d.unsafe_rate * 100, color=col, ls=ls, marker=mk, ms=2.0, lw=0.8, label=lab,
                mfc="white" if ls == ":" else col)
    for fam, col in [("P: Always-edge (CQR)", C["edge"]), ("P: Always-cloud (CQR)", C["cloud"]), ("P: HERA", C["hera"])]:
        r = g[(g.family == fam) & np.isclose(g.param, 0.1)].iloc[0]
        ax.scatter(r.false_maint * 100, r.unsafe_rate * 100, s=22, facecolor="none", edgecolor=col, lw=0.8, zorder=6)
    r = g[(g.family == "P: Cloud point - margin") & (g.param == 0)].iloc[0]
    ax.annotate("point, fixed\nthreshold", (r.false_maint * 100, r.unsafe_rate * 100), xytext=(2, 22), fontsize=5.5,
                arrowprops=dict(arrowstyle="-", lw=0.5, color="#555"))
    ax.set_xlabel("false maintenance (% healthy)"); ax.set_ylabel("unsafe decisions (% critical)")
    ax.set_title("(a) decision trade-off", fontsize=6.5, pad=2)
    ax.legend(frameon=False, fontsize=5.2, loc="upper right", handlelength=1.6, labelspacing=0.25)
    ax.grid(alpha=0.25, lw=0.4)
    # (b) escalation budget at fixed alpha = 0.1
    ax = axs[1]
    styles = {"Width gate": (C["edge"], "s", "--"), "Linear score gate": (C["amber"], "^", "--"),
              "Anomaly trigger": (C["red"], "v", ":"), "Random gate": ("#a0aec0", None, ":")}
    for fam, (col, mk, ls) in styles.items():
        d = g[g.family == fam].sort_values("escalation_rate")
        ax.plot(d.escalation_rate * 100, d.unsafe_rate * 100, color=col, ls=ls, marker=mk, ms=1.8, lw=0.8,
                label=fam.replace(" gate", "").replace(" trigger", " trig."), markevery=3)
    r = g[(g.family == "HERA (decision-sufficiency)") & np.isclose(g.param, 0.1)].iloc[0]
    ax.scatter(r.escalation_rate * 100, r.unsafe_rate * 100, marker="*", s=40, color=C["hera"], zorder=6, label="HERA")
    ax.set_xlabel("escalated snapshots (%)"); ax.set_ylabel("unsafe decisions (% critical)")
    ax.set_title("(b) gates at $\\alpha{=}0.1$", fontsize=6.5, pad=2)
    ax.set_ylim(0, 15); ax.grid(alpha=0.25, lw=0.4)
    ax.legend(frameon=False, fontsize=5.2, loc="upper left", ncol=2, handlelength=1.4, columnspacing=0.6)
    fig.subplots_adjust(wspace=0.42)
    fig.savefig(FIG / "fig4_tradeoff.pdf", bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


def fig_shap():
    p = RES / "shap_importance.csv"
    if not p.exists():
        return
    d = pd.read_csv(p).head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(W1, 1.45))
    cols = d.group.map({"horizontal": C["edge"], "vertical": C["cloud"], "DT context": C["hera"]})
    ax.barh(d.name.str.replace("DT:", "DT: "), d.mean_abs_shap_s, color=cols, height=0.7)
    ax.set_xlabel("mean |SHAP| on predicted RUL (s)")
    fig.savefig(FIG / "fig5_shap.pdf", bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


def choose():
    """Deterministic choice of illustrated bearings: the three test bearings with
    the most degradation-region snapshots (seed 0), and for Fig. 2 the bearing
    with the median escalation rate among them. Not chosen by performance."""
    tr = pd.read_csv(RES / "trajectories.csv.gz")
    n_deg = tr[tr.y < R_MAX].groupby("bearing").size().sort_values(ascending=False)
    bs = sorted(n_deg.index[:3])
    esc = tr[tr.bearing.isin(bs)].groupby("bearing").esc.mean().sort_values()
    out = {"fig3_bearings": bs, "fig2_bearing": esc.index[1], "rule": choose.__doc__.strip()}
    (RES / "figure_choices.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    choose()
    fig1(); fig2(); fig3(); fig4(); fig_shap()
    for f in FIG.glob("*.pdf"):
        shutil.copy(f, ROOT / "paper" / "figures" / f.name)
    print("figures:", sorted(p.name for p in FIG.glob("*.pdf")))
