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

FIG = ROOT / "figures"
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


DSETS = [("femto", "FEMTO/PRONOSTIA"), ("xjtu", "XJTU-SY")]


def info(ds):
    return json.loads((ROOT / "results" / ds / "run_info.json").read_text())


def choose(ds):
    """Deterministic choice of illustrated bearings: the two test bearings with the
    most degradation-region snapshots (seed 0). Not chosen by performance."""
    tr = pd.read_csv(ROOT / "results" / ds / "trajectories.csv.gz")
    rmax = info(ds)["r_max"]
    n_deg = tr[tr.y < rmax].groupby("bearing").size().sort_values(ascending=False)
    return sorted(n_deg.index[:2])


def fig_traj():
    fig, axs = plt.subplots(1, 2, figsize=(W1, 1.45))
    k = 0
    choices = {}
    for ds, title in DSETS:
        tr = pd.read_csv(ROOT / "results" / ds / "trajectories.csv.gz")
        I = info(ds)
        bs = choose(ds)[:1]
        choices[ds] = bs
        for b in bs:
            ax = axs[k]; k += 1
            d = tr[tr.bearing == b].sort_values("t")
            d = d[d.y < I["r_max"] + 1]
            snap = 10 if ds == "femto" else 60
            d = d.iloc[-min(len(d), int(1.5 * I["r_max"] / snap)):]
            x = -(d.t.max() - d.t) * snap / 60
            ax.fill_between(x, d.c_lo / 60, d.c_hi / 60, color=C["cloud"], alpha=0.25, lw=0, label="90% CQR-CV+ interval")
            ax.plot(x, d.c_med / 60, color=C["cloud"], lw=0.8, label="predicted RUL (median)")
            ax.plot(x, d.y / 60, color=C["true"], lw=0.9, ls="--", label="true RUL (capped)")
            ax.axhline(I["h_crit"] / 60, color=C["red"], lw=0.5, ls=":")
            ax.set_title(f"{title.split('/')[0]} {b.replace('Bearing', 'B')}", fontsize=6.5, pad=2)
            ax.set_xlabel("time to failure (min)")
            if k == 1:
                ax.set_ylabel("RUL (min)")
    axs[0].legend(loc="upper left", frameon=False, ncol=2, bbox_to_anchor=(-0.05, 1.55), handlelength=1.4, fontsize=5.8)
    fig.subplots_adjust(wspace=0.28)
    fig.savefig(FIG / "fig3_trajectories.pdf", bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    (ROOT / "results" / "figure_choices.json").write_text(json.dumps({"trajectories": choices,
                                                                      "rule": choose.__doc__.strip()}, indent=2))


def fig_tradeoff():
    fig, axs = plt.subplots(1, 4, figsize=(7.16, 1.75))
    pareto = {"P: Always-edge (CQR)": (C["edge"], "o", "-", "edge, CQR-CV+ ($\\alpha$)"),
              "P: Always-cloud (CQR)": (C["cloud"], "s", "-", "cloud, CQR-CV+ ($\\alpha$)"),
              "P: HERA": (C["hera"], "^", "-", "HERA ($\\alpha$)"),
              "P: Edge point - margin": (C["edge"], "o", ":", "edge point$-\\delta$"),
              "P: Cloud point - margin": (C["cloud"], "s", ":", "cloud point$-\\delta$")}
    gates = {"Width gate": (C["edge"], "s", "--"), "Linear score gate": (C["amber"], "^", "--"),
             "Anomaly trigger": (C["red"], "v", ":"), "Random gate": ("#a0aec0", None, ":")}
    for j, (ds, title) in enumerate(DSETS):
        g = pd.read_csv(ROOT / "results" / ds / "tradeoff_curves.csv").groupby(["family", "param"]) \
            .mean(numeric_only=True).reset_index()
        ax = axs[2 * j]
        for fam, (col, mk, ls, lab) in pareto.items():
            d = g[g.family == fam].sort_values("false_maint")
            ax.plot(d.false_maint * 100, d.unsafe_rate * 100, color=col, ls=ls, marker=mk, ms=1.8, lw=0.8, label=lab,
                    mfc="white" if ls == ":" else col)
        for fam, col in [("P: Always-edge (CQR)", C["edge"]), ("P: Always-cloud (CQR)", C["cloud"]), ("P: HERA", C["hera"])]:
            r = g[(g.family == fam) & np.isclose(g.param, 0.1)].iloc[0]
            ax.scatter(r.false_maint * 100, r.unsafe_rate * 100, s=20, facecolor="none", edgecolor=col, lw=0.8, zorder=6)
        r = g[(g.family == "P: Cloud point - margin") & (g.param == 0)].iloc[0]
        ax.scatter(r.false_maint * 100, r.unsafe_rate * 100, marker="x", s=18, color="k", zorder=7,
                   label="point, fixed thr.")
        ax.set_xlabel("false maintenance (%)")
        ax.set_ylabel("unsafe decisions (%)")
        ax.set_title(f"({'ac'[j]}) {title.split('/')[0]}: decision frontier", fontsize=6.3, pad=2)
        ax.grid(alpha=0.25, lw=0.4)
        ax = axs[2 * j + 1]
        for fam, (col, mk, ls) in gates.items():
            d = g[g.family == fam].sort_values("escalation_rate")
            ax.plot(d.escalation_rate * 100, d.unsafe_rate * 100, color=col, ls=ls, marker=mk, ms=1.6, lw=0.8,
                    label=fam.replace(" gate", "").replace(" trigger", " trig."), markevery=3)
        h = g[g.family == "HERA (decision-sufficiency)"].sort_values("escalation_rate")
        ax.plot(h.escalation_rate * 100, h.unsafe_rate * 100, color=C["hera"], lw=0.6, alpha=0.6)
        r = h[np.isclose(h.param, 0.1)].iloc[0]
        ax.scatter(r.escalation_rate * 100, r.unsafe_rate * 100, marker="*", s=40, color=C["hera"], zorder=6,
                   label="HERA ($\\alpha_e$ sweep, $\\star$=0.1)")
        ax.set_xlabel("escalated snapshots (%)")
        ax.set_ylim(bottom=0, top=max(2.0, g[g.family.isin(list(gates))].unsafe_rate.max() * 110))
        ax.set_title(f"({'bd'[j]}) {title.split('/')[0]}: gates at $\\alpha{{=}}0.1$", fontsize=6.3, pad=2)
        ax.grid(alpha=0.25, lw=0.4)
    h1, l1 = axs[0].get_legend_handles_labels()
    h2, l2 = axs[1].get_legend_handles_labels()
    fig.legend(h1 + h2, l1 + l2, loc="upper center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 1.13),
               fontsize=5.6, handlelength=1.6, columnspacing=0.9)
    fig.subplots_adjust(wspace=0.36)
    fig.savefig(FIG / "fig4_tradeoff.pdf", bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


def fig_shap():
    p = ROOT / "results" / "femto" / "shap_importance.csv"
    if not p.exists():
        return
    d = pd.read_csv(p).head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(W1, 1.35))
    cols = d.group.map({"horizontal": C["edge"], "vertical": C["cloud"], "DT context": C["hera"]})
    ax.barh(d.name.str.replace("DT:", "DT: "), d.mean_abs_shap_s, color=cols, height=0.7)
    ax.set_xlabel("mean |SHAP| on predicted RUL (s)")
    fig.savefig(FIG / "fig5_shap.pdf", bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


if __name__ == "__main__":
    fig1(); fig_traj(); fig_tradeoff(); fig_shap()
    for f in FIG.glob("*.pdf"):
        shutil.copy(f, ROOT / "paper" / "figures" / f.name)
    print("figures:", sorted(p.name for p in FIG.glob("*.pdf")))
