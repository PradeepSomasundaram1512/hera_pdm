"""Robustness of the key policy comparisons to (a) training seeds and (b) the architecture choice.

(a) Selected edge/cloud models with five seeds (0-4) instead of three.
(b) Runner-up edge and fog/cloud architectures (second-lowest inner-validation pinball loss in
    selected_models.json), three seeds. Hyper-parameters are the per-architecture settings already
    chosen on inner validation bearings; nothing is re-tuned and no test data are used for selection.

For every configuration the script recomputes, with the unchanged evaluation code, the pooled unsafe
rate, false maintenance (FM) and escalation of HERA, the matched width gate, the calibrated edge-only and
cloud-only policies and the point-threshold policy, plus per-bearing paired Wilcoxon tests (unit = bearing,
seeds averaged first, zero differences dropped, two-sided). Output: results/robustness_seeds_arch.json

    python experiments/robustness_seeds_arch.py train     # CV+ / point-model runs (idempotent)
    python experiments/robustness_seeds_arch.py analyze_all
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
OUT = ROOT / "results" / "robustness_seeds_arch.json"


def runner_up(ds):
    sel = json.loads((ROOT / "results" / ds / "selected_models.json").read_text())
    vp = sel["val_pinball"]
    edge = sorted([a for a in vp if a.startswith("Edge") and a != sel["edge_model"]], key=vp.get)[0]
    cloud = sorted([a for a in vp if not a.startswith("Edge") and a != sel["cloud_model"]], key=vp.get)[0]
    return sel, edge, cloud


def train():
    code = r'''
import sys, json
sys.path.insert(0, "ROOT"); sys.path.insert(0, "ROOT/src")
import run_experiments as R
from concurrent.futures import ProcessPoolExecutor, as_completed
jobs = json.loads(sys.argv[1])
with ProcessPoolExecutor(5) as ex:
    futs = [ex.submit(R.run_cvplus, f, s, tags) for f, s, tags in jobs]
    for fu in as_completed(futs):
        print(fu.result(), flush=True)
'''.replace("ROOT", str(ROOT))
    for ds in ("xjtu", "femto"):
        sel, e2, c2 = runner_up(ds)
        env = dict(os.environ, HERA_DATASET=ds)
        n_folds = {"femto": 4, "xjtu": 5}[ds]
        # (a) point model for seeds 3-4 (split-trained, as in the main pipeline)
        subprocess.run([PY, str(ROOT / "run_experiments.py"), "--stage", "train", "--seeds", "3", "4",
                        "--workers", "5", "--tags", sel["cloud_model"] + "+DT-point"], env=env, check=True)
        jobs = [(f, s, [sel["edge_model"], sel["cloud_model"] + "+DT"]) for f in range(n_folds) for s in (3, 4)]
        # (b) runner-up architectures, seeds 0-2
        jobs += [(f, s, [e2, c2 + "+DT"]) for f in range(n_folds) for s in (0, 1, 2)]
        print(f"== {ds}: runner-up edge {e2}, cloud {c2}; {len(jobs)} CV+ jobs", flush=True)
        subprocess.run([PY, "-c", code, json.dumps(jobs)], env=env, check=True, cwd=ROOT)


def analyze():
    from scipy.stats import wilcoxon
    out = {}
    for ds in os.environ.get("ROBUST_DS", "femto,xjtu").split(","):
        os.environ["HERA_DATASET"] = ds
        for m in [k for k in list(sys.modules) if k in ("utils", "evaluation", "escalation", "digital_twin")]:
            del sys.modules[m]
        sys.path.insert(0, str(ROOT / "src"))
        import evaluation as ev
        from escalation import CONTINUE, INSPECT, SCHEDULE, URGENT
        sel, e2, c2 = runner_up(ds)
        configs = {"selected_3seeds": (sel["edge_model"], sel["cloud_model"], [0, 1, 2]),
                   "selected_5seeds": (sel["edge_model"], sel["cloud_model"], [0, 1, 2, 3, 4]),
                   "runnerup_3seeds": (e2, c2, [0, 1, 2])}
        orig_load = ev.load
        for cname, (edge, cloud, seeds) in configs.items():
            ev.EDGE_TAG, ev.CLOUD = edge, cloud
            ev.CV_TAGS = {edge, cloud + "+DT", cloud}

            def load(tag, seed, calib="auto", _c=cloud):
                # the no-DT cloud model only feeds the HERA-no-DT ablation, which is not evaluated here
                if tag == _c and not (ev.CVP / f"fold0_seed{seed}_{tag}.npz").exists() and \
                        not (ev.PRED / f"fold0_seed{seed}_{tag}.npz").exists():
                    tag = _c + "+DT"
                return orig_load(tag, seed, calib)
            ev.load = load
            rng = np.random.default_rng(0)
            names = {"hera": "HERA-full", "width": "Width gate (matched)", "edge": "Always-edge",
                     "cloud": "Always-cloud", "point": "Cloud point+threshold"}
            pooled, pb = {k: [] for k in names}, []
            for s in seeds:
                D = ev.policy_actions(s, rng=rng)
                crit, healthy = D["y"] <= ev.H_CRIT, D["y"] > ev.H_PLAN
                pol = {p[0]: p for p in ev.POLICIES}
                acts = {}
                for key, name in names.items():
                    _, ek, ae, ac, ie, ic, _m = pol[name]
                    act, esc, _I = ev.resolve(D, name, ek, ae, ac, ie, ic)
                    acts[key] = (act, esc)
                    nm = (act == CONTINUE) | (act == INSPECT)
                    pooled[key].append([nm[crit].mean(), np.isin(act, [SCHEDULE, URGENT])[healthy].mean(), esc.mean()])
                for bn in np.unique(D["b"]):
                    m = D["b"] == bn
                    row = {"bearing": bn}
                    for key, (act, esc) in acts.items():
                        nm = (act == CONTINUE) | (act == INSPECT)
                        row["unsafe_" + key] = nm[m & crit].mean() if (m & crit).any() else np.nan
                        row["fm_" + key] = np.isin(act, [SCHEDULE, URGENT])[m & healthy].mean() \
                            if (m & healthy).any() else np.nan
                    pb.append(row)
            import pandas as pd
            pbd = pd.DataFrame(pb).groupby("bearing").mean()
            tests = {}
            for met in ("unsafe", "fm"):
                for other in ("point", "width", "edge", "cloud"):
                    d = (pbd[f"{met}_hera"] - pbd[f"{met}_{other}"]).dropna()
                    nz = d[d.abs() > 1e-12]
                    tests[f"{met}_hera_vs_{other}"] = {
                        "n_bearings": int(len(d)), "n_nonzero": int(len(nz)), "mean_diff_pp": float(100 * d.mean()),
                        "p": float(wilcoxon(nz).pvalue) if len(nz) >= 5 else None}
            out[f"{ds}/{cname}"] = {
                "edge": edge, "cloud": cloud, "seeds": seeds,
                "pooled_pct": {k: dict(zip(["unsafe", "fm", "esc"], (100 * np.mean(v, 0)).round(2).tolist()))
                               for k, v in pooled.items()},
                "tests": tests}
            ev.load = orig_load
            print(ds, cname, json.dumps(out[f"{ds}/{cname}"]["pooled_pct"]), flush=True)
    prev = json.loads(OUT.read_text()) if OUT.exists() else {}
    OUT.write_text(json.dumps(prev | out, indent=1))
    print("wrote", OUT)


def analyze_all():
    """One process per dataset: modules cache dataset-specific constants (R_MAX, horizons) at import."""
    OUT.unlink(missing_ok=True)
    for ds in ("femto", "xjtu"):
        subprocess.run([PY, __file__, "analyze"], env=dict(os.environ, ROBUST_DS=ds), check=True)


if __name__ == "__main__":
    {"train": train, "analyze": analyze, "analyze_all": analyze_all}[sys.argv[1]]()
