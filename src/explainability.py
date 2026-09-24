"""Human-in-the-Loop Maintenance Agent: explanations and approval requests.

Attributions: expected-gradients SHAP (shap.GradientExplainer) of the fog/cloud
prognostics model's median RUL output with respect to (i) the input feature
window and (ii) the Digital Twin context vector. The explanation card is a
deterministic template filled with model outputs; no language model is used,
so every quantity in the card is traceable to a validated predictor.
High-impact actions (SCHEDULE, URGENT) are emitted as approval requests only.

    python src/explainability.py    # writes results/shap_importance.csv, results/explanation_cards.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
from digital_twin import CONTEXT_NAMES, build_asset, feature_columns  # noqa: E402
from escalation import ACTION_NAMES, HIGH_IMPACT, decide  # noqa: E402
from prognostics import WindowData, build_model  # noqa: E402
from uncertainty import calibrate  # noqa: E402
from utils import FEATURES, ALPHA, PROC, R_MAX, RESULTS, W_CLOUD, fold_split  # noqa: E402


class MedianWrapper(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x, c):
        return self.m(x, c)[:, 1:2] * R_MAX


def shap_values(model, data: WindowData, idx, bg_idx, seed=0):
    import shap
    torch.manual_seed(seed)
    model.train(False)
    xb, cb, _ = data.batch(torch.as_tensor(bg_idx))
    xs, cs, _ = data.batch(torch.as_tensor(idx))
    ex = shap.GradientExplainer(MedianWrapper(model), [xb, cb])
    sv = ex.shap_values([xs, cs], nsamples=1000, rseed=seed)
    sx, sc = sv[0], sv[1]
    return np.asarray(sx).reshape(len(idx), W_CLOUD, -1), np.asarray(sc).reshape(len(idx), -1)


def explanation_card(asset, t, I, a, top, action, tau=3.0) -> dict:
    lo, med, hi = (float(v) for v in I)
    return {
        "asset": asset, "snapshot": int(t),
        "predicted_RUL_s": round(med), "interval_90pct_s": [round(lo), round(hi)],
        "uncertainty_width_s": round(hi - lo),
        "anomaly_evidence": {"score": round(float(a), 2), "alarm_threshold": round(float(tau), 2), "alarm": bool(a > tau)},
        "top_contributors": top,
        "recommended_action": ACTION_NAMES[int(action)],
        "requires_human_approval": int(action) in HIGH_IMPACT,
        "rationale": (f"The calibrated 90% interval [{lo:.0f}, {hi:.0f}] s "
                      + ("lies entirely below" if hi <= 600 else "reaches below" if lo <= 600 else "stays above")
                      + " the 600 s critical horizon; the recommendation is based on its lower bound."),
    }


def main(fold=2, seed=0, arch=None, n_explain=300):
    sel = json.loads((RESULTS / "selected_models.json").read_text())
    arch = arch or sel["cloud_model"]
    df = pd.read_csv(FEATURES)
    feats = feature_columns(df)
    assets = {b: build_asset(g.reset_index(drop=True), feats) for b, g in df.groupby("bearing")}
    train, calib, test = fold_split(fold, sorted(assets))
    ck = torch.load(RESULTS / "checkpoints" / f"fold{fold}_seed{seed}_{arch}+DT.pt", weights_only=False)
    model = build_model(arch, len(feats), len(CONTEXT_NAMES), width=ck["hp"]["width"])
    model.load_state_dict(ck["state"])
    dte = WindowData(assets, test, W_CLOUD, ck["ctx_stats"])
    p = np.load(RESULTS / "preds" / f"fold{fold}_seed{seed}_{arch}+DT.npz")
    m = np.load(RESULTS / "preds" / f"fold{fold}_meta.npz", allow_pickle=True)
    I = calibrate(p["q_cal"], np.minimum(m["cal_rul"], R_MAX), p["q_test"], ALPHA, "sym")
    from evaluation import tau_anomaly
    tau = tau_anomaly(fold)
    y = np.minimum(m["test_rul"], R_MAX)
    rng = np.random.default_rng(seed)
    # explain the degradation region (where maintenance decisions are made)
    cand = np.where(y < R_MAX)[0]
    idx = np.sort(rng.choice(cand, min(n_explain, len(cand)), replace=False))
    bg = rng.choice(len(y), 200, replace=False)
    sx, sc = shap_values(model, dte, idx, bg, seed)
    imp_feat = np.abs(sx).sum(1).mean(0)          # sum over time, mean |.| over samples
    imp_ctx = np.abs(sc).mean(0)
    imp = pd.DataFrame({"name": feats + [f"DT:{c}" for c in CONTEXT_NAMES],
                        "mean_abs_shap_s": np.concatenate([imp_feat, imp_ctx])})
    imp["group"] = np.where(imp.name.str.startswith("DT:"), "DT context",
                            np.where(imp.name.str.startswith("h_"), "horizontal", "vertical"))
    imp.sort_values("mean_abs_shap_s", ascending=False).to_csv(RESULTS / "shap_importance.csv", index=False)
    # additivity check of the attribution (expected-gradient SHAP is approximately additive)
    with torch.no_grad():
        xs, cs, _ = dte.batch(torch.as_tensor(idx))
        xb, cb, _ = dte.batch(torch.as_tensor(bg))
        f_x = MedianWrapper(model)(xs, cs).numpy().ravel()
        f_b = MedianWrapper(model)(xb, cb).numpy().mean()
    resid = (sx.sum((1, 2)) + sc.sum(1)) - (f_x - f_b)
    # explanation cards for the most critical explained snapshots
    act = decide(I[idx], m["test_a"][idx], tau)
    cards = []
    for j in np.argsort(I[idx, 0])[:3]:
        contrib = np.concatenate([sx[j].sum(0), sc[j]])
        names = feats + [f"DT:{c}" for c in CONTEXT_NAMES]
        top = [{"feature": names[k], "shap_s": round(float(contrib[k]), 1)} for k in np.argsort(-np.abs(contrib))[:3]]
        k = idx[j]
        cards.append(explanation_card(str(m["test_bearing"][k]), m["test_t"][k], I[k], m["test_a"][k], top, act[j], tau))
    out = {"fold": fold, "seed": seed, "model": f"{arch}+DT", "n_explained": int(len(idx)),
           "additivity_mae_s": float(np.abs(resid).mean()),
           "additivity_rel": float(np.abs(resid).mean() / (np.abs(f_x - f_b).mean() + 1e-9)), "cards": cards}
    (RESULTS / "explanation_cards.json").write_text(json.dumps(out, indent=2))
    print(imp.sort_values("mean_abs_shap_s", ascending=False).head(12).to_string(index=False))
    print(json.dumps(out, indent=1)[:2500])


if __name__ == "__main__":
    main()
