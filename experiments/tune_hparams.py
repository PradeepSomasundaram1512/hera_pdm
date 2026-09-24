"""Hyper-parameter selection on inner validation splits.

For folds 0 and 2, two of the fold's *training* bearings are held out as an
inner validation set; calibration and test bearings are never touched, so the
conformal calibration and the reported test metrics remain untouched by tuning.
Selection criterion: mean validation pinball loss (lower is better).
Output: results/hparam_search.csv, results/hparams.json
"""
import itertools
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

INNER = {0: ["Bearing1_7", "Bearing2_6"], 2: ["Bearing1_6", "Bearing2_2"]}
GRID = {"epochs": [8, 20], "width": [32, 64], "noise": [0.0, 0.2]}


def run(job):
    import numpy as np
    import pandas as pd
    import torch

    from digital_twin import CONTEXT_NAMES, build_asset, feature_columns
    from prognostics import WindowData, build_model, predict, train_model
    from utils import PROC, R_MAX, W_CLOUD, W_EDGE, fold_split, set_seed
    torch.set_num_threads(1)
    fold, arch, use_ctx, ep, width, noise = job
    df = pd.read_csv(PROC / "femto_features.csv.gz")
    feats = feature_columns(df)
    assets = {b: build_asset(g.reset_index(drop=True), feats) for b, g in df.groupby("bearing")}
    train, _, _ = fold_split(fold, sorted(assets))
    val = INNER[fold]
    tr = [b for b in train if b not in val]
    W = W_EDGE if arch.startswith("Edge") else W_CLOUD
    dtr = WindowData(assets, tr, W)
    dva = WindowData(assets, val, W, dtr.ctx_stats)
    set_seed(0)
    m = build_model(arch, len(feats), len(CONTEXT_NAMES) if use_ctx else 0, width=width)
    train_model(m, dtr, use_ctx, epochs=ep, noise=noise, weight_decay=1e-3, seed=0)
    q = predict(m, dva, use_ctx) / R_MAX
    y = dva.y.numpy()
    taus = np.array([0.05, 0.5, 0.95])
    d = y[:, None] - q
    return {"fold": fold, "arch": arch, "ctx": use_ctx, "epochs": ep, "width": width, "noise": noise,
            "val_pinball": float(np.maximum(taus * d, (taus - 1) * d).mean()),
            "val_rmse_s": float(np.sqrt(((q[:, 1] - y) ** 2).mean()) * R_MAX),
            "val_raw_cov": float(((y >= q[:, 0]) & (y <= q[:, 2])).mean())}


def main():
    import pandas as pd
    jobs = []
    for fold in INNER:
        for ep, width, noise in itertools.product(*GRID.values()):
            jobs.append((fold, "GRU", True, ep, width, noise))
            jobs.append((fold, "GRU", False, ep, width, noise))
        for ep, noise in itertools.product(GRID["epochs"], GRID["noise"]):
            jobs.append((fold, "Edge-CNN", False, ep, 16, noise))
    with ProcessPoolExecutor(9) as ex:
        rows = list(ex.map(run, jobs))
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results" / "hparam_search.csv", index=False)
    agg = df.groupby(["arch", "ctx", "epochs", "width", "noise"]).mean(numeric_only=True).drop(columns="fold")
    print(agg.sort_values("val_pinball").round(4).to_string())
    best = {}
    for arch, g in agg.reset_index().groupby("arch"):
        b = g.sort_values("val_pinball").iloc[0]
        best[arch] = {"epochs": int(b.epochs), "width": int(b.width), "noise": float(b.noise), "weight_decay": 1e-3}
    (ROOT / "results" / "hparams.json").write_text(json.dumps(best, indent=2))
    print(best)


if __name__ == "__main__":
    main()
