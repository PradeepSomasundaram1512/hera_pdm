"""Per-architecture hyper-parameter selection on inner validation splits.

For two outer folds (utils.INNER), two of the fold's *training* bearings are held
out as inner validation; calibration and test bearings are never used, so the
conformal calibration and the reported test metrics are untouched by tuning.
Each fog/cloud architecture (with DT context) and each edge architecture gets
its own grid; selection criterion: mean validation pinball loss.
The selected setting of an architecture is also used for its no-DT and MSE
variants. Output: <RESULTS>/hparam_search.csv, <RESULTS>/hparams.json
Run: HERA_DATASET=femto python experiments/tune_hparams.py
"""
import itertools
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

CLOUD_GRID = {"epochs": [8, 25], "width": [32, 64, 128], "noise": [0.0, 0.2]}
EDGE_GRID = {"epochs": [8, 25], "width": [16], "noise": [0.0, 0.2]}
CLOUD_ARCHS = ["LSTM", "GRU", "TCN", "CNN-LSTM"]
EDGE_ARCHS = ["Edge-CNN", "Edge-GRU"]


def run(job):
    import numpy as np
    import pandas as pd
    import torch

    from digital_twin import CONTEXT_NAMES, build_asset, feature_columns
    from prognostics import WindowData, build_model, predict, train_model
    from utils import FEATURES, INNER, R_MAX, W_CLOUD, W_EDGE, fold_split, set_seed
    torch.set_num_threads(1)
    fold, arch, ep, width, noise = job
    use_ctx = not arch.startswith("Edge")
    df = pd.read_csv(FEATURES)
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
    return {"fold": fold, "arch": arch, "epochs": ep, "width": width, "noise": noise,
            "val_pinball": float(np.maximum(taus * d, (taus - 1) * d).mean()),
            "val_rmse_s": float(np.sqrt(((q[:, 1] - y) ** 2).mean()) * R_MAX),
            "val_raw_cov": float(((y >= q[:, 0]) & (y <= q[:, 2])).mean())}


def main():
    import pandas as pd

    from utils import INNER, RESULTS
    jobs = []
    for fold in INNER:
        for arch in CLOUD_ARCHS:
            jobs += [(fold, arch, *p) for p in itertools.product(*CLOUD_GRID.values())]
        for arch in EDGE_ARCHS:
            jobs += [(fold, arch, *p) for p in itertools.product(*EDGE_GRID.values())]
    # longest jobs first for better load balance
    jobs.sort(key=lambda j: -(j[2] * j[3] * (3 if j[1] == "TCN" else 1)))
    with ProcessPoolExecutor(9) as ex:
        rows = list(ex.map(run, jobs))
    df = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS / "hparam_search.csv", index=False)
    agg = df.groupby(["arch", "epochs", "width", "noise"]).mean(numeric_only=True).drop(columns="fold").reset_index()
    best = {}
    for arch, g in agg.groupby("arch"):
        b = g.sort_values("val_pinball").iloc[0]
        best[arch] = {"epochs": int(b.epochs), "width": int(b.width), "noise": float(b.noise), "weight_decay": 1e-3}
    (RESULTS / "hparams.json").write_text(json.dumps(best, indent=2))
    # architecture selection on the same inner-validation criterion (no calibration or test data)
    bestrow = agg.loc[agg.groupby("arch").val_pinball.idxmin()].set_index("arch")
    edge = bestrow.loc[EDGE_ARCHS].val_pinball.idxmin()
    cloud = bestrow.loc[CLOUD_ARCHS].val_pinball.idxmin()
    (RESULTS / "selected_models.json").write_text(json.dumps(
        {"edge_model": edge, "cloud_model": cloud, "criterion": "inner-validation pinball loss",
         "val_pinball": bestrow.val_pinball.round(5).to_dict()}, indent=2))
    print("selected:", edge, cloud)
    print(agg.sort_values(["arch", "val_pinball"]).round(4).to_string(index=False))
    print(json.dumps(best, indent=1))


if __name__ == "__main__":
    main()
