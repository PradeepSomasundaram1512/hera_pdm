"""HERA-PdM experiment pipeline.

    python run_experiments.py            # all stages
    python run_experiments.py --stage preprocess|train|analyze|figures
Stages are idempotent: finished training runs are skipped on re-execution.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from digital_twin import CONTEXT_NAMES  # noqa: E402
from utils import FOLDS, RESULTS, SEEDS  # noqa: E402

N_CTX = len(CONTEXT_NAMES)

CLOUD_ARCHS = ["LSTM", "GRU", "TCN", "CNN-LSTM"]
EDGE_ARCHS = ["Edge-CNN", "Edge-GRU"]
# Hyper-parameters selected on inner validation splits (experiments/tune_hparams.py,
# results/hparams.json); the GRU setting is shared by all fog/cloud architectures.
HP_PATH = ROOT / "results" / "hparams.json"


def configs():
    """(tag, arch, window, use_ctx, loss). Tags name the saved prediction files."""
    from utils import W_CLOUD, W_EDGE
    out = [(a, a, W_EDGE, False, "quantile") for a in EDGE_ARCHS]
    for a in CLOUD_ARCHS:
        out += [(f"{a}+DT", a, W_CLOUD, True, "quantile"),
                (f"{a}", a, W_CLOUD, False, "quantile"),
                (f"{a}+DT-point", a, W_CLOUD, True, "mse")]
    return out


def run_fold_seed(fold: int, seed: int) -> str:
    import pandas as pd
    import torch

    from digital_twin import build_asset, feature_columns
    from prognostics import WindowData, build_model, predict, train_model
    from utils import PROC, fold_split, set_seed

    torch.set_num_threads(2)
    out_dir = RESULTS / "preds"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(PROC / "femto_features.csv.gz")
    feats = feature_columns(df)
    assets = {b: build_asset(g.reset_index(drop=True), feats) for b, g in df.groupby("bearing")}
    train, calib, test = fold_split(fold, sorted(assets))
    meta_path = out_dir / f"fold{fold}_meta.npz"
    if not meta_path.exists():
        def meta(names):
            return {k: np.concatenate([assets[n][k] for n in names]) for k in ("rul", "a", "hi")} | {
                "bearing": np.concatenate([[n] * len(assets[n]["rul"]) for n in names]),
                "t": np.concatenate([np.arange(len(assets[n]["rul"])) for n in names])}
        mc, mt = meta(calib), meta(test)
        np.savez(meta_path, **{f"cal_{k}": v for k, v in mc.items()}, **{f"test_{k}": v for k, v in mt.items()})
    log_lines = []
    for tag, arch, W, use_ctx, loss in configs():
        path = out_dir / f"fold{fold}_seed{seed}_{tag}.npz"
        if path.exists():
            continue
        t0 = time.time()
        dtr = WindowData(assets, train, W)
        dca = WindowData(assets, calib, W, dtr.ctx_stats)
        dte = WindowData(assets, test, W, dtr.ctx_stats)
        hp = json.loads(HP_PATH.read_text())["Edge-CNN" if arch.startswith("Edge") else "GRU"]
        set_seed(seed)
        model = build_model(arch, len(feats), N_CTX if use_ctx else 0, width=hp["width"])
        train_model(model, dtr, use_ctx, loss=loss, epochs=hp["epochs"], noise=hp["noise"],
                    weight_decay=hp["weight_decay"], seed=seed)
        np.savez(path, q_cal=predict(model, dca, use_ctx), q_test=predict(model, dte, use_ctx))
        ck = RESULTS / "checkpoints"
        ck.mkdir(exist_ok=True)
        torch.save({"state": model.state_dict(), "ctx_stats": dtr.ctx_stats, "feats": feats, "hp": hp},
                   ck / f"fold{fold}_seed{seed}_{tag}.pt")
        log_lines.append(f"fold{fold} seed{seed} {tag}: {time.time() - t0:.0f}s")
    return "\n".join(log_lines) or f"fold{fold} seed{seed}: cached"


def stage_train(workers: int):
    jobs = [(f, s) for f in range(len(FOLDS)) for s in SEEDS]
    with ProcessPoolExecutor(workers) as ex:
        futs = {ex.submit(run_fold_seed, f, s): (f, s) for f, s in jobs}
        for fu in as_completed(futs):
            print(fu.result(), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["all", "preprocess", "train", "analyze", "figures"])
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    py = sys.executable
    if a.stage in ("all", "preprocess") and not (ROOT / "data/processed/femto_features.csv.gz").exists():
        subprocess.run([py, str(ROOT / "src/preprocessing.py")], check=True)
    if a.stage in ("all", "train"):
        if not HP_PATH.exists():
            subprocess.run([py, str(ROOT / "experiments/tune_hparams.py")], check=True)
        stage_train(a.workers)
    if a.stage in ("all", "analyze"):
        subprocess.run([py, str(ROOT / "src/evaluation.py")], check=True)
    if a.stage in ("all", "figures"):
        subprocess.run([py, str(ROOT / "experiments/make_figures.py")], check=True)
        subprocess.run([py, str(ROOT / "experiments/make_tables.py")], check=True)


if __name__ == "__main__":
    main()
