"""Shared constants, dataset configurations, reproducibility helpers and folds.

The active dataset is chosen with the environment variable HERA_DATASET
(``femto`` | ``xjtu``). HERA_RMAX optionally overrides the RUL cap for the
sensitivity study; the decision horizons then scale with it
(H_CRIT = 0.2 R_MAX, H_PLAN = 0.6 R_MAX, the ratios of the main setting).
Every module reads the constants below at import time, so one process works
on one dataset/configuration.
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
FIGS = ROOT / "figures"


def _femto_folds():
    # Bearing-grouped 4-fold CV balanced over the three operating conditions (7/7/3 bearings).
    folds = [["Bearing1_1", "Bearing1_5", "Bearing2_1", "Bearing2_5", "Bearing3_3"],
             ["Bearing1_2", "Bearing1_6", "Bearing2_2", "Bearing2_6"],
             ["Bearing1_3", "Bearing1_7", "Bearing2_3", "Bearing2_7", "Bearing3_1"],
             ["Bearing1_4", "Bearing2_4", "Bearing3_2"]]
    calib = [["Bearing1_2", "Bearing2_2", "Bearing3_2"], ["Bearing1_3", "Bearing2_3", "Bearing3_3"],
             ["Bearing1_4", "Bearing2_4", "Bearing3_2"], ["Bearing1_5", "Bearing2_5", "Bearing3_3"]]
    inner = {0: ["Bearing1_7", "Bearing2_6"], 2: ["Bearing1_6", "Bearing2_2"]}
    return folds, calib, inner


def _xjtu_folds():
    # 5 folds, one bearing per operating condition in every fold (5/5/5 bearings).
    folds = [[f"Bearing{c}_{i}" for c in (1, 2, 3)] for i in range(1, 6)]
    calib = [folds[(k + 1) % 5] for k in range(5)]
    inner = {0: ["Bearing1_4", "Bearing2_4"], 2: ["Bearing1_1", "Bearing3_5"]}
    return folds, calib, inner


CONFIGS = {
    # snapshot period, RUL cap and horizons in seconds; windows and spans in snapshots
    "femto": dict(name="FEMTO/PRONOSTIA", snapshot_s=10.0, snapshot_samples=2560, r_max=3000.0, baseline_n=50, w_edge=16, w_cloud=64,
                  ewma_fast=6, ewma_slow=60, slope_k=30, folds=_femto_folds),
    "xjtu": dict(name="XJTU-SY", snapshot_s=60.0, snapshot_samples=32768, r_max=6000.0, baseline_n=10, w_edge=8, w_cloud=32,
                 ewma_fast=3, ewma_slow=15, slope_k=10, folds=_xjtu_folds),
}

DATASET = os.environ.get("HERA_DATASET", "femto")
_C = CONFIGS[DATASET]
SNAPSHOT_S = _C["snapshot_s"]
R_MAX = float(os.environ.get("HERA_RMAX", _C["r_max"]))   # piecewise-linear RUL cap (s)
H_CRIT = 0.2 * R_MAX                                        # critical horizon: urgent region (s)
H_PLAN = 0.6 * R_MAX                                        # planning horizon: schedule region (s)
BASELINE_N = _C["baseline_n"]                               # snapshots for the healthy baseline
W_EDGE, W_CLOUD = _C["w_edge"], _C["w_cloud"]
EWMA_FAST, EWMA_SLOW, SLOPE_K = _C["ewma_fast"], _C["ewma_slow"], _C["slope_k"]
FOLDS, CALIB, INNER = _C["folds"]()
ALPHA = 0.10                                                # nominal miscoverage of RUL intervals
QUANTILES = (ALPHA / 2, 0.5, 1 - ALPHA / 2)
SEEDS = (0, 1, 2)
N_INNER = 4                                                 # CV+ inner folds

FEATSET = os.environ.get("HERA_FEATSET", "base")          # "base" (28 features) | "env" (+12 envelope features)
_TAG = DATASET + ("" if FEATSET == "base" else f"_{FEATSET}") + \
    ("" if "HERA_RMAX" not in os.environ else f"_rmax{int(R_MAX)}")
RESULTS = ROOT / "results" / _TAG
FEATURES = PROC / (f"{DATASET}_features.csv.gz" if FEATSET == "base" else f"{DATASET}_features_{FEATSET}.csv.gz")


def n_features() -> int:
    import pandas as pd
    return sum(c[:2] in ("h_", "v_") for c in pd.read_csv(FEATURES, nrows=1).columns)


def set_seed(seed: int) -> None:
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def fold_split(fold: int, all_bearings: list[str]):
    test = FOLDS[fold]
    calib = CALIB[fold]
    assert not set(test) & set(calib)
    train = [b for b in all_bearings if b not in test and b not in calib]
    return train, calib, test


def inner_groups(fold: int, all_bearings: list[str], k: int = N_INNER) -> list[list[str]]:
    """Condition-balanced partition of the fold's non-test bearings into k groups (CV+)."""
    rest = sorted([b for b in all_bearings if b not in FOLDS[fold]], key=lambda b: (b[7], b))
    return [rest[i::k] for i in range(k)]


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=float))
