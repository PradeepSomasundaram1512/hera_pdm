"""Shared constants, reproducibility helpers and cross-validation folds."""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGS = ROOT / "figures"

SNAPSHOT_S = 10.0          # one vibration snapshot every 10 s
R_MAX = 3000.0             # piecewise-linear RUL cap (s)
H_CRIT = 600.0             # critical horizon: urgent maintenance region (s)
H_PLAN = 1800.0            # planning horizon: schedule-maintenance region (s)
BASELINE_N = 50            # snapshots used to fit the asset-specific healthy baseline
W_EDGE = 16                # edge window length (snapshots)
W_CLOUD = 64               # fog/cloud prognostics window length (snapshots)
ALPHA = 0.10               # nominal miscoverage of RUL intervals
QUANTILES = (ALPHA / 2, 0.5, 1 - ALPHA / 2)
SEEDS = (0, 1, 2)

# Bearing-grouped 4-fold cross-validation, balanced over the three operating
# conditions (7/7/3 bearings). Each fold's held-out bearings are never used for
# training or calibration of that fold.
FOLDS = [
    ["Bearing1_1", "Bearing1_5", "Bearing2_1", "Bearing2_5", "Bearing3_3"],
    ["Bearing1_2", "Bearing1_6", "Bearing2_2", "Bearing2_6"],
    ["Bearing1_3", "Bearing1_7", "Bearing2_3", "Bearing2_7", "Bearing3_1"],
    ["Bearing1_4", "Bearing2_4", "Bearing3_2"],
]
# One calibration bearing per operating condition for every fold (drawn from the
# fold's non-test bearings; fixed in advance so that the split is reproducible).
CALIB = [
    ["Bearing1_2", "Bearing2_2", "Bearing3_2"],
    ["Bearing1_3", "Bearing2_3", "Bearing3_3"],
    ["Bearing1_4", "Bearing2_4", "Bearing3_2"],
    ["Bearing1_5", "Bearing2_5", "Bearing3_3"],
]


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


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=float))
