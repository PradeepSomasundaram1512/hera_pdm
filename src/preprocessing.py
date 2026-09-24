"""Preprocessing for FEMTO-ST / PRONOSTIA (IEEE PHM 2012) and XJTU-SY bearings.

Usage: python src/preprocessing.py --dataset femto|xjtu

XJTU-SY (Wang et al., IEEE Trans. Reliab. 2020): 15 bearings run to failure
under 35 Hz/12 kN, 37.5 Hz/11 kN and 40 Hz/10 kN; every minute a 1.28 s
snapshot (32768 samples @ 25.6 kHz) of horizontal and vertical acceleration.
The same 28 condition indicators as for FEMTO are extracted.

FEMTO-ST / PRONOSTIA:

Every 10 s the test rig records a 0.1 s snapshot (2560 samples @ 25.6 kHz) of
horizontal and vertical acceleration. Each snapshot is reduced to a vector of
time- and frequency-domain condition indicators; the resulting sequence
x_1..x_T (one row per snapshot) is the multivariate sensor stream seen by the
edge agent. Run-to-failure records come from Learning_set (6 bearings) and
Full_Test_Set (11 bearings); Test_set is used only to recover the official
challenge truncation point of each test bearing.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "FEMTO"
PROC = ROOT / "data" / "processed"
FS = 25600.0
SNAPSHOT_PERIOD_S = 10.0
BANDS_HZ = [(0, 1000), (1000, 2000), (2000, 4000), (4000, 6000), (6000, 8000), (8000, 12800)]

# Operating conditions documented in the PHM 2012 challenge description.
CONDITIONS = {1: (1800, 4000), 2: (1650, 4200), 3: (1500, 5000)}  # (rpm, radial load N)
TRAIN_SPLIT = ["Bearing1_1", "Bearing1_2", "Bearing2_1", "Bearing2_2", "Bearing3_1", "Bearing3_2"]
TEST_SPLIT = ["Bearing1_3", "Bearing1_4", "Bearing1_5", "Bearing1_6", "Bearing1_7", "Bearing2_3",
              "Bearing2_4", "Bearing2_5", "Bearing2_6", "Bearing2_7", "Bearing3_3"]


def channel_features(x: np.ndarray, prefix: str) -> dict:
    x = x - x.mean()
    rms = np.sqrt(np.mean(x ** 2))
    absx = np.abs(x)
    peak = absx.max()
    spec = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(len(x), 1 / FS)
    tot = spec.sum() + 1e-12
    f = {
        f"{prefix}_rms": rms,
        f"{prefix}_kurt": stats.kurtosis(x, fisher=False),
        f"{prefix}_skew": stats.skew(x),
        f"{prefix}_p2p": x.max() - x.min(),
        f"{prefix}_crest": peak / (rms + 1e-12),
        f"{prefix}_impulse": peak / (absx.mean() + 1e-12),
        f"{prefix}_shape": rms / (absx.mean() + 1e-12),
        f"{prefix}_centroid": float((freqs * spec).sum() / tot),
    }
    for lo, hi in BANDS_HZ:
        m = (freqs >= lo) & (freqs < hi)
        f[f"{prefix}_band{lo // 1000}_{hi // 1000}k"] = float(np.log10(spec[m].sum() + 1e-12))
    return f


def read_snapshot(path: Path) -> np.ndarray:
    sep = ";" if ";" in path.open().readline() else ","
    arr = pd.read_csv(path, sep=sep, header=None).to_numpy()
    return arr[:, 4:6].astype(float)  # horizontal, vertical acceleration (g)


def process_bearing(args) -> pd.DataFrame:
    name, folder = args
    files = sorted(folder.glob("acc_*.csv"))
    rows = []
    for i, fp in enumerate(files):
        sig = read_snapshot(fp)
        feats = {"snapshot": i}
        feats.update(channel_features(sig[:, 0], "h"))
        feats.update(channel_features(sig[:, 1], "v"))
        rows.append(feats)
    df = pd.DataFrame(rows)
    T = len(df)
    df["bearing"] = name
    df["condition"] = int(name[7])
    df["time_s"] = df["snapshot"] * SNAPSHOT_PERIOD_S
    # RUL label: time remaining until the last recorded snapshot (failure = end of run).
    df["rul_s"] = (T - 1 - df["snapshot"]) * SNAPSHOT_PERIOD_S
    df["life_frac"] = df["snapshot"] / (T - 1)
    return df


XJTU_RAW = ROOT / "data" / "raw" / "XJTU"
XJTU_CONDITIONS = {1: ("35Hz12kN", 2100, 12000), 2: ("37.5Hz11kN", 2250, 11000), 3: ("40Hz10kN", 2400, 10000)}


def read_xjtu(path: Path) -> np.ndarray:
    return pd.read_csv(path).to_numpy(dtype=float)[:, :2]   # horizontal, vertical


def process_xjtu_bearing(args) -> pd.DataFrame:
    name, folder = args
    files = sorted(folder.glob("*.csv"), key=lambda p: int(p.stem))   # numeric order (1.csv, 2.csv, ...)
    rows = []
    for i, fp in enumerate(files):
        sig = read_xjtu(fp)
        feats = {"snapshot": i}
        feats.update(channel_features(sig[:, 0], "h"))
        feats.update(channel_features(sig[:, 1], "v"))
        rows.append(feats)
    df = pd.DataFrame(rows)
    T = len(df)
    df["bearing"] = name
    df["condition"] = int(name[7])
    df["time_s"] = df["snapshot"] * 60.0
    df["rul_s"] = (T - 1 - df["snapshot"]) * 60.0
    df["life_frac"] = df["snapshot"] / (T - 1)
    return df


def main_xjtu(workers: int):
    roots = list(XJTU_RAW.rglob("35Hz12kN"))
    assert roots, "XJTU-SY not extracted under data/raw/XJTU"
    base = roots[0].parent
    jobs = []
    for c, (sub, _, _) in XJTU_CONDITIONS.items():
        for i in range(1, 6):
            jobs.append((f"Bearing{c}_{i}", base / sub / f"Bearing{c}_{i}"))
    with ProcessPoolExecutor(workers) as ex:
        dfs = list(ex.map(process_xjtu_bearing, jobs))
    data = pd.concat(dfs, ignore_index=True)
    data.to_csv(PROC / "xjtu_features.csv.gz", index=False)
    summary = data.groupby("bearing").agg(condition=("condition", "first"), snapshots=("snapshot", "size"),
                                          life_s=("time_s", "max")).reset_index()
    summary.to_csv(PROC / "xjtu_dataset_summary.csv", index=False)
    print(summary.to_string(index=False))


def truncation_points() -> dict:
    """Official challenge cut-off index for each test bearing (len(Test_set))."""
    return {b: len(list((RAW / "Test_set" / b).glob("acc_*.csv"))) for b in TEST_SPLIT}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dataset", default="femto", choices=["femto", "xjtu"])
    a = ap.parse_args()
    PROC.mkdir(parents=True, exist_ok=True)
    if a.dataset == "xjtu":
        return main_xjtu(a.workers)
    jobs = [(b, RAW / "Learning_set" / b) for b in TRAIN_SPLIT] + \
           [(b, RAW / "Full_Test_Set" / b) for b in TEST_SPLIT]
    with ProcessPoolExecutor(a.workers) as ex:
        dfs = list(ex.map(process_bearing, jobs))
    data = pd.concat(dfs, ignore_index=True)
    data.to_csv(PROC / "femto_features.csv.gz", index=False)
    cut = truncation_points()
    pd.Series(cut, name="truncation_index").to_csv(PROC / "test_truncation.csv")
    summary = data.groupby("bearing").agg(condition=("condition", "first"), snapshots=("snapshot", "size"),
                                          life_s=("time_s", "max")).reset_index()
    summary["split"] = np.where(summary.bearing.isin(TRAIN_SPLIT), "train", "test")
    summary["official_cut"] = summary.bearing.map(cut)
    summary.to_csv(PROC / "dataset_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
