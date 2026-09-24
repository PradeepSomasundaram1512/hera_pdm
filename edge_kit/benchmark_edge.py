"""On-device benchmark for HERA-PdM (Raspberry Pi 4/5, Jetson, or any Linux/macOS box).
Requires only: python3, numpy, torch (CPU build). Copy the edge_kit folder to the device and run
    python3 benchmark_edge.py --threads 1 --reps 500
Measures, per exported model: single-snapshot inference latency (median/p95/p99, ms),
feature-extraction latency for one raw two-channel snapshot (FEMTO 2560 or XJTU-SY 32768 samples),
model file size and peak resident memory. Input signals are random noise of the correct shape:
timing does not depend on signal content. Results are printed and written to results_<hostname>.json.
"""
import argparse
import json
import platform
import resource
import socket
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
BANDS = [(0, 1000), (1000, 2000), (2000, 4000), (4000, 6000), (6000, 8000), (8000, 12800)]


def features(x, fs=25600.0):
    """Same 14 per-channel indicators as src/preprocessing.py (numpy only)."""
    out = []
    for ch in x.T:
        ch = ch - ch.mean()
        rms = np.sqrt(np.mean(ch ** 2)); a = np.abs(ch); peak = a.max()
        m2 = np.mean(ch ** 2); m3 = np.mean(ch ** 3); m4 = np.mean(ch ** 4)
        spec = np.abs(np.fft.rfft(ch)) ** 2; f = np.fft.rfftfreq(len(ch), 1 / fs); tot = spec.sum() + 1e-12
        out += [rms, m4 / (m2 ** 2 + 1e-12), m3 / (m2 ** 1.5 + 1e-12), ch.max() - ch.min(), peak / (rms + 1e-12),
                peak / (a.mean() + 1e-12), rms / (a.mean() + 1e-12), (f * spec).sum() / tot]
        out += [np.log10(spec[(f >= lo) & (f < hi)].sum() + 1e-12) for lo, hi in BANDS]
    return np.asarray(out, dtype=np.float32)


def stats(ts):
    ts = np.asarray(ts)
    return {"median_ms": float(np.median(ts)), "p95_ms": float(np.percentile(ts, 95)),
            "p99_ms": float(np.percentile(ts, 99))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--reps", type=int, default=500)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    man = json.loads((HERE / "models" / "manifest.json").read_text())
    res = {"host": socket.gethostname(), "machine": platform.machine(), "platform": platform.platform(),
           "processor": platform.processor(), "torch": torch.__version__, "threads": a.threads, "models": {}}
    rng = np.random.default_rng(0)
    for name, info in man.items():
        m = torch.jit.load(str(HERE / "models" / name)); m.eval()
        x = torch.randn(1, info["window"], 28); c = torch.randn(1, info["ctx_dim"])
        with torch.no_grad():
            for _ in range(50):
                m(x, c)
            ts = []
            for _ in range(a.reps):
                t0 = time.perf_counter(); m(x, c); ts.append((time.perf_counter() - t0) * 1e3)
        sig = rng.standard_normal((info["snapshot_samples"], 2))
        fts = []
        for _ in range(min(a.reps, 200)):
            t0 = time.perf_counter(); features(sig); fts.append((time.perf_counter() - t0) * 1e3)
        res["models"][name] = {**info, "inference": stats(ts), "feature_extraction": stats(fts),
                               "file_kb": (HERE / "models" / name).stat().st_size / 1024}
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    res["peak_rss_mb"] = rss / (1024 * 1024 if platform.system() == "Darwin" else 1024)
    out = HERE / f"results_{res['host']}.json"
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
