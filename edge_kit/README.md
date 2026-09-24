# HERA-PdM edge benchmark kit

Measures real on-device cost of the HERA-PdM edge and fog/cloud models (the paper currently reports host-CPU timings only).

## On the development machine
```bash
python edge_kit/export_models.py      # writes edge_kit/models/*.pt + manifest.json (already included)
```

## On the device (Raspberry Pi 4/5 with 64-bit OS, Jetson Nano/Orin, or any Linux box)
```bash
# copy the whole edge_kit/ folder to the device, then:
python3 -m pip install numpy torch     # CPU build of PyTorch for aarch64
python3 benchmark_edge.py --threads 1 --reps 500
```
Output: `results_<hostname>.json` with, per model, inference latency (median/p95/p99 ms), feature-extraction
latency for one raw snapshot, model file size and peak memory. Send that file back and the paper's latency
numbers and system discussion can be updated with measured embedded figures.

Inputs are random tensors of the correct shape; latency does not depend on signal content.
Host reference (Apple M2 Pro, 1 thread): FEMTO edge 0.14 ms, FEMTO cloud 1.05 ms, XJTU-SY edge 0.04 ms,
XJTU-SY cloud 0.52 ms; feature extraction 0.28 ms (2,560 samples) and 2.6 ms (32,768 samples).
