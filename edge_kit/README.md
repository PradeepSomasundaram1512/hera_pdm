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

## ARM64 Linux container run (done; not a substitute for embedded hardware)
Functional-portability and constrained-timing check on the development Mac (Apple M2 Pro), via Colima (vz, aarch64 Ubuntu VM):
```bash
docker build -t hera-edge:arm64 - <<'EOF'
FROM python:3.13-slim
RUN pip install --no-cache-dir numpy==2.5.3 torch==2.14.0 --extra-index-url https://download.pytorch.org/whl/cpu
EOF
python parity_check.py --tag host
docker run --rm --cpus=1 --memory=1g -v "$PWD":/kit -w /kit hera-edge:arm64 python parity_check.py --tag container
python parity_check.py --compare host container
for i in 1 2 3; do docker run --rm --cpus=1 --memory=1g --hostname arm64-container-run$i -v "$PWD":/kit -w /kit \
  hera-edge:arm64 python benchmark_edge.py --threads 1 --reps 1000; done
docker run --rm --cpuset-cpus=0 --memory=1g --hostname arm64-container-pinned -v "$PWD":/kit -w /kit \
  hera-edge:arm64 python benchmark_edge.py --threads 1 --reps 1000
```
Results are in `sim_results/`. Outputs matched the host within 2e-7. Under the `--cpus=1` quota, p99 latency rose to about 50 ms
because of CFS throttling; pinned to one core (`--cpuset-cpus=0`), p99 stayed at or below 2.5 ms. An M2 core is much faster than a
Cortex-A72/A76, so these timings are not Raspberry Pi or Jetson numbers. Run `benchmark_edge.py` on the device for those.
