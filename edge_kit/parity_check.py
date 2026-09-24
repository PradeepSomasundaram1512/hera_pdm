"""Functional portability check: run every exported model on fixed seeded inputs and write the quantile
outputs to parity_<tag>.json. Running it on two platforms and comparing the files shows whether the
TorchScript models give the same predictions there (max absolute difference).
    python3 parity_check.py --tag host        # on the development machine
    python3 parity_check.py --tag device      # on the target
    python3 parity_check.py --compare host device
"""
import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent


def run(tag):
    import torch
    torch.set_num_threads(1)
    man = json.loads((HERE / "models" / "manifest.json").read_text())
    g = torch.Generator().manual_seed(0)
    out = {}
    for name, info in man.items():
        m = torch.jit.load(str(HERE / "models" / name)); m.eval()
        x = torch.randn(32, info["window"], 28, generator=g); c = torch.randn(32, info["ctx_dim"], generator=g)
        with torch.no_grad():
            y = m(x, c)
        out[name] = y.numpy().astype(float).tolist()
    (HERE / f"parity_{tag}.json").write_text(json.dumps(out))
    print("wrote", HERE / f"parity_{tag}.json")


def compare(a, b):
    A = json.loads((HERE / f"parity_{a}.json").read_text()); B = json.loads((HERE / f"parity_{b}.json").read_text())
    for k in A:
        d = np.abs(np.asarray(A[k]) - np.asarray(B[k]))
        print(f"{k}: max |diff| = {d.max():.2e} (scaled output units), shape {np.asarray(A[k]).shape}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag"); ap.add_argument("--compare", nargs=2)
    a = ap.parse_args()
    compare(*a.compare) if a.compare else run(a.tag)
