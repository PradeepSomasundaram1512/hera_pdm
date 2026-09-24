"""Export the selected edge and fog/cloud models (fold 0, seed 0) of both datasets
as TorchScript files for on-device benchmarking.  Run on the development machine:
    python edge_kit/export_models.py
Creates edge_kit/models/<dataset>_<role>.pt and edge_kit/models/manifest.json
"""
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from digital_twin import CONTEXT_NAMES  # noqa: E402
from prognostics import build_model  # noqa: E402

OUT = Path(__file__).parent / "models"
OUT.mkdir(exist_ok=True)
WINDOWS = {"femto": (16, 64), "xjtu": (8, 32)}
manifest = {}
for ds, (w_edge, w_cloud) in WINDOWS.items():
    sel = json.loads((ROOT / "results" / ds / "selected_models.json").read_text())
    for role, tag, W, dctx in [("edge", sel["edge_model"], w_edge, 0),
                               ("cloud", sel["cloud_model"] + "+DT", w_cloud, len(CONTEXT_NAMES))]:
        ck = torch.load(ROOT / "results" / ds / "checkpoints" / f"fold0_seed0_{tag}.pt", weights_only=False)
        m = build_model(tag.replace("+DT", ""), 28, dctx, width=ck["hp"]["width"])
        m.load_state_dict(ck["state"])
        m.eval()

        class Wrap(torch.nn.Module):
            def __init__(self, inner, use_ctx):
                super().__init__()
                self.inner, self.use_ctx = inner, use_ctx

            def forward(self, x, c):
                return self.inner(x, c if self.use_ctx else None)

        ts = torch.jit.trace(Wrap(m, dctx > 0), (torch.zeros(1, W, 28), torch.zeros(1, max(dctx, 1))))
        name = f"{ds}_{role}.pt"
        ts.save(str(OUT / name))
        manifest[name] = {"dataset": ds, "role": role, "model": tag, "window": W, "ctx_dim": max(dctx, 1),
                          "params": sum(p.numel() for p in m.parameters()),
                          "snapshot_samples": 2560 if ds == "femto" else 32768}
(OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
print(json.dumps(manifest, indent=1))
