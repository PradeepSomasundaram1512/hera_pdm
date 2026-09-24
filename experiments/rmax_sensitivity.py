"""R_MAX sensitivity: retrain the selected edge/cloud models (CV+) and the cloud
point model with the RUL cap halved and doubled (horizons scale with the cap),
all three seeds (existing seed-0 runs are reused), then run the light evaluation. Outputs results/<ds>_rmax<R>/light_summary.json.

    python experiments/rmax_sensitivity.py
"""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
GRID = {"femto": [1500, 6000], "xjtu": [3000, 12000]}
SEEDS = ["0", "1", "2"]   # stages are idempotent: seed-0 runs are reused


def main():
    for ds, caps in GRID.items():
        sel = json.loads((ROOT / "results" / ds / "selected_models.json").read_text())
        for cap in caps:
            env = dict(os.environ, HERA_DATASET=ds, HERA_RMAX=str(cap))
            subprocess.run([PY, str(ROOT / "run_experiments.py"), "--stage", "train", "--seeds", *SEEDS, "--workers", "5",
                            "--tags", sel["cloud_model"] + "+DT-point"], env=env, check=True)
            subprocess.run([PY, str(ROOT / "run_experiments.py"), "--stage", "cvplus", "--seeds", *SEEDS, "--workers", "5"],
                           env=env, check=True)
            subprocess.run([PY, str(ROOT / "src" / "evaluation.py"), "--light", "--seeds", *SEEDS], env=env, check=True)


if __name__ == "__main__":
    main()
