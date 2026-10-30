"""Minimal end-to-end example: dataset -> train -> evaluate.

Usage:
    python examples/quickstart.py /path/to/healthy_T1.nii.gz

Runs 3 quick training epochs on CPU and then the evaluation demo.
"""

import subprocess
import sys

from src.data import NiftiSliceDataset, train_val_split

DATA = sys.argv[1] if len(sys.argv) > 1 else "data/healthy_T1.nii.gz"
OUTDIR = "runs/quickstart"

# 1. dataset: load a healthy volume, slice it, split train/val
dataset = NiftiSliceDataset(DATA)
train_ds, val_ds = train_val_split(dataset, val_fraction=0.1)
print(f"Loaded {len(dataset)} slices ({len(train_ds)} train / {len(val_ds)} val)")

# 2. train: a few epochs of the autoencoder on healthy slices
subprocess.run(
    [sys.executable, "src/train.py", "--data", DATA,
     "--epochs", "3", "--outdir", OUTDIR],
    check=True,
)

# 3. evaluate: anomaly scores, maps, and the synthetic-blob demo AUROC
subprocess.run(
    [sys.executable, "src/evaluate.py", "--checkpoint",
     f"{OUTDIR}/checkpoint_best.pt", "--data", DATA, "--outdir", OUTDIR],
    check=True,
)

print(f"Done. See {OUTDIR}/anomaly_map.png")
