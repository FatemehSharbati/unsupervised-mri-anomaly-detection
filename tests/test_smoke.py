"""Smoke tests: CPU-only, no downloads, no data files.

Run with:  python tests/test_smoke.py   (or: python -m pytest tests/)
"""

from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn

from src.data import NiftiSliceDataset, train_val_split
from src.model import BrainAutoencoder, anomaly_scores, reconstruction_error_map


def test_forward():
    model = BrainAutoencoder(input_size=64)
    x = torch.randn(2, 1, 64, 64)
    y = model(x)
    assert y.shape == x.shape, f"shape mismatch: {y.shape} vs {x.shape}"
    assert bool(((y >= 0) & (y <= 1)).all()), "sigmoid output out of [0, 1]"


def test_error_map():
    x = torch.randn(2, 1, 32, 32)
    err = reconstruction_error_map(x, x)
    assert err.shape == (2, 1, 32, 32)
    assert float(err.max()) == 0.0, "identical inputs must give zero error"
    scores = anomaly_scores(err)
    assert scores.shape == (2,)
    assert float(scores.max()) == 0.0


def test_train_step():
    model = BrainAutoencoder(input_size=64)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    x = torch.randn(4, 1, 64, 64)
    opt.zero_grad()
    loss = loss_fn(model(x), x)
    loss.backward()
    opt.step()
    assert loss.item() > 0, "loss must be positive"


def test_nifti_dataset(tmp_dir: Path):
    rng = np.random.default_rng(0)
    vol = rng.normal(size=(32, 32, 10)).astype(np.float32)
    path = tmp_dir / "vol.nii.gz"
    nib.save(nib.Nifti1Image(vol, np.eye(4)), str(path))

    ds = NiftiSliceDataset(str(path), axis=2)
    assert len(ds) == 10, f"expected 10 slices, got {len(ds)}"
    s = ds[0]
    assert s.shape == (1, 32, 32), f"unexpected slice shape {s.shape}"
    # z-score normalization: whole-volume mean ~0
    assert abs(float(ds.slices.mean())) < 1e-4

    train_ds, val_ds = train_val_split(ds, val_fraction=0.2, seed=0)
    assert len(train_ds) + len(val_ds) == 10


if __name__ == "__main__":
    test_forward()
    test_error_map()
    test_train_step()
    with tempfile.TemporaryDirectory() as d:
        test_nifti_dataset(Path(d))
    print("All smoke tests passed.")
