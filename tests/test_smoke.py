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
from torch.utils.data import Dataset

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


def test_anomaly_scores_nonzero():
    # per-slice means of a known error map: 1.0 everywhere -> 1.0, 4.0 -> 4.0
    err = torch.stack([torch.ones(1, 4, 4), torch.full((1, 4, 4), 4.0)])
    scores = anomaly_scores(err)
    assert scores.shape == (2,), f"unexpected scores shape {scores.shape}"
    assert torch.allclose(scores, torch.tensor([1.0, 4.0])), f"scores {scores}"


class _TinyDataset(Dataset):
    """Minimal stand-in dataset for testing train_val_split."""

    def __init__(self, n: int):
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.tensor([idx])


def test_train_val_split_deterministic():
    ds = _TinyDataset(10)
    train1, val1 = train_val_split(ds, val_fraction=0.3, seed=0)
    train2, val2 = train_val_split(ds, val_fraction=0.3, seed=0)
    assert train1.indices == train2.indices, "split must be seed-deterministic"
    assert val1.indices == val2.indices, "split must be seed-deterministic"
    assert set(train1.indices) | set(val1.indices) == set(range(10)), \
        "split must cover every sample"
    assert not (set(train1.indices) & set(val1.indices)), \
        "train and val must be disjoint"


def test_brain_autoencoder_input_size_validation():
    try:
        BrainAutoencoder(input_size=60)
    except ValueError:
        pass
    else:
        raise AssertionError("input_size=60 (not divisible by 8) must raise ValueError")
    BrainAutoencoder(input_size=64)  # must not raise


def test_nifti_dataset(tmp_path: Path):
    rng = np.random.default_rng(0)
    vol = rng.normal(size=(32, 32, 10)).astype(np.float32)
    path = tmp_path / "vol.nii.gz"
    nib.save(nib.Nifti1Image(vol, np.eye(4)), str(path))

    ds = NiftiSliceDataset(str(path), axis=2)
    assert len(ds) == 10, f"expected 10 slices, got {len(ds)}"
    s = ds[0]
    assert s.shape == (1, 32, 32), f"unexpected slice shape {s.shape}"
    # z-score normalization: whole-volume mean ~0
    assert abs(float(ds.slices.mean())) < 1e-4

    train_ds, val_ds = train_val_split(ds, val_fraction=0.2, seed=0)
    assert len(train_ds) + len(val_ds) == 10


def test_nifti_dataset_rejects_4d_volume(tmp_path: Path):
    # A 4D volume (e.g. fMRI time series) must fail loudly instead of
    # silently producing 5D "slices" that break the model downstream.
    rng = np.random.default_rng(0)
    vol = rng.normal(size=(16, 16, 8, 4)).astype(np.float32)
    path = tmp_path / "vol4d.nii.gz"
    nib.save(nib.Nifti1Image(vol, np.eye(4)), str(path))
    try:
        NiftiSliceDataset(str(path))
    except ValueError:
        pass
    else:
        raise AssertionError("4D volume must raise ValueError")


def test_nifti_dataset_rejects_bad_axis(tmp_path: Path):
    rng = np.random.default_rng(0)
    vol = rng.normal(size=(16, 16, 8)).astype(np.float32)
    path = tmp_path / "vol.nii.gz"
    nib.save(nib.Nifti1Image(vol, np.eye(4)), str(path))
    try:
        NiftiSliceDataset(str(path), axis=3)
    except ValueError:
        pass
    else:
        raise AssertionError("axis=3 must raise ValueError")
    ds = NiftiSliceDataset(str(path), axis=-1)  # negative axis is valid
    assert len(ds) == 8


if __name__ == "__main__":
    test_forward()
    test_error_map()
    test_train_step()
    with tempfile.TemporaryDirectory() as d:
        test_nifti_dataset(Path(d))
    print("All smoke tests passed.")
