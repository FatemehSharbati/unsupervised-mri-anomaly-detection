"""Loading brain-MRI volumes and slicing them for the autoencoder."""

from typing import List, Tuple

import nibabel as nib
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Subset


class NiftiSliceDataset(Dataset):
    """2D slices from a NIfTI volume, z-score normalized per volume.

    Args:
        nifti_path: path to a .nii or .nii.gz volume.
        axis: volume axis treated as the slice axis
            (2 = axial for RAS-oriented data).
    """

    def __init__(self, nifti_path: str, axis: int = 2):
        img = nib.load(str(nifti_path))
        data = np.asarray(img.get_fdata(dtype=np.float32))
        mean = float(data.mean())
        std = float(data.std())
        data = (data - mean) / (std + 1e-8)
        slices = np.moveaxis(data, axis, 0)  # (n_slices, H, W)
        self.slices = slices[:, None, :, :].copy()  # add channel dim

    def __len__(self) -> int:
        return int(self.slices.shape[0])

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.from_numpy(self.slices[idx])


def train_val_split(
    dataset: Dataset, val_fraction: float = 0.1, seed: int = 42
) -> Tuple[Subset, Subset]:
    """Split a dataset into train/validation subsets by shuffled index."""
    indices: List[int] = list(range(len(dataset)))
    train_idx, val_idx = train_test_split(
        indices, test_size=val_fraction, random_state=seed
    )
    return Subset(dataset, train_idx), Subset(dataset, val_idx)
