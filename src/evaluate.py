"""Evaluate a trained autoencoder: anomaly scores, maps, and a demo AUROC.

The demo injects small synthetic bright blobs into a few healthy slices and
reports AUROC for separating clean vs. corrupted slices. This is a toy sanity
check of the pipeline, NOT a clinical performance claim.

Example:
    python src/evaluate.py --checkpoint runs/exp1/checkpoint_best.pt \\
        --data /path/to/healthy_T1.nii.gz --outdir runs/exp1
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from src.data import NiftiSliceDataset
from src.model import BrainAutoencoder, anomaly_scores, reconstruction_error_map


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a trained brain-MRI autoencoder.")
    p.add_argument("--checkpoint", required=True, help="Path to a .pt checkpoint.")
    p.add_argument("--data", required=True, help="Path to a .nii/.nii.gz volume.")
    p.add_argument("--outdir", default="runs", help="Where to write outputs.")
    p.add_argument("--n-demo", type=int, default=20,
                   help="Slices to corrupt with synthetic blobs for the demo AUROC.")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def add_synthetic_blob(slice_chw: np.ndarray, rng: np.random.Generator,
                       radius: int = 6, intensity: float = 3.0) -> np.ndarray:
    """Add a small bright disk to a (1, H, W) slice (demo purposes only)."""
    corrupted = slice_chw.copy()
    height, width = corrupted.shape[1], corrupted.shape[2]
    cy = int(rng.integers(radius, height - radius))
    cx = int(rng.integers(radius, width - radius))
    yy, xx = np.ogrid[:height, :width]
    mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2
    corrupted[0][mask] += intensity
    return corrupted


def save_anomaly_figure(original, recon, error, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    for ax, img, title in zip(
        axes,
        [original[0], recon[0], error[0]],
        ["original", "reconstruction", "error map"],
    ):
        ax.imshow(img, cmap="gray")
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = BrainAutoencoder(input_size=int(ckpt["input_size"]))
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    dataset = NiftiSliceDataset(args.data)
    loader = DataLoader(dataset, batch_size=16)

    clean_scores = []
    with torch.no_grad():
        for batch in loader:
            recon = model(batch)
            err = reconstruction_error_map(batch, recon)
            clean_scores.append(anomaly_scores(err))
    clean_scores = torch.cat(clean_scores).numpy()
    print(f"Evaluated {len(clean_scores)} slices. "
          f"Mean anomaly score: {clean_scores.mean():.5f}")

    # --- demo: synthetic blobs -> AUROC (toy check, not a clinical claim) ---
    n_demo = min(args.n_demo, len(dataset))
    demo_scores, demo_labels = [], []
    with torch.no_grad():
        for i in range(n_demo):
            clean = dataset.slices[i]  # (1, H, W) numpy
            corrupted = add_synthetic_blob(clean, rng)
            for arr, label in ((clean, 0), (corrupted, 1)):
                t = torch.from_numpy(arr[None])  # add batch dim
                err = reconstruction_error_map(t, model(t))
                demo_scores.append(float(anomaly_scores(err)[0]))
                demo_labels.append(label)
    auroc = roc_auc_score(demo_labels, demo_scores)
    print(f"Demo AUROC on {n_demo} synthetic-blob slices (toy check): {auroc:.3f}")

    # --- save an example anomaly map for the most anomalous slice ---
    worst = int(np.argmax(clean_scores))
    with torch.no_grad():
        t = torch.from_numpy(dataset.slices[worst][None])
        recon = model(t)
        err = reconstruction_error_map(t, recon)
    save_anomaly_figure(
        t[0].numpy(), recon[0].numpy(), err[0].numpy(),
        outdir / "anomaly_map.png",
    )
    print(f"Wrote {outdir / 'anomaly_map.png'} (most anomalous slice #{worst})")


if __name__ == "__main__":
    main()
