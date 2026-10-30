"""Train the autoencoder on healthy brain-MRI slices (CPU-friendly).

Example:
    python src/train.py --data /path/to/healthy_T1.nii.gz --epochs 20 --outdir runs/exp1
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data import NiftiSliceDataset, train_val_split
from src.model import BrainAutoencoder


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train a brain-MRI autoencoder on healthy slices."
    )
    p.add_argument("--data", required=True, help="Path to a healthy .nii/.nii.gz volume.")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument("--outdir", default="runs", help="Checkpoint directory.")
    return p.parse_args()


def run_epoch(model, loader, optimizer, loss_fn, train: bool):
    model.train(train)
    total, count = 0.0, 0
    for batch in loader:
        if train:
            optimizer.zero_grad()
        recon = model(batch)
        loss = loss_fn(recon, batch)
        if train:
            loss.backward()
            optimizer.step()
        total += loss.item() * batch.shape[0]
        count += batch.shape[0]
    return total / max(count, 1)


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    dataset = NiftiSliceDataset(args.data)
    _, _, height, width = dataset.slices.shape
    if height != width or height % 8 != 0:
        raise ValueError(
            f"Slice size must be square and divisible by 8, got {height}x{width}. "
            "Resample or crop the volume first."
        )

    train_ds, val_ds = train_val_split(dataset, val_fraction=args.val_split)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    model = BrainAutoencoder(input_size=height)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, loss_fn, train=True)
        val_loss = run_epoch(model, val_loader, optimizer, loss_fn, train=False)
        print(f"epoch {epoch:3d}/{args.epochs}  train {train_loss:.5f}  val {val_loss:.5f}")

        torch.save(
            {"epoch": epoch, "input_size": height,
             "model_state": model.state_dict(),
             "optimizer_state": optimizer.state_dict()},
            outdir / "checkpoint_last.pt",
        )
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {"epoch": epoch, "input_size": height,
                 "model_state": model.state_dict()},
                outdir / "checkpoint_best.pt",
            )
    print(f"Done. Best val loss {best_val:.5f}. Checkpoints in {outdir}")


if __name__ == "__main__":
    main()
