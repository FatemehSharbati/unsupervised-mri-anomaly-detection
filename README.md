# Unsupervised MRI Anomaly Detection

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)
![License](https://img.shields.io/badge/license-MIT-green)

Unsupervised detection of abnormalities in brain MRI using a convolutional
autoencoder. The model is trained **only on healthy scans**; at inference time,
brain regions it fails to reconstruct (high reconstruction error) are flagged
as candidate anomalies. No pixel-level labels are needed for training.

## Method

A convolutional autoencoder learns a compressed representation of normal brain
anatomy from 2D axial slices:

- **Encoder:** three conv blocks (16 → 32 → 64 channels), each followed by 2×2
  max-pooling, then a dense bottleneck of 128 units.
- **Decoder:** mirrored transposed-convolution upsampling path with a final
  sigmoid, reconstructing the input slice.
- **Anomaly signal:** per-pixel squared reconstruction error, averaged over
  channels. A slice-level score is the mean of its error map.

Because the network has only ever seen healthy anatomy, pathologies such as
lesions or tumors tend to produce large reconstruction errors exactly where
they are — yielding both a detection score and a localization map.

## Quickstart

```bash
pip install -r requirements.txt

# Train on a healthy T1 volume (NIfTI), CPU-friendly
python src/train.py --data /path/to/healthy_T1.nii.gz --epochs 20 --outdir runs/exp1

# Evaluate: anomaly scores, anomaly maps, and a demo AUROC with synthetic blobs
python src/evaluate.py --checkpoint runs/exp1/checkpoint_best.pt \
    --data /path/to/healthy_T1.nii.gz --outdir runs/exp1
```

Or run the minimal end-to-end example:

```bash
python examples/quickstart.py /path/to/healthy_T1.nii.gz
```

## Repository structure

```
unsupervised-mri-anomaly-detection/
├── src/
│   ├── model.py      # BrainAutoencoder + reconstruction_error_map()
│   ├── data.py       # NiftiSliceDataset (nibabel, z-score per volume) + train/val split
│   ├── train.py      # training CLI (MSE loss, checkpoints to runs/)
│   └── evaluate.py   # anomaly scores, maps, demo AUROC with synthetic anomalies
├── examples/
│   └── quickstart.py # 5-line dataset -> train -> evaluate demo
├── tests/
│   └── test_smoke.py # CPU-only smoke tests (no downloads, no data files)
├── requirements.txt
└── README.md
```

## Datasets

This repo ships no data. Public healthy brain-MRI datasets suitable for training:

- **IXI** — ~600 healthy T1/T2/PD scans (London): https://brain-development.org/ixi-dataset/
- **OASIS** — open-access structural MRI across the adult lifespan: https://www.oasis-brains.org/

Download a T1 volume from either source and point `--data` at the `.nii` /
`.nii.gz` file.

## Results

No metric values are claimed here — anomaly-detection performance depends
entirely on the data, the anatomy covered, and the anomaly type. The
`evaluate.py` script includes a **synthetic** sanity check (small bright blobs
injected into healthy slices) that reports AUROC on that toy setup, purely to
verify the pipeline end to end. Report metrics measured on *your* data, with
the dataset and split clearly stated.

## References

- Baur, C., Denner, S., Wiestler, B., Navab, N., & Albarqouni, S. (2021).
  Autoencoders for unsupervised anomaly segmentation in brain MR images:
  a comparative study. *Medical Image Analysis*, 69, 101952.
- Chen, X., You, S., Tejada, K. C., & Konukoglu, E. (2020). Unsupervised
  lesion detection via image restoration with a normative prior.
  *Medical Image Analysis*, 64, 101713.
- Zimmerer, D., Kohl, S. A. A., Petersen, J., Isensee, F., & Maier-Hein, K. H.
  (2019). Context-encoding variational autoencoders for unsupervised anomaly
  detection. *Medical Imaging with Deep Learning (MIDL)*.

## Notes and limitations

- Educational/portfolio project — **not a clinical tool** and not validated
  for diagnostic use.
- Assumes roughly skull-stripped, bias-corrected volumes; raw scanner output
  benefits from a preprocessing step (see e.g. HD-BET, N4ITK).
- A 2D-slice model is a starting point; 3D or patch-based variants capture
  through-plane context better.

## License

MIT — see [LICENSE](LICENSE).
