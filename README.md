# Vehicle Trajectory Forecasting (TCN Encoder + Conditional VAE)

This repository trains and validates a multimodal vehicle trajectory forecaster.

It is built to predict **multiple plausible futures** (not a single deterministic path) from:
- 50 observed timesteps of graph/node motion features
- ego-vehicle history features

The model outputs 6 trajectory hypotheses and is evaluated with minADE / minFDE.

## 1. What You Will Do

1. Clone the repo.
2. Create the Pixi environment.
3. Verify runtime (CUDA or CPU).
4. Download train/val `.pt` datasets into `datasets/`.
5. Run training and watch per-batch + per-epoch logs.
6. Run validation from the saved checkpoint and see final metrics.

If you follow the commands exactly, you will reach a complete local run.

## 2. Prerequisites

- OS: Linux (this repo is pinned for `linux-64` in `pixi.toml`)
- Git
- Internet access (for Pixi package resolution + Hugging Face dataset download)
- Optional NVIDIA GPU (training also runs on CPU, slower)

## 3. Clone Repository

```bash
git clone https://github.com/aryanipb/tcn-cvae-trajectory-forecasting.git
cd tcn-cvae-trajectory-forecasting
```

## 4. Install Pixi

Install Pixi (official installer):

```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

Restart shell (or source your shell config), then verify:

```bash
pixi --version
```

## 5. Create Project Environment (Pixi)

From project root:

```bash
pixi install
```

Then run bootstrap task used by this repo:

```bash
pixi run bootstrap
```

This installs/aligns Python packages from `requirements.txt` inside the Pixi environment.

## 6. Verify Runtime (GPU/CPU)

```bash
pixi run verify-cuda
```

You will get JSON including:
- `cuda_available`
- `torch_cuda_compiled`
- `gpu_count`
- `nvidia_smi`

If `cuda_available` is `false`, you can still run training/validation on CPU.

## 7. Download Datasets

Download both train and validation files into `datasets/`:

```bash
pixi run download-datasets
```

Expected files after success:
- `datasets/awk9_train_10k.pt`
- `datasets/awk9_val_1k.pt`

The scripts assume these defaults.

## 8. Full Training Run

Start training:

```bash
pixi run train
```

Default training behavior (`scripts/train.py`):
- Train set: `datasets/awk9_train_10k.pt`
- Val set: `datasets/awk9_val_1k.pt`
- Epochs: `2`
- Batch size: `8`
- LR: `3e-4`
- Checkpoint path: `checkpoints/tcn_cvae.pt`

### Logs you should see during training

Per batch:
- `loss_total`
- `loss_recon`
- `loss_kl`
- `loss_diversity`
- `batch_minADE`
- `batch_minFDE`

Per epoch:
- `train_loss`
- `val_minADE`
- `val_minFDE`

Checkpoint save message when validation improves:

```text
[INFO] saved best checkpoint -> checkpoints/tcn_cvae.pt
```

Final completion line:

```text
[DONE] best_val_minADE=...
```

## 9. Full Validation Run

After training, run:

```bash
pixi run validate
```

Default validation behavior (`scripts/validate.py`):
- Val set: `datasets/awk9_val_1k.pt`
- Checkpoint: `checkpoints/tcn_cvae.pt`
- Batch size: `8`

### Logs you should see during validation

Checkpoint load (if file exists):

```text
[INFO] loaded checkpoint from ... (epoch=...)
```

Final validation summary:

```text
[RESULT] samples=... minADE=... minFDE=...
```

That line confirms end-to-end validation is complete.

## 10. Quick Smoke Test (Fast Local Check)

If you want a short sanity run before full training:

```bash
pixi run python scripts/train.py \
  --max-train-samples 20 \
  --max-val-samples 20 \
  --epochs 1 \
  --batch-size 1 \
  --save-path checkpoints/tcn_cvae_smoke.pt

pixi run python scripts/validate.py \
  --max-samples 20 \
  --batch-size 1 \
  --checkpoint checkpoints/tcn_cvae_smoke.pt
```

## 11. Model Concept and Architecture

This section explains the method, not repo layout.

### 11.1 Forecasting Objective

Given observed history, predict a **distribution** of future trajectories:
- observed node tensor (after batching): roughly `B x 50 x N x 5`
- observed ego tensor: `B x 50 x 5`
- predicted trajectories: `B x K x 30 x 2` with `K=6`

Important: the dataset stores 40 future steps, but training uses the first 30 (`[:, :30, :2]`).

### 11.2 Encoder: 2D Dilated TCN Over Time and Neighbor Axis

`TCN2DEncoder` applies residual 2D convolutions with increasing temporal dilation (`1, 2, 4, 8`).

Why this matters:
- Dilations expand temporal receptive field efficiently.
- 2D kernels let the encoder jointly process temporal evolution and local multi-agent structure.
- Residual blocks stabilize deeper temporal feature extraction.

The encoder output is pooled into a compact node context vector.

### 11.3 Conditioning Context

A separate MLP projects the ego feature at the latest observed time.

Then:
- concatenate node context + ego context
- pass through a context head (`Linear + SiLU + LayerNorm`)

This creates a single conditioning vector that represents scene history for forecasting.

### 11.4 Conditional VAE for Multimodal Futures

The CVAE has:
- **Prior** `p(z|context)`
- **Posterior** `q(z|context,target)` (used in training)
- **Decoder** `p(y|z,context)`

Training path:
- posterior latent sampled with reparameterization
- decode into 6 future trajectory modes

Inference path:
- sample from prior
- decode into 6 plausible futures

Mode embeddings are added to latent samples so each mode specializes.

### 11.5 Loss Design

Total loss:

`L = L_recon + lambda_kl * L_kl + lambda_div * L_div`

- `L_recon`: Smooth L1 on the **best** predicted mode (best-of-K by smallest trajectory distance)
- `L_kl`: KL divergence between posterior and prior Gaussian latents
- `L_div`: diversity regularizer from pairwise distances between predicted modes

Best-of-K lets the model fit one valid future while keeping multiple hypotheses.
Diversity term prevents all modes from collapsing to the same path.

### 11.6 Metrics

- `minADE`: minimum (over modes) average displacement error
- `minFDE`: minimum (over modes) final displacement error

Lower is better for both.

## 12. Troubleshooting (Common Failures)

### 12.1 `Dataset file not found`

Cause: missing `.pt` files.

Fix:
```bash
pixi run download-datasets
ls -lh datasets/
```

### 12.2 `checkpoint not found ...; evaluating randomly initialized model`

Cause: validation run before successful training.

Fix:
```bash
pixi run train
pixi run validate
```

### 12.3 CUDA warnings / GPU not detected

- Run `pixi run verify-cuda` and inspect JSON.
- If CUDA is unavailable, continue on CPU (expected slower runtime).
- If GPU is required, verify NVIDIA driver + `nvidia-smi` outside the repo.

### 12.4 First run is slow

Expected on first setup due to:
- Pixi environment resolution
- package downloads
- dataset download and `.pt` deserialization

## 13. Reproducible CLI Reference

Train:
```bash
pixi run train
```

Validate:
```bash
pixi run validate
```

Override defaults (example):
```bash
pixi run python scripts/train.py \
  --dataset-path datasets/awk9_train_10k.pt \
  --val-dataset-path datasets/awk9_val_1k.pt \
  --epochs 5 \
  --batch-size 8 \
  --save-path checkpoints/tcn_cvae.pt
```
