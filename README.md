# Vehicle Trajectory Forecasting with Direct 2D TCN + Conditional VAE

## 1. Overview
This repository implements multimodal vehicle trajectory forecasting.

Model design in this project:
- Node history input to a 2D temporal convolution encoder
- Context-conditioned variational autoencoder for multimodal trajectory generation

## 2. Problem Formulation
Given observed context:
- node features `X in R^{T_obs x N x F_node}`
- ego features `E in R^{T_obs x F_ego}`

Predict `K` plausible future trajectories:
- `Y_hat in R^{K x T_pred x 2}`

Task constants used here:
- `T_obs = 50`
- `N = 9`
- `F_node = 5`
- `F_ego = 5`
- `T_pred = 40`
- `K = 6`

Training objective combines:
- best-of-K trajectory reconstruction
- KL divergence between posterior and prior
- diversity regularization across predicted modes

## 3. Dataset Specification
Source repo:
- https://huggingface.co/datasets/aryanb1702/awk9/

Expected files after download:
- `datasets/awk9_train_10k.pt`
- `datasets/awk9_val_1k.pt`

Observed sample schema:
- `x`: `(50, 9, 5)`
- `edge_attr`: `(50, 9, 4)`
- `ego_features`: `(50, 5)`
- `y`: `(40, 5)`

Target usage:
- Next 40 future steps
- x,y coordinates

Normalization is fixed and consistent across train and validation paths.

## 4. Architecture
### 4.1 2D TCN Encoder
Files:
- `src/trajectory_vae_forecasting/models/tcn_blocks.py`
- `src/trajectory_vae_forecasting/models/tcn_encoder.py`

### 4.2 Context Fusion
File:
- `src/trajectory_vae_forecasting/models/forecaster.py`

### 4.3 Conditional VAE Decoder
File:
- `src/trajectory_vae_forecasting/models/cvae_decoder.py`

### 4.4 Objective
File:
- `src/trajectory_vae_forecasting/models/losses.py`

Total loss:
- `L = L_recon_best_of_K + lambda_kl * L_kl + lambda_div * L_div`

## 5. Repository Structure
```text
vehicle-trajectory-tcn-vae/
  data_processing/
    file1.py
    file2.py
    file3.py
    graphs.py
  datasets/
    awk9_train_10k.pt
    awk9_val_1k.pt
  scripts/
    download_datasets.py
    train.py
    validate.py
    verify_cuda.py
  src/trajectory_vae_forecasting/
    dataset.py
    model.py
    utils.py
    models/
      tcn_blocks.py
      tcn_encoder.py
      cvae_decoder.py
      losses.py
      forecaster.py
  pixi.toml
  pyproject.toml
  requirements.txt
  README.md
```

## 6. Install and Environment
### 6.1 Install Pixi
```bash
curl -fsSL https://pixi.sh/install.sh | bash
pixi --version
```

### 6.2 Create environment
```bash
cd /home/aryan/work/projects/vehicle-trajectory-tcn-vae
pixi install
```

### 6.3 Install requirements in Pixi env
```bash
pixi run bootstrap
```

### 6.4 Verify CUDA/runtime alignment
```bash
pixi run verify-cuda
```

## 7. Download Dataset Into Required Directory
From project root:
```bash
pixi run download-datasets
```

This downloads directly into:
- `datasets/awk9_train_10k.pt`
- `datasets/awk9_val_1k.pt`

Optional direct command:
```bash
pixi run python scripts/download_datasets.py --repo-id aryanb1702/awk9 --out-dir datasets
```

## 8. Training and Validation With New Defaults
The scripts are now configured to use these defaults automatically:
- train dataset: `datasets/awk9_train_10k.pt`
- validation dataset: `datasets/awk9_val_1k.pt`
- validation checkpoint: `checkpoints/tcn_cvae.pt`

Train:
```bash
pixi run train
```

Validate:
```bash
pixi run validate
```

## 9. Per-Batch Metrics
Training prints continuous per-batch metrics:
- `loss_total`
- `loss_recon`
- `loss_kl`
- `loss_diversity`
- `batch_minADE`
- `batch_minFDE`

Epoch summary includes:
- `train_loss`
- `val_minADE`
- `val_minFDE`

## 10. Fast Verification Path
For a quick functional check:
```bash
pixi run python scripts/train.py --max-train-samples 20 --max-val-samples 20 --epochs 1 --batch-size 1 --save-path checkpoints/tcn_cvae_smoke.pt
pixi run python scripts/validate.py --max-samples 20 --batch-size 1 --checkpoint checkpoints/tcn_cvae_smoke.pt
```

## 11. Critical Disclaimer on Initial Runtime
Initial setup and first runs can take noticeable time. This is expected.

Main reasons:
- Pixi resolves and installs a pinned environment
- first import/initialization of heavy ML packages
- large `.pt` file deserialization for dataset loading

After setup, repeated runs are faster. For best iteration speed, use:
```bash
pixi shell
```
and then run `python scripts/train.py` / `python scripts/validate.py` inside that shell.

## 12. Preprocessing Transparency
Reference preprocessing files are included in `data_processing/` for the agroverse 2 motion forecasting dataset:
- `file1.py`
- `file2.py`
- `graphs.py`
- `file3.py`