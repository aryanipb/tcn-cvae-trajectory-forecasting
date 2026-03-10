# Vehicle Trajectory Forecasting with Direct 2D TCN + Conditional VAE

## 1. Overview
This project implements a vehicle trajectory forecasting model for the same task format as the reference `predmodel` training pipeline, with a fully different modeling design:

- No spatial encoder branch is used.
- Node history is fed directly into a 2D temporal convolutional network.
- The resulting context is decoded by a custom conditional variational autoencoder for multimodal trajectory prediction.

The repository includes:
- Modular model implementation
- Minimal non-Modal training script
- Minimal non-Modal validation script
- Dataset preprocessing reference files copied into `data_processing/`
- Reproducible Pixi + requirements-based environment setup
- CUDA compatibility verification utility

## 2. Problem Formulation
Given observed trajectory context for each scenario:
- Node features: `X in R^{T_obs x N x F_node}`
- Ego features: `E in R^{T_obs x F_ego}`

Predict `K` plausible futures for the target agent:
- `Y_hat in R^{K x T_pred x 2}`

where:
- `T_obs = 50`
- `N = 9`
- `F_node = 5`
- `F_ego = 5`
- `T_pred = 30`
- `K = 6`

The training objective combines:
- Best-of-K reconstruction loss on future trajectories
- KL divergence between posterior and context-conditioned prior
- Diversity regularization across generated modes

## 3. Dataset Interface and Processing
### 3.1 Input file format
Training data is loaded from a PyTorch `.pt` file containing a list of `torch_geometric.data.Data` objects.

Observed schema:
- `x`: `(50, 9, 5)`
- `edge_attr`: `(50, 9, 4)`
- `ego_features`: `(50, 5)`
- `y`: `(40, 5)`

The training/validation scripts use:
- first 30 future steps from `y`
- first 2 coordinates as trajectory targets

### 3.2 Normalization
To remain compatible with the task setup, fixed normalization statistics are applied to:
- node features
- ego features
- trajectory targets

The same normalization/de-normalization path is used in train and validation scripts.

### 3.3 Preprocessing references
The following preprocessing files are copied from:
`/home/aryan/work/startup/predmodel/data/agroverse_data_preprocessing`

Included in this repository under `data_processing/`:
- `file1.py`
- `file2.py`
- `graphs.py`
- `file3.py`

These are included for transparency so viewers can inspect how graph data construction was done in your upstream pipeline.

## 4. Architecture
## 4.1 Direct 2D TCN Encoder
Path: `src/trajectory_vae_forecasting/models/tcn_blocks.py`, `tcn_encoder.py`

Pipeline:
1. Input node tensor is rearranged to channel-first format.
2. A stack of residual 2D dilated temporal blocks extracts temporal context.
3. The final encoder output is aggregated over node axis and last observed time index to form compact context.

The model intentionally omits a separate spatial encoder module.

## 4.2 Context Fusion
Path: `src/trajectory_vae_forecasting/models/forecaster.py`

- Ego features are projected with a lightweight MLP.
- Node context and ego context are concatenated.
- A context head maps the fused vector into latent-conditioning space.

## 4.3 Conditional VAE Decoder
Path: `src/trajectory_vae_forecasting/models/cvae_decoder.py`

The decoder learns:
- Prior network `p(z | context)`
- Posterior network `q(z | context, target)`
- Mode-conditioned decoder `p(y | context, z, mode)`

Multimodality is realized by adding mode embeddings to latent samples and decoding all modes jointly.

## 4.4 Loss
Path: `src/trajectory_vae_forecasting/models/losses.py`

Total objective:
- `L = L_recon_best_of_K + lambda_kl * L_kl + lambda_div * L_div`

Where:
- `L_recon_best_of_K`: Smooth L1 on best mode per sample
- `L_kl`: KL divergence between posterior and prior Gaussians
- `L_div`: pairwise trajectory diversity regularizer

## 5. Repository Structure
```text
vehicle-trajectory-tcn-vae/
  data_processing/
    file1.py
    file2.py
    file3.py
    graphs.py
  scripts/
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

## 6. Pixi-Centered Environment Setup
## 6.1 Install Pixi
Official installer:
```bash
curl -fsSL https://pixi.sh/install.sh | bash
```
Then restart shell and verify:
```bash
pixi --version
```

## 6.2 Create and lock environment
From project root:
```bash
cd /home/aryan/work/projects/vehicle-trajectory-tcn-vae
pixi install
```

This provisions the pinned Python and package set from `pixi.toml`.

## 6.3 Install requirements inside Pixi environment
```bash
pixi run bootstrap
```

This executes:
- `python -m pip install --upgrade pip`
- `python -m pip install -r requirements.txt`

Using `pixi run` forces installation into the Pixi-managed environment rather than host Python.

## 6.4 Verify CUDA/runtime alignment
```bash
pixi run verify-cuda
```

The script prints:
- Python version
- Torch version
- CUDA version Torch was compiled with
- CUDA availability
- GPU names
- `nvidia-smi` report
- mismatch flag

If mismatch is reported:
1. verify host NVIDIA driver supports the reported CUDA runtime
2. ensure you run training only via `pixi run ...`
3. update torch wheel pin/index if your cluster requires a different CUDA build

## 7. Training
## 7.1 Full command
```bash
pixi run python scripts/train.py \
  --dataset-path /home/aryan/work/startup/datasetpredmodel/graphs/awk9.pt \
  --val-dataset-path /home/aryan/work/startup/datasetpredmodel/graphs/awk9.pt \
  --max-train-samples 30 \
  --max-val-samples 30 \
  --epochs 2 \
  --batch-size 8 \
  --save-path checkpoints/tcn_cvae.pt
```

### 7.2 Key outputs
Per epoch:
- train loss
- validation minADE
- validation minFDE
- best checkpoint save message

## 8. Validation
```bash
pixi run python scripts/validate.py \
  --dataset-path /home/aryan/work/startup/datasetpredmodel/graphs/awk9.pt \
  --checkpoint checkpoints/tcn_cvae.pt \
  --max-samples 30 \
  --batch-size 8
```

## 9. Smoke-Test Verification Performed
A short smoke test was run on 24 samples with the requested dataset path.

Training test:
- `epochs=1`, `batch_size=6`, `max_train_samples=24`, `max_val_samples=24`
- completed successfully

Validation test:
- same 24 samples used as validation surrogate
- completed successfully

The scripts are therefore verified for:
- data loading
- forward pass
- loss computation
- optimization step
- checkpoint save/load
- metric evaluation path

## 10. Notes on Reproducibility and Stability
- Use only `pixi run ...` to avoid host-env leakage.
- Keep pins synchronized between `pixi.toml`, `pyproject.toml`, and `requirements.txt`.
- Run `pixi run verify-cuda` before long training jobs.
- Keep dataset path explicit in command lines.

## 11. Limitations and Practical Scope
- This repository focuses on architecture replacement and pipeline correctness, not final SOTA performance claims.
- The provided smoke test is a functional check on small samples, not a full convergence study.
- Hyperparameters are intentionally minimal and easy to inspect.

## 12. Citation-Style Summary
If you describe this project in technical documentation, an accurate summary is:

A direct 2D temporal convolution encoder is used to process node histories without a dedicated spatial encoder, and a context-conditioned VAE with mode embeddings generates multimodal trajectory forecasts. Training uses best-of-K reconstruction with KL and diversity regularization under the same task-level dataset contract as the reference pipeline.
