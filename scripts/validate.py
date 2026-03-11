import argparse
from pathlib import Path
import sys

import torch
from torch_geometric.loader import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from trajectory_vae_forecasting.dataset import GraphTrajectoryDataset
from trajectory_vae_forecasting.model import TCNVAEForecaster
from trajectory_vae_forecasting.utils import min_ade, min_fde, move_norm_stats, prepare_batch


def parse_args():
    parser = argparse.ArgumentParser(description="Validate TCN+CVAE trajectory forecasting model")
    parser.add_argument("--dataset-path", type=str, default=str(PROJECT_ROOT / "datasets" / "awk9_val_1k.pt"))
    parser.add_argument("--checkpoint", type=str, default=str(PROJECT_ROOT / "checkpoints" / "tcn_cvae.pt"))
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}")

    dataset = GraphTrajectoryDataset(args.dataset_path, max_samples=args.max_samples)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = TCNVAEForecaster().to(device)
    checkpoint_path = Path(args.checkpoint)
    if checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        print(f"[INFO] loaded checkpoint from {checkpoint_path} (epoch={ckpt.get('epoch', 'n/a')})")
    else:
        print(f"[WARN] checkpoint not found at {checkpoint_path}; evaluating randomly initialized model")

    stats = move_norm_stats(device)
    model.eval()

    ade_sum = 0.0
    fde_sum = 0.0
    count = 0

    with torch.no_grad():
        for batch in loader:
            node, edge, ego, target = prepare_batch(batch, device, stats)
            out = model(node, edge, ego, target=None)
            preds = out["preds"]

            preds_dn = preds * stats["y_sigma"] + stats["y_mu"]
            target_dn = target * stats["y_sigma"] + stats["y_mu"]

            ade_sum += float(min_ade(preds_dn, target_dn).item())
            fde_sum += float(min_fde(preds_dn, target_dn).item())
            count += 1

    ade = ade_sum / max(count, 1)
    fde = fde_sum / max(count, 1)
    print(f"[RESULT] samples={len(dataset)} minADE={ade:.4f} minFDE={fde:.4f}")


if __name__ == "__main__":
    main()
