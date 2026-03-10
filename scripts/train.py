import argparse
import random
from pathlib import Path
import sys

import torch
from torch_geometric.loader import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from trajectory_vae_forecasting.dataset import GraphTrajectoryDataset
from trajectory_vae_forecasting.model import TCNVAEForecaster
from trajectory_vae_forecasting.utils import min_ade, min_fde, move_norm_stats, prepare_batch


def seed_everything(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, device, stats):
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
    return ade_sum / max(count, 1), fde_sum / max(count, 1)


def parse_args():
    parser = argparse.ArgumentParser(description="Train TCN+CVAE trajectory forecasting model")
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--val-dataset-path", type=str, default="")
    parser.add_argument("--max-train-samples", type=int, default=30)
    parser.add_argument("--max-val-samples", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--kl-weight", type=float, default=0.01)
    parser.add_argument("--diversity-weight", type=float, default=0.05)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-path", type=str, default="checkpoints/tcn_cvae.pt")
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}")

    train_ds = GraphTrajectoryDataset(args.dataset_path, max_samples=args.max_train_samples)
    val_path = args.val_dataset_path or args.dataset_path
    val_ds = GraphTrajectoryDataset(val_path, max_samples=args.max_val_samples)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)

    model = TCNVAEForecaster().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    stats = move_norm_stats(device)
    best_val_ade = float("inf")

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        step_count = 0

        for batch in train_loader:
            node, edge, ego, target = prepare_batch(batch, device, stats)

            optimizer.zero_grad(set_to_none=True)
            out = model(node, edge, ego, target=target)
            loss_dict = model.loss(
                out,
                target,
                kl_weight=args.kl_weight,
                diversity_weight=args.diversity_weight,
            )
            loss = loss_dict["total"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += float(loss.item())
            step_count += 1

        train_loss = epoch_loss / max(step_count, 1)
        val_ade, val_fde = evaluate(model, val_loader, device, stats)

        print(
            f"[EPOCH {epoch}] train_loss={train_loss:.4f} "
            f"val_minADE={val_ade:.4f} val_minFDE={val_fde:.4f}"
        )

        if val_ade < best_val_ade:
            best_val_ade = val_ade
            payload = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "best_val_ade": best_val_ade,
                "args": vars(args),
            }
            torch.save(payload, save_path)
            print(f"[INFO] saved best checkpoint -> {save_path}")

    print(f"[DONE] best_val_minADE={best_val_ade:.4f}")


if __name__ == "__main__":
    main()
