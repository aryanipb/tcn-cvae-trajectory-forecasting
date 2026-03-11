import torch


NODE_MU = torch.tensor([-0.2552, -0.0701, -2.5500, 0.6316, -0.0330], dtype=torch.float32)
NODE_SIGMA = torch.tensor([28.8758, 27.7269, 5.3786, 2.8311, 2.1890], dtype=torch.float32)
EGO_MU = torch.tensor([-0.2597, -0.2275, 0.3777, 1.0730, -0.0016], dtype=torch.float32)
EGO_SIGMA = torch.tensor([13.3429, 12.2421, 2.5131, 2.2031, 0.8316], dtype=torch.float32)
Y_MU = torch.tensor([-1.0694, -0.8452], dtype=torch.float32)
Y_SIGMA = torch.tensor([45.0113, 40.9931], dtype=torch.float32)


def move_norm_stats(device: torch.device):
    return {
        "node_mu": NODE_MU.to(device),
        "node_sigma": NODE_SIGMA.to(device),
        "ego_mu": EGO_MU.to(device),
        "ego_sigma": EGO_SIGMA.to(device),
        "y_mu": Y_MU.to(device),
        "y_sigma": Y_SIGMA.to(device),
    }


def prepare_batch(batch, device: torch.device, stats: dict, future_steps: int = 40):
    batch = batch.to(device)
    bsz = batch.num_graphs
    if future_steps <= 0 or future_steps > 40:
        raise ValueError(f"future_steps must be in [1, 40], got {future_steps}")

    node = ((batch.x - stats["node_mu"]) / stats["node_sigma"]).view(bsz, 50, -1, 5)
    edge = batch.edge_attr.view(bsz, 50, -1, 4)
    ego = ((batch.ego_features - stats["ego_mu"]) / stats["ego_sigma"]).view(bsz, 50, -1)

    target = batch.y.reshape(bsz, 40, -1)[:, :future_steps, :2]
    target = (target - stats["y_mu"]) / stats["y_sigma"]
    return node, edge, ego, target


def min_ade(preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    distances = torch.norm(preds - target.unsqueeze(1), dim=-1)
    return distances.mean(dim=-1).min(dim=1).values.mean()


def min_fde(preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    distances = torch.norm(preds[:, :, -1] - target[:, -1].unsqueeze(1), dim=-1)
    return distances.min(dim=1).values.mean()
