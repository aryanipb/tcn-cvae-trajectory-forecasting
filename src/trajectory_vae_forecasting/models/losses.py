import torch
import torch.nn.functional as F


def multimodal_cvae_loss(
    preds: torch.Tensor,
    target: torch.Tensor,
    prior_mu: torch.Tensor,
    prior_logvar: torch.Tensor,
    post_mu: torch.Tensor | None,
    post_logvar: torch.Tensor | None,
    kl_weight: float,
    diversity_weight: float,
) -> dict:
    per_mode_dist = torch.norm(preds - target.unsqueeze(1), dim=-1).mean(dim=-1)
    best_idx = per_mode_dist.argmin(dim=1)
    batch_idx = torch.arange(target.size(0), device=target.device)
    best_pred = preds[batch_idx, best_idx]

    recon_loss = F.smooth_l1_loss(best_pred, target)

    if post_mu is None or post_logvar is None:
        kl_loss = torch.tensor(0.0, device=target.device)
    else:
        var_ratio = torch.exp(post_logvar - prior_logvar)
        mean_diff = (post_mu - prior_mu).pow(2) / torch.exp(prior_logvar)
        kl = 0.5 * (prior_logvar - post_logvar + var_ratio + mean_diff - 1.0)
        kl_loss = kl.sum(dim=-1).mean()

    mode_count = preds.size(1)
    flat = preds.reshape(preds.size(0), mode_count, -1)
    pairwise = torch.cdist(flat, flat, p=2)
    eye = torch.eye(mode_count, device=target.device).unsqueeze(0)
    diversity = (pairwise * (1.0 - eye)).sum(dim=[1, 2]) / (mode_count * (mode_count - 1))
    diversity_loss = torch.exp(-diversity).mean()

    total = recon_loss + kl_weight * kl_loss + diversity_weight * diversity_loss
    return {
        "total": total,
        "recon": recon_loss.detach(),
        "kl": kl_loss.detach(),
        "diversity": diversity_loss.detach(),
    }
