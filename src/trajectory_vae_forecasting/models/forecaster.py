import torch
from torch import nn

from .cvae_decoder import ConditionalTrajectoryVAE
from .losses import multimodal_cvae_loss
from .tcn_encoder import TCN2DEncoder


class TCNVAEForecaster(nn.Module):
    def __init__(
        self,
        node_dim: int = 5,
        ego_dim: int = 5,
        future_steps: int = 30,
        num_modes: int = 6,
        latent_dim: int = 32,
        encoder_channels: tuple[int, ...] = (32, 64, 128, 256),
    ):
        super().__init__()
        self.tcn = TCN2DEncoder(in_channels=node_dim, channels=encoder_channels)
        self.ego_proj = nn.Sequential(
            nn.Linear(ego_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
        )

        context_dim = self.tcn.out_channels + 64
        self.context_head = nn.Sequential(
            nn.Linear(context_dim, 256),
            nn.SiLU(),
            nn.LayerNorm(256),
        )

        self.cvae = ConditionalTrajectoryVAE(
            context_dim=256,
            future_steps=future_steps,
            latent_dim=latent_dim,
            num_modes=num_modes,
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_features: torch.Tensor | None = None,
        ego_features: torch.Tensor | None = None,
        target: torch.Tensor | None = None,
    ) -> dict:
        _ = edge_features
        encoded = self.tcn(node_features)
        node_context = encoded.mean(dim=-1)[:, :, -1]

        if ego_features is None:
            ego_context = torch.zeros(node_context.size(0), 64, device=node_context.device)
        else:
            ego_context = self.ego_proj(ego_features[:, -1, :])

        context = self.context_head(torch.cat([node_context, ego_context], dim=-1))
        return self.cvae(context, target=target)

    def loss(self, model_out: dict, target: torch.Tensor, kl_weight: float = 0.01, diversity_weight: float = 0.05) -> dict:
        return multimodal_cvae_loss(
            preds=model_out["preds"],
            target=target,
            prior_mu=model_out["prior_mu"],
            prior_logvar=model_out["prior_logvar"],
            post_mu=model_out["post_mu"],
            post_logvar=model_out["post_logvar"],
            kl_weight=kl_weight,
            diversity_weight=diversity_weight,
        )
