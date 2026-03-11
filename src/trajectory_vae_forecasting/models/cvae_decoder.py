import torch
from torch import nn


class ConditionalTrajectoryVAE(nn.Module):
    def __init__(
        self,
        context_dim: int,
        future_steps: int = 40,
        latent_dim: int = 32,
        num_modes: int = 6,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.future_steps = future_steps
        self.latent_dim = latent_dim
        self.num_modes = num_modes

        target_dim = future_steps * 2

        self.posterior = nn.Sequential(
            nn.Linear(context_dim + target_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.post_mu = nn.Linear(hidden_dim, latent_dim)
        self.post_logvar = nn.Linear(hidden_dim, latent_dim)

        self.prior = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.prior_mu = nn.Linear(hidden_dim, latent_dim)
        self.prior_logvar = nn.Linear(hidden_dim, latent_dim)

        self.mode_embed = nn.Embedding(num_modes, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(context_dim + latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, target_dim),
        )

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode_modes(self, context: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        bsz = context.size(0)
        mode_ids = torch.arange(self.num_modes, device=context.device)
        mode_bias = self.mode_embed(mode_ids).unsqueeze(0).expand(bsz, -1, -1)

        z_all = z.unsqueeze(1) + mode_bias
        ctx_all = context.unsqueeze(1).expand(bsz, self.num_modes, -1)

        dec_in = torch.cat([ctx_all, z_all], dim=-1)
        out = self.decoder(dec_in.reshape(bsz * self.num_modes, -1))
        return out.view(bsz, self.num_modes, self.future_steps, 2)

    def forward(self, context: torch.Tensor, target: torch.Tensor | None = None) -> dict:
        prior_h = self.prior(context)
        prior_mu = self.prior_mu(prior_h)
        prior_logvar = self.prior_logvar(prior_h)

        if target is not None:
            target_flat = target.reshape(target.size(0), -1)
            post_h = self.posterior(torch.cat([context, target_flat], dim=-1))
            post_mu = self.post_mu(post_h)
            post_logvar = self.post_logvar(post_h)
            z = self.reparameterize(post_mu, post_logvar)
        else:
            post_mu = None
            post_logvar = None
            z = self.reparameterize(prior_mu, prior_logvar)

        preds = self.decode_modes(context, z)
        return {
            "preds": preds,
            "prior_mu": prior_mu,
            "prior_logvar": prior_logvar,
            "post_mu": post_mu,
            "post_logvar": post_logvar,
        }
