from typing import Optional

import torch
import torch.distributions as td
import torch.nn as nn

from rssm import (
    ActionT,
    ContT,
    HiddenT,
    ImageT,
    LatentShape,
    LatentT,
    ResetT,
    Rssm,
    get_mlp_layers,
)
from vae import Decoder

RewardT = torch.Tensor
RewardSymlogT = torch.Tensor
# RewardT (real): [ N 1 ]
# RewardSymlogT (symlog(real)): [ N 1 ]


def symlog(x: torch.Tensor) -> torch.Tensor:
    return x.sign() * (x.abs() + 1).log()


def symexp(x: torch.Tensor) -> torch.Tensor:
    return x.sign() * (x.abs().exp() - 1)


class WorldModel(nn.Module):
    def __init__(
        self,
        img_channels: int,
        ls: LatentShape,
        a_dim: int,
        h_dim: int,
        cnn_depth: int = 32,
        mlp_features: int = 512,
        mlp_layers: int = 2,
        wto: bool = False,
    ) -> None:
        super().__init__()
        self.ls = ls
        self.a_dim = a_dim
        self.h_dim = h_dim

        self.rssm = Rssm(img_channels, ls, a_dim, h_dim, cnn_depth, mlp_features, mlp_layers, wto)

        # The decoders use this for input (z, h) -> this here -> decoder
        self.zh_transform = ZhToModelState(ls, h_dim)

        # All the decoders
        self.decoder = Decoder(img_channels, self.zh_transform.out_features, cnn_depth)
        # NOTE: Weight init here may be 0 init
        self.reward_fc = nn.Sequential(
            *get_mlp_layers(
                self.zh_transform.out_features,
                1,
                final_activation=None,
                hidden_features=mlp_features,
                layers=mlp_layers,
            )
        )
        self.continue_fc = nn.Sequential(
            *get_mlp_layers(
                self.zh_transform.out_features,
                1,
                final_activation=nn.Sigmoid,
                hidden_features=mlp_features,
                layers=mlp_layers,
            )
        )

    def compute_loss(
        self, actions: ActionT, xs: ImageT, rews: RewardT, conts: ContT, resets: ResetT
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # Returns (loss, metrics)
        if len(actions.shape) == 2:
            raise ValueError("Time dimension required")
        _, n, _ = actions.shape
        init_z, init_h = self.rssm.initial_state(n)
        # Shift actions and xs, since RSSM takes (prev_action, next_obs)
        z_posts, z_samples, hiddens = self.rssm(init_z, actions, init_h, xs, resets)
        z_priors = self.rssm.transition(hiddens)

        # Dynamics and representation losses
        dyn_loss_scale = 0.5
        rep_loss_scale = 0.1
        posts_dist = td.Categorical(logits=z_posts)
        posts_dist_sg = td.Categorical(logits=z_posts.detach())
        priors_dist = td.Categorical(logits=z_priors)
        priors_dist_sg = td.Categorical(logits=z_priors.detach())
        # KL takes shape [ T N n_dis n_cls ]
        # KL divergence results in [ T N n_dis ]
        # See equation (5) on Dreamer v3
        one = torch.tensor(1, device=z_posts.device)
        dyn_loss = td.kl_divergence(posts_dist_sg, priors_dist).sum(-1).maximum(one).mean()
        rep_losses = td.kl_divergence(posts_dist, priors_dist_sg).sum(-1).maximum(one)
        rep_loss = rep_losses.mean()
        z_repr_loss = dyn_loss_scale * dyn_loss + rep_loss_scale * rep_loss

        zhs: torch.Tensor = self.zh_transform(z_samples, hiddens)  # [ T N X ] (X is arbitrary)
        t, n, x = zhs.shape
        zhs_f12 = zhs.view(-1, x)
        recon = self.decoder(zhs_f12).view(t, n, *xs.shape[-3:])
        # Loss shape [ T N C 64 64 ]
        recon_losses = (recon - xs).square().sum([2, 3, 4])
        recon_loss = recon_losses.mean()

        rews_pred = self.reward_fc(zhs)  # [ T N 1 ]
        rews_loss = (rews_pred - symlog(rews)).square().mean()

        conts_pred = self.continue_fc(zhs)  # [ T N 1 ]
        conts_loss = torch.nn.functional.binary_cross_entropy(conts_pred, conts, reduction="mean")

        with torch.no_grad():
            low_kl = rep_losses < 1 + 1e-3
            metrics = {
                "Loss/kl": z_repr_loss,
                "Loss/recon": recon_loss,
                "Loss/rew": rews_loss,
                "Loss/cont": conts_loss,
                "Metric/neg_cont_mean": conts_pred[conts == 0].mean(),
                "Metric/low_kl": low_kl.float().mean(),
                "Metric/low_kl_recon_loss": recon_losses[low_kl].mean(),
            }

        return z_repr_loss + recon_loss + rews_loss + conts_loss, metrics


class ZhToModelState(nn.Module):
    def __init__(self, ls: LatentShape, h_dim: int, out_features: Optional[int] = None) -> None:
        super().__init__()
        if out_features is None:
            # No linear projection, only concatenation
            self.out_features = ls[0] * ls[1] + h_dim
            self.linear = None
        else:
            self.out_features = out_features
            self.linear = nn.Linear(ls[0] * ls[1] + h_dim, out_features)

    def forward(self, z: LatentT, h: HiddenT) -> torch.Tensor:
        z = z.flatten(-2)
        zh = torch.cat([z, h], dim=-1)
        if self.linear:
            return self.linear(zh)
        return zh
