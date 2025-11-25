from typing import Optional, Type

import torch
import torch.distributions as td
import torch.nn as nn

from vae import Encoder

LatentShape = tuple[int, int]
EmbedT = torch.Tensor
LatentLogDistT = torch.Tensor
LatentT = torch.Tensor
ActionT = torch.Tensor
HiddenT = torch.Tensor
ImageT = torch.Tensor
ContT = torch.Tensor
ResetT = torch.Tensor

# EmbedT: [ N E1 ]
# LatentLogDistT (logits): [ N n_dis n_cls ] where z_dim = n_dis * n_cls
# LatentT (onehot): [ N n_dis n_cls ]
# ActionT (onehot): [ N n_acts ]
# HiddenT: [ N h_dim ]
# ImageT (0 to 1): [ N C 64 64 ]
# ImageLogT (log(0 to 1)): [ N C 64 64 ]
# ContT (0 to 1): [ N 1 ]
# ResetT (0 or 1, 1-ContT shifted right in T by 1): [ N 1 ]
# Optional [ T ... ] dimension in front where applicable


class LayerNormSiLU(nn.Module):
    def __init__(self, units: int) -> None:
        super().__init__()
        self.fw = nn.Sequential(nn.LayerNorm(units, 1e-3), nn.SiLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fw(x)


def get_mlp_layers(
    in_features: int,
    out_features: int,
    activation: Type[nn.Module] = LayerNormSiLU,
    final_activation: Optional[Type[nn.Module]] = LayerNormSiLU,
    hidden_features: int = 512,
    layers: int = 2,
) -> list[nn.Module]:
    get_act = lambda act, units: act() if act is not LayerNormSiLU else act(units)
    sizes = [in_features] + [hidden_features] * (layers - 1) + [out_features] * (layers > 0)
    res = []
    for i, (ft1, ft2) in enumerate(zip(sizes[:-1], sizes[1:])):
        res.append(nn.Linear(ft1, ft2))
        if i < len(sizes) - 2:
            res.append(get_act(activation, ft2))
        elif final_activation is not None:
            res.append(get_act(final_activation, ft2))
    return res


class Rssm(nn.Module):
    def __init__(
        self,
        img_channels: int,
        ls: LatentShape,
        a_dim: int,
        h_dim: int,
        cnn_depth: int,
        mlp_features: int,
        mlp_layers: int,
        wto: bool = False,
    ) -> None:
        super().__init__()
        self.ls = ls
        self.h_dim = h_dim

        self.recurrent = Recurrent(ls, a_dim, h_dim, mlp_features, mlp_layers if not wto else 0)
        self.image_embedder = Encoder(img_channels, cnn_depth)
        self.representation = Representation(
            ls,
            self.image_embedder.output_size,
            h_dim,
            mlp_features,
            mlp_layers if not wto else 1,
            uniform=0.01,
        )
        self.transition = Transition(ls, h_dim, mlp_features, mlp_layers, uniform=0.01)

    def __call__(
        self,
        prev_z: LatentT,
        prev_a: ActionT,
        prev_h: HiddenT,
        x: Optional[ImageT],
        reset: ResetT,
        stochastic: bool = True,
        temperature: float = 1.0,
    ) -> tuple[LatentLogDistT, LatentT, HiddenT]:
        return super().__call__(
            prev_z, prev_a, prev_h, x, reset, stochastic=stochastic, temperature=temperature
        )

    def forward(
        self,
        prev_z: LatentT,
        prev_a: ActionT,
        prev_h: HiddenT,
        x: Optional[ImageT],
        reset: ResetT,
        stochastic: bool = True,
        temperature: float = 1.0,
    ) -> tuple[LatentLogDistT, LatentT, HiddenT]:
        if len(prev_a.shape) == 2:
            # Apply reset flags
            prev_z = prev_z * (1 - reset).unsqueeze(-1)  # Need to multiply againt dim [ N 1 1 ]
            prev_h = prev_h * (1 - reset)
            # No time dimension
            h = self.recurrent(prev_z, prev_a, prev_h)
            if x is not None:
                e = self.image_embedder(x)
                z_log_dist = self.representation(e, h)
            else:
                z_log_dist = self.transition(h)
            z_dist = td.OneHotCategoricalStraightThrough(logits=z_log_dist / temperature)
            if stochastic:
                z_sample = z_dist.rsample()
            else:
                z_sample = z_dist.mode + z_dist.probs - z_dist.probs.detach()
            return z_dist, z_sample, h
        elif len(prev_a.shape) == 3 and x is None:
            z_sample, h = prev_z, prev_h
            z_log_dists = []
            z_samples = []
            hs = []
            for a, r in zip(prev_a, reset):
                z_log_dist, z_sample, h = self(z_sample, a, h, None, r)
                z_log_dists.append(z_log_dist)
                z_samples.append(z_sample)
                hs.append(h)
            return torch.stack(z_log_dists), torch.stack(z_samples), torch.stack(hs)
        elif len(prev_a.shape) == 3:
            # Special batched impl
            t, n, _, _, _ = x.shape
            embed = self.image_embedder(x.reshape(-1, *x.shape[-3:])).view(t, n, -1)
            hs = []
            z_log_dists = []
            z_samples = []
            z, h = prev_z, prev_h
            for e, a, r in zip(embed, prev_a, reset):
                h = self.recurrent(z * (1 - r).unsqueeze(-1), a, h * (1 - r))
                z_log_dist = self.representation(e, h)
                z_dist = td.OneHotCategoricalStraightThrough(logits=z_log_dist / temperature)
                if stochastic:
                    z_sample = z_dist.rsample()
                else:
                    z_sample = z_dist.mode + z_dist.probs - z_dist.probs.detach()
                z = z_sample
                hs.append(h)
                z_log_dists.append(z_log_dist)
                z_samples.append(z_sample)
            return torch.stack(z_log_dists), torch.stack(z_samples), torch.stack(hs)
        raise ValueError

    def initial_state(self, n: int = 1) -> tuple[LatentT, HiddenT]:
        device = next(self.parameters()).device
        return (
            torch.zeros(n, *self.ls, device=device),
            torch.zeros(n, self.h_dim, device=device),
        )


class Recurrent(nn.Module):
    def __init__(
        self, ls: LatentShape, a_dim: int, h_dim: int, mlp_features: int, mlp_layers: int
    ) -> None:
        super().__init__()
        z_dim = ls[0] * ls[1]
        self.za_fcs = nn.Sequential(
            *get_mlp_layers(
                z_dim + a_dim,
                mlp_features,
                hidden_features=mlp_features,
                layers=mlp_layers,
            )
        )
        self.rnn = nn.GRUCell(mlp_features if mlp_layers > 0 else z_dim + a_dim, h_dim)

    def forward(self, prev_z: LatentT, prev_a: ActionT, prev_h: HiddenT) -> HiddenT:
        assert len(prev_z.shape) == 3  # [ N 32 32 ]
        assert len(prev_a.shape) == 2  # [ N n_acts ]
        za = torch.cat((prev_z.flatten(1), prev_a), dim=1)
        return self.rnn(self.za_fcs(za), prev_h)


class Representation(nn.Module):
    # Stock DV3 does not have skip connection in this part, but it accelerates
    # early learning

    def __init__(
        self,
        ls: LatentShape,
        embed_dim: int,
        h_dim: int,
        mlp_features: int,
        mlp_layers: int,
        uniform: float = 0,
    ) -> None:
        super().__init__()
        self.ls = ls
        self.uniform = uniform
        n_dis, n_cls = ls
        self.eh_to_inter = nn.Sequential(
            *get_mlp_layers(
                embed_dim + h_dim,
                n_dis * n_cls,
                final_activation=None,
                hidden_features=mlp_features,
                layers=mlp_layers,
            )
        )
        if mlp_layers > 1:
            self.e_to_inter = nn.Linear(embed_dim, n_dis * n_cls)
        else:
            self.e_to_inter = None
        self.inter_to_z_dist = nn.Sequential(
            # [ N n_dis*n_cls ] -> [ N n_dis n_cls ]
            nn.Unflatten(-1, (n_dis, n_cls)),
            nn.LogSoftmax(-1),
        )

    def __call__(self, e: EmbedT, h: HiddenT) -> LatentLogDistT:
        return super().__call__(e, h)

    def forward(self, e: EmbedT, h: HiddenT) -> LatentLogDistT:
        assert len(e.shape) == len(h.shape) == 2
        eh = torch.cat((e, h), dim=-1)
        x1 = self.eh_to_inter(eh)
        if self.e_to_inter is not None:
            x1 = x1 + self.e_to_inter(e)
        post_log_probs = self.inter_to_z_dist(x1)
        if self.uniform:
            # Use (1-u) of probs from logits + (u) of uniform
            # Conditions the KL loss better this way
            probs = post_log_probs.exp()
            return ((1 - self.uniform) * probs + self.uniform / self.ls[1]).log()
        return post_log_probs


class Transition(nn.Module):
    def __init__(
        self, ls: LatentShape, h_dim: int, mlp_features: int, mlp_layers: int, uniform: float = 0
    ) -> None:
        super().__init__()
        self.ls = ls
        self.uniform = uniform

        n_dis, n_cls = ls
        self.h_to_z_prior = nn.Sequential(
            *get_mlp_layers(
                h_dim,
                n_dis * n_cls,
                final_activation=None,
                hidden_features=mlp_features,
                layers=mlp_layers,
            ),
            nn.Unflatten(-1, (n_dis, n_cls)),
            nn.LogSoftmax(-1),
        )

    def forward(self, h: HiddenT) -> LatentLogDistT:
        prior_log_probs = self.h_to_z_prior(h)
        if self.uniform:
            probs = prior_log_probs.exp()
            return ((1 - self.uniform) * probs + self.uniform / self.ls[1]).log()
        return prior_log_probs
