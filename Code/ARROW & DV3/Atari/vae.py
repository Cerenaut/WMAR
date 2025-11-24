from typing import Tuple

import torch
import torch.distributions as td
import torch.nn as nn


class CatVae(nn.Module):
    def __init__(self, img_channels: int, n_dis: int, n_cls: int, channels: int) -> None:
        # Using CHW
        super().__init__()
        self.n_dis = n_dis
        self.n_cls = n_cls

        self.encoder = Encoder(img_channels, n_dis, n_cls, channels)
        self.decoder = Decoder(img_channels, n_dis * n_cls, channels)

        self.prior_probs: torch.Tensor
        self.register_buffer("prior_probs", torch.ones(n_dis, n_cls) / n_cls)
        self.prior_dist = td.OneHotCategorical(self.prior_probs)

    def __call__(self, *args, **kwds) -> Tuple[td.OneHotCategorical, torch.Tensor, torch.Tensor]:
        return super().__call__(*args, **kwds)

    def forward(
        self, x: torch.Tensor, stochastic: bool = True
    ) -> Tuple[td.OneHotCategorical, torch.Tensor, torch.Tensor]:
        # Returns (log_latents, latents, reconstruction)
        z: torch.Tensor = self.encoder(x)
        z_probs = z.exp()
        dist = td.OneHotCategorical(logits=z)
        straight_through_gradient = z_probs - z_probs.detach()
        if stochastic:
            z_sample = dist.sample() + straight_through_gradient
        else:
            z_sample = dist.mode + straight_through_gradient

        r = self.decoder(z_sample.flatten(1))

        return dist, z_sample, r

    def loss(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z_dist, _, r = self(x)
        reconstruction_loss = (r - y).square()

        if self.prior_dist.probs.device != self.prior_probs.device:
            self.prior_dist = td.OneHotCategorical(self.prior_probs)
            print("Remaking static dist")
        kl_loss = td.kl_divergence(z_dist, self.prior_dist)

        return (
            reconstruction_loss.sum([1, 2, 3]).mean(),
            kl_loss.sum(1).mean(),
        )


class Encoder(nn.Module):
    # (C, 64, 64) -> (E1,)

    def __init__(
        self,
        img_channels: int,
        channels: int = 32,
    ) -> None:
        super().__init__()
        kernels = (4, 4, 4, 4)
        layers = []
        for i, k in enumerate(kernels):
            if i == 0:
                in_channels = img_channels
            else:
                in_channels = channels * 2 ** (i - 1)
            out_channels = channels * 2**i
            layers.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    k,
                    2,
                    1,
                )
            )
            layers.append(ChLayerNorm(out_channels))
            layers.append(nn.SiLU())
        self.encoder = nn.Sequential(
            *layers,
            nn.Flatten(),  # These are image embeddings
        )
        self.output_size = out_channels * 4 * 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class Decoder(nn.Module):
    # (E2,) -> (C, 64, 64)

    def __init__(
        self,
        img_channels: int,
        in_features: int,
        channels: int = 32,
    ) -> None:
        super().__init__()
        # Prepends a linear [ N in_features ] -> [ N target_features ] =reshape= [ N 4 4 C ] -> conv decoder layers
        kernels = (4, 4, 4, 4)
        layers = []
        for i, k in enumerate(kernels):
            j = len(kernels) - i - 1
            in_channels = channels * 2**j

            if j == 0:
                out_channels = img_channels
            else:
                out_channels = channels * 2 ** (j - 1)

            layers.append(
                nn.ConvTranspose2d(
                    in_channels,
                    out_channels,
                    k,
                    2,
                    1,
                    bias=j == 0,
                )
            )
            if j != 0:
                layers.append(ChLayerNorm(out_channels))
                layers.append(nn.SiLU())

        first_channels = channels * 2 ** (len(kernels) - 1)
        self.decoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, first_channels * 4 * 4),
            nn.Unflatten(1, (first_channels, 4, 4)),
            *layers,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)


class ChLayerNorm(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=1e-3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [ N C H W ] -> [ N H W C ]
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        # [ N H W C ] -> [ N C H W ]
        return x.permute(0, 3, 1, 2)
