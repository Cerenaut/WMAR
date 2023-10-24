from typing import Callable, NamedTuple, Optional

import numpy as np
import torch
import torch.distributions as td
import torch.nn as nn
from torch.optim import Adam, Optimizer
from tqdm import trange

from replay import Replay
from rssm import ActionT, HiddenT, LatentShape, LatentT, get_mlp_layers
from wm import RewardSymlogT, RewardT, WorldModel, symexp, symlog

ActionLogT = torch.Tensor
AcStateT = torch.Tensor
RewardSymlogCatT = torch.Tensor
ReturnT = torch.Tensor
# ActionLogT (log probs): [ N n_acts ]
# AcState (concatenation of flattened LatentT and HiddenT): [ N n_dis*n_cls+h_dim ]
# RewardSymlogCatT (probs): [ N 255 ]
# RewardT (real): [ N 1 ]
N_CRITIC_BINS = 255


def zh_to_ac_state(z: LatentT, h: HiddenT) -> AcStateT:
    return torch.cat((z.flatten(-2), h), dim=-1)


def ac_state_to_zh(state: AcStateT, ls: LatentShape, h_dim: int) -> tuple[LatentT, HiddenT]:
    z, h = state[..., :-h_dim], state[..., -h_dim:]
    return z.unflatten(-1, ls), h


def rew_symlog_to_2hot(x: RewardSymlogT) -> RewardSymlogCatT:
    hi = 20
    scale = N_CRITIC_BINS // 2 / hi
    x = x * scale
    b = x - x.floor()
    a = 1 - b
    res = torch.zeros(*x.shape[:-1], N_CRITIC_BINS, device=x.device)
    # If you get some weird CUDA assert error, it's because `x` is under/overflowing here
    res.scatter_(-1, x.floor().long() + N_CRITIC_BINS // 2, a)
    res.scatter_(-1, x.floor().long() + N_CRITIC_BINS // 2 + 1, b)
    return res


class ActorCritic(nn.Module):
    def __init__(self, in_dim: int, act_space: int) -> None:
        super().__init__()
        self.actor: Callable[[AcStateT], ActionLogT] = nn.Sequential(
            *get_mlp_layers(in_dim, act_space, final_activation=None),
            nn.LogSoftmax(-1),
        )
        self.symlog_bins: torch.Tensor
        self.register_buffer(
            "symlog_bins", torch.linspace(-20, 20, N_CRITIC_BINS).float().unsqueeze(1)
        )

        critic_fcs = get_mlp_layers(in_dim, N_CRITIC_BINS, final_activation=None)
        # DreamerV3 0 init weights of output layer to 0s
        torch.nn.init.constant_(critic_fcs[-1].weight, 0)
        torch.nn.init.constant_(critic_fcs[-1].bias, 0)
        self.critic: Callable[[AcStateT], RewardSymlogCatT] = nn.Sequential(
            *critic_fcs,
            nn.LogSoftmax(-1),
        )

    def __call__(self, state: AcStateT) -> tuple[ActionLogT, RewardT]:
        return super().__call__(state)

    def forward(self, state: AcStateT) -> tuple[ActionLogT, RewardT]:
        # Supports T dimension
        critic_bins = self.critic(state).exp()
        return self.actor(state), symexp(critic_bins @ self.symlog_bins)

    def compute_loss(
        self,
        states: AcStateT,
        actions: ActionT,
        lam_returns: ReturnT,
        scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        action_logs = self.actor(states)
        critic_preds_log = self.critic(states)

        # Actor gradients
        # `action_logs` (log probs): [ T N n_acts ]
        # `action_sample_logs` (log probs): [ T N 1 ]
        action_sample_logs = (action_logs * actions).sum(-1, keepdim=True)
        reinforce = (
            -action_sample_logs
            * (lam_returns - symexp(critic_preds_log.detach().exp() @ self.symlog_bins))
            / scale
        ).mean()

        # [ T N n_acts ]
        entropy = td.Categorical(logits=action_logs).entropy().mean()

        # Critic gradients
        critic_targets = rew_symlog_to_2hot(symlog(lam_returns))
        critic_loss = -(critic_preds_log * critic_targets).sum(-1).mean()

        return reinforce, entropy, critic_loss


class ActorCriticOpt(NamedTuple):
    ac: ActorCritic
    opt: Optimizer


@torch.no_grad()
def dream_rollout(
    wm: WorldModel,
    ac: ActorCritic,
    data: Replay,
    n_sync: int = 1,
    n_steps: int = 16,
    discount: float = 0.997,
    lam: float = 0.95,
    temperature: float = 1.0,
    n_ctx_frames: int = 4,
) -> tuple[AcStateT, ActionT, RewardT, ReturnT]:
    # Returns: (T=n_steps N=n_sync)
    # States [ T N n_dis n_cls ]
    # Actions [ T N 18 ]
    # Rewards [ T N 1 ]
    # Lambda returns: [ T N 1 ]
    z, h = wm.rssm.initial_state(n_sync)
    no_reset = torch.zeros(n_sync, 1, device=z.device)
    # Arbitrary (n_ctx_frames) context frames
    ctx_acts, ctx_images, _, _, ctx_resets = data.minibatch(
        n_ctx_frames, n_sync, mb_device=z.device
    )
    assert ctx_images.shape == (n_ctx_frames, n_sync, 3, 64, 64), ctx_images.shape
    _, z, h = wm.rssm(z, ctx_acts, h, ctx_images, ctx_resets, temperature=temperature)
    z = z[-1]
    h = h[-1]

    states = []
    actions = []
    rewards = []
    returns_preds = []
    conts = []
    for _ in range(n_steps):
        state = zh_to_ac_state(z, h)
        zh = wm.zh_transform(z, h)
        reward = symexp(wm.reward_fc(zh))
        cont = wm.continue_fc(zh)
        action_log, returns_pred = ac(state)
        action_dist = td.OneHotCategorical(logits=action_log)
        action = action_dist.sample()

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        returns_preds.append(returns_pred)
        conts.append(cont)

        _, z, h = wm.rssm(z, action, h, None, no_reset, temperature=temperature)

    states = torch.stack(states)
    actions = torch.stack(actions)
    rewards = torch.stack(rewards)
    returns_preds = torch.stack(returns_preds)

    # Compute returns
    lam_returns = torch.zeros_like(returns_preds, device=returns_preds.device)
    for t in reversed(range(n_steps)):
        if t == n_steps - 1:
            # Set the last step to just equal discount*final returns preds
            _, final_returns_pred = ac(state)
            next_returns_pred = final_returns_pred
            next_r = final_returns_pred
        else:
            next_returns_pred = returns_preds[t + 1]
            next_r = lam_returns[t + 1]
        lam_returns[t] = rewards[t] + discount * conts[t] * (
            (1 - lam) * next_returns_pred + lam * next_r
        )

    return states, actions, rewards, lam_returns


def train_ac_from_wm(
    wm: WorldModel,
    data: Replay,
    steps: int,
    n_sync: int = 16,
    dream_steps: int = 16,
    aco: Optional[ActorCriticOpt] = None,
    lr: float = 3e-5,
) -> tuple[ActorCriticOpt, torch.Tensor]:
    if aco is None:
        ac = ActorCritic(np.prod(wm.ls) + wm.h_dim, wm.a_dim).cuda()
        aco = ActorCriticOpt(ac, Adam(ac.parameters(), lr=lr))
    ac, opt = aco
    for g in opt.param_groups:
        g["lr"] = lr
    scale_ema = None
    lam_returns_mean_ema = None

    progbar = trange(steps, desc="Train AC from WM")
    for step in progbar:
        states, actions, _, lam_returns = dream_rollout(
            wm, ac, data, n_sync=n_sync, n_steps=dream_steps
        )

        scale = torch.quantile(lam_returns, 0.95) - torch.quantile(lam_returns, 0.05)
        lam_returns_mean = lam_returns.mean()
        if scale_ema is None:
            scale_ema = scale
            lam_returns_mean_ema = lam_returns_mean
        else:
            scale_ema = 0.99 * scale_ema + 0.01 * scale
            lam_returns_mean_ema = 0.99 * lam_returns_mean_ema + 0.01 * lam_returns_mean

        one = torch.tensor(1, device=scale.device)
        reinforce, entropy, critic_loss = ac.compute_loss(
            states, actions, lam_returns, torch.max(one, scale_ema)
        )
        loss = reinforce - 3e-4 * entropy + critic_loss

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ac.parameters(), 100)
        opt.step()

        if step % 50 == 0:
            progbar.set_postfix(
                {
                    "Actor entropy": f"{entropy.item():.3f}",
                    "Lam returns mean EMA": f"{lam_returns_mean_ema.item():.3f}",
                }
            )

    return aco, lam_returns_mean_ema
