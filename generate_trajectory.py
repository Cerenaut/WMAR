from typing import Any, Callable, Optional

import cv2
import gymnasium as gym
import ale_py
import numpy as np
import torch
import torch.distributions as td
from gymnasium.vector import AsyncVectorEnv
from gymnasium.wrappers import AtariPreprocessing
from tqdm import tqdm

from ac import ActorCritic, zh_to_ac_state
from rssm import ActionT, ContT, ImageT, ResetT
from wm import RewardT, WorldModel


class EnvironmentSchedule:
    def __init__(self, n_sync: int, templates: list[Callable[[], Any]]) -> None:
        self._step = 0
        self.templates = templates
        self.n_sync = n_sync

    def step(self) -> None:
        self._step += 1

    def funcs(self) -> list[Callable[[], Any]]:
        raise NotImplementedError

    def eval_funcs(self) -> list[list[Callable[[], Any]]]:
        return [[t for _ in range(self.n_sync)] for t in self.templates]

    def is_new_env(self) -> bool:
        raise NotImplementedError


class AllEnvironments(EnvironmentSchedule):
    def __init__(self, n_sync: int, templates: list[Callable[[], Any]]) -> None:
        super().__init__(n_sync, templates)

    def funcs(self) -> list[Callable[[], Any]]:
        res = []
        for _ in range(self.n_sync):
            res.append(np.random.choice(self.templates))
        return res

    def is_new_env(self) -> bool:
        return self._step == 0


class SequentialEnvironments(EnvironmentSchedule):
    def __init__(self, n_sync: int, templates: list[Callable[[], Any]], swap_sched: int) -> None:
        super().__init__(n_sync, templates)
        self.swap_sched = swap_sched

    def funcs(self) -> list[Callable[[], Any]]:
        i = (self._step // self.swap_sched) % len(self.templates)
        return [self.templates[i] for _ in range(self.n_sync)]

    def is_new_env(self) -> bool:
        return (
            self._step % self.swap_sched == 0
            and self._step < len(self.templates) * self.swap_sched
        )


class SyncVectorEnvAtHome:
    @staticmethod
    def resize64(obs):
        if len(obs) == 2:
            obs = obs[0]
        assert len(obs.shape) == 3, obs.shape
        return cv2.resize(obs, (64, 64))

    def __init__(self, create_fns, repeat: int = 1) -> None:
        self.create_fns = create_fns
        self.repeat = repeat
        self.envs = [f() for f in self.create_fns]

    def reset(self) -> np.ndarray:
        return np.stack([self.resize64(e.reset()) for e in self.envs])

    def step(self, act):
        w, x, y, z = [], [], [], []
        for a, e in zip(act, self.envs):
            # obs rew reset
            # _w, _x, _y, _z = e.step(a)
            _x_acc = 0  # reward accumulator
            for _ in range(self.repeat):
                try:
                    dat = e.step(a)
                except IndexError:
                    dat = e.step(a % e.action_space.n)
                if len(dat) == 5:
                    _w, _x, _y1, _y2, _z = dat
                    _y = _y1 or _y2
                else:
                    _w, _x, _y, _z = dat
                _x_acc += _x
                if _y:
                    _w = e.reset()
                    break
            w.append(SyncVectorEnvAtHome.resize64(_w))
            x.append(_x)
            y.append(_y)
            z.append(_z)
        return np.stack(w), np.stack(x), np.stack(y), np.stack(z)


@torch.no_grad()
def evaluate(
    n_sync: int,
    wm: Optional[WorldModel] = None,
    ac: Optional[ActorCritic] = None,
    env_fns: Optional[list[Callable[[], Any]]] = None,
    env_repeat: int = 4,
    n_rollouts: int = 10,
) -> tuple[float, float]:
    _, _, rews, conts, resets = generate_trajectories(
        n_rollouts * 2**13 // n_sync,
        n_sync,
        wm,
        ac,
        env_fns,
        env_repeat,
        n_rollouts,
        no_images=True,
    )
    terms = torch.where(conts == 0)[0]
    starts = torch.where(resets == 1)[0]
    collection = [(t.item(), "E") for t in terms] + [(s.item(), "S") for s in starts]
    collection.sort()
    where_se = []
    for i in range(len(collection) - 1):
        (si, s), (ei, e) = collection[i : i + 2]
        if s == "S" and e == "E":
            where_se.append((si, ei))
    eps_rews = [rews[s : e + 1].sum().item() for s, e in where_se]
    if not eps_rews:
        n_eps = resets.sum().item()
        var = rews.var().item() * rews.numel() / n_eps**2
        return rews.sum().item() / n_eps, np.sqrt(var)
    return np.mean(eps_rews), np.std(eps_rews)


@torch.no_grad()
def generate_trajectories(
    n: int,
    n_sync: int,
    wm: Optional[WorldModel] = None,
    ac: Optional[ActorCritic] = None,
    env_fns: Optional[list[Callable[[], Any]]] = None,
    env_repeat: int = 4,
    target_terminals: Optional[int] = None,
    no_images: bool = False,
) -> tuple[ActionT, Optional[ImageT], RewardT, ContT, ResetT]:
    # Returns [ X ... ] packed as [ N*T ... ] (sort of)
    # To change to [ T N ... ], do reshape and swapaxes
    # `target_terminals` if not None, forces at least some number of environment resets
    # (not including initial resets)

    class DummyList(list):
        def append(self, __object: Any) -> None:
            return

    acts = [[] for _ in range(n_sync)]  # [ N T ] int
    obss = [DummyList() if no_images else [] for _ in range(n_sync)]  # [ N T 64 64 3 ] uint8
    rews = [[] for _ in range(n_sync)]  # [ N T ] float
    conts = [[] for _ in range(n_sync)]  # [ N T ] bool
    # Important: should be 1 for new sequence
    resets = [[] for _ in range(n_sync)]  # [ N T ] bool
    n_samples = 0
    n_terminals = 0

    env = AsyncVectorEnv(
        [
            *map(
                lambda env: lambda: AtariPreprocessing(
                    env(), frame_skip=env_repeat, screen_size=64, grayscale_obs=False
                ),
                [
                    env_fns[i]
                    if env_fns is not None
                    else (
                        lambda: gym.make(
                            "ALE/DonkeyKong-v5",
                            frameskip=1,
                            repeat_action_probability=0,
                        )
                    )
                    for i in range(n_sync)
                ],
            )
        ],
    )
    z = None

    if wm is not None and ac is not None:
        post = " (+WM/AC)"
    else:
        post = ""
    with tqdm(total=n, desc=f"Generating trajectories{post}") as progbar:
        while n_samples < n:
            _n_samples = n_samples
            if target_terminals is not None and n_terminals >= target_terminals:
                break
            if n_samples == 0:  # First step
                n_samples += n_sync
                obs, _ = env.reset()
                for i in range(n_sync):
                    acts[i].append(0)
                    obss[i].append(obs[i])
                    rews[i].append(0)
                    conts[i].append(True)
                    resets[i].append(True)
                reset = np.zeros(n_sync, dtype=bool)
                continue

            n_samples += n_sync
            if wm is None or ac is None:
                act = np.random.randint(0, 18, size=n_sync)
            else:
                if z is None:
                    z, h = wm.rssm.initial_state(n_sync)
                    act_t = torch.zeros(n_sync, 18, device=z.device)
                    act_t[:, 0] = 1  # Previous move would have been all 0s
                # Follow a stochastic policy
                _, z, h = wm.rssm(
                    z,
                    act_t,
                    h,
                    torch.from_numpy(obs / 255).float().permute(0, 3, 1, 2).to(z.device),
                    torch.from_numpy(reset).float().unsqueeze(-1).to(z.device),
                )
                ac_state = zh_to_ac_state(z, h)
                act_prob = ac.actor(ac_state)
                act_prob_dist = td.Categorical(logits=act_prob)
                act = act_prob_dist.sample()
                act_t = torch.nn.functional.one_hot(act, 18)
                act = act.cpu().numpy()

            obs, rew, term, trunc, _ = env.step(act)
            reset = term | trunc
            # When there is an episode termination (`reset`):
            # `obs` is of the new episode
            # `rew` is of the previous episode
            # Final observation information not used due to inconsistency with procgen
            # This loses a frame but this change is insignificant
            for i in range(n_sync):
                acts[i].append(act[i])
                obss[i].append(obs[i])
                rews[i].append(rew[i])
                conts[i].append(True)
                resets[i].append(reset[i])
                if reset[i]:
                    conts[i][-2] = False
                    rews[i][-2] = rew[i]
                    rews[i][-1] = 0
                    n_terminals += 1

            progbar.update(n_samples - _n_samples)

    acts = [np.stack(e) for e in acts]
    obss = [np.stack(e) for e in obss] if not no_images else None
    rews = [np.stack(e) for e in rews]
    conts = [np.stack(e) for e in conts]
    resets = [np.stack(e) for e in resets]

    return (
        torch.nn.functional.one_hot(torch.from_numpy(np.concatenate(acts)[:n]).long(), 18).float(),
        torch.from_numpy(np.concatenate(obss)[:n] / 255).float().permute(0, 3, 1, 2)
        if not no_images
        else None,
        torch.from_numpy(np.concatenate(rews)[:n]).float().unsqueeze(-1),
        torch.from_numpy(np.concatenate(conts)[:n]).float().unsqueeze(-1),
        torch.from_numpy(np.concatenate(resets)[:n]).float().unsqueeze(-1),
    )


def reinterpret_nt_to_t_n(
    acts: ActionT, obss: ImageT, rews: RewardT, conts: ContT, resets: ResetT, t: int, n: int
) -> tuple[ActionT, ImageT, RewardT, ContT, ResetT]:
    if t * n != acts.shape[0]:
        raise ValueError(f"Illegal reinterpret (acts.shape={acts.shape}[0] != {t * n})")
    return (
        acts.reshape(n, t, 18).swapaxes(0, 1),
        obss.reshape(n, t, 3, 64, 64).swapaxes(0, 1),
        rews.reshape(n, t, 1).swapaxes(0, 1),
        conts.reshape(n, t, 1).swapaxes(0, 1),
        resets.reshape(n, t, 1).swapaxes(0, 1),
    )
