from typing import Any, Callable, Optional, List

import os
import cv2

# IMPORTANT: On some HPC systems (e.g., MPICH/Hydra-based clusters), vendor MPI
# libraries can auto-initialize if certain environment variables are present
# (PMI_*, OMPI_*, MPI_*). Procgen/gym does not require MPI. Unset these to avoid
# accidental MPI_Init in system libraries that leads to PMI errors.
for _var in list(os.environ.keys()):
    if _var.startswith(("PMI_", "OMPI_", "MPI_")):
        os.environ.pop(_var, None)

import gym
import numpy as np
import torch
import torch.distributions as td
from tqdm import tqdm

from ac import ActorCritic, zh_to_ac_state
from rssm import ActionT, ContT, ImageT, ResetT
from wm import RewardT, WorldModel
from typing import Tuple


class EnvironmentSchedule:
    def __init__(self, n_sync: int, templates: List[Callable[[], Any]]) -> None:
        self._step = 0
        self.templates = templates
        self.n_sync = n_sync

    def step(self) -> None:
        self._step += 1

    def funcs(self) -> List[Callable[[], Any]]:
        raise NotImplementedError

    def eval_funcs(self) -> List[List[Callable[[], Any]]]:
        return [[t for _ in range(self.n_sync)] for t in self.templates]

    def is_new_env(self) -> bool:
        raise NotImplementedError


class AllEnvironments(EnvironmentSchedule):
    def __init__(self, n_sync: int, templates: List[Callable[[], Any]]) -> None:
        super().__init__(n_sync, templates)

    def funcs(self) -> List[Callable[[], Any]]:
        res = []
        for _ in range(self.n_sync):
            res.append(np.random.choice(self.templates))
        return res

    def is_new_env(self) -> bool:
        return self._step == 0


class SequentialEnvironments(EnvironmentSchedule):
    def __init__(self, n_sync: int, templates: List[Callable[[], Any]], swap_sched: int) -> None:
        super().__init__(n_sync, templates)
        self.swap_sched = swap_sched

    def funcs(self) -> List[Callable[[], Any]]:
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
    env_fns: Optional[List[Callable[[], Any]]] = None,
    env_repeat: int = 1,
    n_rollouts: int = 100,
) -> Tuple[float, float]:
    env = SyncVectorEnvAtHome(
        [
            env_fns[i]
            if env_fns is not None
            else (lambda: gym.make("procgen:procgen-coinrun-v0"))
            for i in range(n_sync)
        ],
        repeat=env_repeat,
    )
    obs = env.reset()
    ep_returns: list[float] = []
    ret = np.zeros(n_sync, dtype=np.float32)
    while len(ep_returns) < n_rollouts:
        if ac is not None:
            policy_device = next(ac.parameters()).device if isinstance(ac, torch.nn.Module) else "cpu"
            obs_t = torch.from_numpy(obs / 255).float().permute(0, 3, 1, 2).to(policy_device)
            logits = ac(obs_t)
            act = torch.argmax(logits, dim=-1).cpu().numpy()
        else:
            act = np.random.randint(0, 15, size=n_sync)

        obs, rew, reset, _ = env.step(act)
        ret += rew
        for i in range(n_sync):
            if reset[i]:
                ep_returns.append(float(ret[i]))
                ret[i] = 0.0
                if len(ep_returns) >= n_rollouts:
                    break
    return float(np.mean(ep_returns)), float(np.std(ep_returns))


@torch.no_grad()
def generate_trajectories(
    n: int,
    n_sync: int,
    wm: Optional[WorldModel] = None,
    ac: Optional[ActorCritic] = None,
    env_fns: Optional[List[Callable[[], Any]]] = None,
    env_repeat: int = 1,
    target_terminals: Optional[int] = None,
    no_images: bool = False,
    deterministic: bool = False,
    collect_eps: float = 0.05,
) -> Tuple[ActionT, ImageT, RewardT, ContT, ResetT]:
    assert env_repeat == 1
    # Returns [ X ... ] packed as [ N*T ... ] (sort of)
    # To change to [ T N ... ], do reshape and swapaxes
    # `target_terminals` if not None, forces at least some number of environment resets
    # (not including initial resets)

    class DummyList(List):
        def append(self, __object: Any) -> None:
            return

    acts = [[] for _ in range(n_sync)]  # [ N T ] int
    obss = [DummyList() if no_images else [] for _ in range(n_sync)]  # [ N T 64 64 3 ] uint8
    rews = [[] for _ in range(n_sync)]  # [ N T ] float
    conts = [[] for _ in range(n_sync)]  # [ N T ] bool
    resets = [[] for _ in range(n_sync)]  # [ N T ] bool
    n_samples = 0
    n_terminals = 0

    
    env = SyncVectorEnvAtHome(
        [
            env_fns[i]
            if env_fns is not None
            else (
                lambda: gym.make(
                    "procgen:procgen-coinrun-v0",
                )
                
            )
            for i in range(n_sync)
        ]
    )
    z = None

    if wm is not None and ac is not None:
        post = " (+WM/AC)"
    else:
        post = ""
    with tqdm(total=n, desc=f"Generating trajectories{post}", disable=True) as progbar:
        while n_samples < n:
            _n_samples = n_samples
            if target_terminals is not None and n_terminals >= target_terminals:
                break
            if n_samples == 0:  # First step
                n_samples += n_sync
                obs = env.reset()
                for i in range(n_sync):
                    acts[i].append(0)
                    obss[i].append(obs[i])
                    rews[i].append(0)
                    conts[i].append(True)
                    resets[i].append(True)
                reset = np.zeros(n_sync, dtype=bool)
                continue

            n_samples += n_sync
            if wm is not None and ac is not None:
                if z is None:
                    z, h = wm.rssm.initial_state(n_sync)
                    act_t = torch.zeros(n_sync, 15, device=z.device)
                    act_t[:, 0] = 1
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
                act_t = torch.nn.functional.one_hot(act, 15)
                act = act.cpu().numpy()
            elif ac is not None:
                with torch.no_grad():
                    policy_device = next(ac.parameters()).device if isinstance(ac, torch.nn.Module) else "cpu"
                    obs_t = torch.from_numpy(obs / 255).float().permute(0, 3, 1, 2).to(policy_device)
                    logits = ac(obs_t)
                    if deterministic:
                        act = logits.argmax(dim=-1).cpu().numpy()
                    else:
                        dist = td.Categorical(logits=logits)
                        act = dist.sample().cpu().numpy()
                        # Epsilon-greedy exploration during data collection to avoid early collapse
                        if collect_eps > 0.0:
                            mask = np.random.rand(n_sync) < collect_eps
                            rand_act = np.random.randint(0, 15, size=n_sync)
                            act = np.where(mask, rand_act, act)
            else:
                act = np.random.randint(0, 15, size=n_sync)

            obs, rew, reset, _ = env.step(act)
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
        torch.nn.functional.one_hot(torch.from_numpy(np.concatenate(acts)[:n]).long(), 15).float(),
        torch.from_numpy(np.concatenate(obss)[:n] / 255).float().permute(0, 3, 1, 2)
        if not no_images
        else None,
        torch.from_numpy(np.concatenate(rews)[:n]).float().unsqueeze(-1),
        torch.from_numpy(np.concatenate(conts)[:n]).float().unsqueeze(-1),
        torch.from_numpy(np.concatenate(resets)[:n]).float().unsqueeze(-1),
    )


def reinterpret_nt_to_t_n(
    acts: ActionT, obss: ImageT, rews: RewardT, conts: ContT, resets: ResetT, t: int, n: int
) -> Tuple[ActionT, ImageT, RewardT, ContT, ResetT]:
    if t * n != acts.shape[0]:
        raise ValueError(f"Illegal reinterpret (acts.shape={acts.shape}[0] != {t * n})")
    return (
        acts.reshape(n, t, 15).swapaxes(0, 1),
        obss.reshape(n, t, 3, 64, 64).swapaxes(0, 1),
        rews.reshape(n, t, 1).swapaxes(0, 1),
        conts.reshape(n, t, 1).swapaxes(0, 1),
        resets.reshape(n, t, 1).swapaxes(0, 1),
    )
