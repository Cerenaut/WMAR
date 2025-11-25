import random

import numpy as np
import torch
from sortedcontainers import SortedList

from rssm import ActionT, ContT, ImageT, ResetT
from wm import RewardT


class Replay:
    def __init__(self) -> None:
        self.n_valid = 0

    def add(
        self, acts: ActionT, obss: ImageT, rews: RewardT, conts: ContT, resets: ResetT
    ) -> None:
        raise NotImplementedError

    def minibatch(
        self, mb_t: int, mb_n: int, mb_device: str = "cuda"
    ) -> tuple[ActionT, ImageT, RewardT, ContT, ResetT]:
        # Data [ T N ... ]
        # Sample minibatches in [ t_size n_size ... ]
        t_size = min(mb_t, self.t)
        t_starts = np.random.randint(0, self.t - t_size + 1, size=mb_n)
        t_stops = t_starts + t_size
        ns = np.random.randint(0, self.n_valid, size=mb_n)

        mb_acts = torch.stack(
            [self.acts[t_start:t_stop, it] for t_start, t_stop, it in zip(t_starts, t_stops, ns)],
            dim=1,
        )
        mb_obss = torch.stack(
            [self.obss[t_start:t_stop, it] for t_start, t_stop, it in zip(t_starts, t_stops, ns)],
            dim=1,
        )
        mb_rews = torch.stack(
            [self.rews[t_start:t_stop, it] for t_start, t_stop, it in zip(t_starts, t_stops, ns)],
            dim=1,
        )
        mb_conts = torch.stack(
            [self.conts[t_start:t_stop, it] for t_start, t_stop, it in zip(t_starts, t_stops, ns)],
            dim=1,
        )
        mb_resets = torch.stack(
            [
                self.resets[t_start:t_stop, it]
                for t_start, t_stop, it in zip(t_starts, t_stops, ns)
            ],
            dim=1,
        )

        return (
            mb_acts.to(mb_device),
            mb_obss.to(mb_device),
            mb_rews.to(mb_device),
            mb_conts.to(mb_device),
            mb_resets.to(mb_device),
        )


class FifoReplay(Replay):
    def __init__(self, t: int, n: int, n_acts: int, store_device: str = "cpu") -> None:
        super().__init__()

        self.t = t
        self.n = n
        self.n_idx = 0
        self.n_valid = 0
        self.acts: ActionT = torch.zeros(t, n, n_acts).to(store_device)
        self.obss: ImageT = torch.zeros(t, n, 3, 64, 64).to(store_device)
        self.rews: RewardT = torch.zeros(t, n, 1).to(store_device)
        self.conts: ContT = torch.zeros(t, n, 1).to(store_device)
        self.resets: ResetT = torch.zeros(t, n, 1).to(store_device)

    def add(
        self, acts: ActionT, obss: ImageT, rews: RewardT, conts: ContT, resets: ResetT
    ) -> None:
        # Incoming shapes [ T N ... ]
        assert acts.shape[0] == self.t
        data_n = acts.shape[1]

        if self.n_idx + data_n <= self.n:
            self.acts[:, self.n_idx : self.n_idx + data_n] = acts
            self.obss[:, self.n_idx : self.n_idx + data_n] = obss
            self.rews[:, self.n_idx : self.n_idx + data_n] = rews
            self.conts[:, self.n_idx : self.n_idx + data_n] = conts
            self.resets[:, self.n_idx : self.n_idx + data_n] = resets
        else:
            n1 = self.n - self.n_idx
            n2 = data_n - n1
            self.acts[:, self.n_idx :] = acts[:, :n1]
            self.obss[:, self.n_idx :] = obss[:, :n1]
            self.rews[:, self.n_idx :] = rews[:, :n1]
            self.conts[:, self.n_idx :] = conts[:, :n1]
            self.resets[:, self.n_idx :] = resets[:, :n1]

            self.acts[:, :n2] = acts[:, -n2:]
            self.obss[:, :n2] = obss[:, -n2:]
            self.rews[:, :n2] = rews[:, -n2:]
            self.conts[:, :n2] = conts[:, -n2:]
            self.resets[:, :n2] = resets[:, -n2:]

        self.n_idx = (self.n_idx + data_n) % self.n
        self.n_valid = min(self.n_valid + data_n, self.n)


class LongTermReplay(Replay):
    Priority = float
    NIndex = int

    def __init__(self, t: int, n: int, n_acts: int, store_device: str = "cpu") -> None:
        super().__init__()

        self.t = t
        self.n = n
        self.acts: ActionT = torch.zeros(t, n, n_acts).to(store_device)
        self.obss: ImageT = torch.zeros(t, n, 3, 64, 64).to(store_device)
        self.rews: RewardT = torch.zeros(t, n, 1).to(store_device)
        self.conts: ContT = torch.zeros(t, n, 1).to(store_device)
        self.resets: ResetT = torch.zeros(t, n, 1).to(store_device)

        self.collection = SortedList([(float("-inf"), _n) for _n in range(n)])
        self.n_valid = 0

    def add(self, acts: ActionT, obss: ImageT, rews: RewardT, conts: ContT, resets: ResetT) -> None:
        assert acts.shape[0] == self.t
        data_n = acts.shape[1]

        for n in range(data_n):
            least_prio, least_index = self.collection[0]
            rand_prio = np.random.randn()
            if rand_prio > least_prio:
                del self.collection[0]
                self.collection.add((rand_prio, least_index))
                self.n_valid = min(self.n, self.n_valid + 1)

                self.acts[:, least_index] = acts[:, n]
                self.obss[:, least_index] = obss[:, n]
                self.rews[:, least_index] = rews[:, n]
                self.conts[:, least_index] = conts[:, n]
                self.resets[:, least_index] = resets[:, n]


class MultiTypeReplay(Replay):
    def __init__(self, *replays: Replay) -> None:
        super().__init__()
        self.replays = replays

    @property
    def n_valid(self) -> int:
        return self.replays[0].n_valid

    @n_valid.setter
    def n_valid(self, _: int) -> None:
        return

    def add(self, acts: ActionT, obss: ImageT, rews: RewardT, conts: ContT, resets: ResetT) -> None:
        for replay in self.replays:
            replay.add(acts, obss, rews, conts, resets)

    def minibatch(self, mb_t: int, mb_n: int, mb_device: str = "cuda") -> tuple[ActionT, ImageT, RewardT, ContT, ResetT]:
        return random.choice(self.replays).minibatch(mb_t, mb_n, mb_device)
