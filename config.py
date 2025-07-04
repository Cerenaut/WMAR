import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Type, TypeVar, Union

import gym

import generate_trajectory
import replay
from generate_trajectory import EnvironmentSchedule
from replay import FifoReplay, LongTermReplay, MultiTypeReplay, Replay

T = TypeVar("T", bound="Serialisable")


@dataclass
class Serialisable:
    @classmethod
    def from_file(cls: Type[T], path: Path) -> T:
        with open(path, "r") as fp:
            data = json.load(fp)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls: Type[T], data: dict[str, Any]) -> T:
        return cls(**data)

    def save(self, path: Path) -> None:
        with open(path, "w") as fp:
            json.dump(self.to_dict(), fp, indent=4)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EnvConfig(Serialisable):
    # Name like "CoinRun+NB+RT+MA"
    name: str
    kwargs: dict[str, Any] = field(default_factory=dict)
    rew_scale: float = 1

    def __post_init__(self) -> None:
        assert self.rew_scale == 1

    def get_function(self) -> Callable[[], Any]:
        default = {
            "use_backgrounds": True,
            "restrict_themes": False,
            "use_monochrome_assets": False,
        }
        mods = {
            "NB": {"use_backgrounds": False},
            "RT": {"restrict_themes": True},
            "MA": {"use_monochrome_assets": True}
        }
        parts = self.name.split("+")
        assert parts[0] == "CoinRun"
        for part in parts[1:]:
            default.update(mods[part])
        return lambda: gym.make(
            "procgen:procgen-coinrun-v0",
            **default,
        )


@dataclass
class EnvScheduleConfig(Serialisable):
    env_schedule_type: Type[EnvironmentSchedule]
    env_configs: list[EnvConfig]
    kwargs: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls: type[T], data: dict[str, Any]) -> T:
        data = data.copy()
        data["env_schedule_type"] = getattr(generate_trajectory, data["env_schedule_type"])
        data["env_configs"] = [EnvConfig.from_dict(d) for d in data["env_configs"]]
        return cls(**data)

    def __post_init__(self) -> None:
        assert self.env_schedule_type != EnvironmentSchedule
        assert len(self.env_configs)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["env_schedule_type"] = self.env_schedule_type.__name__
        data["env_configs"] = [c.to_dict() for c in self.env_configs]
        return data


@dataclass
class RbConfig(Serialisable):
    rb_type: Union[Type[FifoReplay], Type[LongTermReplay]]
    rb_device: str = "cuda"

    @classmethod
    def from_dict(cls: Type[T], data: dict[str, Any]) -> T:
        data = data.copy()
        data["rb_type"] = getattr(replay, data["rb_type"])
        return cls(**data)

    def __post_init__(self) -> None:
        assert self.rb_type in {FifoReplay, LongTermReplay}

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["rb_type"] = self.rb_type.__name__
        return data


@dataclass
class Config(Serialisable):
    esc: EnvScheduleConfig

    seed: int = 1337

    epochs: int = 10_000
    wm_lr: float = 4e-4
    log_frequency: int = 800
    steps_per_batch: int = 1600
    ac_train_steps: int = 800
    ac_train_sync: int = 128
    # False = do not create fresh ac
    # True = create fresh ac every epoch
    # int = create fresh ac every n epochs
    fresh_ac: Union[bool, int] = False

    n_sync: int = 2
    gen_seq_len: int = 4096
    env_repeat: int = 1
    data_n: int = 16
    data_n_max: int = 512
    data_t: int = 512

    mb_t_size: int = 32
    mb_n_size: int = 16

    random_policy: Union[Literal["first"], Literal["new"]] = "first"

    pretrain_enabled: bool = True
    pretrain_data_multiplier: int = 4
    pretrain_mb_t_size: int = 8
    pretrain_mb_n_size: int = 16
    pretrain_steps: int = 32_000

    gru_units: int = 512
    cnn_depth: int = 32
    mlp_features: int = 512
    mlp_layers: int = 2
    wall_time_optimisation: bool = False

    action_space: int = 18
    replay_buffers: list[RbConfig] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        data = data.copy()
        data["esc"] = EnvScheduleConfig.from_dict(data["esc"])
        data["replay_buffers"] = [RbConfig.from_dict(d) for d in data["replay_buffers"]]
        return cls(**data)

    def __post_init__(self) -> None:
        assert self.n_sync * self.gen_seq_len == self.data_n * self.data_t
        assert self.random_policy in {"first", "new"}
        assert self.replay_buffers != []
        assert self.env_repeat == 1, "Env repeat disabled for procgen"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["esc"] = self.esc.to_dict()
        data["replay_buffers"] = [c.to_dict() for c in self.replay_buffers]
        return data

    def get_env_schedule(self) -> EnvironmentSchedule:
        return self.esc.env_schedule_type(
            self.n_sync, [e.get_function() for e in self.esc.env_configs], **self.esc.kwargs
        )

    def get_replay_buffer(self) -> Replay:
        if len(self.replay_buffers) > 1:
            return MultiTypeReplay(
                *[
                    rc.rb_type(self.data_t, self.data_n_max, self.action_space, rc.rb_device)
                    for rc in self.replay_buffers
                ]
            )
        rc = self.replay_buffers[0]
        return rc.rb_type(self.data_t, self.data_n_max, self.action_space, rc.rb_device)
