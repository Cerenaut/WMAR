import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Type, TypeVar, Union, Dict, List
import warnings
# Must be before any Gym imports!
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module=r"gym\.utils\.passive_env_checker"
)
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    module=r"gym\.utils\.passive_env_checker"
)

import logging
import gymnasium as gym

# silence everything below ERROR in passive_env_checker
logging.getLogger("gym.utils.passive_env_checker").setLevel(logging.ERROR)

# — or — silence all gym warnings —
gym.logger.set_level(gym.logger.ERROR)

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
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        return cls(**data)

    def save(self, path: Path) -> None:
        with open(path, "w") as fp:
            json.dump(self.to_dict(), fp, indent=4)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EnvConfig(Serialisable):
    # Name like "CoinRun+NB+RT+MA"
    name: str
    kwargs: Dict[str, Any] = field(default_factory=Dict)
    rew_scale: float = 1

    def __post_init__(self) -> None:
        assert self.rew_scale == 1

    def get_function(self) -> Callable[[], Any]:
        default = {
            "use_backgrounds": True,
            "restrict_themes": False,
            "use_monochrome_assets": False,
            "use_generated_assets": False,   # Procgen option
            "center_agent": True,            # Procgen option

        }
        mods = {
            "NB": {"use_backgrounds": False},
            "RT": {"restrict_themes": True},
            "MA": {"use_monochrome_assets": True},
            "UGA": {"use_generated_assets": True},   # “use-generated-assets”
            "CA":  {"center_agent": False}        # “center-agent = False”

        }
        parts = self.name.split("+")
        assert parts[0] == "CoinRun"
        for part in parts[1:]:
            default.update(mods[part])
        class GymV26Adapter(gym.Env):
            metadata = {"render_modes": [None]}

            def __init__(self, legacy_env):
                super().__init__()
                self.legacy_env = legacy_env
                # spaces
                self.action_space = getattr(legacy_env, "action_space", None)
                self.observation_space = getattr(legacy_env, "observation_space", None)
                # render_mode attribute
                try:
                    self.render_mode = getattr(legacy_env, "render_mode", None)
                except Exception:
                    self.render_mode = None

            def reset(self, *, seed: int | None = None, options: dict | None = None):
                if seed is not None and hasattr(self.legacy_env, "seed"):
                    try:
                        self.legacy_env.seed(seed)
                    except Exception:
                        pass
                out = self.legacy_env.reset()
                if isinstance(out, tuple) and len(out) == 2:
                    obs, info = out
                else:
                    obs, info = out, {}
                return obs, info

            def step(self, action):
                out = self.legacy_env.step(action)
                if len(out) == 5:
                    return out
                obs, reward, done, info = out
                terminated, truncated = bool(done), False
                return obs, reward, terminated, truncated, info

            def render(self):
                if hasattr(self.legacy_env, "render"):
                    return self.legacy_env.render()
                return None

            def close(self):
                if hasattr(self.legacy_env, "close"):
                    return self.legacy_env.close()

        def create_env():
            # Build procgen directly without registry, then adapt to Gymnasium API
            try:
                from procgen.env import ProcgenGym3Env  # type: ignore
                from gym3 import ToGymEnv  # type: ignore
            except Exception as e:
                raise RuntimeError(f"Procgen/gym3 packages not available: {e}")

            base = self.name.split("+")[0].lower()  # e.g., 'coinrun'
            # Map our default kwargs directly to ProcgenEnv
            pg = ProcgenGym3Env(num=1, env_name=base, **default)
            legacy_gym = ToGymEnv(pg)  # legacy Gym API from gym3 expects gym3 env
            return GymV26Adapter(legacy_gym)
        
        return create_env


@dataclass
class EnvScheduleConfig(Serialisable):
    env_schedule_type: Type[EnvironmentSchedule]
    env_configs: List[EnvConfig]
    kwargs: Dict[str, Any] = field(default_factory=Dict)

    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        data = data.copy()
        data["env_schedule_type"] = getattr(generate_trajectory, data["env_schedule_type"])
        data["env_configs"] = [EnvConfig.from_dict(d) for d in data["env_configs"]]
        return cls(**data)

    def __post_init__(self) -> None:
        assert self.env_schedule_type != EnvironmentSchedule
        assert len(self.env_configs)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["env_schedule_type"] = self.env_schedule_type.__name__
        data["env_configs"] = [c.to_dict() for c in self.env_configs]
        return data


@dataclass
class RbConfig(Serialisable):
    rb_type: Union[Type[FifoReplay], Type[LongTermReplay]]
    rb_device: str = "cuda"

    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        data = data.copy()
        data["rb_type"] = getattr(replay, data["rb_type"])
        return cls(**data)

    def __post_init__(self) -> None:
        assert self.rb_type in {FifoReplay, LongTermReplay}

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["rb_type"] = self.rb_type.__name__
        return data


@dataclass
class Config(Serialisable):
    esc: EnvScheduleConfig
    algorithm: Literal["dv3", "arrow", "sac"] = "dv3"
    # SAC hyper-parameters…
    sac_lr: float = 3e-4
    sac_batch_size: int = 256
    sac_dv3_data_n_max: int = 1024 # to match total memory for WMAR
    sac_tau: float = 0.005
    sac_gamma: float = 0.99
    sac_alpha: float = 0.2
    img_size: int = 64

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
    replay_buffers: List[RbConfig] = field(default_factory=List)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        data = data.copy()
        data["esc"] = EnvScheduleConfig.from_dict(data["esc"])
        data["replay_buffers"] = [RbConfig.from_dict(d) for d in data["replay_buffers"]]
        return cls(**data)

    def __post_init__(self) -> None:
        assert self.n_sync * self.gen_seq_len == self.data_n * self.data_t
        assert self.random_policy in {"first", "new"}
        assert self.replay_buffers != []
        assert self.env_repeat == 1, "Env repeat disabled for procgen"

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["esc"] = self.esc.to_dict()
        data["replay_buffers"] = [c.to_dict() for c in self.replay_buffers]
        return data

    def get_env_schedule(self) -> EnvironmentSchedule:
        return self.esc.env_schedule_type(
            self.n_sync, [e.get_function() for e in self.esc.env_configs], **self.esc.kwargs
        )

    def get_replay_buffer(self) -> Replay:
        if self.algorithm == "arrow":
            return MultiTypeReplay(
                *[
                    rc.rb_type(self.data_t, self.data_n_max, self.action_space, rc.rb_device)
                    for rc in self.replay_buffers
                ]
            )
        if self.algorithm == "dv3" or self.algorithm == "sac":
            rc = self.replay_buffers[0]
            return rc.rb_type(self.data_t, self.sac_dv3_data_n_max, self.action_space, rc.rb_device)
        #self.sac_dv3_data_n_max
