#!/usr/bin/env python3
"""
Final working SAC implementation using Tianshou for CoinRun environments.
Uses DQN as fallback for discrete actions with proper Tianshou trainer.
"""
import os as _os
# Guard against stray MPI/PMI environments on some clusters causing MPI_Init attempts
for _k in list(_os.environ.keys()):
    if _k.startswith(("PMI_", "PMIX_", "OMPI_", "MPICH_")):
        _os.environ.pop(_k, None)
_os.environ.setdefault("OMP_NUM_THREADS", "1")

from math import e
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module=r"gym\.utils\.passive_env_checker")
warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"gym\.utils\.passive_env_checker")

import time
import json
import os
import socket
from datetime import datetime
from pathlib import Path
import argparse
from typing import Dict, List, Callable, Any
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import gym
try:
    from gym.wrappers import FrameStack  # gym <= 0.26
except Exception:  # pragma: no cover
    from gymnasium.wrappers import FrameStack  # gymnasium
# Try to import procgen
try:
    import procgen
    print("[IMPORT] Successfully imported procgen")
except ImportError:
    print("[IMPORT] Warning: procgen not available")

import tianshou as ts
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.policy import DiscreteSACPolicy
from tianshou.utils.net.common import Net
from tianshou.trainer import offpolicy_trainer


def select_device():
    """Select device with preference: CUDA > CPU (never MPS)"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"[DEVICE] Selected CUDA device: {torch.cuda.get_device_name()}")
    else:
        device = torch.device('cpu')
        print(f"[DEVICE] Selected CPU device (CUDA not available)")
    
    print(f"[DEVICE] Final device: {device}")
    return device


class CoinRunEnvironmentFactory:
    """Factory for creating CoinRun environments with various modifiers"""
    
    def __init__(self, name: str, kwargs: Dict[str, Any] = None, rew_scale: float = 0.10):
        self.name = name
        self.kwargs = kwargs or {}
        self.rew_scale = rew_scale
        #print(f"[ENV_FACTORY] Creating factory for {name} with rew_scale={rew_scale}")
    
    def get_function(self) -> Callable:
        """Create environment function based on name and modifiers"""
        default = {
            "use_backgrounds": True,
            "restrict_themes": False,
            "use_monochrome_assets": False,
            "use_generated_assets": False,
            "center_agent": True,
        }
        
        mods = {
            "NB": {"use_backgrounds": False},
            "RT": {"restrict_themes": True},
            "MA": {"use_monochrome_assets": True},
            "UGA": {"use_generated_assets": True},
            "CA": {"center_agent": False},
        }
        
        parts = self.name.split("+")
        assert parts[0] == "CoinRun", f"Environment name must start with 'CoinRun', got {parts[0]}"
        
        # Apply modifiers
        for part in parts[1:]:
            if part in mods:
                default.update(mods[part])
                #print(f"[ENV_FACTORY] Applied modifier {part}: {mods[part]}")
        
        #print(f"[ENV_FACTORY] Final env config for {self.name}: {default}")
        
        def env_fn():
            try:
                # Merge user-provided kwargs on top of defaults
                env_kwargs = {**default, **(self.kwargs or {})}
                env = gym.make("procgen:procgen-coinrun-v0", **env_kwargs)
                ##print(f"[ENV_FACTORY] Successfully created {self.name}")
                # Add compatibility wrapper for procgen environments
                env = GymCompatibilityWrapper(env)
                # Stack frames to provide temporal information (velocity, motion)
                env = FrameStack(env, 4)
            except Exception as e:
                print(f"[ENV_FACTORY] Warning: procgen not available or failed to create, using CartPole-v1 for testing")
                env = gym.make("CartPole-v1")
                env = GymCompatibilityWrapper(env)
            
            # Always apply reward scaling wrapper (default 0.10)
            env = RewardScaleWrapper(env, self.rew_scale)
            return env
        
        return env_fn


class RewardScaleWrapper(gym.Wrapper):
    """Wrapper to scale rewards"""
    
    def __init__(self, env, scale: float):
        super().__init__(env)
        self.scale = scale
    
    def step(self, action):
        out = self.env.step(action)
        if isinstance(out, tuple) and len(out) == 4:
            obs, reward, done, info = out
            return obs, reward * self.scale, done, info
        elif isinstance(out, tuple) and len(out) == 5:
            obs, reward, terminated, truncated, info = out
            return obs, reward * self.scale, terminated, truncated, info
        else:
            raise ValueError(f"Unexpected step return format from inner env: {type(out)} len={len(out) if isinstance(out, tuple) else 'NA'}")


class GymCompatibilityWrapper(gym.Wrapper):
    """Wrapper to add missing attributes for Tianshou compatibility"""
    
    def __init__(self, env):
        super().__init__(env)
        # Add render_mode attribute if it doesn't exist
        if not hasattr(self, 'render_mode'):
            self.render_mode = None
    
    def __getattr__(self, name):
        """Handle missing attributes that Tianshou expects"""
        if name == 'render_mode':
            return None
        # Try to get the attribute from the underlying environment
        try:
            return getattr(self.env, name)
        except AttributeError:
            # If the underlying environment doesn't have it, return None for render_mode
            if name == 'render_mode':
                return None
            raise
    
    def reset(self, **kwargs):
        """Override reset to preserve seeding when supported, with graceful fallback"""
        # First try passing all kwargs (including seed/options) through
        try:
            result = self.env.reset(**kwargs)
        except TypeError:
            # If that fails, try without seed/options
            filtered = {k: v for k, v in kwargs.items() if k not in ['seed', 'options']}
            try:
                result = self.env.reset(**filtered)
            except TypeError:
                # Final fallback: no kwargs
                result = self.env.reset()
        
        # Handle different return formats
        if isinstance(result, tuple) and len(result) == 2:
            # Already in (obs, info) format
            return result
        else:
            # Single observation, wrap in tuple with empty info
            return result, {}
    
    def step(self, action):
        """Override step to handle older gym interfaces"""
        result = self.env.step(action)
        
        # Handle different return formats
        if len(result) == 4:
            # Old format: (obs, reward, done, info)
            obs, reward, done, info = result
            # Convert to new format: (obs, reward, terminated, truncated, info)
            terminated = done
            truncated = False
            return obs, reward, terminated, truncated, info
        elif len(result) == 5:
            # Already in new format: (obs, reward, terminated, truncated, info)
            return result
        else:
            raise ValueError(f"Unexpected step return format: {result}")





class SequentialEnvironments:
    """Environment scheduler that cycles through environments"""
    
    def __init__(self, env_configs: List[Dict], swap_sched: int = 90):
        self.env_configs = env_configs
        self.swap_sched = swap_sched
        self.current_env_idx = 0
        self.step_count = 0
        
        print(f"[ENV_SCHEDULE] Initialized with {len(env_configs)} environments")
        print(f"[ENV_SCHEDULE] Swap schedule: every {swap_sched} steps")
        for i, config in enumerate(env_configs):
            print(f"[ENV_SCHEDULE] Env {i}: {config['name']}")
    
    def funcs(self) -> List[Callable]:
        """Get current environment functions"""
        current_config = self.env_configs[self.current_env_idx]
        factory = CoinRunEnvironmentFactory(**current_config)
        return [factory.get_function()]
    
    def eval_funcs(self) -> List[List[Callable]]:
        """Get all environment functions for evaluation"""
        eval_funcs = []
        for config in self.env_configs:
            # Use evaluation-specific kwargs if provided; otherwise fix levels by default
            fixed_eval_kwargs = {"start_level": 0, "num_levels": 200}
            eval_kwargs = config.get("eval_kwargs", config.get("kwargs", {}))
            # Ensure fixed seeds/levels override
            eval_kwargs = {**eval_kwargs, **fixed_eval_kwargs}
            factory = CoinRunEnvironmentFactory(name=config["name"], kwargs=eval_kwargs, rew_scale=config.get("rew_scale", 1.0))
            eval_funcs.append([factory.get_function()])
        return eval_funcs
    
    def step(self):
        """Step the environment scheduler"""
        self.step_count += 1
        if self.step_count % self.swap_sched == 0:
            old_idx = self.current_env_idx
            self.current_env_idx = (self.current_env_idx + 1) % len(self.env_configs)
            print(f"[ENV_SCHEDULE] Swapped from env {old_idx} ({self.env_configs[old_idx]['name']}) "
                  f"to env {self.current_env_idx} ({self.env_configs[self.current_env_idx]['name']})")


def evaluate_policy(policy, env_funcs: List[List[Callable]], n_rollouts: int = 256, reward_scale: float = 1.0) -> tuple:
    """Evaluate policy on all environments"""
    print(f"[EVAL] Starting evaluation with {n_rollouts} rollouts per environment")
    
    all_means = []
    all_stds = []
    
    for env_idx, env_fn_list in enumerate(env_funcs):
        print(f"[EVAL] Evaluating environment {env_idx}")
        
        # Create evaluation environments - use fewer parallel envs for evaluation
        n_eval_envs = min(4, n_rollouts)  # Use at most 4 parallel environments
        eval_envs = DummyVectorEnv([env_fn_list[0] for _ in range(n_eval_envs)])
        
        # Create collector
        eval_collector = Collector(policy, eval_envs)
        
        # For SAC, evaluation uses the policy as-is (stochastic)
        
        try:
            result = eval_collector.collect(n_episode=n_rollouts)
            
            episode_rewards = result["rews"]
            if len(episode_rewards) == 0:
                print(f"[EVAL] Warning: No episodes completed for environment {env_idx}")
                mean_reward = 0.0
                std_reward = 0.0
            else:
                # Adjust by inverse of reward scaling used in envs (default 1.0 = no scaling)
                mean_reward = float(np.mean(episode_rewards) / reward_scale)
                std_reward = float(np.std(episode_rewards) / reward_scale)
            
            print(f"[EVAL] Env {env_idx}: Mean reward = {mean_reward:.3f} ± {std_reward:.3f} (n={len(episode_rewards)})")
            
        except Exception as e:
            print(f"[EVAL] Error during evaluation for environment {env_idx}: {e}")
            mean_reward = 0.0
            std_reward = 0.0
        
        all_means.append(mean_reward)
        all_stds.append(std_reward)
        
        eval_envs.close()
    
    return all_means, all_stds


def train_sac(config):
    """Main SAC training function using Tianshou trainer"""
    print(f"[SAC] Starting SAC training with config")
    #print(f"[SAC] Algorithm: {config.get('algorithm', 'sac')}")
    #print(f"[SAC] Seed: {config.get('seed', 123456789)}")
    #print(f"[SAC] Epochs: {config.get('epochs', 100)}")
    
    # Set device
    device = select_device()
    
    # Set random seeds
    seed = config.get('seed', 123456789)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    #print(f"[SAC] Set random seed to {seed}")
    
    # Create environment scheduler
    esc_config = config.get('esc', {})
    env_configs = esc_config.get('env_configs', [])
    swap_sched = esc_config.get('kwargs', {}).get('swap_sched', 90)
    
    env_scheduler = SequentialEnvironments(env_configs, swap_sched)
    
    # Create training environments
    n_envs = config.get('n_sync', 4)  # Use n_sync as specified in config

    #print(f"[SAC] Creating {n_envs} training environments...")
    train_env_fns = env_scheduler.funcs()
    
    #print(f"[SAC] Creating DummyVectorEnv with {n_envs} environments...")
    train_envs = DummyVectorEnv([train_env_fns[0] for _ in range(n_envs)])
    #print(f"[SAC] Successfully created training environments")
    
    # Get environment info
    #print(f"[SAC] Getting environment info...")
    sample_env = train_env_fns[0]()
    state_shape = sample_env.observation_space.shape
    
    if hasattr(sample_env.action_space, 'n'):
        action_shape = sample_env.action_space.n
        is_discrete = True
    else:
        action_shape = sample_env.action_space.shape[0]
        is_discrete = False
    
    sample_env.close()
    #print(f"[SAC] Environment info collected successfully")
    
    #print(f"[SAC] Environment info:")
    #print(f"[SAC]   State shape: {state_shape}")
    #print(f"[SAC]   Action shape: {action_shape}")
    #print(f"[SAC]   Action space type: {'Discrete' if is_discrete else 'Continuous'}")
    
    # Build Discrete SAC actor/critics
    if len(state_shape) in (3, 4):
        print(f"[SAC] Creating CNN-based actor/critics for image environment: {state_shape}")
        class BaseCNN(nn.Module):
            def __init__(self, state_shape, device="cuda"):
                super().__init__()
                self.device = device
                if len(state_shape) == 3:
                    height, width, channels = state_shape
                else:
                    stack, height, width, base_channels = state_shape
                    channels = stack * base_channels
                self.channels, self.height, self.width = channels, height, width
                self.conv1 = nn.Conv2d(channels, 32, kernel_size=8, stride=4)
                self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
                self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
                def conv2d_size_out(size, kernel_size, stride):
                    return (size - (kernel_size - 1) - 1) // stride + 1
                convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(self.width, 8, 4), 4, 2), 3, 1)
                convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(self.height, 8, 4), 4, 2), 3, 1)
                linear_input_size = convw * convh * 64
                self.fc1 = nn.Linear(linear_input_size, 512)
            def forward_backbone(self, obs: torch.Tensor) -> torch.Tensor:
                x = torch.relu(self.conv1(obs))
                x = torch.relu(self.conv2(x))
                x = torch.relu(self.conv3(x))
                x = x.reshape(x.size(0), -1)
                x = torch.relu(self.fc1(x))
                return x
            def preprocess(self, obs):
                if not isinstance(obs, torch.Tensor):
                    obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
                if obs.ndim == 3:
                    obs = obs.unsqueeze(0)
                if obs.max() > 1.0:
                    obs = obs / 255.0
                obs = obs.contiguous()
                if obs.ndim == 5:
                    b, s, h, w, c = obs.shape
                    obs = obs.permute(0, 4, 2, 3, 1).contiguous().reshape(b, c * s, h, w)
                elif obs.ndim == 4:
                    if obs.shape[1] == self.channels and obs.shape[2] == self.height:
                        pass
                    else:
                        obs = obs.permute(0, 3, 1, 2)
                return obs
        class ActorCNN(BaseCNN):
            def __init__(self, state_shape, action_dim, device="cuda"):
                super().__init__(state_shape, device)
                self.fc_out = nn.Linear(512, action_dim)
            def forward(self, obs, state=None, info={}):
                obs = self.preprocess(obs)
                x = self.forward_backbone(obs)
                logits = self.fc_out(x)
                return logits, state
        class CriticCNN(BaseCNN):
            def __init__(self, state_shape, action_dim, device="cuda"):
                super().__init__(state_shape, device)
                self.fc_out = nn.Linear(512, action_dim)
            def forward(self, obs, action=None):
                obs = self.preprocess(obs)
                x = self.forward_backbone(obs)
                q = self.fc_out(x)
                return q
        actor = ActorCNN(state_shape, action_shape, device).to(device)
        critic1 = CriticCNN(state_shape, action_shape, device).to(device)
        critic2 = CriticCNN(state_shape, action_shape, device).to(device)
    else:
        print(f"[SAC] Creating MLP-based actor/critics for state environment: {state_shape}")
        hidden = config.get('mlp_features', 512)
        layers = config.get('mlp_layers', 2)
        def make_mlp(in_dim, out_dim):
            mods = []
            dims = [in_dim] + [hidden] * (layers - 1) + [out_dim]
            for i in range(len(dims) - 1):
                mods.append(nn.Linear(dims[i], dims[i+1]))
                if i < len(dims) - 2:
                    mods.append(nn.ReLU())
            return nn.Sequential(*mods)
        state_dim = int(np.prod(state_shape))
        class ActorMLP(nn.Module):
            def __init__(self, in_dim, action_dim):
                super().__init__()
                self.net = make_mlp(in_dim, action_dim)
            def forward(self, obs, state=None, info={}):
                if not isinstance(obs, torch.Tensor):
                    obs = torch.as_tensor(obs, device=device, dtype=torch.float32)
                x = obs.view(obs.shape[0], -1) if obs.ndim > 1 else obs.reshape(1, -1)
                logits = self.net(x)
                return logits, state
        class CriticMLP(nn.Module):
            def __init__(self, in_dim, action_dim):
                super().__init__()
                self.net = make_mlp(in_dim, action_dim)
            def forward(self, obs, action=None):
                if not isinstance(obs, torch.Tensor):
                    obs = torch.as_tensor(obs, device=device, dtype=torch.float32)
                x = obs.view(obs.shape[0], -1) if obs.ndim > 1 else obs.reshape(1, -1)
                q = self.net(x)
                return q
        actor = ActorMLP(state_dim, action_shape).to(device)
        critic1 = CriticMLP(state_dim, action_shape).to(device)
        critic2 = CriticMLP(state_dim, action_shape).to(device)

    # Optimizers for SAC
    actor_optim = torch.optim.Adam(actor.parameters(), lr=3e-4, weight_decay=1e-5)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=2e-4, weight_decay=3e-4)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=2e-4, weight_decay=3e-4)

    # Create Discrete SAC policy
    temp_env = train_env_fns[0]()
    policy = DiscreteSACPolicy(
        actor=actor,
        actor_optim=actor_optim,
        critic1=critic1,
        critic1_optim=critic1_optim,
        critic2=critic2,
        critic2_optim=critic2_optim,
        tau=float(config.get('sac_tau', 0.005)),
        gamma=float(config.get('sac_gamma', 0.99)),
        alpha=float(config.get('sac_alpha', 0.2)),
        estimation_step=int(config.get('estimation_step', 3)),
        reward_normalization=False,
    ).to(device)
    temp_env.close()
    
    #print(f"[SAC] Created DiscreteSACPolicy")
    
    buffer_size = config.get('sac_dv3_data_n_max', 1024) * 512
    rb_device = 'cpu'  
    replay_buffer = VectorReplayBuffer(
        total_size=buffer_size,
        buffer_num=len(train_envs),
        device=rb_device,
        stack_num=1,                 # env already provides FrameStack(4)
        save_only_last_obs=True      # reduce memory
    )    
    print(f"[SAC] Created replay buffer with size {buffer_size} on {rb_device}")
    
    #Create collectors
    #print(f"[SAC] Creating train collector...")
    train_collector = Collector(policy, train_envs, replay_buffer)
    #print(f"[SAC] Successfully created train collector")

    # Prefill replay buffer with random experience to stabilize early learning
    warmup_steps = int(config.get('sac_warmup_steps', 20000))
    if warmup_steps > 0:
        print(f"[SAC] Warmup: collecting {warmup_steps} random steps to prefill replay buffer")
        train_collector.collect(n_step=warmup_steps, random=True)
        # reset to initial exploration schedule will be handled in train_fn
        #print(f"[SAC] Warmup complete. Current buffer size: {len(replay_buffer)}")
    print(f"[SAC] Warmup complete. Current buffer size: {len(replay_buffer)}")
    # Create evaluation environments for all configurations
    #print(f"[SAC] Creating test environments...")
    eval_funcs = env_scheduler.eval_funcs()
    print(f"[SAC] Created {len(eval_funcs)} evaluation environments")
    test_envs = DummyVectorEnv([eval_funcs[0][0] for _ in range(n_envs)])  # Use same n_envs as training
    print(f"[SAC] Created {len(test_envs)} test environments")
    #print(f"[SAC] Creating test collector...")
    test_collector = Collector(policy, test_envs)
    print(f"[SAC] Created test collector")
    #print(f"[SAC] Successfully created test collector")
    
    # Setup logging
    current_time = datetime.now().strftime("%b%d_%H-%M-%S")
    run_name = f"sac_{current_time}_{socket.gethostname()}_seed{seed}"
    log_dir = os.path.join("runs", "sac", run_name)
    writer = SummaryWriter(log_dir=log_dir)
    
    #print(f"[SAC] Logging to {log_dir}")
    
    # Save config
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    # Training parameters
    epochs = config.get('epochs', 100)
    steps_per_batch = config.get('steps_per_batch', 1000)
    batch_size = 256
    
    print(f"[SAC] Training parameters:")
    print(f"[SAC]   Epochs: {epochs}")
    print(f"[SAC]   Steps per batch: {steps_per_batch}")
    print(f"[SAC]   Batch size: {batch_size}")
    
    # Use Tianshou's offpolicy trainer
    #print(f"[SAC] Starting training with Tianshou trainer...")
    
    def save_best_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_dir, "best_policy.pth"))
    
    def stop_fn(mean_rewards):
        # Stop if mean reward exceeds threshold (adjust for your environment)
        return mean_rewards >= 450  # For CartPole, max is around 500
    
    # Configure step-based evaluation schedule
    eval_every_steps = int(config.get('eval_every_steps', 10000))
    next_eval_step = eval_every_steps
    
    def train_fn(epoch, env_step):
        nonlocal next_eval_step
        
        # Step environment scheduler
        if epoch > 0:
            env_scheduler.step()
        
        # Trigger evaluation by global steps (e.g., 10k, 20k, ...)
        while env_step >= next_eval_step:
            print(f"[SAC] Epoch {epoch}: training step {env_step}")
            print(f"[SAC] ===== Running evaluation at step {env_step} =====")
            eval_funcs = env_scheduler.eval_funcs()
            # Use reward_scale that matches training envs
            eval_rew_scale = env_scheduler.env_configs[0].get('rew_scale', 1.0) if env_scheduler.env_configs else 1.0
            all_means, all_stds = evaluate_policy(policy, eval_funcs, n_rollouts=256, reward_scale=eval_rew_scale)
            
            # Log to tensorboard
            for i, (mean_rew, std_rew) in enumerate(zip(all_means, all_stds)):
                writer.add_scalar(f"Perf/eval_rew_eps_mean_{i}", mean_rew, env_step)
                writer.add_scalar(f"Perf/eval_rew_eps_std_{i}", std_rew, env_step)
                env_name = env_scheduler.env_configs[i]['name'] if i < len(env_scheduler.env_configs) else f"env_{i}"
                print(f"[SAC] Env {i} ({env_name}): Mean = {mean_rew:.3f} ± {std_rew:.3f} steps: {env_step}")
            print(f"[SAC] ===== Evaluation completed at step {env_step} =====")
            next_eval_step += eval_every_steps
        
    # Run the trainer
    #print(f"[SAC] Starting offpolicy_trainer with {epochs} epochs, {steps_per_batch} steps per epoch")
    #print(f"[SAC] Batch size: {batch_size}, Update per step: 1.0")
    
    try:
        result = offpolicy_trainer(
            policy=policy,
            train_collector=train_collector,
            test_collector=test_collector,
            max_epoch=epochs,
            step_per_epoch=steps_per_batch,
            step_per_collect=steps_per_batch*3,
            update_per_step=0.5,
            episode_per_test=256,
            batch_size=batch_size,
            save_best_fn=save_best_fn,
            stop_fn=stop_fn,
            train_fn=train_fn,
            verbose=True
        )
    except Exception as e:
        print(f"[SAC] Error during training: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    print(f"\n[SAC] ===== Training completed =====")
    print(f"[SAC] Best reward achieved: {result['best_reward']:.3f}")
    print(f"[SAC] Total training time: {result['duration']}")
    
    # Run final evaluation on all environments
    print(f"[SAC] Running final evaluation on all environments...")
    final_means, final_stds = evaluate_policy(policy, eval_funcs, n_rollouts=256)
    
    print(f"[SAC] Final evaluation results:")
    for i, (mean_rew, std_rew) in enumerate(zip(final_means, final_stds)):
        print(f"[SAC]   Env {i} ({env_configs[i]['name']}): {mean_rew:.3f} ± {std_rew:.3f}")
    
    # Save final model
    torch.save(policy.state_dict(), os.path.join(log_dir, "final_policy.pth"))
    
    # Close environments and writer
    train_envs.close()
    test_envs.close()
    writer.close()
    
    print(f"[SAC] Training finished. Results saved to {log_dir}")


def load_config(config_path: str) -> dict:
    """Load configuration from JSON file"""
    print(f"[CONFIG] Loading config from {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print(f"[CONFIG] Loaded config with {len(config)} keys")
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SAC agent on CoinRun environments")
    parser.add_argument("--config", type=str, required=True, help="Path to config JSON file")
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Verify it's a SAC config
    if config.get('algorithm') != 'sac':
        print(f"[ERROR] Config algorithm is '{config.get('algorithm')}', expected 'sac'")
        exit(1)
    
    # Run training
    train_sac(config)