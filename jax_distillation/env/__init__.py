"""Gymnasium and JAX-native environments for distillation column control.

This module provides two environment APIs:

1. Gymnasium API (DistillationColumnEnv):
   - Standard Gymnasium interface
   - Compatible with stable-baselines3 and other Gym-based RL libraries
   - Uses NumPy arrays

2. JAX-native API (DistillationEnvJax):
   - Gymnax-style functional interface
   - Full JIT/vmap compatibility for GPU-accelerated training
   - Uses JAX arrays only
   - 100x faster for parallel environments
"""

from jax_distillation.env.base_env import (
    DistillationColumnEnv,
    make_env,
)
from jax_distillation.env.jax_env import (
    DistillationEnvJax,
    EnvState,
    EnvParams,
    EnvInfo,
    make_env_fns,
)
from jax_distillation.env.spaces import (
    create_action_space,
    create_observation_space,
    ObservationNormalizer,
    ActionDenormalizer,
)
from jax_distillation.env.rewards import (
    RewardConfig,
    create_default_reward_config,
    compute_reward,
)
from jax_distillation.env.wrappers import (
    wrap_env,
    NormalizedActionWrapper,
    FrameStackWrapper,
)

__all__ = [
    # Gymnasium API
    "DistillationColumnEnv",
    "make_env",
    # JAX-native API
    "DistillationEnvJax",
    "EnvState",
    "EnvParams",
    "EnvInfo",
    "make_env_fns",
    # Spaces
    "create_action_space",
    "create_observation_space",
    "ObservationNormalizer",
    "ActionDenormalizer",
    # Rewards
    "RewardConfig",
    "create_default_reward_config",
    "compute_reward",
    # Wrappers
    "wrap_env",
    "NormalizedActionWrapper",
    "FrameStackWrapper",
]
