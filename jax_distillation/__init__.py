"""JAX-native distillation column simulator for reinforcement learning research.

This package provides:
- High-fidelity, JAX-native distillation column simulation
- Full JIT-compilability and vmap-compatibility
- Gymnasium-compatible RL environment
- Validation tools for physical consistency

Quick start:
    >>> from jax_distillation import DistillationColumnEnv
    >>> env = DistillationColumnEnv()
    >>> obs, info = env.reset()
    >>> obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

For simulation without RL:
    >>> from jax_distillation.column import (
    ...     create_teaching_column_config,
    ...     create_initial_column_state,
    ...     create_default_action,
    ...     column_step,
    ... )
    >>> config = create_teaching_column_config()
    >>> state = create_initial_column_state(config)
    >>> action = create_default_action()
    >>> new_state, outputs = column_step(state, action, config)
"""

__version__ = "0.1.0"

# Core types
from jax_distillation.core.types import (
    TrayState,
    ColumnState,
    ColumnParams,
    Action,
    ThermoParams,
)

# Column simulation
from jax_distillation.column.config import (
    ColumnConfig,
    create_teaching_column_config,
)
from jax_distillation.column.column import (
    FullColumnState,
    ColumnAction,
    ColumnOutputs,
    column_step,
    create_initial_column_state,
    create_default_action,
    simulate_column,
)

# Gymnasium environment
from jax_distillation.env.base_env import (
    DistillationColumnEnv,
    make_env,
)
from jax_distillation.env.rewards import (
    RewardConfig,
    create_default_reward_config,
)

# Thermodynamics
from jax_distillation.core.thermodynamics import (
    create_methanol_water_thermo,
)

__all__ = [
    # Version
    "__version__",
    # Core types
    "TrayState",
    "ColumnState",
    "ColumnParams",
    "Action",
    "ThermoParams",
    # Column
    "ColumnConfig",
    "create_teaching_column_config",
    "FullColumnState",
    "ColumnAction",
    "ColumnOutputs",
    "column_step",
    "create_initial_column_state",
    "create_default_action",
    "simulate_column",
    # Environment
    "DistillationColumnEnv",
    "make_env",
    "RewardConfig",
    "create_default_reward_config",
    # Thermodynamics
    "create_methanol_water_thermo",
]
