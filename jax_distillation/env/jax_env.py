"""Gymnax-style JAX-native RL environment for distillation column.

Follows the Gymnax functional API pattern for full JIT/vmap compatibility.
See: https://github.com/RobertTLange/gymnax

Usage:
    from jax_distillation.env import DistillationEnvJax, EnvState, EnvParams

    # Create environment and params
    env = DistillationEnvJax()
    params = env.default_params

    # Reset (JIT-compilable)
    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key, params)

    # Step (JIT-compilable)
    key, subkey = jax.random.split(key)
    obs, state, reward, done, info = env.step(subkey, state, action, params)

    # Vectorized (parallel envs)
    reset_fn = jax.vmap(env.reset, in_axes=(0, None))
    step_fn = jax.vmap(env.step, in_axes=(0, 0, 0, None))

    keys = jax.random.split(key, 64)
    obs, states = reset_fn(keys, params)
    obs, states, rewards, dones, infos = step_fn(keys, states, actions, params)
"""

import jax
import jax.numpy as jnp
from typing import NamedTuple, Tuple
from functools import partial

from jax_distillation.column.config import (
    ColumnConfig,
    create_teaching_column_config,
)
from jax_distillation.column.column import (
    FullColumnState,
    ColumnAction,
    ColumnOutputs,
    make_column_step_fn,
    create_initial_column_state,
)


class EnvState(NamedTuple):
    """Environment state (pytree-compatible).

    Attributes:
        column_state: Full column physics state.
        prev_x_D: Previous distillate composition for stability reward.
        prev_x_B: Previous bottoms composition for stability reward.
        time: Current timestep in episode.
    """

    column_state: FullColumnState
    prev_x_D: jnp.ndarray
    prev_x_B: jnp.ndarray
    time: jnp.ndarray


class EnvParams(NamedTuple):
    """Static environment parameters.

    Attributes:
        max_steps: Maximum steps per episode.
        x_D_target: Target distillate purity.
        x_B_target: Target bottoms purity.
        energy_weight: Weight for energy penalty.
        stability_weight: Weight for stability penalty.
        purity_tolerance_D: Tolerance for distillate purity.
        purity_tolerance_B: Tolerance for bottoms purity.
        Q_R_max: Maximum reboiler duty for normalization.
    """

    max_steps: int = 1000
    x_D_target: float = 0.95
    x_B_target: float = 0.05
    energy_weight: float = 0.01
    stability_weight: float = 0.1
    purity_tolerance_D: float = 0.02
    purity_tolerance_B: float = 0.02
    Q_R_max: float = 20000.0


class EnvInfo(NamedTuple):
    """Info returned from step (pytree-compatible).

    Attributes:
        x_D: Current distillate composition.
        x_B: Current bottoms composition.
        Q_R: Reboiler duty used.
        D: Distillate flow rate.
        B: Bottoms flow rate.
    """

    x_D: jnp.ndarray
    x_B: jnp.ndarray
    Q_R: jnp.ndarray
    D: jnp.ndarray
    B: jnp.ndarray


class DistillationEnvJax:
    """Gymnax-style distillation column environment.

    This environment follows the Gymnax functional API pattern where:
    - reset() and step() are pure functions
    - All operations use JAX arrays (no NumPy)
    - No Python conditionals on traced values
    - Fully JIT-compilable and vmap-compatible

    Attributes:
        column_config: Configuration for the distillation column.
        default_params: Default environment parameters.
    """

    def __init__(self, column_config: ColumnConfig = None):
        """Initialize the environment.

        Args:
            column_config: Column configuration. Uses teaching column if None.
        """
        self.column_config = column_config or create_teaching_column_config()
        self._step_fn = make_column_step_fn(self.column_config)

    @property
    def default_params(self) -> EnvParams:
        """Get default environment parameters."""
        return EnvParams()

    @partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key: jnp.ndarray, params: EnvParams
    ) -> Tuple[jnp.ndarray, EnvState]:
        """Reset environment to initial state (JIT-compilable).

        Args:
            key: Random key (unused currently, for API compatibility).
            params: Environment parameters.

        Returns:
            Tuple of (observation, env_state).
        """
        column_state = create_initial_column_state(self.column_config)

        # Get initial compositions for stability tracking
        # Use condenser/reboiler compositions as estimates
        initial_x_D = column_state.condenser.x
        initial_x_B = column_state.reboiler.x

        state = EnvState(
            column_state=column_state,
            prev_x_D=initial_x_D,
            prev_x_B=initial_x_B,
            time=jnp.array(0),
        )

        obs = self._get_obs(state)
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: jnp.ndarray,
        state: EnvState,
        action: jnp.ndarray,
        params: EnvParams,
    ) -> Tuple[jnp.ndarray, EnvState, jnp.ndarray, jnp.ndarray, EnvInfo]:
        """Execute one environment step (JIT-compilable).

        Args:
            key: Random key (unused currently, for API compatibility).
            state: Current environment state.
            action: Control action array [Q_R, reflux_ratio, B_setpoint, D_setpoint].
            params: Environment parameters.

        Returns:
            Tuple of (observation, new_state, reward, done, info).
        """
        # Convert action array to ColumnAction
        column_action = ColumnAction(
            Q_R=action[0],
            reflux_ratio=action[1],
            B_setpoint=action[2],
            D_setpoint=action[3],
        )

        # Step physics
        new_column_state, outputs = self._step_fn(
            state.column_state, column_action
        )

        # Get current compositions
        x_D = outputs.x_D
        x_B = outputs.x_B

        # Compute reward (pure JAX)
        reward = self._compute_reward(
            x_D=x_D,
            x_B=x_B,
            Q_R=outputs.Q_R,
            dx_D=x_D - state.prev_x_D,
            dx_B=x_B - state.prev_x_B,
            tray_M=new_column_state.tray_M,
            params=params,
        )

        # Check done (use JAX operations, not Python if)
        new_time = state.time + 1
        done = new_time >= params.max_steps

        # Create new state
        new_state = EnvState(
            column_state=new_column_state,
            prev_x_D=x_D,
            prev_x_B=x_B,
            time=new_time,
        )

        obs = self._get_obs(new_state)

        info = EnvInfo(
            x_D=x_D,
            x_B=x_B,
            Q_R=outputs.Q_R,
            D=outputs.D,
            B=outputs.B,
        )

        return obs, new_state, reward, done, info

    def _get_obs(self, state: EnvState) -> jnp.ndarray:
        """Extract observation from state (pure JAX).

        Observation includes:
        - Normalized tray temperatures
        - Tray compositions
        - Reboiler and condenser states

        Args:
            state: Environment state.

        Returns:
            Observation array.
        """
        cs = state.column_state

        # Normalize temperatures (inline, no stateful normalizer)
        T_norm = (cs.tray_T - 300.0) / 100.0

        # Reboiler state (normalized)
        reb_T_norm = jnp.array([(cs.reboiler.T - 300.0) / 100.0])
        reb_x = jnp.array([cs.reboiler.x])

        # Condenser state (normalized)
        cond_T_norm = jnp.array([(cs.condenser.T - 300.0) / 100.0])
        cond_x = jnp.array([cs.condenser.x])

        return jnp.concatenate([
            T_norm,  # Tray temperatures
            cs.tray_x,  # Tray compositions
            reb_T_norm,  # Reboiler temperature
            reb_x,  # Reboiler composition
            cond_T_norm,  # Condenser temperature
            cond_x,  # Condenser composition
        ])

    def _compute_reward(
        self,
        x_D: jnp.ndarray,
        x_B: jnp.ndarray,
        Q_R: jnp.ndarray,
        dx_D: jnp.ndarray,
        dx_B: jnp.ndarray,
        tray_M: jnp.ndarray,
        params: EnvParams,
    ) -> jnp.ndarray:
        """Compute reward (pure JAX, no Python conditionals).

        Reward components:
        - Purity reward: tracks target compositions
        - Energy penalty: penalizes high reboiler duty
        - Stability penalty: penalizes large composition changes
        - Constraint penalty: penalizes low tray holdups

        Args:
            x_D: Distillate composition.
            x_B: Bottoms composition.
            Q_R: Reboiler duty.
            dx_D: Change in distillate composition.
            dx_B: Change in bottoms composition.
            tray_M: Tray holdups.
            params: Environment parameters.

        Returns:
            Scalar reward.
        """
        # Purity reward (quadratic with tolerance zone)
        error_D = jnp.abs(x_D - params.x_D_target)
        error_B = jnp.abs(x_B - params.x_B_target)

        reward_D = jnp.maximum(0.0, 1.0 - (error_D / params.purity_tolerance_D) ** 2)
        reward_B = jnp.maximum(0.0, 1.0 - (error_B / params.purity_tolerance_B) ** 2)
        purity_reward = jnp.sqrt(reward_D * reward_B)

        # Energy penalty (normalized)
        energy_penalty = params.energy_weight * jnp.minimum(1.0, Q_R / params.Q_R_max)

        # Stability penalty (penalize oscillations)
        stability_penalty = params.stability_weight * (dx_D ** 2 + dx_B ** 2)

        # Constraint penalty (penalize low holdups)
        min_holdup = 0.1
        holdup_violations = jnp.sum(jnp.maximum(0.0, min_holdup - tray_M))
        constraint_penalty = holdup_violations

        return purity_reward - energy_penalty - stability_penalty - constraint_penalty

    @property
    def observation_space_shape(self) -> Tuple[int]:
        """Observation space shape (for compatibility).

        Returns:
            Shape tuple.
        """
        n_trays = self.column_config.geometry.n_trays
        # T + x for trays + reb T + reb x + cond T + cond x
        return (n_trays * 2 + 4,)

    @property
    def action_space_shape(self) -> Tuple[int]:
        """Action space shape (for compatibility).

        Returns:
            Shape tuple.
        """
        return (4,)  # Q_R, reflux_ratio, B_setpoint, D_setpoint

    @property
    def action_space_low(self) -> jnp.ndarray:
        """Action space lower bounds."""
        return jnp.array([0.0, 0.5, 0.0, 0.0])

    @property
    def action_space_high(self) -> jnp.ndarray:
        """Action space upper bounds."""
        return jnp.array([20000.0, 10.0, 0.2, 0.2])


def make_env_fns(column_config: ColumnConfig = None):
    """Create pure functional reset and step functions.

    This is an alternative to the class-based API for users who prefer
    a purely functional interface.

    Args:
        column_config: Column configuration. Uses teaching column if None.

    Returns:
        Tuple of (reset_fn, step_fn, default_params).

    Example:
        reset_fn, step_fn, default_params = make_env_fns()

        # Single environment
        obs, state = reset_fn(key, default_params)
        obs, state, reward, done, info = step_fn(key, state, action, default_params)

        # Vectorized
        batch_reset = jax.vmap(reset_fn, in_axes=(0, None))
        batch_step = jax.vmap(step_fn, in_axes=(0, 0, 0, None))
    """
    env = DistillationEnvJax(column_config)

    return env.reset, env.step, env.default_params
