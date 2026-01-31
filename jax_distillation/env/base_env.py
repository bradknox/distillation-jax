"""Gymnasium environment for distillation column control.

This module provides a Gymnasium-compatible environment for RL training
on distillation column control tasks.
"""

import gymnasium as gym
import numpy as np
import jax.numpy as jnp
from typing import Any

from jax_distillation.column.config import (
    ColumnConfig,
    create_teaching_column_config,
)
from jax_distillation.column.column import (
    FullColumnState,
    ColumnAction,
    ColumnOutputs,
    column_step,
    make_column_step_fn,
    create_initial_column_state,
    create_default_action,
)
from jax_distillation.env.spaces import (
    create_action_space,
    create_reduced_action_space,
    create_observation_space,
    ObservationNormalizer,
    ActionDenormalizer,
)
from jax_distillation.env.rewards import (
    RewardConfig,
    create_default_reward_config,
    compute_reward,
)


class DistillationColumnEnv(gym.Env):
    """Gymnasium environment for distillation column control.

    This environment simulates a distillation column and provides
    a standard RL interface for training control policies.

    Observation Space:
        Normalized state information including tray temperatures,
        compositions, and product qualities.

    Action Space:
        Control inputs: reboiler duty, reflux ratio, and flow setpoints.

    Reward:
        Configurable combination of:
        - Product purity (tracking target compositions)
        - Energy efficiency (minimizing reboiler duty)
        - Stability (smooth operation)
        - Constraint satisfaction

    Episode Termination:
        - Maximum steps reached
        - Constraint violations (optional)
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(
        self,
        config: ColumnConfig | None = None,
        reward_config: RewardConfig | None = None,
        max_episode_steps: int = 1000,
        use_reduced_action_space: bool = False,
        include_flows_in_obs: bool = True,
        include_holdups_in_obs: bool = False,
        terminate_on_constraint_violation: bool = False,
        render_mode: str | None = None,
    ):
        """Initialize the distillation column environment.

        Args:
            config: Column configuration. Uses teaching column if None.
            reward_config: Reward configuration. Uses defaults if None.
            max_episode_steps: Maximum steps per episode.
            use_reduced_action_space: If True, only control Q_R and reflux ratio.
            include_flows_in_obs: Include flow rates in observation.
            include_holdups_in_obs: Include liquid holdups in observation.
            terminate_on_constraint_violation: End episode on constraint violation.
            render_mode: Rendering mode ("human" or "rgb_array").
        """
        super().__init__()

        # Configuration
        self.config = config or create_teaching_column_config()
        self.reward_config = reward_config or create_default_reward_config()
        self.max_episode_steps = max_episode_steps
        self.terminate_on_constraint = terminate_on_constraint_violation
        self.render_mode = render_mode

        # Space parameters
        self.use_reduced_action_space = use_reduced_action_space
        self.include_flows_in_obs = include_flows_in_obs
        self.include_holdups_in_obs = include_holdups_in_obs

        # Create spaces
        n_trays = self.config.geometry.n_trays
        if use_reduced_action_space:
            self.action_space = create_reduced_action_space()
        else:
            self.action_space = create_action_space()

        self.observation_space = create_observation_space(
            n_trays=n_trays,
            include_flows=include_flows_in_obs,
            include_holdups=include_holdups_in_obs,
        )

        # Normalizers
        self.obs_normalizer = ObservationNormalizer()
        self.action_denormalizer = ActionDenormalizer()

        # State
        self._state: FullColumnState | None = None
        self._prev_x_D: float = 0.0
        self._prev_x_B: float = 0.0
        self._step_count: int = 0

        # Default action values for reduced action space
        self._default_B_setpoint = 0.03
        self._default_D_setpoint = 0.02

        # Create JIT-compiled step function for performance
        import jax
        self._jit_step_fn = jax.jit(make_column_step_fn(self.config))

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the environment to initial state.

        Args:
            seed: Random seed for reproducibility.
            options: Additional options (e.g., custom initial state).

        Returns:
            Tuple of (initial_observation, info_dict).
        """
        super().reset(seed=seed)

        # Handle options
        if options and "initial_state" in options:
            self._state = options["initial_state"]
        else:
            self._state = create_initial_column_state(self.config)

        self._step_count = 0

        # Get initial products for stability tracking
        action = self._get_default_action()
        _, outputs = self._jit_step_fn(self._state, action)
        self._prev_x_D = float(outputs.x_D)
        self._prev_x_B = float(outputs.x_B)

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Execute one environment step.

        Args:
            action: Control action from the agent.

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        # Convert action to ColumnAction
        column_action = self._process_action(action)

        # Step the simulation (using JIT-compiled function)
        self._state, outputs = self._jit_step_fn(self._state, column_action)
        self._step_count += 1

        # Get observation
        observation = self._get_observation()

        # Compute reward
        x_D = float(outputs.x_D)
        x_B = float(outputs.x_B)
        dx_D = x_D - self._prev_x_D
        dx_B = x_B - self._prev_x_B

        reward, reward_components = compute_reward(
            x_D=x_D,
            x_B=x_B,
            Q_R=float(outputs.Q_R),
            dx_D=dx_D,
            dx_B=dx_B,
            tray_M=np.array(self._state.tray_M),
            tray_T=np.array(self._state.tray_T),
            config=self.reward_config,
        )

        # Update previous values
        self._prev_x_D = x_D
        self._prev_x_B = x_B

        # Check termination conditions
        terminated = False
        if self.terminate_on_constraint and reward_components["constraint"] < -1.0:
            terminated = True

        truncated = self._step_count >= self.max_episode_steps

        # Info
        info = self._get_info()
        info["reward_components"] = reward_components
        info["outputs"] = {
            "D": float(outputs.D),
            "x_D": x_D,
            "B": float(outputs.B),
            "x_B": x_B,
            "Q_R": float(outputs.Q_R),
            "Q_C": float(outputs.Q_C),
        }

        return observation, reward, terminated, truncated, info

    def _process_action(self, action: np.ndarray) -> ColumnAction:
        """Convert numpy action to ColumnAction.

        Args:
            action: Raw action from agent.

        Returns:
            ColumnAction for simulation.
        """
        # Clip action to valid range
        action = np.clip(action, self.action_space.low, self.action_space.high)

        if self.use_reduced_action_space:
            Q_R = float(action[0])
            reflux_ratio = float(action[1])
            B_setpoint = self._default_B_setpoint
            D_setpoint = self._default_D_setpoint
        else:
            Q_R = float(action[0])
            reflux_ratio = float(action[1])
            B_setpoint = float(action[2])
            D_setpoint = float(action[3])

        return ColumnAction(
            Q_R=jnp.array(Q_R),
            reflux_ratio=jnp.array(reflux_ratio),
            B_setpoint=jnp.array(B_setpoint),
            D_setpoint=jnp.array(D_setpoint),
        )

    def _get_default_action(self) -> ColumnAction:
        """Get default action for initial evaluation."""
        return create_default_action(
            Q_R=5000.0,
            reflux_ratio=3.0,
            B_setpoint=self._default_B_setpoint,
            D_setpoint=self._default_D_setpoint,
        )

    def _get_observation(self) -> np.ndarray:
        """Extract observation from current state.

        Returns:
            Normalized observation array.
        """
        state = self._state
        n_trays = self.config.geometry.n_trays

        obs_parts = []

        # Tray temperatures (normalized)
        tray_T_norm = self.obs_normalizer.normalize_temperature(np.array(state.tray_T))
        obs_parts.append(tray_T_norm)

        # Tray compositions (already in [0, 1])
        obs_parts.append(np.array(state.tray_x))

        # Reboiler state
        reb_T_norm = self.obs_normalizer.normalize_temperature(
            np.array([float(state.reboiler.T)])
        )
        obs_parts.append(reb_T_norm)
        obs_parts.append(np.array([float(state.reboiler.x)]))

        # Condenser state
        cond_T_norm = self.obs_normalizer.normalize_temperature(
            np.array([float(state.condenser.T)])
        )
        obs_parts.append(cond_T_norm)
        obs_parts.append(np.array([float(state.condenser.x)]))

        # Flow information (if included)
        if self.include_flows_in_obs:
            # Use condenser/reboiler states as proxies for product flows
            # In full implementation, would compute actual flows
            D_norm = self.obs_normalizer.normalize_flow(np.array([0.02]))  # Approximate
            B_norm = self.obs_normalizer.normalize_flow(np.array([0.03]))
            x_D = np.array([float(state.condenser.x)])
            x_B = np.array([float(state.reboiler.x)])
            obs_parts.extend([D_norm, B_norm, x_D, x_B])

        # Holdups (if included)
        if self.include_holdups_in_obs:
            tray_M_norm = self.obs_normalizer.normalize_holdup(np.array(state.tray_M))
            reb_M_norm = self.obs_normalizer.normalize_holdup(
                np.array([float(state.reboiler.M)])
            )
            cond_M_norm = self.obs_normalizer.normalize_holdup(
                np.array([float(state.condenser.M)])
            )
            obs_parts.extend([tray_M_norm, reb_M_norm, cond_M_norm])

        observation = np.concatenate(obs_parts).astype(np.float32)

        # Clip to valid range
        observation = np.clip(observation, 0.0, 1.0)

        return observation

    def _get_info(self) -> dict[str, Any]:
        """Get additional information about current state.

        Returns:
            Info dictionary.
        """
        state = self._state
        return {
            "step": self._step_count,
            "time": float(state.t),
            "reboiler_holdup": float(state.reboiler.M),
            "condenser_holdup": float(state.condenser.M),
            "tray_holdup_mean": float(np.mean(np.array(state.tray_M))),
        }

    def render(self) -> np.ndarray | None:
        """Render the environment.

        Returns:
            RGB array if render_mode is "rgb_array", None otherwise.
        """
        if self.render_mode is None:
            return None

        # Simple text rendering for "human" mode
        if self.render_mode == "human":
            state = self._state
            print(f"\nStep {self._step_count}:")
            print(f"  Distillate x: {float(state.condenser.x):.3f}")
            print(f"  Bottoms x: {float(state.reboiler.x):.3f}")
            print(f"  Reboiler T: {float(state.reboiler.T):.1f} K")
            print(f"  Condenser T: {float(state.condenser.T):.1f} K")
            return None

        # For "rgb_array" mode, return a simple visualization
        if self.render_mode == "rgb_array":
            # Create a simple column visualization
            n_trays = self.config.geometry.n_trays
            height = 100 + n_trays * 20
            width = 200
            img = np.ones((height, width, 3), dtype=np.uint8) * 255

            # Draw column outline
            img[20:height - 20, 50:150, :] = 200

            # Draw trays with color based on composition
            state = self._state
            for i, x in enumerate(state.tray_x):
                y = 30 + i * 20
                color = int(255 * float(x))
                img[y : y + 15, 60:140, 0] = color
                img[y : y + 15, 60:140, 1] = 0
                img[y : y + 15, 60:140, 2] = 255 - color

            return img

        return None

    def close(self):
        """Clean up environment resources."""
        pass


def make_env(
    config_kwargs: dict[str, Any] | None = None,
    env_kwargs: dict[str, Any] | None = None,
) -> DistillationColumnEnv:
    """Factory function to create environment.

    Args:
        config_kwargs: Arguments for create_teaching_column_config.
        env_kwargs: Arguments for DistillationColumnEnv.

    Returns:
        Configured environment instance.
    """
    config_kwargs = config_kwargs or {}
    env_kwargs = env_kwargs or {}

    config = create_teaching_column_config(**config_kwargs)
    return DistillationColumnEnv(config=config, **env_kwargs)


# Register environment with Gymnasium
gym.register(
    id="DistillationColumn-v0",
    entry_point="jax_distillation.env.base_env:DistillationColumnEnv",
    max_episode_steps=1000,
)
