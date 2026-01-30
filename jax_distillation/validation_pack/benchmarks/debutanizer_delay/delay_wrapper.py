"""Gymnasium wrapper for delayed composition measurements.

This module implements a wrapper that adds realistic measurement
delays to composition observations, simulating gas chromatograph
analyzer behavior in industrial distillation columns.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
from gymnasium import spaces
import numpy as np


@dataclass
class DelayConfig:
    """Configuration for delayed measurements.

    Attributes:
        dead_time: Fixed dead time [s] (if deterministic).
        dead_time_mean: Mean dead time [s] (if stochastic).
        dead_time_std: Std dev of dead time [s] (if stochastic).
        sample_period: Time between measurements [s].
        measurement_noise_std: Std dev of measurement noise.
        missing_probability: Probability of missing measurement.
        initial_value_mode: How to handle pre-delay period
            ("nan", "feed", "last")
        seed: Random seed for reproducibility.
    """

    dead_time: float = 60.0  # Default 1 minute
    dead_time_mean: Optional[float] = None
    dead_time_std: float = 0.0
    sample_period: float = 60.0  # Default 1 minute sample period
    measurement_noise_std: float = 0.0
    missing_probability: float = 0.0
    initial_value_mode: str = "feed"  # "nan", "feed", or "last"
    seed: Optional[int] = None


class DelayWrapper(gym.Wrapper):
    """Gymnasium wrapper adding delayed composition measurements.

    This wrapper maintains a buffer of historical composition values
    and provides delayed observations based on the configured dead time.

    The observation space is modified to include:
    - Original observations (temperatures, flows, etc.)
    - delayed_x_D: Delayed distillate composition
    - delayed_x_B: Delayed bottoms composition
    - time_since_measurement: Time since last composition update

    The info dict includes:
    - true_x_D: Actual (undelayed) distillate composition
    - true_x_B: Actual (undelayed) bottoms composition
    """

    def __init__(
        self,
        env: gym.Env,
        config: Optional[DelayConfig] = None,
    ):
        """Initialize delay wrapper.

        Args:
            env: Base Gymnasium environment.
            config: Delay configuration. Uses defaults if None.
        """
        super().__init__(env)

        if config is None:
            config = DelayConfig()
        self.config = config

        # Get simulation dt from environment
        # Try to extract from env attributes, or use default
        self._dt = getattr(env, "dt", 1.0)
        if hasattr(env, "unwrapped") and hasattr(env.unwrapped, "config"):
            try:
                self._dt = float(env.unwrapped.config.simulation.dt)
            except AttributeError:
                pass

        # Calculate buffer size
        max_dead_time = config.dead_time
        if config.dead_time_mean is not None:
            max_dead_time = config.dead_time_mean + 3 * config.dead_time_std
        self._buffer_size = max(1, int(max_dead_time / self._dt) + 10)

        # Initialize buffers
        self._x_D_buffer = np.zeros(self._buffer_size)
        self._x_B_buffer = np.zeros(self._buffer_size)
        self._buffer_index = 0
        self._simulation_time = 0.0

        # Measurement state
        self._last_measurement_time = -float("inf")
        self._last_measured_x_D = np.nan
        self._last_measured_x_B = np.nan

        # Random state
        self._rng = np.random.default_rng(config.seed)

        # Modify observation space to include delayed measurements
        self._setup_observation_space()

    def _setup_observation_space(self):
        """Set up modified observation space with delay info."""
        base_space = self.env.observation_space

        # Add delay-related observations
        delay_space = spaces.Dict({
            "delayed_x_D": spaces.Box(low=0.0, high=1.0, shape=(), dtype=np.float32),
            "delayed_x_B": spaces.Box(low=0.0, high=1.0, shape=(), dtype=np.float32),
            "time_since_measurement": spaces.Box(
                low=0.0, high=float("inf"), shape=(), dtype=np.float32
            ),
        })

        if isinstance(base_space, spaces.Dict):
            # Merge with existing dict space
            combined_spaces = dict(base_space.spaces)
            combined_spaces.update(delay_space.spaces)
            self.observation_space = spaces.Dict(combined_spaces)
        else:
            # Wrap in dict
            self.observation_space = spaces.Dict({
                "base": base_space,
                **delay_space.spaces,
            })

    def _get_dead_time(self) -> float:
        """Get dead time (possibly stochastic).

        Returns:
            Dead time in seconds.
        """
        if self.config.dead_time_mean is not None:
            dt = self._rng.normal(
                self.config.dead_time_mean,
                self.config.dead_time_std,
            )
            return max(0.0, dt)
        return self.config.dead_time

    def _add_measurement_noise(self, value: float) -> float:
        """Add measurement noise to a value.

        Args:
            value: True measurement value.

        Returns:
            Noisy measurement.
        """
        if self.config.measurement_noise_std > 0:
            noise = self._rng.normal(0, self.config.measurement_noise_std)
            return np.clip(value + noise, 0.0, 1.0)
        return value

    def _is_measurement_missing(self) -> bool:
        """Check if current measurement is missing.

        Returns:
            True if measurement should be marked as missing.
        """
        if self.config.missing_probability > 0:
            return self._rng.random() < self.config.missing_probability
        return False

    def _store_composition(self, x_D: float, x_B: float):
        """Store current composition in buffer.

        Args:
            x_D: Current distillate composition.
            x_B: Current bottoms composition.
        """
        self._x_D_buffer[self._buffer_index] = x_D
        self._x_B_buffer[self._buffer_index] = x_B
        self._buffer_index = (self._buffer_index + 1) % self._buffer_size

    def _get_delayed_composition(self, dead_time: float) -> Tuple[float, float]:
        """Get composition from buffer with specified delay.

        Args:
            dead_time: Delay time in seconds.

        Returns:
            Tuple of (delayed_x_D, delayed_x_B).
        """
        # Calculate index offset
        steps_back = int(dead_time / self._dt)
        if steps_back >= self._buffer_size:
            steps_back = self._buffer_size - 1

        delayed_index = (self._buffer_index - 1 - steps_back) % self._buffer_size

        return self._x_D_buffer[delayed_index], self._x_B_buffer[delayed_index]

    def _update_measurement(self):
        """Update measurement if sample period has elapsed."""
        time_since_last = self._simulation_time - self._last_measurement_time

        if time_since_last >= self.config.sample_period:
            # Time for a new measurement
            if not self._is_measurement_missing():
                dead_time = self._get_dead_time()

                # Check if we have enough history
                if self._simulation_time >= dead_time:
                    x_D, x_B = self._get_delayed_composition(dead_time)
                    self._last_measured_x_D = self._add_measurement_noise(x_D)
                    self._last_measured_x_B = self._add_measurement_noise(x_B)
                else:
                    # Not enough history yet, use initial value mode
                    if self.config.initial_value_mode == "nan":
                        self._last_measured_x_D = np.nan
                        self._last_measured_x_B = np.nan
                    elif self.config.initial_value_mode == "feed":
                        # Use feed composition as initial guess
                        self._last_measured_x_D = 0.5  # Placeholder
                        self._last_measured_x_B = 0.5  # Placeholder
                    # "last" mode keeps previous value

            self._last_measurement_time = self._simulation_time

    def _extract_composition(self, obs: Any, info: Dict) -> Tuple[float, float]:
        """Extract true composition from observation or info.

        Args:
            obs: Observation from base environment.
            info: Info dict from base environment.

        Returns:
            Tuple of (x_D, x_B).
        """
        # Try to get from info first
        if "x_D" in info:
            return float(info["x_D"]), float(info.get("x_B", 0.5))

        # Try to extract from observation
        if isinstance(obs, dict):
            if "x_D" in obs:
                return float(obs["x_D"]), float(obs.get("x_B", 0.5))

        # Fallback to defaults
        return 0.5, 0.5

    def _wrap_observation(
        self,
        obs: Any,
        true_x_D: float,
        true_x_B: float,
    ) -> Dict[str, Any]:
        """Wrap observation with delayed measurements.

        Args:
            obs: Base observation.
            true_x_D: True distillate composition.
            true_x_B: True bottoms composition.

        Returns:
            Wrapped observation dict.
        """
        time_since = self._simulation_time - self._last_measurement_time

        if isinstance(obs, dict):
            wrapped = dict(obs)
        else:
            wrapped = {"base": obs}

        wrapped["delayed_x_D"] = np.float32(self._last_measured_x_D)
        wrapped["delayed_x_B"] = np.float32(self._last_measured_x_B)
        wrapped["time_since_measurement"] = np.float32(time_since)

        return wrapped

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset environment and delay buffers.

        Args:
            seed: Random seed.
            options: Reset options.

        Returns:
            Tuple of (observation, info).
        """
        # Reset base environment
        obs, info = self.env.reset(seed=seed, options=options)

        # Reset delay state
        self._x_D_buffer[:] = 0.0
        self._x_B_buffer[:] = 0.0
        self._buffer_index = 0
        self._simulation_time = 0.0
        self._last_measurement_time = -float("inf")

        # Initialize with feed composition
        true_x_D, true_x_B = self._extract_composition(obs, info)
        self._last_measured_x_D = true_x_D if self.config.initial_value_mode == "feed" else np.nan
        self._last_measured_x_B = true_x_B if self.config.initial_value_mode == "feed" else np.nan

        # Store initial composition
        self._store_composition(true_x_D, true_x_B)

        # Update measurement
        self._update_measurement()

        # Wrap observation
        wrapped_obs = self._wrap_observation(obs, true_x_D, true_x_B)

        # Add true values to info
        info["true_x_D"] = true_x_D
        info["true_x_B"] = true_x_B

        return wrapped_obs, info

    def step(
        self,
        action: Any,
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Step environment with delay handling.

        Args:
            action: Action to take.

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        # Step base environment
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Update simulation time
        self._simulation_time += self._dt

        # Extract and store true composition
        true_x_D, true_x_B = self._extract_composition(obs, info)
        self._store_composition(true_x_D, true_x_B)

        # Update measurement
        self._update_measurement()

        # Wrap observation
        wrapped_obs = self._wrap_observation(obs, true_x_D, true_x_B)

        # Add true values to info
        info["true_x_D"] = true_x_D
        info["true_x_B"] = true_x_B

        return wrapped_obs, float(reward), terminated, truncated, info


def create_delayed_env(
    env: Optional[gym.Env] = None,
    dead_time: float = 60.0,
    sample_period: float = 60.0,
    seed: Optional[int] = None,
) -> DelayWrapper:
    """Create a delayed measurement environment.

    Args:
        env: Base environment. Creates default if None.
        dead_time: Measurement dead time [s].
        sample_period: Time between measurements [s].
        seed: Random seed.

    Returns:
        DelayWrapper around the environment.
    """
    if env is None:
        from jax_distillation.env import DistillationColumnEnv
        env = DistillationColumnEnv()

    config = DelayConfig(
        dead_time=dead_time,
        sample_period=sample_period,
        seed=seed,
    )

    return DelayWrapper(env, config)
