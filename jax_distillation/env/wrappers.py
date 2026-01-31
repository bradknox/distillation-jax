"""Gymnasium wrappers for distillation column environment.

Provides observation and action transformations:
- Normalization
- Frame stacking
- Action smoothing
- Observation filtering
"""

import gymnasium as gym
import numpy as np
from collections import deque
from typing import Any


class NormalizedObservationWrapper(gym.ObservationWrapper):
    """Normalize observations to zero mean and unit variance.

    Uses running statistics to normalize observations, useful for
    policy gradient methods.
    """

    def __init__(
        self,
        env: gym.Env,
        epsilon: float = 1e-8,
        clip: float = 10.0,
    ):
        """Initialize wrapper.

        Args:
            env: Environment to wrap.
            epsilon: Small constant for numerical stability.
            clip: Clip normalized values to [-clip, clip].
        """
        super().__init__(env)
        self.epsilon = epsilon
        self.clip = clip

        obs_shape = env.observation_space.shape
        self.running_mean = np.zeros(obs_shape, dtype=np.float64)
        self.running_var = np.ones(obs_shape, dtype=np.float64)
        self.count = 0

    def observation(self, obs: np.ndarray) -> np.ndarray:
        """Normalize observation."""
        self._update_stats(obs)
        normalized = (obs - self.running_mean) / np.sqrt(self.running_var + self.epsilon)
        return np.clip(normalized, -self.clip, self.clip).astype(np.float32)

    def _update_stats(self, obs: np.ndarray):
        """Update running statistics with new observation."""
        self.count += 1
        delta = obs - self.running_mean
        self.running_mean += delta / self.count
        delta2 = obs - self.running_mean
        self.running_var += (delta * delta2 - self.running_var) / self.count


class NormalizedActionWrapper(gym.ActionWrapper):
    """Normalize actions from [-1, 1] to environment's action space."""

    def __init__(self, env: gym.Env):
        """Initialize wrapper.

        Args:
            env: Environment to wrap.
        """
        super().__init__(env)

        # Store original action space bounds
        self.low = env.action_space.low
        self.high = env.action_space.high

        # Create normalized action space
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=env.action_space.shape,
            dtype=np.float32,
        )

    def action(self, action: np.ndarray) -> np.ndarray:
        """Convert normalized action to environment scale."""
        # Scale from [-1, 1] to [low, high]
        return self.low + (action + 1.0) * 0.5 * (self.high - self.low)

    def reverse_action(self, action: np.ndarray) -> np.ndarray:
        """Convert environment action to normalized scale."""
        return 2.0 * (action - self.low) / (self.high - self.low) - 1.0


class FrameStackWrapper(gym.ObservationWrapper):
    """Stack multiple observations for temporal context.

    Useful for learning dynamics from observation sequences.
    """

    def __init__(
        self,
        env: gym.Env,
        n_frames: int = 4,
    ):
        """Initialize wrapper.

        Args:
            env: Environment to wrap.
            n_frames: Number of frames to stack.
        """
        super().__init__(env)
        self.n_frames = n_frames
        self.frames = deque(maxlen=n_frames)

        # Update observation space
        low = np.repeat(env.observation_space.low[np.newaxis, :], n_frames, axis=0)
        high = np.repeat(env.observation_space.high[np.newaxis, :], n_frames, axis=0)
        self.observation_space = gym.spaces.Box(
            low=low.flatten(),
            high=high.flatten(),
            dtype=np.float32,
        )

    def reset(self, **kwargs) -> tuple[np.ndarray, dict]:
        """Reset and initialize frame buffer."""
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.n_frames):
            self.frames.append(obs)
        return self.observation(obs), info

    def observation(self, obs: np.ndarray) -> np.ndarray:
        """Stack frames into single observation."""
        self.frames.append(obs)
        return np.concatenate(list(self.frames)).astype(np.float32)


class ActionSmoothingWrapper(gym.ActionWrapper):
    """Smooth actions using exponential moving average.

    Prevents abrupt control changes that could destabilize the column.
    """

    def __init__(
        self,
        env: gym.Env,
        smoothing_factor: float = 0.3,
    ):
        """Initialize wrapper.

        Args:
            env: Environment to wrap.
            smoothing_factor: EMA factor (0 = no smoothing, 1 = full smoothing).
        """
        super().__init__(env)
        self.smoothing_factor = smoothing_factor
        self.prev_action = None

    def reset(self, **kwargs) -> tuple[np.ndarray, dict]:
        """Reset and clear previous action."""
        self.prev_action = None
        return self.env.reset(**kwargs)

    def action(self, action: np.ndarray) -> np.ndarray:
        """Apply exponential smoothing to action."""
        if self.prev_action is None:
            self.prev_action = action.copy()
            return action

        smoothed = (
            self.smoothing_factor * self.prev_action
            + (1 - self.smoothing_factor) * action
        )
        self.prev_action = smoothed.copy()
        return smoothed


class RewardScaleWrapper(gym.RewardWrapper):
    """Scale reward by a constant factor."""

    def __init__(
        self,
        env: gym.Env,
        scale: float = 1.0,
    ):
        """Initialize wrapper.

        Args:
            env: Environment to wrap.
            scale: Reward scaling factor.
        """
        super().__init__(env)
        self.scale = scale

    def reward(self, reward: float) -> float:
        """Scale reward."""
        return reward * self.scale


class ClipRewardWrapper(gym.RewardWrapper):
    """Clip reward to a specified range."""

    def __init__(
        self,
        env: gym.Env,
        min_reward: float = -10.0,
        max_reward: float = 10.0,
    ):
        """Initialize wrapper.

        Args:
            env: Environment to wrap.
            min_reward: Minimum reward value.
            max_reward: Maximum reward value.
        """
        super().__init__(env)
        self.min_reward = min_reward
        self.max_reward = max_reward

    def reward(self, reward: float) -> float:
        """Clip reward."""
        return np.clip(reward, self.min_reward, self.max_reward)


class TimeFeatureWrapper(gym.ObservationWrapper):
    """Add normalized time feature to observation.

    Useful for time-varying setpoints or episode-aware policies.
    """

    def __init__(
        self,
        env: gym.Env,
        max_episode_steps: int = 1000,
    ):
        """Initialize wrapper.

        Args:
            env: Environment to wrap.
            max_episode_steps: Maximum episode length for normalization.
        """
        super().__init__(env)
        self.max_episode_steps = max_episode_steps
        self.current_step = 0

        # Extend observation space
        old_space = env.observation_space
        self.observation_space = gym.spaces.Box(
            low=np.append(old_space.low, 0.0),
            high=np.append(old_space.high, 1.0),
            dtype=np.float32,
        )

    def reset(self, **kwargs) -> tuple[np.ndarray, dict]:
        """Reset step counter."""
        self.current_step = 0
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info

    def observation(self, obs: np.ndarray) -> np.ndarray:
        """Add time feature to observation."""
        time_feature = self.current_step / self.max_episode_steps
        return np.append(obs, time_feature).astype(np.float32)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Step and increment counter."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.current_step += 1
        return self.observation(obs), reward, terminated, truncated, info


class RecordEpisodeStatisticsWrapper(gym.Wrapper):
    """Record episode statistics for logging.

    Tracks cumulative reward, episode length, and custom metrics.
    """

    def __init__(self, env: gym.Env):
        """Initialize wrapper."""
        super().__init__(env)
        self.episode_returns = []
        self.episode_lengths = []
        self.current_return = 0.0
        self.current_length = 0

    def reset(self, **kwargs) -> tuple[np.ndarray, dict]:
        """Reset and record previous episode stats."""
        if self.current_length > 0:
            self.episode_returns.append(self.current_return)
            self.episode_lengths.append(self.current_length)

        self.current_return = 0.0
        self.current_length = 0
        return self.env.reset(**kwargs)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Step and track statistics."""
        obs, reward, terminated, truncated, info = self.env.step(action)

        self.current_return += reward
        self.current_length += 1

        if terminated or truncated:
            info["episode"] = {
                "return": self.current_return,
                "length": self.current_length,
            }

        return obs, reward, terminated, truncated, info


def wrap_env(
    env: gym.Env,
    normalize_obs: bool = False,
    normalize_action: bool = True,
    frame_stack: int = 0,
    action_smoothing: float = 0.0,
    reward_scale: float = 1.0,
    add_time_feature: bool = False,
    record_stats: bool = True,
) -> gym.Env:
    """Apply common wrappers to environment.

    Args:
        env: Base environment.
        normalize_obs: Apply running observation normalization.
        normalize_action: Normalize action space to [-1, 1].
        frame_stack: Number of frames to stack (0 = disabled).
        action_smoothing: Action smoothing factor (0 = disabled).
        reward_scale: Reward scaling factor.
        add_time_feature: Add time feature to observation.
        record_stats: Record episode statistics.

    Returns:
        Wrapped environment.
    """
    if record_stats:
        env = RecordEpisodeStatisticsWrapper(env)

    if normalize_obs:
        env = NormalizedObservationWrapper(env)

    if normalize_action:
        env = NormalizedActionWrapper(env)

    if frame_stack > 1:
        env = FrameStackWrapper(env, n_frames=frame_stack)

    if action_smoothing > 0:
        env = ActionSmoothingWrapper(env, smoothing_factor=action_smoothing)

    if reward_scale != 1.0:
        env = RewardScaleWrapper(env, scale=reward_scale)

    if add_time_feature:
        max_steps = getattr(env, "max_episode_steps", 1000)
        env = TimeFeatureWrapper(env, max_episode_steps=max_steps)

    return env
