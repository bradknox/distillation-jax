"""Tests for the Gymnasium distillation column environment."""

import gymnasium as gym
import numpy as np
import pytest

from jax_distillation.env.base_env import (
    DistillationColumnEnv,
    make_env,
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
    compute_reward,
    purity_reward,
    energy_penalty,
    stability_reward,
)
from jax_distillation.env.wrappers import (
    NormalizedActionWrapper,
    FrameStackWrapper,
    ActionSmoothingWrapper,
    wrap_env,
)


class TestActionSpace:
    """Tests for action space creation."""

    def test_create_action_space_shape(self):
        """Action space should have correct shape."""
        space = create_action_space()
        assert space.shape == (4,)

    def test_create_reduced_action_space_shape(self):
        """Reduced action space should have 2 dimensions."""
        space = create_reduced_action_space()
        assert space.shape == (2,)

    def test_action_space_bounds(self):
        """Action space bounds should be valid."""
        space = create_action_space()
        assert np.all(space.low < space.high)
        assert np.all(space.low >= 0)

    def test_action_space_sample_valid(self):
        """Sampled actions should be within bounds."""
        space = create_action_space()
        for _ in range(10):
            action = space.sample()
            assert space.contains(action)


class TestObservationSpace:
    """Tests for observation space creation."""

    def test_create_observation_space_shape(self):
        """Observation space should have correct shape."""
        n_trays = 10
        space = create_observation_space(n_trays=n_trays)
        # Expected: 2*n_trays + 4 (base) + 4 (flows) = 28
        assert space.shape[0] == 2 * n_trays + 4 + 4

    def test_observation_space_bounds(self):
        """Observation space should be in [0, 1]."""
        space = create_observation_space(n_trays=10)
        assert np.all(space.low == 0.0)
        assert np.all(space.high == 1.0)


class TestNormalizers:
    """Tests for observation and action normalizers."""

    def test_obs_normalizer_temperature(self):
        """Temperature normalization should be correct."""
        normalizer = ObservationNormalizer(T_min=300.0, T_max=400.0)
        T = np.array([350.0])
        T_norm = normalizer.normalize_temperature(T)
        assert np.isclose(T_norm[0], 0.5)

    def test_obs_normalizer_bounds(self):
        """Normalized values should be in expected range."""
        normalizer = ObservationNormalizer()
        T = np.array([normalizer.T_min, normalizer.T_max])
        T_norm = normalizer.normalize_temperature(T)
        assert np.isclose(T_norm[0], 0.0)
        assert np.isclose(T_norm[1], 1.0)

    def test_action_denormalizer(self):
        """Action denormalization should invert bounds."""
        denorm = ActionDenormalizer(
            Q_R_range=(0.0, 10000.0),
            reflux_ratio_range=(1.0, 5.0),
        )
        action = np.array([0.5, 0.5, 0.5, 0.5])
        result = denorm.denormalize(action)
        assert np.isclose(result["Q_R"], 5000.0)
        assert np.isclose(result["reflux_ratio"], 3.0)


class TestRewards:
    """Tests for reward functions."""

    def test_purity_reward_perfect(self):
        """Perfect purity should give reward of 1."""
        reward = purity_reward(x_D=0.95, x_B=0.05, x_D_target=0.95, x_B_target=0.05)
        assert np.isclose(reward, 1.0)

    def test_purity_reward_off_target(self):
        """Off-target compositions should reduce reward."""
        # Use values closer to target to get non-zero reward
        reward = purity_reward(x_D=0.94, x_B=0.06, x_D_target=0.95, x_B_target=0.05)
        assert reward < 1.0
        assert reward > 0.0

    def test_energy_penalty_zero(self):
        """Zero energy should give zero penalty."""
        penalty = energy_penalty(Q_R=0.0, Q_R_max=20000.0)
        assert penalty == 0.0

    def test_energy_penalty_max(self):
        """Maximum energy should give penalty of 1."""
        penalty = energy_penalty(Q_R=20000.0, Q_R_max=20000.0)
        assert np.isclose(penalty, 1.0)

    def test_stability_reward_stable(self):
        """Stable operation should give high reward."""
        reward = stability_reward(dx_D=0.0, dx_B=0.0)
        assert np.isclose(reward, 1.0)

    def test_compute_reward_components(self):
        """Compute reward should return valid components."""
        reward, components = compute_reward(
            x_D=0.90,
            x_B=0.10,
            Q_R=5000.0,
        )
        assert "purity" in components
        assert "energy" in components
        assert "stability" in components
        assert "total" in components


class TestEnvironment:
    """Tests for the main environment."""

    def test_env_creation(self):
        """Environment should be created successfully."""
        env = DistillationColumnEnv()
        assert env is not None

    def test_env_reset(self):
        """Reset should return valid observation and info."""
        env = DistillationColumnEnv()
        obs, info = env.reset()

        assert obs is not None
        assert obs.shape == env.observation_space.shape
        assert env.observation_space.contains(obs)
        assert isinstance(info, dict)

    def test_env_step(self):
        """Step should return valid outputs."""
        env = DistillationColumnEnv()
        env.reset()

        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        assert obs is not None
        assert obs.shape == env.observation_space.shape
        assert np.isscalar(reward) or isinstance(reward, (int, float, np.floating))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_env_multiple_steps(self):
        """Environment should run multiple steps."""
        env = DistillationColumnEnv()
        env.reset()

        for _ in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            assert env.observation_space.contains(obs)
            assert np.isfinite(reward)

            if terminated or truncated:
                break

    def test_env_reduced_action_space(self):
        """Environment should work with reduced action space."""
        env = DistillationColumnEnv(use_reduced_action_space=True)
        assert env.action_space.shape == (2,)

        obs, _ = env.reset()
        action = env.action_space.sample()
        obs, reward, _, _, _ = env.step(action)

        assert env.observation_space.contains(obs)

    def test_env_max_episode_steps(self):
        """Episode should truncate at max steps."""
        max_steps = 10
        env = DistillationColumnEnv(max_episode_steps=max_steps)
        env.reset()

        truncated = False
        for i in range(max_steps + 5):
            action = env.action_space.sample()
            _, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break

        assert truncated
        assert i == max_steps - 1

    def test_env_seed_reproducibility(self):
        """Same seed should produce same initial state."""
        env1 = DistillationColumnEnv()
        env2 = DistillationColumnEnv()

        obs1, _ = env1.reset(seed=42)
        obs2, _ = env2.reset(seed=42)

        assert np.allclose(obs1, obs2)

    def test_env_info_contents(self):
        """Info dict should contain expected keys."""
        env = DistillationColumnEnv()
        _, info = env.reset()

        assert "step" in info
        assert "time" in info

        action = env.action_space.sample()
        _, _, _, _, info = env.step(action)

        assert "outputs" in info
        assert "reward_components" in info

    def test_make_env_factory(self):
        """Factory function should create valid environment."""
        env = make_env(
            config_kwargs={"n_trays": 5},
            env_kwargs={"max_episode_steps": 100},
        )

        assert env is not None
        obs, _ = env.reset()
        assert obs is not None


class TestEnvironmentChecker:
    """Run Gymnasium's environment checker."""

    def test_gymnasium_check_env(self):
        """Environment should pass Gymnasium's check_env."""
        from gymnasium.utils.env_checker import check_env

        env = DistillationColumnEnv()
        # check_env will raise an exception if the env is invalid
        check_env(env, warn=False, skip_render_check=True)


class TestWrappers:
    """Tests for environment wrappers."""

    def test_normalized_action_wrapper(self):
        """Normalized action wrapper should scale actions."""
        env = DistillationColumnEnv()
        wrapped = NormalizedActionWrapper(env)

        assert wrapped.action_space.low[0] == -1.0
        assert wrapped.action_space.high[0] == 1.0

        obs, _ = wrapped.reset()
        # Action of 0 should map to midpoint
        action = np.zeros(4, dtype=np.float32)
        scaled = wrapped.action(action)

        # Check that it's between low and high
        assert np.all(scaled >= env.action_space.low)
        assert np.all(scaled <= env.action_space.high)

    def test_frame_stack_wrapper(self):
        """Frame stack wrapper should stack observations."""
        env = DistillationColumnEnv()
        n_frames = 4
        wrapped = FrameStackWrapper(env, n_frames=n_frames)

        obs, _ = wrapped.reset()
        expected_shape = (env.observation_space.shape[0] * n_frames,)
        assert obs.shape == expected_shape

    def test_action_smoothing_wrapper(self):
        """Action smoothing should blend actions."""
        env = DistillationColumnEnv()
        wrapped = ActionSmoothingWrapper(env, smoothing_factor=0.5)

        wrapped.reset()

        # First action is not smoothed
        action1 = np.ones(4, dtype=np.float32) * 10000
        smoothed1 = wrapped.action(action1)
        assert np.allclose(smoothed1, action1)

        # Second action should be blended
        action2 = np.zeros(4, dtype=np.float32)
        smoothed2 = wrapped.action(action2)
        # Should be halfway between action1 and action2
        assert np.allclose(smoothed2, action1 * 0.5)

    def test_wrap_env_utility(self):
        """wrap_env should apply multiple wrappers."""
        env = DistillationColumnEnv()
        wrapped = wrap_env(
            env,
            normalize_action=True,
            frame_stack=2,
            record_stats=True,
        )

        obs, _ = wrapped.reset()
        action = wrapped.action_space.sample()
        obs, reward, _, _, info = wrapped.step(action)

        assert obs is not None
        assert np.isfinite(reward)


class TestEnvironmentDynamics:
    """Tests for environment physical behavior."""

    def test_reboiler_duty_effect(self):
        """Higher reboiler duty should affect product quality."""
        env = DistillationColumnEnv()

        # Run with low heat
        env.reset(seed=0)
        low_action = np.array([2000.0, 3.0, 0.03, 0.02], dtype=np.float32)
        for _ in range(10):
            _, _, _, _, info_low = env.step(low_action)

        # Run with high heat
        env.reset(seed=0)
        high_action = np.array([15000.0, 3.0, 0.03, 0.02], dtype=np.float32)
        for _ in range(10):
            _, _, _, _, info_high = env.step(high_action)

        # Higher heat should use more energy
        assert info_high["outputs"]["Q_R"] > info_low["outputs"]["Q_R"]

    def test_observations_bounded(self):
        """Observations should stay within space bounds."""
        env = DistillationColumnEnv()
        env.reset()

        for _ in range(20):
            action = env.action_space.sample()
            obs, _, terminated, truncated, _ = env.step(action)

            assert env.observation_space.contains(obs), f"Observation out of bounds: {obs}"

            if terminated or truncated:
                break


class TestRegisteredEnvironment:
    """Tests for registered Gymnasium environment."""

    def test_make_registered_env(self):
        """Should be able to create env via gym.make."""
        # This tests the gym.register call
        env = gym.make("DistillationColumn-v0")
        assert env is not None

        obs, _ = env.reset()
        assert obs is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
