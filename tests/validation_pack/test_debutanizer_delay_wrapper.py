"""Tests for debutanizer delay wrapper."""

import pytest
import numpy as np

from jax_distillation.validation_pack.benchmarks.debutanizer_delay import (
    DelayWrapper,
    DelayConfig,
    run_delay_validation,
)


class TestDelayConfig:
    """Tests for delay configuration."""

    def test_default_config(self):
        """Test default delay configuration."""
        config = DelayConfig()

        assert config.dead_time > 0
        assert config.sample_period > 0

    def test_custom_config(self):
        """Test custom delay configuration."""
        config = DelayConfig(
            dead_time=120.0,
            sample_period=60.0,
            measurement_noise_std=0.01,
        )

        assert config.dead_time == 120.0
        assert config.sample_period == 60.0
        assert config.measurement_noise_std == 0.01


class TestDelayWrapper:
    """Tests for delay wrapper functionality."""

    def test_wrapper_creates(self):
        """Test that wrapper can be created."""
        pytest.importorskip("gymnasium")

        from jax_distillation.validation_pack.benchmarks.debutanizer_delay.delay_wrapper import (
            create_delayed_env,
        )

        env = create_delayed_env(dead_time=30.0, sample_period=30.0)
        assert env is not None
        env.close()

    def test_wrapper_reset(self):
        """Test that wrapper reset works."""
        pytest.importorskip("gymnasium")

        from jax_distillation.validation_pack.benchmarks.debutanizer_delay.delay_wrapper import (
            create_delayed_env,
        )

        env = create_delayed_env(dead_time=30.0, sample_period=30.0)
        obs, info = env.reset()

        assert obs is not None
        assert isinstance(obs, dict)
        assert "delayed_x_D" in obs
        assert "delayed_x_B" in obs
        assert "time_since_measurement" in obs

        env.close()

    def test_wrapper_step(self):
        """Test that wrapper step works."""
        pytest.importorskip("gymnasium")

        from jax_distillation.validation_pack.benchmarks.debutanizer_delay.delay_wrapper import (
            create_delayed_env,
        )

        env = create_delayed_env(dead_time=30.0, sample_period=30.0)
        obs, info = env.reset()

        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        assert obs is not None
        assert "delayed_x_D" in obs
        assert "true_x_D" in info  # True value in info

        env.close()

    def test_observation_space_valid(self):
        """Test that observation space is properly defined."""
        pytest.importorskip("gymnasium")

        from jax_distillation.validation_pack.benchmarks.debutanizer_delay.delay_wrapper import (
            create_delayed_env,
        )

        env = create_delayed_env(dead_time=30.0, sample_period=30.0)
        obs, _ = env.reset()

        # Observation should be in observation space
        assert env.observation_space.contains(obs)

        env.close()


class TestDelayValidation:
    """Tests for delay wrapper validation."""

    def test_validation_runs(self):
        """Test that delay validation runs."""
        pytest.importorskip("gymnasium")

        result = run_delay_validation(
            dead_time=30.0,
            sample_period=30.0,
            n_steps=50,
        )

        assert result is not None
        assert hasattr(result, "delay_correct")
        assert hasattr(result, "deterministic")
        assert hasattr(result, "api_compliant")

    def test_api_compliant(self):
        """Test that wrapper is Gymnasium API compliant."""
        pytest.importorskip("gymnasium")

        result = run_delay_validation(
            dead_time=30.0,
            sample_period=30.0,
            n_steps=50,
        )

        assert result.api_compliant, "Wrapper not Gymnasium API compliant"

    def test_deterministic_with_seed(self):
        """Test that wrapper is deterministic with fixed seed."""
        pytest.importorskip("gymnasium")

        result = run_delay_validation(
            dead_time=30.0,
            sample_period=30.0,
            n_steps=50,
            seed=42,
        )

        assert result.deterministic, "Wrapper not deterministic with fixed seed"

    @pytest.mark.slow
    def test_full_validation(self):
        """Run complete delay validation."""
        pytest.importorskip("gymnasium")

        result = run_delay_validation(
            dead_time=60.0,
            sample_period=60.0,
            n_steps=100,
        )

        print(f"Delay correct: {result.delay_correct}")
        print(f"Sample-hold correct: {result.sample_hold_correct}")
        print(f"Deterministic: {result.deterministic}")
        print(f"API compliant: {result.api_compliant}")

        # All should pass
        all_passed = (
            result.delay_correct
            and result.sample_hold_correct
            and result.deterministic
            and result.api_compliant
        )
        assert all_passed, "Some delay validation checks failed"
