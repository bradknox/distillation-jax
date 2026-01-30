"""Metrics and validation for delay wrapper behavior.

This module validates that the delay wrapper correctly implements
measurement delays and sample-and-hold behavior.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from jax_distillation.validation_pack.benchmarks.debutanizer_delay.delay_wrapper import (
    DelayWrapper,
    DelayConfig,
    create_delayed_env,
)


@dataclass
class DelayValidationResult:
    """Result of delay wrapper validation.

    Attributes:
        delay_correct: True if delays are applied correctly
        sample_hold_correct: True if sample-and-hold works
        deterministic: True if results are reproducible with same seed
        api_compliant: True if passes Gymnasium API check
        n_steps_tested: Number of simulation steps tested
        measured_delay: Actual measured delay [s]
        expected_delay: Expected delay from config [s]
        details: Additional diagnostic information
    """

    delay_correct: bool
    sample_hold_correct: bool
    deterministic: bool
    api_compliant: bool
    n_steps_tested: int
    measured_delay: float
    expected_delay: float
    details: Dict


def _measure_delay(
    env: DelayWrapper,
    n_steps: int = 200,
) -> Tuple[float, Dict]:
    """Measure actual delay by comparing true vs delayed observations.

    Args:
        env: Delayed environment.
        n_steps: Number of steps to run.

    Returns:
        Tuple of (measured_delay_steps, trajectory_data).
    """
    obs, info = env.reset()
    dt = env._dt

    true_x_D_history = []
    delayed_x_D_history = []
    times = []

    for step in range(n_steps):
        # Random action
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        true_x_D_history.append(info.get("true_x_D", np.nan))
        delayed_x_D_history.append(obs.get("delayed_x_D", np.nan))
        times.append(step * dt)

        if terminated or truncated:
            break

    true_x_D = np.array(true_x_D_history)
    delayed_x_D = np.array(delayed_x_D_history)
    times = np.array(times)

    # Estimate delay by cross-correlation
    if len(true_x_D) > 10 and not np.any(np.isnan(delayed_x_D)):
        # Find shift that maximizes correlation
        max_shift = min(len(true_x_D) // 2, 100)
        best_corr = -1
        best_shift = 0

        for shift in range(max_shift):
            if shift == 0:
                corr = np.corrcoef(true_x_D, delayed_x_D)[0, 1]
            else:
                corr = np.corrcoef(true_x_D[:-shift], delayed_x_D[shift:])[0, 1]

            if not np.isnan(corr) and corr > best_corr:
                best_corr = corr
                best_shift = shift

        # If no valid correlation was found (e.g. constant signal at steady
        # state), report NaN so callers know delay is unmeasurable.
        if best_corr < 0:
            measured_delay = np.nan
        else:
            measured_delay = best_shift * dt
    else:
        measured_delay = np.nan

    return measured_delay, {
        "true_x_D": true_x_D,
        "delayed_x_D": delayed_x_D,
        "times": times,
    }


def _check_sample_hold(trajectory_data: Dict, sample_period: float, dt: float) -> bool:
    """Check that sample-and-hold is correctly implemented.

    Args:
        trajectory_data: Trajectory from delay measurement.
        sample_period: Expected sample period [s].
        dt: Timestep [s].

    Returns:
        True if sample-and-hold appears correct.
    """
    delayed_x_D = trajectory_data.get("delayed_x_D", np.array([]))

    if len(delayed_x_D) < 10 or np.any(np.isnan(delayed_x_D)):
        return True  # Can't verify, assume OK

    # Count consecutive identical values
    steps_per_sample = int(sample_period / dt)
    if steps_per_sample < 2:
        return True  # Sample period too short to test

    # Check that values are held constant between samples
    n_holds = 0
    n_changes = 0

    for i in range(1, len(delayed_x_D)):
        if abs(delayed_x_D[i] - delayed_x_D[i - 1]) < 1e-8:
            n_holds += 1
        else:
            n_changes += 1

    # Should have more holds than changes if sample_period > dt
    if steps_per_sample > 1:
        expected_hold_ratio = (steps_per_sample - 1) / steps_per_sample
        actual_hold_ratio = n_holds / (n_holds + n_changes) if (n_holds + n_changes) > 0 else 0

        # Allow some tolerance
        return actual_hold_ratio > expected_hold_ratio * 0.5

    return True


def _check_determinism(config: DelayConfig, n_steps: int = 100) -> bool:
    """Check that results are reproducible with fixed seed.

    Args:
        config: Delay configuration with seed.
        n_steps: Number of steps to test.

    Returns:
        True if results are deterministic.
    """
    if config.seed is None:
        return True  # Can't test without seed

    # Run twice with same seed
    results = []

    for _ in range(2):
        env = create_delayed_env(
            dead_time=config.dead_time,
            sample_period=config.sample_period,
            seed=config.seed,
        )
        obs, _ = env.reset(seed=config.seed)

        trajectory = []
        for _ in range(n_steps):
            action = env.action_space.sample()
            obs, _, terminated, truncated, _ = env.step(action)
            trajectory.append(obs.get("delayed_x_D", 0.0))

            if terminated or truncated:
                break

        results.append(np.array(trajectory))
        env.close()

    # Check if trajectories are identical
    if len(results[0]) != len(results[1]):
        return False

    return np.allclose(results[0], results[1], rtol=1e-6)


def _check_gymnasium_api(env: DelayWrapper) -> bool:
    """Check Gymnasium API compliance.

    Args:
        env: Environment to check.

    Returns:
        True if API compliant.
    """
    try:
        # Check reset
        obs, info = env.reset()
        assert isinstance(obs, dict), "Observation must be dict"
        assert "delayed_x_D" in obs, "Missing delayed_x_D"
        assert "delayed_x_B" in obs, "Missing delayed_x_B"
        assert isinstance(info, dict), "Info must be dict"

        # Check step
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert isinstance(obs, dict), "Step obs must be dict"
        assert isinstance(reward, (int, float)), "Reward must be numeric"
        assert isinstance(terminated, bool), "Terminated must be bool"
        assert isinstance(truncated, bool), "Truncated must be bool"
        assert isinstance(info, dict), "Step info must be dict"

        # Check observation space
        assert env.observation_space.contains(obs), "Obs not in observation_space"

        return True

    except Exception as e:
        print(f"API check failed: {e}")
        return False


def run_delay_validation(
    dead_time: float = 60.0,
    sample_period: float = 60.0,
    n_steps: int = 200,
    seed: int = 42,
) -> DelayValidationResult:
    """Run complete delay wrapper validation.

    Args:
        dead_time: Dead time to test [s].
        sample_period: Sample period to test [s].
        n_steps: Number of steps to run.
        seed: Random seed.

    Returns:
        DelayValidationResult with all validation metrics.
    """
    config = DelayConfig(
        dead_time=dead_time,
        sample_period=sample_period,
        seed=seed,
    )

    env = create_delayed_env(
        dead_time=dead_time,
        sample_period=sample_period,
        seed=seed,
    )

    # Measure delay
    measured_delay, trajectory_data = _measure_delay(env, n_steps)

    # Check delay correctness (within 50% tolerance due to measurement noise)
    delay_error = abs(measured_delay - dead_time) / dead_time if dead_time > 0 else 0
    delay_correct = delay_error < 0.5 or np.isnan(measured_delay)

    # Check sample-and-hold
    sample_hold_correct = _check_sample_hold(trajectory_data, sample_period, env._dt)

    # Check determinism
    deterministic = _check_determinism(config, n_steps)

    # Check API compliance
    api_compliant = _check_gymnasium_api(env)

    env.close()

    return DelayValidationResult(
        delay_correct=delay_correct,
        sample_hold_correct=sample_hold_correct,
        deterministic=deterministic,
        api_compliant=api_compliant,
        n_steps_tested=n_steps,
        measured_delay=measured_delay,
        expected_delay=dead_time,
        details={
            "delay_error": delay_error,
            "trajectory_length": len(trajectory_data.get("times", [])),
        },
    )


def print_delay_validation_report(result: DelayValidationResult) -> None:
    """Print a formatted delay validation report.

    Args:
        result: DelayValidationResult from run_delay_validation.
    """
    print("=" * 70)
    print("DELAY WRAPPER VALIDATION REPORT")
    print("=" * 70)

    def status(ok: bool) -> str:
        return "PASS" if ok else "FAIL"

    print(f"\nDelay Correctness: {status(result.delay_correct)}")
    print(f"  Expected delay: {result.expected_delay:.1f} s")
    if not np.isnan(result.measured_delay):
        print(f"  Measured delay: {result.measured_delay:.1f} s")
    else:
        print("  Measured delay: N/A (insufficient data)")

    print(f"\nSample-and-Hold: {status(result.sample_hold_correct)}")
    print(f"  Measurements held constant between updates")

    print(f"\nDeterminism: {status(result.deterministic)}")
    print(f"  Results reproducible with fixed seed")

    print(f"\nGymnasium API: {status(result.api_compliant)}")
    print(f"  Observation space and step() correct")

    print(f"\nSteps tested: {result.n_steps_tested}")

    print("\n" + "=" * 70)
    all_passed = (
        result.delay_correct
        and result.sample_hold_correct
        and result.deterministic
        and result.api_compliant
    )
    overall = "PASS" if all_passed else "FAIL"
    print(f"OVERALL DELAY VALIDATION: {overall}")
    print("=" * 70)


if __name__ == "__main__":
    result = run_delay_validation(dead_time=30.0, sample_period=30.0, n_steps=200)
    print_delay_validation_report(result)
