"""Tests for JIT compilation of column step function.

This test verifies that make_column_step_fn produces a function that:
1. JIT compiles without error
2. Works with jax.lax.scan
3. Produces correct results
"""

import jax
import jax.numpy as jnp
import pytest

from jax_distillation.column.config import create_teaching_column_config
from jax_distillation.column.column import (
    column_step,
    make_column_step_fn,
    create_initial_column_state,
    create_default_action,
    FullColumnState,
)


class TestJITCompilation:
    """Tests for JIT compilation."""

    def test_make_column_step_fn_returns_callable(self):
        """make_column_step_fn should return a callable function."""
        config = create_teaching_column_config()
        step_fn = make_column_step_fn(config)

        assert callable(step_fn), "step_fn should be callable"

    def test_jit_compiles_without_error(self):
        """Step function should JIT compile without error."""
        config = create_teaching_column_config()
        step_fn = make_column_step_fn(config)

        # This should not raise ConcretizationTypeError
        jit_step = jax.jit(step_fn)

        # Run to trigger compilation
        state = create_initial_column_state(config)
        action = create_default_action()
        new_state, outputs = jit_step(state, action)

        assert new_state is not None
        assert outputs is not None

    def test_jit_matches_eager_execution(self):
        """JIT-compiled results should match eager execution."""
        config = create_teaching_column_config()
        step_fn = make_column_step_fn(config)
        jit_step = jax.jit(step_fn)

        state = create_initial_column_state(config)
        action = create_default_action()

        # Eager execution
        new_state_eager, outputs_eager = step_fn(state, action)

        # JIT execution
        new_state_jit, outputs_jit = jit_step(state, action)

        # Compare results
        assert jnp.allclose(new_state_eager.tray_M, new_state_jit.tray_M, rtol=1e-5)
        assert jnp.allclose(new_state_eager.tray_x, new_state_jit.tray_x, rtol=1e-5)
        assert jnp.allclose(new_state_eager.tray_T, new_state_jit.tray_T, rtol=1e-5)
        assert jnp.allclose(outputs_eager.x_D, outputs_jit.x_D, rtol=1e-5)
        assert jnp.allclose(outputs_eager.x_B, outputs_jit.x_B, rtol=1e-5)

    def test_scan_with_jit(self):
        """lax.scan should work inside JIT with step function."""
        config = create_teaching_column_config()
        step_fn = make_column_step_fn(config)

        # Use static_argnums for n_steps since scan length must be static
        @jax.jit
        def run_100_steps(state, action):
            def step(s, _):
                new_s, out = step_fn(s, action)
                return new_s, out

            final_state, outputs = jax.lax.scan(step, state, None, length=100)
            return final_state, outputs

        state = create_initial_column_state(config)
        action = create_default_action()

        # This should not raise ConcretizationTypeError
        final_state, outputs = run_100_steps(state, action)

        assert final_state is not None
        assert outputs.x_D.shape == (100,), "Should have 100 output values"
        assert jnp.all(jnp.isfinite(outputs.x_D)), "All outputs should be finite"

    def test_scan_produces_valid_trajectory(self):
        """Scan should produce physically valid trajectory."""
        config = create_teaching_column_config()
        step_fn = make_column_step_fn(config)

        @jax.jit
        def simulate_50_steps(state, action):
            def step(s, _):
                new_s, out = step_fn(s, action)
                return new_s, (new_s, out)

            _, (states, outputs) = jax.lax.scan(step, state, None, length=50)
            return states, outputs

        state = create_initial_column_state(config)
        action = create_default_action(Q_R=5000.0)

        states, outputs = simulate_50_steps(state, action)

        # Check all states are valid
        assert jnp.all(jnp.isfinite(states.tray_M)), "All tray holdups should be finite"
        assert jnp.all(states.tray_M > 0), "All tray holdups should be positive"
        assert jnp.all((states.tray_x >= 0) & (states.tray_x <= 1)), \
            "All compositions should be in [0, 1]"

    def test_different_configs_produce_different_step_fns(self):
        """Different configs should produce independent step functions."""
        config1 = create_teaching_column_config(n_trays=8)
        config2 = create_teaching_column_config(n_trays=12)

        step_fn1 = make_column_step_fn(config1)
        step_fn2 = make_column_step_fn(config2)

        state1 = create_initial_column_state(config1)
        state2 = create_initial_column_state(config2)
        action = create_default_action()

        new_state1, outputs1 = step_fn1(state1, action)
        new_state2, outputs2 = step_fn2(state2, action)

        # Different number of trays
        assert new_state1.tray_M.shape != new_state2.tray_M.shape

    def test_jit_second_call_uses_cache(self):
        """Second JIT call should be faster (uses cached compilation)."""
        import time

        config = create_teaching_column_config()
        step_fn = make_column_step_fn(config)
        jit_step = jax.jit(step_fn)

        state = create_initial_column_state(config)
        action = create_default_action()

        # First call (compiles)
        start1 = time.perf_counter()
        _ = jit_step(state, action)
        jax.block_until_ready(_[0].tray_M)
        time1 = time.perf_counter() - start1

        # Second call (uses cache)
        start2 = time.perf_counter()
        _ = jit_step(state, action)
        jax.block_until_ready(_[0].tray_M)
        time2 = time.perf_counter() - start2

        print(f"First call: {time1:.3f}s")
        print(f"Second call: {time2:.3f}s")
        print(f"Speedup: {time1/time2:.1f}x")

        # Second call should be significantly faster
        assert time2 < time1, "Second call should be faster (cached)"


class TestScanStability:
    """Tests for stability with lax.scan."""

    def test_long_simulation_stable(self):
        """Long simulation should remain numerically stable."""
        config = create_teaching_column_config()
        step_fn = make_column_step_fn(config)

        @jax.jit
        def run_long(state, action):
            def step(s, _):
                new_s, out = step_fn(s, action)
                return new_s, out

            return jax.lax.scan(step, state, None, length=1000)

        state = create_initial_column_state(config)
        action = create_default_action(Q_R=5000.0)

        final_state, outputs = run_long(state, action)

        # Check no NaN/Inf
        assert jnp.all(jnp.isfinite(final_state.tray_M)), "No NaN/Inf in holdups"
        assert jnp.all(jnp.isfinite(final_state.tray_x)), "No NaN/Inf in compositions"
        assert jnp.all(jnp.isfinite(outputs.x_D)), "No NaN/Inf in x_D"
        assert jnp.all(jnp.isfinite(outputs.x_B)), "No NaN/Inf in x_B"

    def test_varied_actions_stable(self):
        """Simulation with varied actions should remain stable."""
        config = create_teaching_column_config()
        step_fn = make_column_step_fn(config)

        # Create sequence of actions
        n_steps = 200
        Q_R_values = jnp.linspace(3000, 8000, n_steps)

        def step_with_varying_action(carry, Q_R):
            state = carry
            action = create_default_action(Q_R=Q_R)
            new_state, outputs = step_fn(state, action)
            return new_state, outputs

        state = create_initial_column_state(config)

        @jax.jit
        def run_varied(state):
            return jax.lax.scan(step_with_varying_action, state, Q_R_values)

        final_state, outputs = run_varied(state)

        # Check stability
        assert jnp.all(jnp.isfinite(final_state.tray_M))
        assert jnp.all(final_state.tray_M > 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
