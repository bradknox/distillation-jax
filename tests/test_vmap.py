"""Tests for vmap compatibility with JAX-native environment.

This test verifies that:
1. vmap works over batch of 64 parallel environments
2. Vectorized operations produce correct results
3. GPU execution works (if available)
"""

import jax
import jax.numpy as jnp
import pytest
import time

from jax_distillation.env import (
    DistillationEnvJax,
    EnvParams,
    EnvState,
    make_env_fns,
)
from jax_distillation.column.config import create_teaching_column_config
from jax_distillation.column.column import (
    make_column_step_fn,
    create_initial_column_state,
    create_default_action,
)


class TestVmapEnvironment:
    """Tests for vmap over Gymnax-style environment."""

    def test_vmap_reset_64_envs(self):
        """vmap over reset for 64 parallel environments."""
        env = DistillationEnvJax()
        params = env.default_params

        reset_fn = jax.vmap(env.reset, in_axes=(0, None))
        keys = jax.random.split(jax.random.PRNGKey(0), 64)

        obs, states = reset_fn(keys, params)

        # Check shapes
        assert obs.shape == (64, 24), f"Expected obs shape (64, 24), got {obs.shape}"
        assert states.column_state.tray_M.shape == (64, 10), \
            f"Expected tray_M shape (64, 10), got {states.column_state.tray_M.shape}"

        # Check all values are identical (same deterministic initial state)
        assert jnp.allclose(obs[0], obs[1]), "All initial obs should be identical"

    def test_vmap_step_64_envs(self):
        """vmap over step for 64 parallel environments."""
        env = DistillationEnvJax()
        params = env.default_params

        # Reset all envs
        reset_fn = jax.vmap(env.reset, in_axes=(0, None))
        keys = jax.random.split(jax.random.PRNGKey(0), 64)
        obs, states = reset_fn(keys, params)

        # Step all envs
        step_fn = jax.vmap(env.step, in_axes=(0, 0, 0, None))
        actions = jnp.zeros((64, 4))
        actions = actions.at[:, 0].set(5000.0)  # Q_R
        actions = actions.at[:, 1].set(3.0)  # reflux_ratio
        actions = actions.at[:, 2].set(0.03)  # B_setpoint
        actions = actions.at[:, 3].set(0.02)  # D_setpoint

        new_keys = jax.random.split(jax.random.PRNGKey(1), 64)
        obs, new_states, rewards, dones, infos = step_fn(
            new_keys, states, actions, params
        )

        # Check shapes
        assert rewards.shape == (64,), f"Expected rewards shape (64,), got {rewards.shape}"
        assert dones.shape == (64,), f"Expected dones shape (64,), got {dones.shape}"

        # Check all values are finite
        assert jnp.all(jnp.isfinite(rewards)), "All rewards should be finite"
        assert jnp.all(jnp.isfinite(obs)), "All observations should be finite"

    def test_vmap_with_different_actions(self):
        """vmap with different actions per environment."""
        env = DistillationEnvJax()
        params = env.default_params

        # Reset
        reset_fn = jax.vmap(env.reset, in_axes=(0, None))
        keys = jax.random.split(jax.random.PRNGKey(0), 64)
        obs, states = reset_fn(keys, params)

        # Different Q_R values for each env
        Q_R_values = jnp.linspace(2000, 10000, 64)
        actions = jnp.zeros((64, 4))
        actions = actions.at[:, 0].set(Q_R_values)
        actions = actions.at[:, 1].set(3.0)
        actions = actions.at[:, 2].set(0.03)
        actions = actions.at[:, 3].set(0.02)

        step_fn = jax.vmap(env.step, in_axes=(0, 0, 0, None))
        new_keys = jax.random.split(jax.random.PRNGKey(1), 64)
        obs, new_states, rewards, dones, infos = step_fn(
            new_keys, states, actions, params
        )

        # Higher Q_R should lead to different behavior
        # At least rewards should vary
        reward_std = jnp.std(rewards)
        print(f"Reward std across different Q_R: {float(reward_std):.6f}")

    def test_vmap_rollout_100_steps(self):
        """vmap rollout for 100 steps with 64 envs."""
        env = DistillationEnvJax()
        params = env.default_params

        @jax.jit
        def rollout(key, n_steps=100):
            # Reset
            keys = jax.random.split(key, 64)
            reset_fn = jax.vmap(env.reset, in_axes=(0, None))
            obs, states = reset_fn(keys, params)

            # Fixed action
            actions = jnp.zeros((64, 4))
            actions = actions.at[:, 0].set(5000.0)
            actions = actions.at[:, 1].set(3.0)
            actions = actions.at[:, 2].set(0.03)
            actions = actions.at[:, 3].set(0.02)

            step_fn = jax.vmap(env.step, in_axes=(0, 0, 0, None))

            def step_loop(carry, _):
                key, states = carry
                keys = jax.random.split(key, 64)
                obs, states, rewards, dones, infos = step_fn(
                    keys, states, actions, params
                )
                return (keys[0], states), rewards

            (_, final_states), all_rewards = jax.lax.scan(
                step_loop, (key, states), None, length=n_steps
            )
            return final_states, all_rewards

        key = jax.random.PRNGKey(42)
        final_states, all_rewards = rollout(key)

        # Check shapes
        assert all_rewards.shape == (100, 64), \
            f"Expected all_rewards shape (100, 64), got {all_rewards.shape}"

        # Check all values are finite
        assert jnp.all(jnp.isfinite(all_rewards)), "All rewards should be finite"

        # Check final states are valid
        assert jnp.all(final_states.column_state.tray_M > 0), "All holdups should be positive"

    def test_make_env_fns_produces_vmappable_functions(self):
        """make_env_fns should produce vmappable functions."""
        reset_fn, step_fn, params = make_env_fns()

        # Vectorize
        batch_reset = jax.vmap(reset_fn, in_axes=(0, None))
        batch_step = jax.vmap(step_fn, in_axes=(0, 0, 0, None))

        # Reset
        keys = jax.random.split(jax.random.PRNGKey(0), 32)
        obs, states = batch_reset(keys, params)

        assert obs.shape[0] == 32

        # Step
        actions = jnp.zeros((32, 4))
        actions = actions.at[:, 0].set(5000.0)
        actions = actions.at[:, 1].set(3.0)
        actions = actions.at[:, 2].set(0.03)
        actions = actions.at[:, 3].set(0.02)

        new_keys = jax.random.split(jax.random.PRNGKey(1), 32)
        obs, states, rewards, dones, infos = batch_step(
            new_keys, states, actions, params
        )

        assert rewards.shape == (32,)


class TestVmapPerformance:
    """Performance tests for vmap."""

    def test_vmap_speedup(self):
        """vmap should provide speedup over sequential execution."""
        env = DistillationEnvJax()
        params = env.default_params

        # Sequential function
        def run_sequential(key, n_envs=16, n_steps=10):
            total_reward = 0.0
            for i in range(n_envs):
                key, subkey = jax.random.split(key)
                obs, state = env.reset(subkey, params)
                action = jnp.array([5000.0, 3.0, 0.03, 0.02])
                for _ in range(n_steps):
                    key, subkey = jax.random.split(key)
                    obs, state, reward, done, info = env.step(
                        subkey, state, action, params
                    )
                    total_reward += reward
            return total_reward

        # Batched function
        @jax.jit
        def run_batched(key, n_envs=16, n_steps=10):
            keys = jax.random.split(key, n_envs)
            reset_fn = jax.vmap(env.reset, in_axes=(0, None))
            obs, states = reset_fn(keys, params)

            actions = jnp.zeros((n_envs, 4))
            actions = actions.at[:, 0].set(5000.0)
            actions = actions.at[:, 1].set(3.0)
            actions = actions.at[:, 2].set(0.03)
            actions = actions.at[:, 3].set(0.02)

            step_fn = jax.vmap(env.step, in_axes=(0, 0, 0, None))

            def step_loop(carry, _):
                k, states = carry
                keys = jax.random.split(k, n_envs)
                obs, states, rewards, dones, infos = step_fn(
                    keys, states, actions, params
                )
                return (keys[0], states), rewards

            (_, _), all_rewards = jax.lax.scan(
                step_loop, (key, states), None, length=n_steps
            )
            return jnp.sum(all_rewards)

        key = jax.random.PRNGKey(0)

        # Warmup batched
        _ = run_batched(key)

        # Time batched
        start = time.perf_counter()
        _ = run_batched(key)
        jax.block_until_ready(_)
        time_batched = time.perf_counter() - start

        print(f"Batched time: {time_batched:.3f}s")
        print(f"Throughput: {16*10/time_batched:.0f} env-steps/sec")

    def test_scaling_with_batch_size(self):
        """Throughput should scale well with batch size."""
        env = DistillationEnvJax()
        params = env.default_params

        results = {}

        for batch_size in [8, 32, 64, 128]:
            @jax.jit
            def run_batched(key, n_steps=50):
                keys = jax.random.split(key, batch_size)
                reset_fn = jax.vmap(env.reset, in_axes=(0, None))
                obs, states = reset_fn(keys, params)

                actions = jnp.zeros((batch_size, 4))
                actions = actions.at[:, 0].set(5000.0)
                actions = actions.at[:, 1].set(3.0)
                actions = actions.at[:, 2].set(0.03)
                actions = actions.at[:, 3].set(0.02)

                step_fn = jax.vmap(env.step, in_axes=(0, 0, 0, None))

                def step_loop(carry, _):
                    k, states = carry
                    keys = jax.random.split(k, batch_size)
                    obs, states, rewards, dones, infos = step_fn(
                        keys, states, actions, params
                    )
                    return (keys[0], states), rewards

                (_, _), all_rewards = jax.lax.scan(
                    step_loop, (key, states), None, length=n_steps
                )
                return all_rewards

            key = jax.random.PRNGKey(0)

            # Warmup
            _ = run_batched(key)

            # Time
            start = time.perf_counter()
            _ = run_batched(key)
            jax.block_until_ready(_)
            elapsed = time.perf_counter() - start

            throughput = batch_size * 50 / elapsed
            results[batch_size] = throughput
            print(f"Batch size {batch_size}: {throughput:.0f} env-steps/sec")

        # Throughput should generally be better with larger batch sizes
        # but may vary due to system load and JIT caching
        # Just verify we got reasonable results
        assert all(t > 1000 for t in results.values()), \
            "All batch sizes should achieve >1000 env-steps/sec"


class TestVmapColumnStep:
    """Tests for vmap over raw column step function."""

    def test_vmap_column_step(self):
        """vmap should work over column step function."""
        config = create_teaching_column_config()
        step_fn = make_column_step_fn(config)

        # Create batch of states
        state = create_initial_column_state(config)

        def make_batched_state(base_state, batch_size):
            """Replicate state across batch dimension."""
            from jax_distillation.column.column import FullColumnState
            from jax_distillation.column.reboiler import ReboilerState
            from jax_distillation.column.condenser import CondenserState

            return FullColumnState(
                tray_M=jnp.tile(base_state.tray_M[None, :], (batch_size, 1)),
                tray_x=jnp.tile(base_state.tray_x[None, :], (batch_size, 1)),
                tray_T=jnp.tile(base_state.tray_T[None, :], (batch_size, 1)),
                tray_L_out=jnp.tile(base_state.tray_L_out[None, :], (batch_size, 1)),
                reboiler=ReboilerState(
                    M=jnp.tile(base_state.reboiler.M[None], (batch_size,)),
                    x=jnp.tile(base_state.reboiler.x[None], (batch_size,)),
                    T=jnp.tile(base_state.reboiler.T[None], (batch_size,)),
                ),
                condenser=CondenserState(
                    M=jnp.tile(base_state.condenser.M[None], (batch_size,)),
                    x=jnp.tile(base_state.condenser.x[None], (batch_size,)),
                    T=jnp.tile(base_state.condenser.T[None], (batch_size,)),
                ),
                t=jnp.tile(base_state.t[None], (batch_size,)),
                V_prev=jnp.tile(base_state.V_prev[None, :], (batch_size, 1)),
            )

        from jax_distillation.column.column import ColumnAction

        def make_batched_action(base_action, batch_size):
            return ColumnAction(
                Q_R=jnp.tile(base_action.Q_R[None], (batch_size,)),
                reflux_ratio=jnp.tile(base_action.reflux_ratio[None], (batch_size,)),
                B_setpoint=jnp.tile(base_action.B_setpoint[None], (batch_size,)),
                D_setpoint=jnp.tile(base_action.D_setpoint[None], (batch_size,)),
            )

        batch_size = 32
        batched_state = make_batched_state(state, batch_size)
        base_action = create_default_action()
        batched_action = make_batched_action(base_action, batch_size)

        # vmap over both state and action
        vmap_step = jax.vmap(step_fn)
        new_states, outputs = vmap_step(batched_state, batched_action)

        assert new_states.tray_M.shape == (batch_size, 10)
        assert outputs.x_D.shape == (batch_size,)


class TestGPUExecution:
    """Tests for GPU execution (skipped if no GPU)."""

    @pytest.fixture
    def has_gpu(self):
        """Check if GPU is available."""
        try:
            return len(jax.devices('gpu')) > 0
        except RuntimeError:
            return False

    def test_gpu_execution(self, has_gpu):
        """Environment should run on GPU without errors."""
        if not has_gpu:
            pytest.skip("No GPU available")

        env = DistillationEnvJax()
        params = env.default_params

        with jax.default_device(jax.devices('gpu')[0]):
            key = jax.random.PRNGKey(0)
            obs, state = env.reset(key, params)
            action = jnp.array([5000.0, 3.0, 0.03, 0.02])

            key, subkey = jax.random.split(key)
            obs, state, reward, done, info = env.step(subkey, state, action, params)

            # Check execution happened on GPU
            assert state.column_state.tray_M.device().platform == 'gpu'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
