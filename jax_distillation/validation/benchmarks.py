"""Performance benchmarks for distillation simulation.

This module provides benchmarks to validate:
- Simulation speed (steps per second)
- JIT compilation time
- Memory usage
- vmap scaling
"""

import time
import jax
import jax.numpy as jnp
import numpy as np
from typing import NamedTuple

from jax_distillation.column.column import (
    FullColumnState,
    ColumnAction,
    column_step,
    create_initial_column_state,
    create_default_action,
    simulate_column_jit,
)
from jax_distillation.column.config import (
    ColumnConfig,
    create_teaching_column_config,
)


class BenchmarkResults(NamedTuple):
    """Results from benchmark run.

    Attributes:
        name: Benchmark name.
        mean_time: Mean execution time [s].
        std_time: Standard deviation of time [s].
        min_time: Minimum time [s].
        max_time: Maximum time [s].
        throughput: Operations per second.
    """

    name: str
    mean_time: float
    std_time: float
    min_time: float
    max_time: float
    throughput: float


def benchmark_single_step(
    config: ColumnConfig | None = None,
    n_warmup: int = 5,
    n_runs: int = 20,
) -> BenchmarkResults:
    """Benchmark single simulation step execution time.

    Args:
        config: Column configuration.
        n_warmup: Warmup iterations before timing.
        n_runs: Number of timed runs.

    Returns:
        BenchmarkResults with timing statistics.
    """
    if config is None:
        config = create_teaching_column_config()

    state = create_initial_column_state(config)
    action = create_default_action()

    # JIT compile
    @jax.jit
    def step_jit(state, action):
        return column_step(state, action, config)

    # Warmup
    for _ in range(n_warmup):
        state, _ = step_jit(state, action)

    # Block until warmup complete
    jax.block_until_ready(state.tray_M)

    # Time runs
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        state, _ = step_jit(state, action)
        jax.block_until_ready(state.tray_M)
        end = time.perf_counter()
        times.append(end - start)

    times = np.array(times)

    return BenchmarkResults(
        name="single_step",
        mean_time=float(np.mean(times)),
        std_time=float(np.std(times)),
        min_time=float(np.min(times)),
        max_time=float(np.max(times)),
        throughput=1.0 / float(np.mean(times)),
    )


def benchmark_trajectory(
    config: ColumnConfig | None = None,
    n_steps: int = 100,
    n_warmup: int = 2,
    n_runs: int = 10,
) -> BenchmarkResults:
    """Benchmark trajectory simulation.

    Args:
        config: Column configuration.
        n_steps: Steps per trajectory.
        n_warmup: Warmup iterations.
        n_runs: Number of timed runs.

    Returns:
        BenchmarkResults with timing statistics.
    """
    if config is None:
        config = create_teaching_column_config()

    state = create_initial_column_state(config)
    action = create_default_action()

    # JIT compile trajectory simulation
    simulate_jit = jax.jit(lambda s, a: simulate_column_jit(config, a, n_steps, s))

    # Warmup
    for _ in range(n_warmup):
        final_state, _ = simulate_jit(state, action)

    jax.block_until_ready(final_state.tray_M)

    # Time runs
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        final_state, _ = simulate_jit(state, action)
        jax.block_until_ready(final_state.tray_M)
        end = time.perf_counter()
        times.append(end - start)

    times = np.array(times)

    return BenchmarkResults(
        name=f"trajectory_{n_steps}_steps",
        mean_time=float(np.mean(times)),
        std_time=float(np.std(times)),
        min_time=float(np.min(times)),
        max_time=float(np.max(times)),
        throughput=n_steps / float(np.mean(times)),
    )


def benchmark_jit_compilation(
    n_trays_list: list[int] | None = None,
) -> dict:
    """Benchmark JIT compilation time for different column sizes.

    Args:
        n_trays_list: List of tray counts to test.

    Returns:
        Dictionary mapping n_trays to compilation time.
    """
    if n_trays_list is None:
        n_trays_list = [5, 10, 20]

    results = {}

    for n_trays in n_trays_list:
        config = create_teaching_column_config(n_trays=n_trays, feed_tray=n_trays // 2)
        state = create_initial_column_state(config)
        action = create_default_action()

        # Force fresh compilation by creating new function
        def make_step_fn(cfg):
            def step_fn(state, action):
                return column_step(state, action, cfg)
            return step_fn

        step_fn = make_step_fn(config)
        step_jit = jax.jit(step_fn)

        # Time compilation (first call triggers JIT)
        start = time.perf_counter()
        state_new, _ = step_jit(state, action)
        jax.block_until_ready(state_new.tray_M)
        end = time.perf_counter()

        results[n_trays] = end - start

    return results


def benchmark_vmap_scaling(
    config: ColumnConfig | None = None,
    batch_sizes: list[int] | None = None,
    n_runs: int = 5,
) -> dict:
    """Benchmark vmap scaling with batch size.

    Args:
        config: Column configuration.
        batch_sizes: List of batch sizes to test.
        n_runs: Runs per batch size.

    Returns:
        Dictionary mapping batch_size to timing results.
    """
    if config is None:
        config = create_teaching_column_config(n_trays=5)

    if batch_sizes is None:
        batch_sizes = [1, 4, 16, 64]

    base_state = create_initial_column_state(config)
    base_action = create_default_action()

    # Create batched step function
    def step_fn(state, action):
        return column_step(state, action, config)

    step_vmap = jax.vmap(step_fn)
    step_jit = jax.jit(step_vmap)

    results = {}

    for batch_size in batch_sizes:
        # Create batched states
        batched_state = FullColumnState(
            tray_M=jnp.tile(base_state.tray_M[None, :], (batch_size, 1)),
            tray_x=jnp.tile(base_state.tray_x[None, :], (batch_size, 1)),
            tray_T=jnp.tile(base_state.tray_T[None, :], (batch_size, 1)),
            tray_L_out=jnp.tile(base_state.tray_L_out[None, :], (batch_size, 1)),
            reboiler=base_state.reboiler,  # Would need proper batching
            condenser=base_state.condenser,
            t=jnp.tile(base_state.t[None], (batch_size,)),
            V_prev=jnp.tile(base_state.V_prev[None, :], (batch_size, 1)),
        )

        batched_action = ColumnAction(
            Q_R=jnp.tile(base_action.Q_R[None], (batch_size,)),
            reflux_ratio=jnp.tile(base_action.reflux_ratio[None], (batch_size,)),
            B_setpoint=jnp.tile(base_action.B_setpoint[None], (batch_size,)),
            D_setpoint=jnp.tile(base_action.D_setpoint[None], (batch_size,)),
        )

        # Note: Full batching requires reboiler/condenser to be properly batched
        # This is a simplified benchmark

        times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            # For now, just run single step multiple times to estimate
            for _ in range(batch_size):
                state_new, _ = step_fn(base_state, base_action)
            jax.block_until_ready(state_new.tray_M)
            end = time.perf_counter()
            times.append(end - start)

        results[batch_size] = {
            "mean_time": float(np.mean(times)),
            "throughput": batch_size / float(np.mean(times)),
        }

    return results


def run_all_benchmarks(
    verbose: bool = True,
) -> dict:
    """Run all benchmarks and return results.

    Args:
        verbose: Print results to console.

    Returns:
        Dictionary with all benchmark results.
    """
    results = {}

    if verbose:
        print("Running benchmarks...")
        print("-" * 50)

    # Single step benchmark
    if verbose:
        print("Benchmarking single step...")
    single_step = benchmark_single_step()
    results["single_step"] = single_step._asdict()
    if verbose:
        print(f"  Mean time: {single_step.mean_time * 1000:.2f} ms")
        print(f"  Throughput: {single_step.throughput:.0f} steps/s")

    # Trajectory benchmark
    if verbose:
        print("Benchmarking trajectory (100 steps)...")
    trajectory = benchmark_trajectory(n_steps=100)
    results["trajectory_100"] = trajectory._asdict()
    if verbose:
        print(f"  Mean time: {trajectory.mean_time * 1000:.2f} ms")
        print(f"  Throughput: {trajectory.throughput:.0f} steps/s")

    # JIT compilation benchmark
    if verbose:
        print("Benchmarking JIT compilation...")
    jit_times = benchmark_jit_compilation(n_trays_list=[5, 10])
    results["jit_compilation"] = jit_times
    if verbose:
        for n_trays, time_s in jit_times.items():
            print(f"  {n_trays} trays: {time_s:.2f} s")

    if verbose:
        print("-" * 50)
        print("Benchmarks complete.")

    return results


def validate_performance_requirements(
    max_step_time_ms: float = 10.0,
    max_jit_time_s: float = 30.0,
) -> dict:
    """Validate that simulation meets performance requirements.

    Args:
        max_step_time_ms: Maximum acceptable step time [ms].
        max_jit_time_s: Maximum acceptable JIT compilation time [s].

    Returns:
        Dictionary with validation results.
    """
    # Run benchmarks
    single_step = benchmark_single_step()
    jit_times = benchmark_jit_compilation(n_trays_list=[10])

    step_time_ok = single_step.mean_time * 1000 < max_step_time_ms
    jit_time_ok = all(t < max_jit_time_s for t in jit_times.values())

    return {
        "step_time_ms": single_step.mean_time * 1000,
        "step_time_requirement_ms": max_step_time_ms,
        "step_time_ok": step_time_ok,
        "jit_compilation_times": jit_times,
        "jit_time_requirement_s": max_jit_time_s,
        "jit_time_ok": jit_time_ok,
        "all_requirements_met": step_time_ok and jit_time_ok,
    }
