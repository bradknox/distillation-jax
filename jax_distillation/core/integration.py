"""Numerical integration methods for distillation simulation.

This module provides fixed-step integrators compatible with JAX:
- RK4 (4th-order Runge-Kutta) - default
- Euler (1st-order) - for debugging

All integrators are pure functions compatible with JIT compilation.
"""

import jax
import jax.numpy as jnp
from typing import Callable, TypeVar

State = TypeVar("State")


def rk4_step(
    f: Callable[[State, float], State],
    state: State,
    t: float,
    dt: float,
) -> State:
    """Single RK4 integration step.

    Implements the classic 4th-order Runge-Kutta method:
    k1 = f(y, t)
    k2 = f(y + dt/2 * k1, t + dt/2)
    k3 = f(y + dt/2 * k2, t + dt/2)
    k4 = f(y + dt * k3, t + dt)
    y_new = y + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

    Args:
        f: Derivative function f(state, t) -> d(state)/dt.
        state: Current state (pytree).
        t: Current time.
        dt: Time step.

    Returns:
        Updated state after one step.
    """
    k1 = f(state, t)
    k2 = f(
        jax.tree.map(lambda y, dy: y + 0.5 * dt * dy, state, k1),
        t + 0.5 * dt,
    )
    k3 = f(
        jax.tree.map(lambda y, dy: y + 0.5 * dt * dy, state, k2),
        t + 0.5 * dt,
    )
    k4 = f(
        jax.tree.map(lambda y, dy: y + dt * dy, state, k3),
        t + dt,
    )

    # Combine: y_new = y + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    new_state = jax.tree.map(
        lambda y, dy1, dy2, dy3, dy4: y + (dt / 6.0) * (dy1 + 2.0 * dy2 + 2.0 * dy3 + dy4),
        state,
        k1,
        k2,
        k3,
        k4,
    )

    return new_state


def euler_step(
    f: Callable[[State, float], State],
    state: State,
    t: float,
    dt: float,
) -> State:
    """Single Euler integration step.

    Simple first-order forward Euler:
    y_new = y + dt * f(y, t)

    Args:
        f: Derivative function f(state, t) -> d(state)/dt.
        state: Current state (pytree).
        t: Current time.
        dt: Time step.

    Returns:
        Updated state after one step.
    """
    deriv = f(state, t)
    new_state = jax.tree.map(lambda y, dy: y + dt * dy, state, deriv)
    return new_state


def integrate(
    f: Callable[[State, float], State],
    state: State,
    t0: float,
    t1: float,
    n_steps: int,
    method: str = "rk4",
) -> State:
    """Integrate from t0 to t1 using n_steps.

    Args:
        f: Derivative function f(state, t) -> d(state)/dt.
        state: Initial state (pytree).
        t0: Start time.
        t1: End time.
        n_steps: Number of integration steps.
        method: Integration method ("rk4" or "euler").

    Returns:
        Final state at t1.
    """
    dt = (t1 - t0) / n_steps

    if method == "euler":
        step_fn = euler_step
    else:
        step_fn = rk4_step

    def body_fn(carry, _):
        state, t = carry
        new_state = step_fn(f, state, t, dt)
        return (new_state, t + dt), None

    (final_state, _), _ = jax.lax.scan(body_fn, (state, t0), None, length=n_steps)

    return final_state


def integrate_with_trajectory(
    f: Callable[[State, float], State],
    state: State,
    t0: float,
    t1: float,
    n_steps: int,
    method: str = "rk4",
) -> tuple[State, State]:
    """Integrate and return full trajectory.

    Args:
        f: Derivative function f(state, t) -> d(state)/dt.
        state: Initial state (pytree).
        t0: Start time.
        t1: End time.
        n_steps: Number of integration steps.
        method: Integration method ("rk4" or "euler").

    Returns:
        Tuple of (final_state, trajectory) where trajectory is a pytree
        with an extra leading dimension of size n_steps+1.
    """
    dt = (t1 - t0) / n_steps

    if method == "euler":
        step_fn = euler_step
    else:
        step_fn = rk4_step

    def body_fn(carry, _):
        state, t = carry
        new_state = step_fn(f, state, t, dt)
        return (new_state, t + dt), new_state

    (final_state, _), trajectory = jax.lax.scan(body_fn, (state, t0), None, length=n_steps)

    # Prepend initial state to trajectory
    trajectory = jax.tree.map(
        lambda init, traj: jnp.concatenate([init[None, ...], traj], axis=0),
        state,
        trajectory,
    )

    return final_state, trajectory


def adaptive_substeps(
    f: Callable[[State, float], State],
    state: State,
    t: float,
    dt: float,
    tau_min: float,
    method: str = "rk4",
) -> State:
    """Integrate with automatic substep selection based on time constant.

    Chooses n_substeps such that dt_sub < tau_min / 10 for numerical stability.

    Args:
        f: Derivative function.
        state: Current state.
        t: Current time.
        dt: Total time step.
        tau_min: Minimum time constant in the system.
        method: Integration method.

    Returns:
        State at t + dt.
    """
    # Choose substeps for stability: dt_sub < tau_min / 10
    dt_sub_target = tau_min / 10.0
    n_substeps = jnp.maximum(1, jnp.ceil(dt / dt_sub_target).astype(int))

    return integrate(f, state, t, t + dt, int(n_substeps), method=method)


# =============================================================================
# Specialized integrators for column simulation
# =============================================================================


def integrate_column_state(
    derivative_fn: Callable,
    state_arrays: tuple[jnp.ndarray, ...],
    dt: float,
    n_substeps: int,
) -> tuple[jnp.ndarray, ...]:
    """Integrate column state arrays using RK4.

    Specialized integrator for column simulation where state is
    a tuple of arrays (M, x, T) for each tray.

    Args:
        derivative_fn: Function returning (dM/dt, dx/dt, dT/dt) arrays.
        state_arrays: Tuple of (M_array, x_array, T_array).
        dt: Total time step.
        n_substeps: Number of substeps.

    Returns:
        Updated state arrays.
    """
    dt_sub = dt / n_substeps

    def rk4_arrays(arrays, _):
        M, x, T = arrays

        # k1
        dM1, dx1, dT1 = derivative_fn(M, x, T)

        # k2
        dM2, dx2, dT2 = derivative_fn(
            M + 0.5 * dt_sub * dM1,
            x + 0.5 * dt_sub * dx1,
            T + 0.5 * dt_sub * dT1,
        )

        # k3
        dM3, dx3, dT3 = derivative_fn(
            M + 0.5 * dt_sub * dM2,
            x + 0.5 * dt_sub * dx2,
            T + 0.5 * dt_sub * dT2,
        )

        # k4
        dM4, dx4, dT4 = derivative_fn(
            M + dt_sub * dM3,
            x + dt_sub * dx3,
            T + dt_sub * dT3,
        )

        # Update
        M_new = M + (dt_sub / 6.0) * (dM1 + 2.0 * dM2 + 2.0 * dM3 + dM4)
        x_new = x + (dt_sub / 6.0) * (dx1 + 2.0 * dx2 + 2.0 * dx3 + dx4)
        T_new = T + (dt_sub / 6.0) * (dT1 + 2.0 * dT2 + 2.0 * dT3 + dT4)

        return (M_new, x_new, T_new), None

    (M_final, x_final, T_final), _ = jax.lax.scan(
        rk4_arrays, state_arrays, None, length=n_substeps
    )

    return M_final, x_final, T_final
