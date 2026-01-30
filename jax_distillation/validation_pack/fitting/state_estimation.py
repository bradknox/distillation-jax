"""State estimation for distillation columns.

This module provides state estimators for estimating unmeasured
states from available measurements, accounting for measurement
delays and noise.

Estimators:
- Extended Kalman Filter (EKF)
- Moving Horizon Estimator (MHE) - placeholder
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np


@dataclass
class EKFState:
    """State of Extended Kalman Filter.

    Attributes:
        x: State estimate [n_states,]
        P: State covariance [n_states, n_states]
        t: Current time
    """

    x: np.ndarray
    P: np.ndarray
    t: float


class ExtendedKalmanFilter:
    """Extended Kalman Filter for nonlinear state estimation.

    Implements the discrete-time EKF for estimating column states
    from noisy measurements.
    """

    def __init__(
        self,
        f: Callable,  # State transition function
        h: Callable,  # Measurement function
        Q: np.ndarray,  # Process noise covariance
        R: np.ndarray,  # Measurement noise covariance
        n_states: int,
        n_measurements: int,
    ):
        """Initialize EKF.

        Args:
            f: State transition function f(x, u) -> x_next
            h: Measurement function h(x) -> y
            Q: Process noise covariance [n_states, n_states]
            R: Measurement noise covariance [n_meas, n_meas]
            n_states: Number of state variables
            n_measurements: Number of measurements
        """
        self.f = f
        self.h = h
        self.Q = Q
        self.R = R
        self.n_states = n_states
        self.n_meas = n_measurements

        # For numerical Jacobian computation
        self._eps = 1e-6

    def _numerical_jacobian(
        self,
        func: Callable,
        x: np.ndarray,
        *args,
    ) -> np.ndarray:
        """Compute Jacobian numerically using finite differences.

        Args:
            func: Function to differentiate
            x: Point at which to evaluate Jacobian
            *args: Additional arguments to func

        Returns:
            Jacobian matrix
        """
        n = len(x)
        f0 = func(x, *args)
        m = len(f0) if hasattr(f0, "__len__") else 1
        J = np.zeros((m, n))

        for i in range(n):
            x_plus = x.copy()
            x_plus[i] += self._eps
            f_plus = func(x_plus, *args)
            J[:, i] = (f_plus - f0) / self._eps

        return J

    def initialize(
        self,
        x0: np.ndarray,
        P0: Optional[np.ndarray] = None,
        t0: float = 0.0,
    ) -> EKFState:
        """Initialize filter state.

        Args:
            x0: Initial state estimate
            P0: Initial covariance (defaults to identity)
            t0: Initial time

        Returns:
            Initial EKFState
        """
        if P0 is None:
            P0 = np.eye(self.n_states)

        return EKFState(x=x0.copy(), P=P0.copy(), t=t0)

    def predict(
        self,
        state: EKFState,
        u: np.ndarray,
        dt: float,
    ) -> EKFState:
        """Prediction step.

        Args:
            state: Current filter state
            u: Control input
            dt: Time step

        Returns:
            Predicted state
        """
        # Predict state
        x_pred = self.f(state.x, u)

        # Compute Jacobian of f
        F = self._numerical_jacobian(self.f, state.x, u)

        # Predict covariance
        P_pred = F @ state.P @ F.T + self.Q

        return EKFState(x=x_pred, P=P_pred, t=state.t + dt)

    def update(
        self,
        state: EKFState,
        y: np.ndarray,
    ) -> EKFState:
        """Measurement update step.

        Args:
            state: Predicted state
            y: Measurement vector

        Returns:
            Updated state
        """
        # Predicted measurement
        y_pred = self.h(state.x)

        # Compute Jacobian of h
        H = self._numerical_jacobian(self.h, state.x)

        # Innovation
        innovation = y - y_pred

        # Innovation covariance
        S = H @ state.P @ H.T + self.R

        # Kalman gain
        try:
            K = state.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            K = state.P @ H.T @ np.linalg.pinv(S)

        # Update state
        x_upd = state.x + K @ innovation

        # Update covariance (Joseph form for numerical stability)
        I_KH = np.eye(self.n_states) - K @ H
        P_upd = I_KH @ state.P @ I_KH.T + K @ self.R @ K.T

        return EKFState(x=x_upd, P=P_upd, t=state.t)

    def step(
        self,
        state: EKFState,
        u: np.ndarray,
        y: np.ndarray,
        dt: float,
    ) -> EKFState:
        """Complete filter step (predict + update).

        Args:
            state: Current state
            u: Control input
            y: Measurement
            dt: Time step

        Returns:
            Updated state
        """
        predicted = self.predict(state, u, dt)
        updated = self.update(predicted, y)
        return updated


class MovingHorizonEstimator:
    """Moving Horizon Estimator for state estimation with constraints.

    MHE solves an optimization problem over a moving window of
    past measurements to estimate current state.

    Note: This is a placeholder implementation. Full MHE requires
    solving a constrained optimization at each step.
    """

    def __init__(
        self,
        horizon: int = 10,
        n_states: int = 10,
        n_measurements: int = 4,
    ):
        """Initialize MHE.

        Args:
            horizon: Number of past measurements to use
            n_states: Number of state variables
            n_measurements: Number of measurements
        """
        self.horizon = horizon
        self.n_states = n_states
        self.n_meas = n_measurements

        # History buffers
        self._y_history: List[np.ndarray] = []
        self._u_history: List[np.ndarray] = []

    def add_measurement(self, y: np.ndarray, u: np.ndarray):
        """Add new measurement to history.

        Args:
            y: Measurement vector
            u: Control input
        """
        self._y_history.append(y.copy())
        self._u_history.append(u.copy())

        # Keep only horizon
        if len(self._y_history) > self.horizon:
            self._y_history.pop(0)
            self._u_history.pop(0)

    def estimate(self) -> Optional[np.ndarray]:
        """Estimate current state from measurement history.

        Returns:
            State estimate, or None if insufficient data.
        """
        if len(self._y_history) < 2:
            return None

        # Placeholder: return last measurement extended to state dimension
        # Full implementation would solve optimization problem
        y_last = self._y_history[-1]
        x_est = np.zeros(self.n_states)
        x_est[: min(len(y_last), self.n_states)] = y_last[: min(len(y_last), self.n_states)]

        return x_est


def run_state_estimation(
    measurements: np.ndarray,
    inputs: np.ndarray,
    times: np.ndarray,
    estimator: str = "ekf",
    **kwargs,
) -> Dict:
    """Run state estimation on time series data.

    Args:
        measurements: Measurement matrix [n_times, n_meas]
        inputs: Input matrix [n_times, n_inputs]
        times: Time vector [n_times]
        estimator: Estimator type ("ekf" or "mhe")
        **kwargs: Additional arguments for estimator

    Returns:
        Dict with state estimates and diagnostics
    """
    n_times = len(times)
    n_meas = measurements.shape[1] if measurements.ndim > 1 else 1
    n_states = kwargs.get("n_states", n_meas * 2)

    # Placeholder: Simple smoothing as demonstration
    # Full implementation would use actual EKF or MHE

    state_estimates = np.zeros((n_times, n_states))

    # Simple exponential smoothing for demonstration
    alpha = 0.3
    for i in range(n_times):
        if i == 0:
            state_estimates[i, :n_meas] = measurements[i]
        else:
            state_estimates[i, :n_meas] = (
                alpha * measurements[i] + (1 - alpha) * state_estimates[i - 1, :n_meas]
            )

    return {
        "times": times,
        "state_estimates": state_estimates,
        "measurements": measurements,
        "method": estimator,
    }
