"""Wood-Berry MIMO transfer function model implementation.

This module implements the classic Wood-Berry 2x2 transfer function
model for distillation column dynamics.

Reference:
    Wood, R.K. and Berry, M.W. (1973). "Terminal composition control of a
    binary distillation column." Chemical Engineering Science, 28(9), 1707-1717.
"""

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from scipy import signal


@dataclass
class TransferFunctionParams:
    """Parameters for a first-order-plus-dead-time transfer function.

    G(s) = K * exp(-theta * s) / (tau * s + 1)

    Attributes:
        K: Static gain
        tau: Time constant [min]
        theta: Dead time [min]
    """

    K: float
    tau: float
    theta: float


@dataclass
class WoodBerryCoefficients:
    """Wood-Berry model coefficients (published values).

    All time constants and dead times in minutes.
    """

    # G11: R → x_D
    G11: TransferFunctionParams = None

    # G12: S → x_D
    G12: TransferFunctionParams = None

    # G21: R → x_B
    G21: TransferFunctionParams = None

    # G22: S → x_B
    G22: TransferFunctionParams = None

    def __post_init__(self):
        if self.G11 is None:
            self.G11 = TransferFunctionParams(K=12.8, tau=16.7, theta=1.0)
        if self.G12 is None:
            self.G12 = TransferFunctionParams(K=-18.9, tau=21.0, theta=3.0)
        if self.G21 is None:
            self.G21 = TransferFunctionParams(K=6.6, tau=10.9, theta=7.0)
        if self.G22 is None:
            self.G22 = TransferFunctionParams(K=-19.4, tau=14.4, theta=3.0)


def get_wood_berry_coefficients() -> WoodBerryCoefficients:
    """Get the canonical Wood-Berry transfer function coefficients.

    Returns:
        WoodBerryCoefficients with published values.
    """
    return WoodBerryCoefficients()


class WoodBerryModel:
    """Wood-Berry 2x2 MIMO transfer function model.

    Simulates the linearized dynamics of a binary distillation column
    using first-order-plus-dead-time transfer functions.

    Inputs:
        R: Reflux flow deviation
        S: Steam flow deviation

    Outputs:
        x_D: Distillate composition deviation
        x_B: Bottoms composition deviation
    """

    def __init__(
        self,
        coefficients: WoodBerryCoefficients = None,
        dt: float = 0.1,
        time_unit: str = "min",
    ):
        """Initialize Wood-Berry model.

        Args:
            coefficients: Model coefficients. Uses published values if None.
            dt: Simulation timestep [time_unit].
            time_unit: Time unit for dt ("min" or "s").
        """
        if coefficients is None:
            coefficients = get_wood_berry_coefficients()

        self.coefficients = coefficients
        self.dt = dt
        self.time_unit = time_unit

        # Convert dt to minutes if needed (coefficients are in minutes)
        if time_unit == "s":
            self._dt_min = dt / 60.0
        else:
            self._dt_min = dt

        # Initialize state-space representations
        self._init_state_space()

    def _init_state_space(self):
        """Initialize state-space representations for each transfer function."""
        self._systems = {}
        self._states = {}
        self._delay_buffers = {}

        for name, params in [
            ("G11", self.coefficients.G11),
            ("G12", self.coefficients.G12),
            ("G21", self.coefficients.G21),
            ("G22", self.coefficients.G22),
        ]:
            # First-order system (without dead time): G = K / (tau*s + 1)
            num = [params.K]
            den = [params.tau, 1.0]
            sys_c = signal.TransferFunction(num, den)

            # Discretize
            sys_d = signal.cont2discrete((num, den), self._dt_min, method="tustin")
            self._systems[name] = sys_d

            # State (for first-order, single state)
            self._states[name] = 0.0

            # Delay buffer (circular buffer for dead time)
            n_delay = max(1, int(params.theta / self._dt_min))
            self._delay_buffers[name] = {
                "buffer": np.zeros(n_delay),
                "index": 0,
                "size": n_delay,
            }

    def reset(self):
        """Reset model to initial conditions (zero state)."""
        for name in self._states:
            self._states[name] = 0.0
            buf = self._delay_buffers[name]
            buf["buffer"][:] = 0.0
            buf["index"] = 0

    def _apply_delay(self, name: str, input_val: float) -> float:
        """Apply dead time to a signal using circular buffer.

        Args:
            name: Transfer function name (G11, G12, etc.)
            input_val: Current input value

        Returns:
            Delayed output value
        """
        buf = self._delay_buffers[name]
        idx = buf["index"]

        # Get delayed output
        delayed = buf["buffer"][idx]

        # Store new input
        buf["buffer"][idx] = input_val

        # Advance index
        buf["index"] = (idx + 1) % buf["size"]

        return delayed

    def _first_order_step(self, name: str, u: float) -> float:
        """Simulate one step of first-order dynamics.

        Args:
            name: Transfer function name
            u: Input value

        Returns:
            Output value
        """
        params = getattr(self.coefficients, name)

        # Simple Euler discretization for first-order system
        # dy/dt = (K*u - y) / tau
        y = self._states[name]
        dydt = (params.K * u - y) / params.tau
        y_new = y + dydt * self._dt_min

        self._states[name] = y_new
        return y_new

    def step(self, R: float, S: float) -> Tuple[float, float]:
        """Simulate one timestep.

        Args:
            R: Reflux deviation input
            S: Steam deviation input

        Returns:
            Tuple of (x_D, x_B) output deviations
        """
        # Apply dead times to inputs
        R_delayed_11 = self._apply_delay("G11", R)
        S_delayed_12 = self._apply_delay("G12", S)
        R_delayed_21 = self._apply_delay("G21", R)
        S_delayed_22 = self._apply_delay("G22", S)

        # First-order dynamics
        y11 = self._first_order_step("G11", R_delayed_11)
        y12 = self._first_order_step("G12", S_delayed_12)
        y21 = self._first_order_step("G21", R_delayed_21)
        y22 = self._first_order_step("G22", S_delayed_22)

        # Sum contributions
        x_D = y11 + y12
        x_B = y21 + y22

        return x_D, x_B

    def simulate(
        self,
        R_trajectory: np.ndarray,
        S_trajectory: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate over a trajectory of inputs.

        Args:
            R_trajectory: Reflux input trajectory
            S_trajectory: Steam input trajectory

        Returns:
            Tuple of (x_D_trajectory, x_B_trajectory)
        """
        n_steps = len(R_trajectory)
        x_D = np.zeros(n_steps)
        x_B = np.zeros(n_steps)

        for i in range(n_steps):
            x_D[i], x_B[i] = self.step(R_trajectory[i], S_trajectory[i])

        return x_D, x_B


def simulate_wood_berry_step_response(
    input_var: str = "R",
    step_size: float = 1.0,
    total_time: float = 100.0,
    dt: float = 0.1,
) -> Dict:
    """Simulate step response of Wood-Berry model.

    Args:
        input_var: Which input to step ("R" or "S")
        step_size: Size of step change
        total_time: Total simulation time [min]
        dt: Timestep [min]

    Returns:
        Dict with time, x_D, x_B trajectories and metadata
    """
    model = WoodBerryModel(dt=dt)
    n_steps = int(total_time / dt)

    times = np.arange(n_steps) * dt
    R_traj = np.zeros(n_steps)
    S_traj = np.zeros(n_steps)

    if input_var == "R":
        R_traj[:] = step_size
    else:
        S_traj[:] = step_size

    x_D, x_B = model.simulate(R_traj, S_traj)

    # Expected final values
    coef = model.coefficients
    if input_var == "R":
        expected_x_D = coef.G11.K * step_size
        expected_x_B = coef.G21.K * step_size
    else:
        expected_x_D = coef.G12.K * step_size
        expected_x_B = coef.G22.K * step_size

    return {
        "times": times,
        "x_D": x_D,
        "x_B": x_B,
        "input_var": input_var,
        "step_size": step_size,
        "expected_x_D_final": expected_x_D,
        "expected_x_B_final": expected_x_B,
        "actual_x_D_final": x_D[-1],
        "actual_x_B_final": x_B[-1],
    }


def print_wood_berry_summary():
    """Print summary of Wood-Berry model coefficients."""
    coef = get_wood_berry_coefficients()

    print("=" * 60)
    print("WOOD-BERRY MODEL COEFFICIENTS")
    print("=" * 60)
    print("\nTransfer Function Matrix:")
    print("  [x_D]   [G11  G12] [R]")
    print("  [x_B] = [G21  G22] [S]")

    print("\nParameters (K, tau [min], theta [min]):")
    print(f"  G11 (R→x_D): K={coef.G11.K:+6.1f}, τ={coef.G11.tau:5.1f}, θ={coef.G11.theta:4.1f}")
    print(f"  G12 (S→x_D): K={coef.G12.K:+6.1f}, τ={coef.G12.tau:5.1f}, θ={coef.G12.theta:4.1f}")
    print(f"  G21 (R→x_B): K={coef.G21.K:+6.1f}, τ={coef.G21.tau:5.1f}, θ={coef.G21.theta:4.1f}")
    print(f"  G22 (S→x_B): K={coef.G22.K:+6.1f}, τ={coef.G22.tau:5.1f}, θ={coef.G22.theta:4.1f}")

    print("\nGain Signs (Physical Interpretation):")
    print("  G11 > 0: ↑Reflux → ↑Distillate purity")
    print("  G12 < 0: ↑Steam  → ↓Distillate purity (coupling)")
    print("  G21 > 0: ↑Reflux → ↑Bottoms impurity (coupling)")
    print("  G22 < 0: ↑Steam  → ↓Bottoms impurity")
    print("=" * 60)


if __name__ == "__main__":
    print_wood_berry_summary()

    print("\nStep response to R=1:")
    result = simulate_wood_berry_step_response("R", step_size=1.0, total_time=100.0)
    print(f"  x_D final: {result['actual_x_D_final']:.2f} (expected: {result['expected_x_D_final']:.2f})")
    print(f"  x_B final: {result['actual_x_B_final']:.2f} (expected: {result['expected_x_B_final']:.2f})")

    print("\nStep response to S=1:")
    result = simulate_wood_berry_step_response("S", step_size=1.0, total_time=100.0)
    print(f"  x_D final: {result['actual_x_D_final']:.2f} (expected: {result['expected_x_D_final']:.2f})")
    print(f"  x_B final: {result['actual_x_B_final']:.2f} (expected: {result['expected_x_B_final']:.2f})")
