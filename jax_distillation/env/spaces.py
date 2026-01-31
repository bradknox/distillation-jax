"""Action and observation space definitions for DistillationColumnEnv.

This module defines the Gymnasium-compatible spaces for:
- Actions: Control inputs (reboiler duty, reflux ratio, setpoints)
- Observations: State measurements (temperatures, compositions, flows)

Spaces are configurable for different control scenarios.
"""

import numpy as np
from gymnasium import spaces


def create_action_space(
    Q_R_range: tuple[float, float] = (0.0, 20000.0),
    reflux_ratio_range: tuple[float, float] = (1.0, 10.0),
    B_setpoint_range: tuple[float, float] = (0.01, 0.1),
    D_setpoint_range: tuple[float, float] = (0.01, 0.1),
    continuous: bool = True,
) -> spaces.Space:
    """Create action space for column control.

    Args:
        Q_R_range: Reboiler duty range [W].
        reflux_ratio_range: Reflux ratio range.
        B_setpoint_range: Bottoms flow setpoint range [mol/s].
        D_setpoint_range: Distillate flow setpoint range [mol/s].
        continuous: If True, use Box space; else use MultiDiscrete.

    Returns:
        Gymnasium Space for actions.

    Action dimensions:
        0: Q_R - Reboiler duty [W]
        1: reflux_ratio - Reflux ratio (R/D)
        2: B_setpoint - Bottoms flow setpoint [mol/s]
        3: D_setpoint - Distillate flow setpoint [mol/s]
    """
    if continuous:
        low = np.array(
            [Q_R_range[0], reflux_ratio_range[0], B_setpoint_range[0], D_setpoint_range[0]],
            dtype=np.float32,
        )
        high = np.array(
            [Q_R_range[1], reflux_ratio_range[1], B_setpoint_range[1], D_setpoint_range[1]],
            dtype=np.float32,
        )
        return spaces.Box(low=low, high=high, dtype=np.float32)
    else:
        # Discrete action space with 10 levels per action
        return spaces.MultiDiscrete([10, 10, 10, 10])


def create_reduced_action_space(
    Q_R_range: tuple[float, float] = (0.0, 20000.0),
    reflux_ratio_range: tuple[float, float] = (1.0, 10.0),
) -> spaces.Space:
    """Create reduced action space with only Q_R and reflux ratio.

    For simpler control scenarios where flow setpoints are fixed.

    Args:
        Q_R_range: Reboiler duty range [W].
        reflux_ratio_range: Reflux ratio range.

    Returns:
        Gymnasium Box space for actions.

    Action dimensions:
        0: Q_R - Reboiler duty [W]
        1: reflux_ratio - Reflux ratio (R/D)
    """
    low = np.array([Q_R_range[0], reflux_ratio_range[0]], dtype=np.float32)
    high = np.array([Q_R_range[1], reflux_ratio_range[1]], dtype=np.float32)
    return spaces.Box(low=low, high=high, dtype=np.float32)


def create_observation_space(
    n_trays: int,
    include_flows: bool = True,
    include_holdups: bool = False,
) -> spaces.Space:
    """Create observation space for column state.

    Args:
        n_trays: Number of trays in the column.
        include_flows: Include flow rates in observation.
        include_holdups: Include liquid holdups in observation.

    Returns:
        Gymnasium Box space for observations.

    Observation dimensions (with include_flows=True, include_holdups=False):
        [0:n_trays]: Tray temperatures [K], normalized to [0, 1]
        [n_trays:2*n_trays]: Tray compositions [mol frac]
        [2*n_trays]: Reboiler temperature [K], normalized
        [2*n_trays+1]: Reboiler composition [mol frac]
        [2*n_trays+2]: Condenser temperature [K], normalized
        [2*n_trays+3]: Condenser composition [mol frac]
        [2*n_trays+4]: Distillate flow [mol/s], normalized
        [2*n_trays+5]: Bottoms flow [mol/s], normalized
        [2*n_trays+6]: Distillate composition [mol frac]
        [2*n_trays+7]: Bottoms composition [mol frac]
    """
    # Base dimensions: tray T, tray x, reboiler T/x, condenser T/x
    obs_dim = 2 * n_trays + 4

    if include_flows:
        obs_dim += 4  # D, B, x_D, x_B

    if include_holdups:
        obs_dim += n_trays + 2  # tray holdups + reboiler + condenser

    # All observations normalized to approximately [0, 1]
    low = np.zeros(obs_dim, dtype=np.float32)
    high = np.ones(obs_dim, dtype=np.float32)

    return spaces.Box(low=low, high=high, dtype=np.float32)


def create_minimal_observation_space() -> spaces.Space:
    """Create minimal observation space for simple scenarios.

    Includes only product compositions and key temperatures.

    Returns:
        Gymnasium Box space.

    Observation dimensions:
        0: Distillate composition x_D [mol frac]
        1: Bottoms composition x_B [mol frac]
        2: Top tray temperature (normalized)
        3: Bottom tray temperature (normalized)
        4: Reboiler duty (normalized)
        5: Reflux ratio (normalized)
    """
    low = np.zeros(6, dtype=np.float32)
    high = np.ones(6, dtype=np.float32)
    return spaces.Box(low=low, high=high, dtype=np.float32)


class ObservationNormalizer:
    """Normalizes observations to [0, 1] range."""

    def __init__(
        self,
        T_min: float = 300.0,
        T_max: float = 400.0,
        flow_max: float = 0.2,
        holdup_max: float = 20.0,
        Q_max: float = 20000.0,
        reflux_ratio_max: float = 10.0,
    ):
        """Initialize normalizer with expected ranges.

        Args:
            T_min: Minimum expected temperature [K].
            T_max: Maximum expected temperature [K].
            flow_max: Maximum expected flow rate [mol/s].
            holdup_max: Maximum expected holdup [mol].
            Q_max: Maximum reboiler duty [W].
            reflux_ratio_max: Maximum reflux ratio.
        """
        self.T_min = T_min
        self.T_max = T_max
        self.flow_max = flow_max
        self.holdup_max = holdup_max
        self.Q_max = Q_max
        self.reflux_ratio_max = reflux_ratio_max

    def normalize_temperature(self, T: np.ndarray) -> np.ndarray:
        """Normalize temperature to [0, 1]."""
        return (T - self.T_min) / (self.T_max - self.T_min)

    def normalize_flow(self, F: np.ndarray) -> np.ndarray:
        """Normalize flow rate to [0, 1]."""
        return F / self.flow_max

    def normalize_holdup(self, M: np.ndarray) -> np.ndarray:
        """Normalize holdup to [0, 1]."""
        return M / self.holdup_max

    def normalize_duty(self, Q: np.ndarray) -> np.ndarray:
        """Normalize duty to [0, 1]."""
        return Q / self.Q_max

    def normalize_reflux_ratio(self, rr: np.ndarray) -> np.ndarray:
        """Normalize reflux ratio to [0, 1]."""
        return rr / self.reflux_ratio_max


class ActionDenormalizer:
    """Converts normalized actions to physical units."""

    def __init__(
        self,
        Q_R_range: tuple[float, float] = (0.0, 20000.0),
        reflux_ratio_range: tuple[float, float] = (1.0, 10.0),
        B_setpoint_range: tuple[float, float] = (0.01, 0.1),
        D_setpoint_range: tuple[float, float] = (0.01, 0.1),
    ):
        """Initialize with action ranges."""
        self.Q_R_range = Q_R_range
        self.reflux_ratio_range = reflux_ratio_range
        self.B_setpoint_range = B_setpoint_range
        self.D_setpoint_range = D_setpoint_range

    def denormalize(self, action: np.ndarray) -> dict[str, float]:
        """Convert normalized action [0, 1]^n to physical units.

        Args:
            action: Normalized action array.

        Returns:
            Dictionary with physical action values.
        """
        return {
            "Q_R": self._lerp(action[0], *self.Q_R_range),
            "reflux_ratio": self._lerp(action[1], *self.reflux_ratio_range),
            "B_setpoint": self._lerp(action[2], *self.B_setpoint_range),
            "D_setpoint": self._lerp(action[3], *self.D_setpoint_range),
        }

    def denormalize_reduced(self, action: np.ndarray) -> dict[str, float]:
        """Convert reduced normalized action to physical units.

        Args:
            action: Normalized action array (Q_R, reflux_ratio only).

        Returns:
            Dictionary with physical action values.
        """
        return {
            "Q_R": self._lerp(action[0], *self.Q_R_range),
            "reflux_ratio": self._lerp(action[1], *self.reflux_ratio_range),
        }

    @staticmethod
    def _lerp(t: float, low: float, high: float) -> float:
        """Linear interpolation."""
        return low + t * (high - low)
