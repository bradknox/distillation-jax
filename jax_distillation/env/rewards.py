"""Reward functions for distillation column RL environment.

This module provides configurable reward components:
- Product purity tracking
- Energy minimization
- Constraint violation penalties
- Stability rewards

Rewards can be combined with different weights for various objectives.
"""

import jax.numpy as jnp
import numpy as np
from typing import NamedTuple


class RewardConfig(NamedTuple):
    """Configuration for reward calculation.

    Attributes:
        x_D_target: Target distillate composition [mol frac].
        x_B_target: Target bottoms composition [mol frac].
        purity_weight: Weight for purity reward.
        energy_weight: Weight for energy penalty.
        stability_weight: Weight for stability reward.
        constraint_weight: Weight for constraint violations.
        Q_R_max: Maximum reboiler duty for normalization [W].
        tolerance_D: Tolerance for distillate purity.
        tolerance_B: Tolerance for bottoms purity.
    """

    x_D_target: float = 0.95
    x_B_target: float = 0.05
    purity_weight: float = 1.0
    energy_weight: float = 0.1
    stability_weight: float = 0.1
    constraint_weight: float = 10.0
    Q_R_max: float = 20000.0
    tolerance_D: float = 0.02
    tolerance_B: float = 0.02


def create_default_reward_config() -> RewardConfig:
    """Create default reward configuration."""
    return RewardConfig()


# =============================================================================
# Reward Components
# =============================================================================


def purity_reward(
    x_D: float,
    x_B: float,
    x_D_target: float = 0.95,
    x_B_target: float = 0.05,
    tolerance_D: float = 0.02,
    tolerance_B: float = 0.02,
) -> float:
    """Compute reward based on product purity.

    Reward is 1.0 when both products are within tolerance of targets,
    and decreases quadratically with deviation.

    Args:
        x_D: Actual distillate composition [mol frac].
        x_B: Actual bottoms composition [mol frac].
        x_D_target: Target distillate composition.
        x_B_target: Target bottoms composition.
        tolerance_D: Acceptable deviation for distillate.
        tolerance_B: Acceptable deviation for bottoms.

    Returns:
        Purity reward in [0, 1].
    """
    # Error from targets
    error_D = abs(x_D - x_D_target)
    error_B = abs(x_B - x_B_target)

    # Reward within tolerance is 1, then decays
    reward_D = max(0.0, 1.0 - (error_D / tolerance_D) ** 2)
    reward_B = max(0.0, 1.0 - (error_B / tolerance_B) ** 2)

    # Combined reward (geometric mean)
    return (reward_D * reward_B) ** 0.5


def purity_reward_jax(
    x_D: jnp.ndarray,
    x_B: jnp.ndarray,
    x_D_target: float = 0.95,
    x_B_target: float = 0.05,
    tolerance_D: float = 0.02,
    tolerance_B: float = 0.02,
) -> jnp.ndarray:
    """JAX-compatible purity reward."""
    error_D = jnp.abs(x_D - x_D_target)
    error_B = jnp.abs(x_B - x_B_target)

    reward_D = jnp.maximum(0.0, 1.0 - (error_D / tolerance_D) ** 2)
    reward_B = jnp.maximum(0.0, 1.0 - (error_B / tolerance_B) ** 2)

    return jnp.sqrt(reward_D * reward_B)


def energy_penalty(
    Q_R: float,
    Q_R_max: float = 20000.0,
) -> float:
    """Compute energy penalty (higher energy = lower reward).

    Args:
        Q_R: Reboiler duty [W].
        Q_R_max: Maximum/reference reboiler duty [W].

    Returns:
        Energy penalty in [0, 1], where 0 is no penalty (Q_R=0).
    """
    return min(1.0, Q_R / Q_R_max)


def energy_penalty_jax(
    Q_R: jnp.ndarray,
    Q_R_max: float = 20000.0,
) -> jnp.ndarray:
    """JAX-compatible energy penalty."""
    return jnp.minimum(1.0, Q_R / Q_R_max)


def stability_reward(
    dx_D: float,
    dx_B: float,
    threshold: float = 0.01,
) -> float:
    """Reward for maintaining stable operation.

    Penalizes large changes in product compositions between steps.

    Args:
        dx_D: Change in distillate composition from last step.
        dx_B: Change in bottoms composition from last step.
        threshold: Threshold for acceptable change.

    Returns:
        Stability reward in [0, 1].
    """
    change = abs(dx_D) + abs(dx_B)
    return max(0.0, 1.0 - change / threshold)


def stability_reward_jax(
    dx_D: jnp.ndarray,
    dx_B: jnp.ndarray,
    threshold: float = 0.01,
) -> jnp.ndarray:
    """JAX-compatible stability reward."""
    change = jnp.abs(dx_D) + jnp.abs(dx_B)
    return jnp.maximum(0.0, 1.0 - change / threshold)


def constraint_violation_penalty(
    tray_M: np.ndarray,
    M_min: float = 0.1,
    M_max: float = 20.0,
    tray_T: np.ndarray | None = None,
    T_min: float = 300.0,
    T_max: float = 400.0,
) -> float:
    """Compute penalty for constraint violations.

    Args:
        tray_M: Liquid holdups on all trays [mol].
        M_min: Minimum acceptable holdup [mol].
        M_max: Maximum acceptable holdup [mol].
        tray_T: Temperatures on all trays [K] (optional).
        T_min: Minimum acceptable temperature [K].
        T_max: Maximum acceptable temperature [K].

    Returns:
        Constraint violation penalty (0 = no violations).
    """
    penalty = 0.0

    # Holdup violations
    holdup_low = np.sum(np.maximum(0, M_min - tray_M))
    holdup_high = np.sum(np.maximum(0, tray_M - M_max))
    penalty += holdup_low + holdup_high

    # Temperature violations
    if tray_T is not None:
        temp_low = np.sum(np.maximum(0, T_min - tray_T))
        temp_high = np.sum(np.maximum(0, tray_T - T_max))
        penalty += (temp_low + temp_high) / 10.0  # Scale temperature penalty

    return penalty


def constraint_violation_penalty_jax(
    tray_M: jnp.ndarray,
    M_min: float = 0.1,
    M_max: float = 20.0,
    tray_T: jnp.ndarray | None = None,
    T_min: float = 300.0,
    T_max: float = 400.0,
) -> jnp.ndarray:
    """JAX-compatible constraint violation penalty."""
    # Holdup violations
    holdup_low = jnp.sum(jnp.maximum(0.0, M_min - tray_M))
    holdup_high = jnp.sum(jnp.maximum(0.0, tray_M - M_max))
    penalty = holdup_low + holdup_high

    # Temperature violations
    if tray_T is not None:
        temp_low = jnp.sum(jnp.maximum(0.0, T_min - tray_T))
        temp_high = jnp.sum(jnp.maximum(0.0, tray_T - T_max))
        penalty = penalty + (temp_low + temp_high) / 10.0

    return penalty


# =============================================================================
# Combined Reward Functions
# =============================================================================


def compute_reward(
    x_D: float,
    x_B: float,
    Q_R: float,
    dx_D: float = 0.0,
    dx_B: float = 0.0,
    tray_M: np.ndarray | None = None,
    tray_T: np.ndarray | None = None,
    config: RewardConfig | None = None,
) -> tuple[float, dict[str, float]]:
    """Compute total reward with component breakdown.

    Args:
        x_D: Distillate composition [mol frac].
        x_B: Bottoms composition [mol frac].
        Q_R: Reboiler duty [W].
        dx_D: Change in distillate composition.
        dx_B: Change in bottoms composition.
        tray_M: Tray holdups [mol] (for constraints).
        tray_T: Tray temperatures [K] (for constraints).
        config: Reward configuration.

    Returns:
        Tuple of (total_reward, component_dict).
    """
    if config is None:
        config = create_default_reward_config()

    # Compute components
    r_purity = purity_reward(
        x_D, x_B, config.x_D_target, config.x_B_target, config.tolerance_D, config.tolerance_B
    )

    r_energy = energy_penalty(Q_R, config.Q_R_max)

    r_stability = stability_reward(dx_D, dx_B)

    r_constraint = 0.0
    if tray_M is not None:
        r_constraint = constraint_violation_penalty(tray_M, tray_T=tray_T)

    # Combine with weights
    total_reward = (
        config.purity_weight * r_purity
        - config.energy_weight * r_energy
        + config.stability_weight * r_stability
        - config.constraint_weight * r_constraint
    )

    components = {
        "purity": r_purity,
        "energy": -r_energy,
        "stability": r_stability,
        "constraint": -r_constraint,
        "total": total_reward,
    }

    return total_reward, components


def compute_reward_jax(
    x_D: jnp.ndarray,
    x_B: jnp.ndarray,
    Q_R: jnp.ndarray,
    dx_D: jnp.ndarray,
    dx_B: jnp.ndarray,
    tray_M: jnp.ndarray,
    config: RewardConfig,
) -> jnp.ndarray:
    """JAX-compatible reward computation for JIT.

    Args:
        x_D: Distillate composition.
        x_B: Bottoms composition.
        Q_R: Reboiler duty.
        dx_D: Change in distillate composition.
        dx_B: Change in bottoms composition.
        tray_M: Tray holdups.
        config: Reward configuration.

    Returns:
        Total reward scalar.
    """
    r_purity = purity_reward_jax(
        x_D, x_B, config.x_D_target, config.x_B_target, config.tolerance_D, config.tolerance_B
    )

    r_energy = energy_penalty_jax(Q_R, config.Q_R_max)

    r_stability = stability_reward_jax(dx_D, dx_B)

    r_constraint = constraint_violation_penalty_jax(tray_M)

    total_reward = (
        config.purity_weight * r_purity
        - config.energy_weight * r_energy
        + config.stability_weight * r_stability
        - config.constraint_weight * r_constraint
    )

    return total_reward


# =============================================================================
# Specialized Reward Functions
# =============================================================================


def tracking_reward(
    x_D: float,
    x_D_setpoint: float,
    x_B: float,
    x_B_setpoint: float,
    weight_D: float = 1.0,
    weight_B: float = 1.0,
) -> float:
    """Reward for tracking time-varying setpoints.

    Args:
        x_D: Actual distillate composition.
        x_D_setpoint: Setpoint for distillate composition.
        x_B: Actual bottoms composition.
        x_B_setpoint: Setpoint for bottoms composition.
        weight_D: Weight for distillate tracking error.
        weight_B: Weight for bottoms tracking error.

    Returns:
        Tracking reward (negative of weighted squared error).
    """
    error_D = (x_D - x_D_setpoint) ** 2
    error_B = (x_B - x_B_setpoint) ** 2

    return -(weight_D * error_D + weight_B * error_B)


def separation_efficiency_reward(
    x_D: float,
    x_B: float,
    z_F: float = 0.5,
) -> float:
    """Reward based on separation efficiency.

    Measures how well the column separates feed into pure products.

    Args:
        x_D: Distillate composition.
        x_B: Bottoms composition.
        z_F: Feed composition.

    Returns:
        Separation efficiency reward.
    """
    # Perfect separation: x_D = 1, x_B = 0
    # Efficiency measured by product purity improvement over feed
    enrichment = max(0, x_D - z_F)
    stripping = max(0, z_F - x_B)

    # Combined efficiency (normalized to [0, 1])
    max_enrichment = 1.0 - z_F
    max_stripping = z_F

    if max_enrichment > 0 and max_stripping > 0:
        efficiency = 0.5 * (enrichment / max_enrichment + stripping / max_stripping)
    else:
        efficiency = 0.0

    return efficiency


def economic_reward(
    D: float,
    x_D: float,
    B: float,
    x_B: float,
    Q_R: float,
    distillate_price: float = 100.0,
    bottoms_price: float = 10.0,
    energy_cost: float = 0.001,
    x_D_spec: float = 0.95,
    x_B_spec: float = 0.05,
) -> float:
    """Economic reward based on product value minus energy cost.

    Args:
        D: Distillate flow rate [mol/s].
        x_D: Distillate composition.
        B: Bottoms flow rate [mol/s].
        x_B: Bottoms composition.
        Q_R: Reboiler duty [W].
        distillate_price: Price per mol of on-spec distillate.
        bottoms_price: Price per mol of on-spec bottoms.
        energy_cost: Cost per Watt of reboiler duty.
        x_D_spec: Specification for distillate.
        x_B_spec: Specification for bottoms.

    Returns:
        Economic reward (profit rate).
    """
    # Only count product value if it meets spec
    D_value = distillate_price * D if x_D >= x_D_spec else 0.0
    B_value = bottoms_price * B if x_B <= x_B_spec else 0.0

    # Energy cost
    energy_expense = energy_cost * Q_R

    return D_value + B_value - energy_expense
