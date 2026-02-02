"""Complete distillation column model.

This module assembles all column components:
- Multiple trays connected in series
- Reboiler at the bottom
- Condenser at the top
- Feed tray handling

The column_step function performs a complete timestep update.
All functions are pure and JIT-compilable.
"""

import jax
import jax.numpy as jnp
from typing import NamedTuple, Optional

from jax_distillation.column.config import ColumnConfig
from jax_distillation.column.tray import (
    TrayState,
    TrayInflows,
    total_material_balance,
    component_material_balance,
    murphree_vapor_efficiency,
)
from jax_distillation.column.reboiler import (
    ReboilerState,
    ReboilerInputs,
    ReboilerOutputs,
    reboiler_step,
    create_initial_reboiler_state,
)
from jax_distillation.column.condenser import (
    CondenserState,
    CondenserInputs,
    CondenserOutputs,
    condenser_step,
    create_initial_condenser_state,
)
from jax_distillation.core.thermodynamics import (
    bubble_point_temperature,
    equilibrium_vapor_composition,
    mixture_molecular_weight,
)
from jax_distillation.core.hydraulics import (
    liquid_density,
    vapor_density,
    surface_tension,
    static_liquid_outflow,
    update_liquid_outflow,
    flooding_velocity,
    flooding_ratio,
    weep_point_velocity,
    is_weeping,
)
from jax_distillation.core.types import ThermoParams


class FullColumnState(NamedTuple):
    """Complete state of the distillation column.

    Attributes:
        tray_M: Liquid holdup on each tray [mol], shape (n_trays,).
        tray_x: Liquid composition on each tray [mol frac], shape (n_trays,).
        tray_T: Temperature of each tray [K], shape (n_trays,).
        tray_L_out: Dynamic liquid outflow from each tray [mol/s], shape (n_trays,).
        reboiler: Reboiler state.
        condenser: Condenser state.
        t: Current simulation time [s].
        V_prev: Previous vapor flow for hydraulic coupling [mol/s], shape (n_trays+1,).
    """

    tray_M: jnp.ndarray
    tray_x: jnp.ndarray
    tray_T: jnp.ndarray
    tray_L_out: jnp.ndarray
    reboiler: ReboilerState
    condenser: CondenserState
    t: jnp.ndarray
    V_prev: jnp.ndarray


class ColumnAction(NamedTuple):
    """Control actions for the column.

    Attributes:
        Q_R: Reboiler duty [W].
        reflux_ratio: Reflux ratio (R/D).
        B_setpoint: Bottoms flow setpoint [mol/s].
        D_setpoint: Distillate flow setpoint [mol/s].
    """

    Q_R: jnp.ndarray
    reflux_ratio: jnp.ndarray
    B_setpoint: jnp.ndarray
    D_setpoint: jnp.ndarray


class ColumnOutputs(NamedTuple):
    """Outputs from column simulation step.

    Attributes:
        D: Distillate flow [mol/s].
        x_D: Distillate composition [mol fraction].
        B: Bottoms flow [mol/s].
        x_B: Bottoms composition [mol fraction].
        Q_R: Reboiler duty [W].
        Q_C: Condenser duty [W].
        V: Vapor flow profile [mol/s], shape (n_trays+1,).
        L: Liquid flow profile [mol/s], shape (n_trays+1,).
        flood_fraction: Fraction of flooding velocity for each tray (0-1+), shape (n_trays,).
        weeping: Boolean array indicating weeping condition for each tray, shape (n_trays,).
    """

    D: jnp.ndarray
    x_D: jnp.ndarray
    B: jnp.ndarray
    x_B: jnp.ndarray
    Q_R: jnp.ndarray
    Q_C: jnp.ndarray
    V: jnp.ndarray
    L: jnp.ndarray
    flood_fraction: jnp.ndarray
    weeping: jnp.ndarray


class StaticColumnParams(NamedTuple):
    """Static (non-traced) parameters for JIT compilation.

    These parameters must be concrete Python values (not JAX traced values)
    for use with jax.lax.scan length parameter.

    Attributes:
        n_trays: Number of trays (static int).
        feed_tray: Feed tray location (static int).
        n_substeps: Number of substeps per timestep (static int).
    """

    n_trays: int
    feed_tray: int
    n_substeps: int


# =============================================================================
# Column Flow Calculations
# =============================================================================


def compute_feed_split(
    F: jnp.ndarray,
    z_F: jnp.ndarray,
    q: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Split feed into liquid and vapor portions based on feed quality.

    Args:
        F: Total feed flow [mol/s].
        z_F: Feed composition [mol fraction].
        q: Feed quality (1=saturated liquid, 0=saturated vapor).

    Returns:
        Tuple of (L_feed, x_feed, V_feed, y_feed).
    """
    L_feed = q * F
    V_feed = (1.0 - q) * F

    # Composition is the same for both phases at feed conditions
    x_feed = z_F
    y_feed = z_F

    return L_feed, x_feed, V_feed, y_feed


def compute_tray_flows(
    tray_M: jnp.ndarray,
    tray_x: jnp.ndarray,
    tray_T: jnp.ndarray,
    V_from_reboiler: jnp.ndarray,
    y_from_reboiler: jnp.ndarray,
    R: jnp.ndarray,
    x_R: jnp.ndarray,
    config: ColumnConfig,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute vapor and liquid flows throughout the column.

    Tray numbering: 1 = top tray, n_trays = bottom tray
    Array indexing: [0] = top tray, [n_trays-1] = bottom tray

    Args:
        tray_M: Liquid holdup on each tray [mol].
        tray_x: Liquid composition on each tray [mol frac].
        tray_T: Temperature on each tray [K].
        V_from_reboiler: Vapor from reboiler [mol/s].
        y_from_reboiler: Vapor composition from reboiler [mol frac].
        R: Reflux from condenser [mol/s].
        x_R: Reflux composition [mol frac].
        config: Column configuration.

    Returns:
        Tuple of (V_out, y_out, L_out, x_out) for each tray.
        V_out[i] = vapor leaving tray i (going up)
        L_out[i] = liquid leaving tray i (going down)
    """
    n_trays = config.geometry.n_trays
    feed_tray = config.geometry.feed_tray
    E_M = config.simulation.murphree_efficiency
    P = config.P
    thermo = config.thermo

    # Get feed split
    L_feed, x_feed, V_feed, y_feed = compute_feed_split(
        config.feed.F, config.feed.z_F, config.feed.q
    )

    # Initialize flow arrays
    V_out = jnp.zeros(n_trays)
    y_out = jnp.zeros(n_trays)
    L_out = jnp.zeros(n_trays)
    x_out = jnp.zeros(n_trays)

    # Process trays from bottom to top for vapor flows
    # Process trays from top to bottom for liquid flows

    def process_vapor(carry, tray_idx):
        """Compute vapor flow going up from tray_idx."""
        V_in, y_in = carry

        # Tray index (0 = top, n_trays-1 = bottom)
        # We process bottom-up: tray_idx goes from n_trays-1 to 0

        x = tray_x[tray_idx]
        T = tray_T[tray_idx]

        # Equilibrium vapor composition
        y_eq = equilibrium_vapor_composition(x, T, P, thermo)

        # Murphree efficiency: actual vapor composition
        y_actual = y_in + E_M * (y_eq - y_in)

        # Assume constant molar overflow (CMO) for vapor
        # In reality, V changes due to enthalpy balance, but CMO is good first approx
        # Add feed vapor at feed tray (tray numbering: feed_tray is 1-indexed)
        feed_tray_idx = feed_tray - 1  # Convert to 0-indexed
        is_feed_tray = (tray_idx == feed_tray_idx)
        V_out_tray = V_in + is_feed_tray * V_feed

        return (V_out_tray, y_actual), (V_out_tray, y_actual)

    def process_liquid(carry, tray_idx):
        """Compute liquid flow going down from tray_idx."""
        L_in, x_in = carry

        # Liquid composition equals tray composition (well-mixed)
        x = tray_x[tray_idx]

        # CMO assumption for liquid
        # Add feed liquid at feed tray
        feed_tray_idx = feed_tray - 1
        is_feed_tray = (tray_idx == feed_tray_idx)
        L_out_tray = L_in + is_feed_tray * L_feed

        return (L_out_tray, x), (L_out_tray, x)

    # Process vapor from bottom to top
    tray_indices_bottom_up = jnp.arange(n_trays - 1, -1, -1)
    _, (V_arr, y_arr) = jax.lax.scan(
        process_vapor,
        (V_from_reboiler, y_from_reboiler),
        tray_indices_bottom_up,
    )
    # Reverse to get proper ordering (top=0, bottom=n-1)
    V_out = V_arr[::-1]
    y_out = y_arr[::-1]

    # Process liquid from top to bottom
    tray_indices_top_down = jnp.arange(n_trays)
    _, (L_arr, x_arr) = jax.lax.scan(
        process_liquid,
        (R, x_R),
        tray_indices_top_down,
    )
    L_out = L_arr
    x_out = x_arr

    return V_out, y_out, L_out, x_out


# =============================================================================
# Column State Derivatives
# =============================================================================


def compute_tray_derivatives(
    tray_M: jnp.ndarray,
    tray_x: jnp.ndarray,
    tray_T: jnp.ndarray,
    V_in: jnp.ndarray,
    y_in: jnp.ndarray,
    L_in: jnp.ndarray,
    x_in: jnp.ndarray,
    V_out: jnp.ndarray,
    y_out: jnp.ndarray,
    L_out: jnp.ndarray,
    x_out: jnp.ndarray,
    config: ColumnConfig,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute time derivatives for all trays.

    Args:
        tray_M, tray_x, tray_T: Current tray states.
        V_in, y_in: Vapor inflows to each tray (from below).
        L_in, x_in: Liquid inflows to each tray (from above).
        V_out, y_out: Vapor outflows from each tray (going up).
        L_out, x_out: Liquid outflows from each tray (going down).
        config: Column configuration.

    Returns:
        Tuple of (dM_dt, dx_dt, dT_dt) arrays.
    """
    n_trays = config.geometry.n_trays
    feed_tray_idx = config.geometry.feed_tray - 1

    # Get feed
    L_feed, x_feed, V_feed, y_feed = compute_feed_split(
        config.feed.F, config.feed.z_F, config.feed.q
    )

    def tray_deriv(tray_idx):
        M = tray_M[tray_idx]
        x = tray_x[tray_idx]

        # Flows into this tray
        V_in_tray = V_in[tray_idx]
        y_in_tray = y_in[tray_idx]
        L_in_tray = L_in[tray_idx]
        x_in_tray = x_in[tray_idx]

        # Flows out of this tray
        V_out_tray = V_out[tray_idx]
        y_out_tray = y_out[tray_idx]
        L_out_tray = L_out[tray_idx]
        x_out_tray = x_out[tray_idx]

        # Feed addition at feed tray
        is_feed = (tray_idx == feed_tray_idx)
        F_L = is_feed * L_feed
        F_V = is_feed * V_feed
        z_F = config.feed.z_F

        # Material balance: dM/dt = L_in + V_in + F - L_out - V_out
        dM_dt = L_in_tray + V_in_tray + F_L + F_V - L_out_tray - V_out_tray

        # Component balance: d(Mx)/dt = L_in*x_in + V_in*y_in + F*z_F - L_out*x - V_out*y
        dMx_dt = (
            L_in_tray * x_in_tray
            + V_in_tray * y_in_tray
            + (F_L + F_V) * z_F
            - L_out_tray * x_out_tray
            - V_out_tray * y_out_tray
        )

        # Convert to dx/dt: dx/dt = (dMx/dt - x*dM/dt) / M
        dx_dt = (dMx_dt - x * dM_dt) / jnp.maximum(M, 1e-6)

        # Temperature (quasi-steady: T adjusts to bubble point)
        # For now, use dT/dt = 0 as temperature will be reset to bubble point
        dT_dt = jnp.array(0.0)

        return dM_dt, dx_dt, dT_dt

    # Vectorize over trays
    dM_dt, dx_dt, dT_dt = jax.vmap(tray_deriv)(jnp.arange(n_trays))

    return dM_dt, dx_dt, dT_dt


# =============================================================================
# Main Column Step Function
# =============================================================================


def column_step(
    state: FullColumnState,
    action: ColumnAction,
    config: ColumnConfig,
) -> tuple[FullColumnState, ColumnOutputs]:
    """Perform one timestep of the complete column simulation.

    WARNING: Do NOT call this function in a Python for/while loop!
    That causes massive overhead. Use simulate_column() or simulate_column_jit()
    instead, which use jax.lax.scan for efficient batched execution.

    Args:
        state: Current column state.
        action: Control actions.
        config: Column configuration.

    Returns:
        Tuple of (new_state, outputs).
    """
    dt = config.simulation.dt
    n_substeps = config.simulation.n_substeps
    dt_sub = dt / n_substeps
    P = config.P
    thermo = config.thermo
    hydraulics = config.hydraulics

    # Current state
    tray_M = state.tray_M
    tray_x = state.tray_x
    tray_T = state.tray_T
    tray_L_out = state.tray_L_out
    reboiler = state.reboiler
    condenser = state.condenser
    V_prev = state.V_prev

    def substep(carry, _):
        tray_M, tray_x, tray_T, tray_L_out, reboiler, condenser, V_prev = carry

        # 1. Reboiler step
        # Use dynamic liquid outflow from bottom tray (Francis weir-based)
        L_to_reboiler = tray_L_out[-1]
        x_to_reboiler = tray_x[-1]
        T_to_reboiler = tray_T[-1]

        reboiler_inputs = ReboilerInputs(
            L_in=L_to_reboiler,
            x_in=x_to_reboiler,
            T_in=T_to_reboiler,
            Q_R=action.Q_R,
            B_setpoint=action.B_setpoint,
        )

        new_reboiler, reboiler_outputs = reboiler_step(
            reboiler,
            reboiler_inputs,
            P,
            config.controllers.M_setpoint_reboiler,
            thermo,
            dt_sub,
            config.controllers.Kp_level_reboiler,
        )

        # 2. Preliminary condenser step to get reflux
        # Need to estimate V_in to condenser first
        # Use previous V from top tray as estimate
        V_to_condenser = V_prev[0]

        preliminary_condenser_inputs = CondenserInputs(
            V_in=V_to_condenser,
            y_in=tray_x[0],  # Approximate
            T_V_in=tray_T[0],
            reflux_ratio=action.reflux_ratio,
            D_setpoint=action.D_setpoint,
        )

        _, preliminary_condenser_outputs = condenser_step(
            condenser,
            preliminary_condenser_inputs,
            P,
            config.controllers.M_setpoint_condenser,
            thermo,
            jnp.array(0.0),  # Zero dt for just getting flows
            Kp_level=config.controllers.Kp_level_condenser,
        )

        # Get reflux from condenser
        R = preliminary_condenser_outputs.R
        x_R = preliminary_condenser_outputs.x_R

        # 3. Compute tray flows using CMO approximation
        V_out, y_out, L_out, x_out = compute_tray_flows(
            tray_M,
            tray_x,
            tray_T,
            reboiler_outputs.V_out,
            reboiler_outputs.y_out,
            R,
            x_R,
            config,
        )

        # 4. Condenser step with actual vapor flow
        condenser_inputs = CondenserInputs(
            V_in=V_out[0],
            y_in=y_out[0],
            T_V_in=tray_T[0],
            reflux_ratio=action.reflux_ratio,
            D_setpoint=action.D_setpoint,
        )

        new_condenser, condenser_outputs = condenser_step(
            condenser,
            condenser_inputs,
            P,
            config.controllers.M_setpoint_condenser,
            thermo,
            dt_sub,
            Kp_level=config.controllers.Kp_level_condenser,
        )

        # 5. Update dynamic liquid outflows using Francis weir and hydraulic coupling
        # Compute static L from Francis weir for each tray
        def compute_static_L(M, x, T):
            rho_L = liquid_density(x, T, thermo)
            mw = mixture_molecular_weight(x, thermo)
            return static_liquid_outflow(M, hydraulics, rho_L, mw)

        L_static = jax.vmap(compute_static_L)(tray_M, tray_x, tray_T)

        # Vapor inflows to each tray (from below)
        V_in_all = jnp.concatenate([
            V_out[1:],  # V from trays below
            reboiler_outputs.V_out[None],  # V from reboiler to bottom tray
        ])

        # Update dynamic L with hydraulic coupling
        def update_L_single(L_out_i, L_static_i, V_in_i, V_prev_i):
            return update_liquid_outflow(
                L_out_i, L_static_i, V_in_i, V_prev_i,
                hydraulics.tau_L, hydraulics.j, dt_sub
            )

        new_tray_L_out = jax.vmap(update_L_single)(
            tray_L_out, L_static, V_in_all, V_prev[1:]  # V_prev[1:] matches tray indices
        )

        # 5b. Limit tray liquid outflow to prevent mass creation at low holdup.
        # Analogous to reboiler vapor limiting: liquid outflow cannot drain
        # the tray below a minimum holdup. Conservative limit based only on
        # the tray's own holdup (not inflows) to avoid cascading drain issues
        # when upstream trays are simultaneously limited.
        M_min_tray = jnp.array(0.01)
        L_max_tray = jnp.maximum(
            (tray_M - M_min_tray) / jnp.maximum(dt_sub, 1e-10),
            0.0,
        )
        new_tray_L_out = jnp.minimum(new_tray_L_out, L_max_tray)

        # 5c. Correct reboiler state for mass conservation at boundary.
        # Step 1 used tray_L_out[-1] (pre-update) as reboiler L_in, but
        # the bottom tray's actual outflow is new_tray_L_out[-1] (post-update).
        # Correct the reboiler holdup so mass leaving bottom tray equals
        # mass entering reboiler.
        dL_reb = new_tray_L_out[-1] - tray_L_out[-1]
        M_reb_corrected = new_reboiler.M + dt_sub * dL_reb
        M_reb_corrected = jnp.maximum(M_reb_corrected, 1e-6)

        Mx_reb_corrected = (
            new_reboiler.M * new_reboiler.x
            + dt_sub * dL_reb * tray_x[-1]
        )
        x_reb_corrected = jnp.clip(
            Mx_reb_corrected / jnp.maximum(M_reb_corrected, 1e-6), 0.0, 1.0
        )
        T_reb_corrected = bubble_point_temperature(
            x_reb_corrected, P, thermo, T_init=new_reboiler.T
        )
        new_reboiler = ReboilerState(
            M=M_reb_corrected, x=x_reb_corrected, T=T_reb_corrected
        )

        # 6. Compute tray inflows for material balance
        n_trays = config.geometry.n_trays

        # Vapor inflows (from below)
        V_in = V_in_all
        y_in = jnp.concatenate([
            y_out[1:],
            reboiler_outputs.y_out[None],
        ])

        # Liquid inflows (from above) - use dynamic L values
        L_in = jnp.concatenate([
            condenser_outputs.R[None],  # R from condenser to top tray
            new_tray_L_out[:-1],  # Dynamic L from trays above
        ])
        x_in = jnp.concatenate([
            condenser_outputs.x_R[None],
            x_out[:-1],
        ])

        # 7. Compute tray derivatives using dynamic flows
        dM_dt, dx_dt, dT_dt = compute_tray_derivatives(
            tray_M,
            tray_x,
            tray_T,
            V_in,
            y_in,
            L_in,
            x_in,
            V_out,
            y_out,
            new_tray_L_out,  # Use dynamic L for outflows
            x_out,
            config,
        )

        # 8. Integrate tray states (Euler for substep)
        new_tray_M = tray_M + dt_sub * dM_dt
        new_tray_M = jnp.maximum(new_tray_M, 1e-6)

        new_tray_x = tray_x + dt_sub * dx_dt
        new_tray_x = jnp.clip(new_tray_x, 0.0, 1.0)

        # Update temperatures from bubble point
        def update_temp(x):
            return bubble_point_temperature(x, P, thermo, T_init=jnp.array(350.0))

        new_tray_T = jax.vmap(update_temp)(new_tray_x)

        # Update V_prev for next substep
        new_V_prev = jnp.concatenate([
            V_out[0:1],  # V leaving top tray (to condenser)
            V_in_all,    # V into each tray
        ])

        return (
            new_tray_M, new_tray_x, new_tray_T, new_tray_L_out,
            new_reboiler, new_condenser, new_V_prev
        ), None

    # Run substeps
    (
        tray_M_final, tray_x_final, tray_T_final, tray_L_out_final,
        reboiler_final, condenser_final, V_prev_final
    ), _ = jax.lax.scan(
        substep,
        (tray_M, tray_x, tray_T, tray_L_out, reboiler, condenser, V_prev),
        None,
        length=n_substeps,
    )

    # Create new state
    new_state = FullColumnState(
        tray_M=tray_M_final,
        tray_x=tray_x_final,
        tray_T=tray_T_final,
        tray_L_out=tray_L_out_final,
        reboiler=reboiler_final,
        condenser=condenser_final,
        t=state.t + dt,
        V_prev=V_prev_final,
    )

    # Compute final outputs
    # Get reflux from condenser for flow computation
    preliminary_cond_inputs = CondenserInputs(
        V_in=V_prev_final[0],
        y_in=tray_x_final[0],
        T_V_in=tray_T_final[0],
        reflux_ratio=action.reflux_ratio,
        D_setpoint=action.D_setpoint,
    )
    _, preliminary_cond_outputs = condenser_step(
        condenser_final,
        preliminary_cond_inputs,
        P,
        config.controllers.M_setpoint_condenser,
        thermo,
        jnp.array(0.0),
        Kp_level=config.controllers.Kp_level_condenser,
    )

    # Compute reboiler outputs FIRST to get accurate V_out for flow computation
    reboiler_inputs_for_flows = ReboilerInputs(
        L_in=tray_L_out_final[-1],
        x_in=tray_x_final[-1],
        T_in=tray_T_final[-1],
        Q_R=action.Q_R,
        B_setpoint=action.B_setpoint,
    )
    _, reboiler_outputs_for_flows = reboiler_step(
        reboiler_final,
        reboiler_inputs_for_flows,
        P,
        config.controllers.M_setpoint_reboiler,
        thermo,
        jnp.array(0.0),  # Zero dt just to get outputs
        config.controllers.Kp_level_reboiler,
    )

    # Recompute flows for final state to get accurate outputs
    V_out_final, y_out_final, L_out_final, x_out_final = compute_tray_flows(
        tray_M_final,
        tray_x_final,
        tray_T_final,
        reboiler_outputs_for_flows.V_out,  # Use actual reboiler vapor output
        reboiler_outputs_for_flows.y_out,  # Use actual reboiler vapor composition
        preliminary_cond_outputs.R,
        preliminary_cond_outputs.x_R,
        config,
    )

    # Get final condenser outputs
    condenser_inputs_final = CondenserInputs(
        V_in=V_out_final[0],
        y_in=y_out_final[0],
        T_V_in=tray_T_final[0],
        reflux_ratio=action.reflux_ratio,
        D_setpoint=action.D_setpoint,
    )
    _, condenser_outputs_final = condenser_step(
        condenser_final,
        condenser_inputs_final,
        P,
        config.controllers.M_setpoint_condenser,
        thermo,
        jnp.array(0.0),  # Zero dt just to get outputs
        Kp_level=config.controllers.Kp_level_condenser,
    )

    # Get final reboiler outputs using dynamic liquid flow
    reboiler_inputs_final = ReboilerInputs(
        L_in=tray_L_out_final[-1],  # Use dynamic liquid flow
        x_in=tray_x_final[-1],
        T_in=tray_T_final[-1],
        Q_R=action.Q_R,
        B_setpoint=action.B_setpoint,
    )
    _, reboiler_outputs_final = reboiler_step(
        reboiler_final,
        reboiler_inputs_final,
        P,
        config.controllers.M_setpoint_reboiler,
        thermo,
        jnp.array(0.0),
        config.controllers.Kp_level_reboiler,
    )

    # Compute flooding and weeping indicators for each tray
    def compute_flood_weep(x, T, V):
        """Compute flooding fraction and weeping status for a tray."""
        # Compute vapor composition for density calculation
        y = equilibrium_vapor_composition(x, T, P, thermo)

        # Densities
        rho_L = liquid_density(x, T, thermo)
        rho_V = vapor_density(y, T, P, thermo)

        # Surface tension
        sigma = surface_tension(x, T)

        # Molecular weights for vapor
        mw_V = mixture_molecular_weight(y, thermo)

        # Net area (active area minus downcomer)
        net_area = hydraulics.tray_area - hydraulics.downcomer_area

        # Flooding velocity
        U_nf = flooding_velocity(rho_L, rho_V, sigma, hydraulics.C_sbf)

        # Flooding ratio
        flood_frac = flooding_ratio(V, net_area, rho_V, mw_V, U_nf)

        # Weeping check
        U_min = weep_point_velocity(rho_V, hydraulics.hole_diameter, hydraulics.K2_weep)
        weep = is_weeping(V, net_area, rho_V, mw_V, U_min)

        return flood_frac, weep

    # Vectorize over trays
    flood_frac_arr, weeping_arr = jax.vmap(compute_flood_weep)(
        tray_x_final, tray_T_final, V_out_final
    )

    outputs = ColumnOutputs(
        D=condenser_outputs_final.D,
        x_D=condenser_outputs_final.x_D,
        B=reboiler_outputs_final.B,
        x_B=reboiler_outputs_final.x_B,
        Q_R=action.Q_R,
        Q_C=condenser_outputs_final.Q_C,
        V=V_out_final,
        L=tray_L_out_final,  # Use dynamic liquid flows
        flood_fraction=flood_frac_arr,
        weeping=weeping_arr,
    )

    return new_state, outputs


def make_column_step_fn(config: ColumnConfig):
    """Create a JIT-compilable step function with config baked in.

    This factory function solves the traced value issue with jax.lax.scan
    by capturing static parameters (n_substeps) in a closure rather than
    passing them as arguments.

    Usage:
        config = create_teaching_column_config()
        step_fn = make_column_step_fn(config)
        jit_step = jax.jit(step_fn)
        new_state, outputs = jit_step(state, action)

        # For vectorized simulation:
        @jax.jit
        def run_n_steps(state, action, n_steps):
            def step(s, _):
                return step_fn(s, action), None
            final, _ = jax.lax.scan(step, state, None, length=n_steps)
            return final

    Args:
        config: Column configuration (captured in closure).

    Returns:
        A pure function (state, action) -> (new_state, outputs) that is
        JIT-compilable without traced value errors.
    """
    # Extract static parameters at function creation time (not traced)
    static_params = StaticColumnParams(
        n_trays=config.geometry.n_trays,
        feed_tray=config.geometry.feed_tray,
        n_substeps=config.simulation.n_substeps,
    )

    def step_fn(state: FullColumnState, action: ColumnAction) -> tuple[FullColumnState, ColumnOutputs]:
        """JIT-compilable step function with config in closure."""
        # Call column_step with config from closure
        return column_step(state, action, config)

    return step_fn


# =============================================================================
# Initialization Functions
# =============================================================================


def create_initial_column_state(
    config: ColumnConfig,
    tray_holdup: Optional[float] = None,
    action: Optional[ColumnAction] = None,
) -> FullColumnState:
    """Create initial column state with linear composition profile.

    Args:
        config: Column configuration.
        tray_holdup: Initial liquid holdup per tray [mol]. If None, computes
            a physically appropriate value based on the weir height to ensure
            liquid can flow over the weir (Francis weir requires liquid height
            above the weir crest).
        action: Control action for deriving consistent initial flows. If
            provided, computes V from Q_R and thermodynamics, then derives
            L_above = RR * D and L_below = L_above + q*F for self-consistent
            CMO initialization. If None, uses heuristic V_est = F * 3.2.

    Returns:
        Initial FullColumnState.
    """
    n_trays = config.geometry.n_trays
    P = config.P
    thermo = config.thermo
    z_F = config.feed.z_F

    # Linear composition profile (higher at top, lower at bottom).
    # Use wide spread (±0.45) to approximate the sharp separation
    # profile of a real column, reducing startup transients.
    x_top = jnp.minimum(z_F + 0.45, 0.95)
    x_bottom = jnp.maximum(z_F - 0.45, 0.05)
    tray_x = jnp.linspace(x_top, x_bottom, n_trays)

    # Temperatures from bubble points (computed early for use in holdup calc)
    def get_bubble_T(x):
        return bubble_point_temperature(x, P, thermo, T_init=jnp.array(350.0))

    tray_T = jax.vmap(get_bubble_T)(tray_x)

    # Estimate steady-state liquid flows using CMO assumption.
    if action is not None:
        # Derive V, L from the control action's operating point.
        # At CMO steady state: V = (RR+1)*D, L_above = RR*D.
        # Using action.D_setpoint gives the exact steady-state flows,
        # so L matches R from the condenser at equilibrium. This prevents
        # persistent holdup drift when tau_L is very large.
        V_est = (action.reflux_ratio + 1.0) * action.D_setpoint
        D_est = action.D_setpoint
        L_above = action.reflux_ratio * D_est
    else:
        # Heuristic when action is not known.
        V_est = config.feed.F * 3.2
        D_est = config.feed.F * 0.5
        L_above = jnp.maximum(V_est - D_est, config.feed.F)
    L_below = L_above + config.feed.q * config.feed.F
    feed_idx = config.geometry.feed_tray - 1  # 0-indexed
    tray_indices = jnp.arange(n_trays)
    L_target = jnp.where(tray_indices < feed_idx, L_above, L_below)

    if tray_holdup is None:
        # Compute self-consistent holdups: for each tray, find M such that
        # Francis_weir(M) = L_target. This ensures initial holdups and flows
        # are mutually consistent, avoiding startup transients that would
        # otherwise drain trays or flood the reboiler.
        def _invert_francis_weir(L_target_i, x_i, T_i):
            rho_L = liquid_density(x_i, T_i, thermo)
            mw = mixture_molecular_weight(x_i, thermo)
            C_w = config.hydraulics.weir_coefficient
            L_w = config.hydraulics.weir_length
            A = config.hydraulics.tray_area
            h_weir = config.hydraulics.weir_height
            # L = C_w * L_w * h_ow^1.5 * rho_L / mw
            # => h_ow = (L * mw / (C_w * L_w * rho_L))^(2/3)
            h_ow = jnp.power(
                L_target_i * mw / (C_w * L_w * rho_L),
                2.0 / 3.0,
            )
            h_liquid = h_ow + h_weir
            M = h_liquid * A * rho_L / mw
            return jnp.maximum(M, 1.0)

        tray_M = jax.vmap(_invert_francis_weir)(L_target, tray_x, tray_T)
    else:
        tray_M = jnp.ones(n_trays) * tray_holdup

    # Initialize tray_L_out at CMO steady-state estimates (consistent with holdups)
    tray_L_out = L_target

    # Initialize V_prev (shape: n_trays+1 for vapor from reboiler to all trays)
    V_prev = jnp.ones(n_trays + 1) * V_est

    # Reboiler (at bottom conditions)
    # Use jnp.asarray to ensure JAX compatibility without float() conversion
    reboiler = ReboilerState(
        M=jnp.asarray(config.controllers.M_setpoint_reboiler),
        x=jnp.asarray(x_bottom),
        T=jnp.asarray(tray_T[-1] + 5.0),  # Slightly hotter than bottom tray
    )

    # Condenser (at top conditions)
    condenser = CondenserState(
        M=jnp.asarray(config.controllers.M_setpoint_condenser),
        x=jnp.asarray(x_top),
        T=jnp.asarray(tray_T[0] - 5.0),  # Slightly cooler than top tray
    )

    return FullColumnState(
        tray_M=tray_M,
        tray_x=tray_x,
        tray_T=tray_T,
        tray_L_out=tray_L_out,
        reboiler=reboiler,
        condenser=condenser,
        t=jnp.array(0.0),
        V_prev=V_prev,
    )


def create_default_action(
    Q_R: float = 5000.0,
    reflux_ratio: float = 3.0,
    B_setpoint: float = 0.03,
    D_setpoint: float = 0.02,
) -> ColumnAction:
    """Create default control action.

    Args:
        Q_R: Reboiler duty [W].
        reflux_ratio: Reflux ratio (R/D).
        B_setpoint: Bottoms flow setpoint [mol/s].
        D_setpoint: Distillate flow setpoint [mol/s].

    Returns:
        ColumnAction with specified values.
    """
    return ColumnAction(
        Q_R=jnp.array(Q_R),
        reflux_ratio=jnp.array(reflux_ratio),
        B_setpoint=jnp.array(B_setpoint),
        D_setpoint=jnp.array(D_setpoint),
    )


# =============================================================================
# Simulation Utilities
# =============================================================================


def simulate_column(
    config: ColumnConfig,
    action: ColumnAction,
    n_steps: int,
    initial_state: FullColumnState | None = None,
) -> tuple[FullColumnState, list[ColumnOutputs]]:
    """Run column simulation for multiple steps.

    Args:
        config: Column configuration.
        action: Control action (held constant).
        n_steps: Number of simulation steps.
        initial_state: Initial state (creates default if None).

    Returns:
        Tuple of (final_state, list of outputs).
    """
    if initial_state is None:
        initial_state = create_initial_column_state(config)

    state = initial_state
    outputs_list = []

    for _ in range(n_steps):
        state, outputs = column_step(state, action, config)
        outputs_list.append(outputs)

    return state, outputs_list


def simulate_column_jit(
    config: ColumnConfig,
    action: ColumnAction,
    n_steps: int,
    initial_state: FullColumnState,
) -> tuple[FullColumnState, ColumnOutputs]:
    """JIT-compiled simulation returning final state and outputs.

    Uses lax.scan for efficient compilation.

    Args:
        config: Column configuration.
        action: Control action (held constant).
        n_steps: Number of simulation steps.
        initial_state: Initial state.

    Returns:
        Tuple of (final_state, stacked_outputs).
    """

    def step_fn(state, _):
        new_state, outputs = column_step(state, action, config)
        return new_state, outputs

    final_state, outputs = jax.lax.scan(step_fn, initial_state, None, length=n_steps)

    return final_state, outputs
