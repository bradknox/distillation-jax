---
layout: default
title: Implementation Plan
---

# Phase 2 Implementation Plan: JAX-Based Distillation Column Simulator

## Overview

Implement a high-fidelity, JAX-native distillation column simulator for RL research with sim-to-real transfer capability. The simulator must be fully JIT-compilable, vmap-compatible, and Gymnasium-compatible.

## Critical Files

- `project_plan.md` - Full specification with code examples
- `phase1.md` - Research foundation with all equations and parameters
- `CLAUDE.md` - Project-specific guidelines

---

## Detailed To-Do List

### Stage 0: Project Setup ✅

- [x] Create directory structure (`jax_distillation/` with core/, column/, env/, utils/, validation/ subdirs)
- [x] Create `pyproject.toml` with dependencies (jax, chex, gymnasium, numpy, matplotlib)
- [x] Create all `__init__.py` files
- [x] Create `core/types.py` with JAX-compatible dataclasses (TrayState, ColumnState, ColumnParams, Action)
- [x] Create `tests/` directory structure

### Stage 1: Thermodynamics Module (`core/thermodynamics.py`) ✅

- [x] Implement Antoine equation for vapor pressure (NIST form)
  - Include parameters for: methanol, water, ethanol, benzene, toluene
  - Add temperature range validation/clamping
- [x] Implement NRTL activity coefficient model (binary)
  - $\tau_{12}$, $\tau_{21}$ with temperature dependence
  - $G_{12}$, $G_{21}$ calculations
  - $\ln(\gamma_1)$, $\ln(\gamma_2)$ equations
  - Default params for methanol-water ($\alpha=0.1$)
- [x] Implement ideal Raoult's Law as baseline option
- [x] Implement K-value calculation: $K_k = \gamma_k \cdot P_k^{sat} / P$
- [x] Implement equilibrium vapor composition: $y^* = K_1 x / (K_1 x + K_2(1-x))$
- [x] Implement bubble point solver (bracketing + Newton)
- [x] Implement enthalpy models:
  - Liquid enthalpy: $h_L(x,T)$
  - Vapor enthalpy: $h_V(y,T)$ with heat of vaporization
- [x] Write `tests/test_thermodynamics.py`:
  - Antoine matches NIST values
  - Bubble point converges for known mixtures
  - NRTL matches published VLE points
  - Enthalpy consistency checks

### Stage 2: Hydraulics Module (`core/hydraulics.py`) ✅

- [x] Implement Francis weir formula for liquid outflow
  - $Q_L = C_w \cdot L_w \cdot h_{ow}^{3/2}$
  - Molar conversion: $L = Q_L \cdot \rho_L / MW$
- [x] Implement hydraulic coupling dynamics
  - $dL_{out} = (1/\tau_L) \cdot dM + j \cdot dV_{in}$
  - Parameters: $\tau_L$ (0.5-15s), $j$ (-5 to +5)
- [x] Implement flooding correlation (Fair)
  - $U_{n,f} = C_{sbf} \cdot (\sigma/20)^{0.2} \cdot ((\rho_L - \rho_V)/\rho_V)^{0.5}$
- [x] Implement weeping constraint
  - $U_{min} = (K_2 - 0.9(25.4 - d_h)) / \sqrt{\rho_V}$
- [x] Implement liquid/vapor density calculations
- [x] Write `tests/test_hydraulics.py`:
  - Francis weir matches textbook examples
  - Flow rates physically reasonable
  - Flooding/weeping triggers at correct thresholds

### Stage 3: Single Tray Dynamics (`column/tray.py`) ✅

- [x] Implement total material balance: $dM/dt = L_{in} + V_{in} - L_{out} - V_{out}$ (+ feed if applicable)
- [x] Implement component balance: $d(Mx)/dt$ equation with feed handling
- [x] Implement Murphree vapor efficiency: $y = y_{in} + E_M(y^* - y_{in})$
- [x] Implement energy balance: $dU/dt$ with enthalpy flows
- [x] Create `tray_step(tray_state, inflows, params, dt)` function
- [x] Write `tests/test_tray.py`:
  - Mass conservation
  - Energy conservation
  - Steady state matches analytical
  - Correct time constants

### Stage 4: Reboiler and Condenser ✅

- [x] Implement reboiler model (`column/reboiler.py`):
  - Holdup dynamics: $dM_B/dt = L_N - (V_{N+1} + B)$
  - Component balance with equilibrium vapor
  - Energy balance with $Q_R$ input
- [x] Implement condenser model (`column/condenser.py`):
  - Total condenser assumption
  - Reflux drum dynamics: $dM_D/dt = V_1 - (R + D)$
  - Component and energy balance
- [x] Implement level control (PI controllers for drum/sump)
- [x] Write `tests/test_reboiler.py` and condenser tests:
  - Reaches specified vapor rate
  - Total condensation achieved
  - Level dynamics stable

### Stage 5: Column Assembly (`column/column.py`) ✅

- [x] Create `column/config.py` with ColumnConfig dataclass
  - Geometry: n_trays, feed_tray, column_diameter, weir_height, tray_spacing
  - Mixture selection
  - Operating conditions: feed_rate, feed_composition, feed_temperature
  - Simulation: dt, n_substeps
- [x] Implement tray stacking with correct indexing (1=top, N=bottom)
- [x] Connect condenser at top, reboiler at bottom
- [x] Implement feed tray handling (liquid/vapor split by quality q)
- [x] Implement `column_step(state, action, params, dt)`:
  - Compute all flows
  - Compute MESH derivatives
  - Integrate with RK4
  - Apply physical constraints
- [x] Ensure JIT-compilability and vmap-compatibility
- [x] Create `core/integration.py` with RK4 integrator
- [x] Write `tests/test_column.py`:
  - Reaches steady state from startup
  - Mass balance closes (<0.1% error)
  - Energy balance closes (<1% error)
  - Temperature profile monotonic

### Stage 6: Gymnasium Environment (`env/`) ✅

- [x] Create `env/spaces.py` with action/observation space definitions
- [x] Create `env/rewards.py` with configurable reward components:
  - Product purity tracking
  - Energy minimization
  - Constraint violation penalties
- [x] Create `env/base_env.py` with DistillationColumnEnv:
  - `__init__`: config, params, spaces
  - `reset`: initial state, seeding
  - `step`: action processing, JAX transition, reward, termination
  - `_get_obs`: state to observation
  - `_get_info`: debugging info
- [x] Create `env/wrappers.py` for observation/action transformations
- [x] Write `tests/test_env.py`:
  - Passes gymnasium.utils.env_checker.check_env()
  - Actions have expected effects
  - Rewards align with objectives
  - Termination conditions work

### Stage 7: Validation Suite (`validation/`) ✅

- [x] Create `validation/conservation.py`:
  - Continuous mass balance monitoring
  - Energy balance closure tracking
- [x] Create `validation/steady_state.py`:
  - Total reflux analytical comparison
  - Published simulation result comparison
- [x] Create `validation/dynamic_response.py`:
  - Step response tests (reflux ratio, reboiler duty)
  - Verify correct response directions
  - Time constant validation
- [x] Create `validation/benchmarks.py`:
  - Armfield-style experiments
  - Pressure drop vs boil-up
  - Efficiency vs boil-up
- [x] Run full validation checklist:
  - [x] No NaN/Inf for 10,000+ steps
  - [x] vmap produces identical results
  - [x] JIT compilation succeeds (<30s)
  - [x] Single step <1ms CPU

### Stage 8: Examples and Documentation ✅

- [x] Create `examples/basic_simulation.py` - forward simulation demo
- [x] Create `examples/step_response.py` - step response analysis
- [x] Create `examples/vectorized_sim.py` - parallel simulation demo
- [x] Create `examples/rl_training.py` - basic RL training with stable-baselines3 or PureJaxRL
- [x] Create `README.md` with:
  - Installation instructions
  - Quick start example
  - API overview
  - Validation results summary
- [x] Add docstrings to all public functions
- [x] Create mixture data files (`data/mixtures/`)
- [x] Create column config files (`data/columns/teaching_column.json`)

---

## Verification Plan

1. **Unit tests**: Run `pytest tests/` after each stage ✅ (181 tests passing)
2. **Physical consistency**: Verify mass/energy balance after Stages 3-5 ✅
3. **Gymnasium check**: Run `check_env()` after Stage 6 ✅
4. **Full validation**: Run validation suite after Stage 7 ✅
5. **End-to-end**: Run RL training example after Stage 8 ✅

## Key Constraints

- All core functions must be JIT-compilable (no Python side effects) ✅
- Use chex.dataclass for all state representations ✅
- Fixed-step RK4 for determinism ✅
- $dt_{int} \le 0.1 \cdot \tau_L$ for numerical stability ✅
- Default mixture: methanol-water with NRTL ($\alpha=0.1$) ✅

---

## Implementation Summary

**Completed**: January 2025

**Test Results**: 181 tests passing (160 core + 21 physics/performance)

**Files Created**:
- `jax_distillation/core/types.py` - JAX dataclasses
- `jax_distillation/core/thermodynamics.py` - VLE calculations
- `jax_distillation/core/hydraulics.py` - Tray hydraulics
- `jax_distillation/core/integration.py` - RK4 integrator
- `jax_distillation/column/tray.py` - Single tray dynamics
- `jax_distillation/column/reboiler.py` - Reboiler model
- `jax_distillation/column/condenser.py` - Condenser model
- `jax_distillation/column/config.py` - Configuration dataclasses
- `jax_distillation/column/column.py` - Full column assembly
- `jax_distillation/env/spaces.py` - Gym spaces
- `jax_distillation/env/rewards.py` - Reward functions
- `jax_distillation/env/base_env.py` - Gymnasium environment
- `jax_distillation/env/jax_env.py` - Gymnax-style JAX environment
- `jax_distillation/env/wrappers.py` - Environment wrappers
- `jax_distillation/validation/conservation.py` - Conservation checks
- `jax_distillation/validation/steady_state.py` - Steady state validation
- `jax_distillation/validation/dynamic_response.py` - Dynamic response tests
- `jax_distillation/validation/benchmarks.py` - Performance benchmarks
- `examples/basic_simulation.py` - Basic simulation example
- `examples/step_response.py` - Step response example

**Additional Files Created**:
- `examples/vectorized_sim.py` - Parallel simulation with vmap
- `examples/rl_training.py` - RL training example (stable-baselines3)
- `examples/purejax_training.py` - Pure JAX RL training example
- `README.md` - Project documentation
- `data/mixtures/methanol_water.json` - Methanol-water mixture data
- `data/mixtures/ethanol_water.json` - Ethanol-water mixture data
- `data/columns/teaching_column.json` - Teaching column configuration
- `data/columns/pilot_column.json` - Pilot scale column configuration

**Status**: Phase 2 implementation complete!

---

## Phase 3: Validation Pack

### Summary

Phase 3 added comprehensive validation against public benchmarks:

- **NIST Thermodynamics Validation**: Antoine equation validated against NIST WebBook data
- **Skogestad Column A (COLA) Benchmark**: 40-tray binary distillation column reference
- **Wood-Berry MIMO Benchmark**: Classic 2x2 transfer function model for control validation
- **Debutanizer Delay Wrapper**: Realistic measurement delay environment for RL

### Physics Fixes Applied

- Hydraulic dynamics calibrated to match Wittgens & Skogestad (2000)
- JIT compilation via `make_column_step_fn()` for deterministic, vectorized simulation
- vmap compatibility verified for 64+ parallel environments
- Mass/energy balance closure validated

### Validation Pack Files Created

- `jax_distillation/validation_pack/thermodynamics/nist_validation.py`
- `jax_distillation/validation_pack/benchmarks/skogestad_cola/` - COLA benchmark
- `jax_distillation/validation_pack/benchmarks/wood_berry/` - Wood-Berry benchmark
- `jax_distillation/validation_pack/benchmarks/debutanizer/` - Debutanizer wrapper
- `jax_distillation/validation_pack/fitting/` - Parameter fitting pipeline
- `scripts/run_public_validation.py` - Validation runner
- `scripts/build_credibility_report.py` - Report generator

**Status**: Phase 3 validation pack complete!
