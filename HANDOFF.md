# JAX Distillation Simulator - Handoff Documentation

## Overview

This document provides guidance for two audiences: RL practitioners interested in using the simulator for training, and chemical engineers who need to understand the physical model. The JAX distillation simulator provides a JIT-compilable, vectorizable implementation of a binary distillation column suitable for reinforcement learning research.

---

## For RL Practitioners

### Using DistillationEnvJax with vmap for Parallel Training

The pure JAX environment (`DistillationEnvJax`) is designed for high-throughput parallel training:

```python
from examples.purejax_training import DistillationEnvJax
import jax
import jax.numpy as jnp

# Create environment
env = DistillationEnvJax()

# Initialize single environment
key = jax.random.PRNGKey(0)
obs, state = env.reset(key)

# For parallel environments, use vmap
n_envs = 1024
keys = jax.random.split(key, n_envs)

# Vectorized reset
v_reset = jax.vmap(env.reset)
obs_batch, state_batch = v_reset(keys)

# Vectorized step
v_step = jax.vmap(env.step)
action_batch = jnp.zeros((n_envs, 4))  # 4 action dimensions
obs_batch, state_batch, reward_batch, done_batch, info_batch = v_step(
    keys, state_batch, action_batch
)
```

### Using DistillationColumnEnv with stable-baselines3

For compatibility with stable-baselines3 and other Gymnasium-based libraries:

```python
from jax_distillation.envs.gymnasium_env import DistillationColumnEnv
from stable_baselines3 import PPO

# Create Gymnasium-compatible environment
env = DistillationColumnEnv()

# Standard SB3 training
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)
```

### Throughput Expectations

- **Pure JAX (single GPU):** ~50,000-100,000 env-steps/sec with vmap batch size 1024
- **Gymnasium wrapper:** ~1,000-5,000 env-steps/sec (CPU-bound due to Python overhead)
- **JIT compilation:** First step takes ~10-30s for compilation, subsequent steps are fast

### Example Training Commands

```bash
# Pure JAX training example
python examples/purejax_training.py --n_envs 1024 --n_steps 1000000

# Gymnasium-compatible training with SB3
python examples/sb3_training.py --algo ppo --timesteps 1000000
```

### Action Space

The environment uses 4 continuous actions:
1. `Q_R`: Reboiler duty [W] (normalized)
2. `reflux_ratio`: Reflux ratio L/D
3. `B_setpoint`: Bottoms flow setpoint [mol/s] (normalized)
4. `D_setpoint`: Distillate flow setpoint [mol/s] (normalized)

### Observation Space

Observations include:
- Tray temperatures (n_trays values)
- Reboiler and condenser states (holdups, compositions, temperatures)
- Product compositions (x_D, x_B)
- Current flows (D, B)

### Column Outputs (for direct simulation)

When using `column_step()` directly, the `ColumnOutputs` namedtuple includes:
- `D`, `x_D`: Distillate flow and composition
- `B`, `x_B`: Bottoms flow and composition
- `Q_R`, `Q_C`: Reboiler and condenser duties
- `V`, `L`: Vapor and liquid flow profiles
- `flood_fraction`: Fraction of flooding velocity per tray (0-1+, >0.85 indicates near-flooding)
- `weeping`: Boolean array indicating weeping condition per tray

**For RL:** The `flood_fraction` and `weeping` outputs can be used to add constraint penalties or early termination when the column approaches operational limits.

---

## For Chemical Engineers

### Model Assumptions and Limitations

**Physical Model:**
- Binary mixture separation (methanol-water by default)
- NRTL activity coefficient model for VLE (α=0.1)
- Antoine equation for vapor pressure (NIST parameters)
- Francis weir for liquid overflow hydraulics
- Kettle reboiler with level control
- Total condenser with reflux drum

**Simplifications:**
- Constant molar overflow (CMO) approximation for vapor flow
- Negligible vapor holdup on trays
- Constant pressure (typically 1 bar)
- Murphree tray efficiency (adjustable, default 1.0)
- Simplified energy balance (enthalpy flow without heat losses)

**Hydraulic Model:**
- Francis weir equation for liquid outflow
- Configurable hydraulic time constant τ_L (default: 3.0 s)
- Vapor-liquid coupling parameter j (default: 0.0)

### Interpreting Validation Results

See `artifacts/credibility_report.md` for detailed validation results. Key points:

1. **Numerical Verification:** Mass balance closure 0.069% (PASS, target < 0.1%)
2. **Thermodynamic Validation:** Antoine equations validated against NIST WebBook data
3. **COLA Benchmark:** Qualitative agreement; quantitative differences due to different VLE models (NRTL vs constant α=1.5)
4. **Temperature Profile:** Monotonically increasing from top to bottom at steady state
5. **Numerical Stability:** No NaN/Inf over 10,000+ timesteps
6. **Step Response:** Correct signs (↑reflux → ↑x_D, ↑reboiler duty → ↓x_B)

### How to Fit Parameters When Plant Data Is Available

1. **Collect Data:**
   - Steady-state temperatures along column
   - Inlet/outlet flows and compositions
   - Step response data (e.g., reflux step → composition response)

2. **Run Fitting Pipeline:**
   ```bash
   python scripts/run_fit_pipeline_demo.py --data path/to/plant_data.csv
   ```

3. **Fittable Parameters:**
   - Murphree tray efficiency (η)
   - Hydraulic time constant (τ_L)
   - NRTL binary interaction parameters (if mixture differs)
   - Heat loss coefficient (if significant)

4. **Validation:**
   - Compare model predictions to held-out test data
   - Check mass/energy balance closure
   - Verify step response dynamics match plant

### Credibility Statement

**What IS validated:**
- Mass balance closure (numerical)
- VLE calculations against NIST reference data
- Qualitative behavior against Skogestad Column A benchmark
- Basic control response directions

**What is NOT validated:**
- Absolute accuracy against any specific physical column
- Hydraulic correlations for specific tray geometry
- Tray efficiency for specific packing/plates
- Heat loss coefficients

See `artifacts/credibility_report.md` for the full credibility assessment.

---

## For All Users

### How to Run Validation Suite

```bash
# Quick validation (subset of tests)
python scripts/run_public_validation.py --quick

# Full validation suite
pytest tests/validation_pack/ -v

# Generate credibility report
python scripts/build_credibility_report.py
```

### Troubleshooting Common Issues

**Issue:** "ConcretizationTypeError" during JAX compilation
- **Cause:** Dynamic shapes or control flow with traced values
- **Fix:** Ensure all array shapes are static; use `jax.lax.cond` for control flow

**Issue:** Simulation blows up (NaN/Inf values)
- **Cause:** Numerical instability, usually from extreme inputs
- **Fix:** Check action bounds; reduce timestep; increase n_substeps

**Issue:** Slow first step
- **Cause:** JIT compilation
- **Fix:** This is expected; subsequent steps are fast

**Issue:** Mass balance error > 20%
- **Cause:** Simulation not at true steady state; hydraulic configuration issues
- **Fix:** Run longer; check initial conditions; verify hydraulic parameters match geometry

### Key Files

```
jax_distillation/
├── column/
│   ├── column.py        # Main column simulation
│   ├── reboiler.py      # Reboiler model
│   ├── condenser.py     # Condenser model
│   └── config.py        # Configuration dataclasses
├── core/
│   ├── thermodynamics.py # VLE calculations
│   └── hydraulics.py    # Weir and flooding correlations
├── envs/
│   ├── gymnasium_env.py  # Gymnasium wrapper
│   └── purejax_env.py   # Pure JAX environment
└── validation_pack/      # Validation benchmarks

artifacts/
├── credibility_report.md # Validation summary

scripts/
├── build_credibility_report.py
├── run_public_validation.py
└── run_fit_pipeline_demo.py
```

### Contact / Issue Reporting

Report issues at: https://github.com/anthropics/claude-code/issues

---

## Known Issues (Resolved)

The following issues from earlier development have been resolved:

1. ~~**COLA benchmark mass balance:** Current configuration achieves ~50% mass balance error.~~
   - **RESOLVED:** Mass balance closure is now 0.069% (well under the 0.1% target)

2. ~~**Temperature profile non-monotonicity:** Temperature profile may not be strictly monotonic.~~
   - **RESOLVED:** Temperature profiles are now monotonically increasing from top to bottom at steady state

3. ~~**Numerical stability:** Simulation can become unstable with certain parameter combinations.~~
   - **RESOLVED:** Stable for 10,000+ timesteps with default configuration

---

## Current Limitations

The following items are known limitations in the current implementation:

1. **Energy balance closure:** Energy tracking is implemented but may show larger errors during transients due to simplified enthalpy model. At steady state, errors should be < 1%.

2. **Flooding/weeping integration:** Detection functions are now integrated into `ColumnOutputs.flood_fraction` and `ColumnOutputs.weeping`. However, these are informational outputs only - the dynamics are not modified when flooding/weeping occurs (this is a modeling choice, not a bug).

3. **NRMSE values:** Wood-Berry and COLA NRMSE values show moderate mismatch (0.3-0.6) due to differences between our full nonlinear NRTL model and the linearized/simplified reference models. This is expected - qualitative behavior (gain signs, response direction) is the key validation.
