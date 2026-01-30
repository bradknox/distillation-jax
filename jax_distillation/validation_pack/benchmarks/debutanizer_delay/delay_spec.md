# Debutanizer Delay Wrapper Specification

## Overview

Industrial distillation columns typically have significant measurement delays
for composition analysis due to:

1. **Transport delay**: Time for sample to travel from column to analyzer
2. **Analysis delay**: Time for gas chromatograph or other analyzer to process sample

This wrapper adds realistic delayed composition observations to the
Gymnasium environment for RL training that accounts for these delays.

## Reference

```
Fortuna, L., Graziani, S., Rizzo, A., and Xibilia, M.G. (2007).
"Soft Sensors for Monitoring and Control of Industrial Processes."
Springer-Verlag London. ISBN 978-1-84628-479-3.
```

## Typical Delay Characteristics

### Debutanizer Column (Industrial)
- Gas chromatograph cycle time: 15-30 minutes
- Transport delay: 1-5 minutes
- Total delay: 15-35 minutes

### Teaching Column (Laboratory)
- Simplified or on-line analyzers: 1-5 minutes
- Transport delay: negligible
- Total delay: 1-5 minutes

## Wrapper Features

### 1. Fixed Dead Time
```python
delay_config = DelayConfig(
    dead_time=900.0,  # 15 minutes in seconds
    sample_period=900.0,  # Update every 15 minutes
)
```

### 2. Variable Dead Time
```python
delay_config = DelayConfig(
    dead_time_mean=900.0,
    dead_time_std=60.0,  # ±1 minute variation
)
```

### 3. Sample-and-Hold
The analyzer provides discrete measurements at fixed intervals:
- Measurement available at t = t_sample
- Measurement reflects composition at t = t_sample - dead_time
- Measurement held constant until next sample

### 4. Optional Noise
```python
delay_config = DelayConfig(
    measurement_noise_std=0.001,  # Analyzer precision
)
```

### 5. Missing Measurements
```python
delay_config = DelayConfig(
    missing_probability=0.05,  # 5% chance of failed analysis
)
```

## Observation Space Modification

The wrapper modifies the observation space to include:

1. **delayed_x_D**: Last available distillate composition measurement
2. **delayed_x_B**: Last available bottoms composition measurement
3. **time_since_last_update**: Time since last composition measurement
4. **next_update_in**: Time until next measurement (if known)

The "true" compositions (x_D, x_B) are still available in the info dict
for training analysis but should not be used by the policy.

## Implementation Notes

### Buffer Management
- Circular buffer stores recent composition history
- Buffer size = max_dead_time / dt
- At each step, query buffer for appropriately delayed value

### Determinism
- With fixed RNG seed, delay behavior is reproducible
- Important for RL training reproducibility

### Edge Cases
- At simulation start: no measurement available yet
  - Option 1: Return NaN (explicit uncertainty)
  - Option 2: Return initial guess (e.g., feed composition)
  - Option 3: Wait until first measurement (delay first obs)

## Acceptance Criteria

1. **Delay Correctness**: Measurement at time t reflects composition at time t - θ
2. **Sample-and-Hold**: Measurement constant between updates
3. **Determinism**: Same seed → same trajectory
4. **Gymnasium API**: Wrapper passes env_checker
5. **Performance**: Minimal overhead (<10% slowdown)

## Usage Example

```python
from jax_distillation.env import DistillationColumnEnv
from jax_distillation.validation_pack.benchmarks.debutanizer_delay import DelayWrapper, DelayConfig

# Create base environment
env = DistillationColumnEnv()

# Wrap with delay
delay_config = DelayConfig(
    dead_time=900.0,  # 15 minutes
    sample_period=900.0,
)
env = DelayWrapper(env, delay_config)

# Use as normal Gymnasium env
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(action)

# True values available in info for analysis
true_x_D = info["true_x_D"]
delayed_x_D = obs[...]["delayed_x_D"]
```

## Research Context

The debutanizer delay benchmark is relevant for:

1. **Soft sensor development**: Predicting composition from secondary measurements
2. **Model predictive control**: Compensating for measurement delays
3. **Reinforcement learning**: Learning policies robust to delayed feedback
4. **State estimation**: Designing observers for delayed systems

This wrapper enables training RL agents that can handle realistic
industrial measurement delays, a key step toward sim-to-real transfer.
