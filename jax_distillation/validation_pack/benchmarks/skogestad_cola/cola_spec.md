# Skogestad Column A (COLA) Benchmark Specification

## Overview

Column A is a standard 40-tray binary distillation column benchmark used extensively
in control-oriented distillation research. It was introduced by Skogestad and has
become a canonical test case for multivariable control studies.

## Source

https://skoge.folk.ntnu.no/book/matlab_m/cola/cola.html

## Citation

```
Skogestad, S. (2007). "The dos and don'ts of distillation column control."
Chemical Engineering Research and Design, 85(1), 13-23.
https://doi.org/10.1205/cherd06133

Skogestad, S. and Postlethwaite, I. (2005). "Multivariable Feedback Control:
Analysis and Design." John Wiley & Sons, 2nd edition.
```

## Column Configuration

| Parameter | Value | Units |
|-----------|-------|-------|
| Number of trays | 40 | - |
| Feed tray | 21 | (from bottom) |
| Total feed rate | 1.0 | mol/s |
| Feed composition | 0.5 | mol fraction (light) |
| Feed quality | 1.0 | (saturated liquid) |
| Relative volatility | 1.5 | - |
| Reflux ratio | ~2.7 | - |
| Boilup ratio | ~2.7 | - |

## Nominal Operating Point

| Variable | Value | Units |
|----------|-------|-------|
| Distillate rate (D) | 0.5 | mol/s |
| Bottoms rate (B) | 0.5 | mol/s |
| Distillate purity (x_D) | 0.99 | mol fraction |
| Bottoms purity (x_B) | 0.01 | mol fraction |
| Reflux flow (L) | 2.7059 | mol/s |
| Vapor flow (V) | 3.2059 | mol/s |

## Model Assumptions

1. **Binary mixture** with constant relative volatility
2. **Constant molar overflow** (CMO) - equimolar overflow
3. **Perfect mixing** on each tray
4. **Negligible vapor holdup** - only liquid holdup considered
5. **Instantaneous vapor-liquid equilibrium**
6. **No heat losses** (adiabatic column)
7. **Constant pressure** throughout the column

## VLE Model

Simple constant relative volatility model:
```
y = (α * x) / (1 + (α - 1) * x)
```

where α = 1.5 (constant).

## Control Configuration

Column A is typically studied with L-V control:
- **Controlled variables**: x_D (distillate purity), x_B (bottoms impurity)
- **Manipulated variables**: L (reflux flow), V (boilup rate)

Alternative configurations studied:
- D-V control
- L-B control
- Ratio control (L/D, V/B)

## Validation Scenarios

### Scenario 1: Steady-State Initialization

Starting from the nominal operating point, verify:
- x_D ≈ 0.99
- x_B ≈ 0.01
- Mass balance: D + B = F

### Scenario 2: Reflux Step Response

Apply a 1% step increase in reflux L:
- x_D should increase (higher purity)
- x_B should increase slightly (coupling effect)
- Dominant time constant: 10-20 minutes

### Scenario 3: Boilup Step Response

Apply a 1% step increase in vapor boilup V:
- x_D should decrease slightly
- x_B should decrease (lower impurity)
- Similar time constants to reflux step

## Expected Behavior

### Temperature Profile
- Monotonically increasing from top (condenser) to bottom (reboiler)
- Feed tray may show a small kink in profile

### Composition Profile
- S-shaped composition profile
- Steepest gradient near feed tray
- Asymptotic approach to product purities at column ends

### Dynamic Response
- MIMO system with significant interaction
- Response to reflux change: primarily affects x_D
- Response to boilup change: primarily affects x_B
- Cross-coupling effects present but smaller than main effects

## Acceptance Criteria for Validation

1. **Steady-state accuracy**: x_D, x_B within 5% of published values
2. **Step response direction**: Correct sign of change
3. **Time constant order**: Within factor of 2 of expected values
4. **NRMSE < 10%** for dynamic trajectories (or documented explanation)

## Notes for JAX Implementation

The JAX simulator uses a different VLE model (NRTL activity coefficients
vs. constant α), so some quantitative differences are expected. The key
is to verify:

1. Qualitative behavior matches
2. Correct physical trends
3. Similar time scales
4. Mass/energy conservation

When differences exist, they should be documented with explanations
related to model differences (not implementation bugs).
