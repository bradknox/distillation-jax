# Wood-Berry Distillation Column Benchmark Specification

## Overview

The Wood-Berry model is a classic 2x2 MIMO (Multiple-Input Multiple-Output)
transfer function model representing the linearized dynamics of a binary
distillation column. It is one of the most widely used benchmarks for
multivariable control system design.

## Source

```
Wood, R.K. and Berry, M.W. (1973). "Terminal composition control of a
binary distillation column." Chemical Engineering Science, 28(9), 1707-1717.
https://doi.org/10.1016/0009-2509(73)80002-0
```

## Transfer Function Model

The Wood-Berry model relates two inputs to two outputs via a 2x2 transfer function matrix:

### Inputs (Manipulated Variables)
- **R**: Reflux flow rate (deviation from nominal)
- **S**: Steam flow rate (deviation from nominal)

### Outputs (Controlled Variables)
- **x_D**: Distillate composition (deviation from nominal)
- **x_B**: Bottoms composition (deviation from nominal)

### Transfer Functions

```
[x_D(s)]   [G11(s)  G12(s)] [R(s)]
[x_B(s)] = [G21(s)  G22(s)] [S(s)]
```

Where:

```
G11(s) = 12.8 * exp(-s) / (16.7s + 1)
G12(s) = -18.9 * exp(-3s) / (21.0s + 1)
G21(s) = 6.6 * exp(-7s) / (10.9s + 1)
G22(s) = -19.4 * exp(-3s) / (14.4s + 1)
```

### Parameters Table

| Transfer | Gain (K) | Time Constant (τ) | Dead Time (θ) |
|----------|----------|-------------------|---------------|
| G11      | 12.8     | 16.7 min          | 1.0 min       |
| G12      | -18.9    | 21.0 min          | 3.0 min       |
| G21      | 6.6      | 10.9 min          | 7.0 min       |
| G22      | -19.4    | 14.4 min          | 3.0 min       |

## Physical Interpretation

### Gain Signs
- **G11 > 0**: Increasing reflux increases distillate purity (correct)
- **G12 < 0**: Increasing steam decreases distillate purity (coupling, less reflux effect)
- **G21 > 0**: Increasing reflux increases bottoms impurity (coupling, less stripping)
- **G22 < 0**: Increasing steam decreases bottoms impurity (correct)

### Time Constants
- G11, G12: ~17-21 min (rectifying section dynamics)
- G21, G22: ~11-14 min (stripping section dynamics)

### Dead Times
- G11: 1 min (fast reflux effect on distillate)
- G12, G22: 3 min (steam affects compositions with delay)
- G21: 7 min (reflux effect on bottoms takes longest)

## Control-Relevant Properties

### Relative Gain Array (RGA)
```
Λ = [2.01   -1.01]
    [-1.01   2.01]
```

The RGA suggests:
- Strong interaction between loops
- 1-1/2-2 pairing (R→x_D, S→x_B) is preferred
- Diagonal elements > 1 indicates interaction will make control harder

### Condition Number
The system has a high condition number (~40), indicating:
- Ill-conditioning at some frequencies
- Sensitivity to model uncertainty
- Challenge for multivariable control design

## Validation Scenarios

### Scenario 1: Step in Reflux (R)
- Apply unit step to R at t=0
- Expected responses:
  - x_D: rises after 1 min delay, settles to +12.8 with τ ≈ 16.7 min
  - x_B: rises after 7 min delay, settles to +6.6 with τ ≈ 10.9 min

### Scenario 2: Step in Steam (S)
- Apply unit step to S at t=0
- Expected responses:
  - x_D: drops after 3 min delay, settles to -18.9 with τ ≈ 21.0 min
  - x_B: drops after 3 min delay, settles to -19.4 with τ ≈ 14.4 min

## Acceptance Criteria for JAX Simulator Comparison

Since the JAX simulator is a nonlinear model and Wood-Berry is linearized:

1. **Gain direction**: Signs must match (all 4 transfer functions)
2. **Gain magnitude**: Within factor of 2-3 of Wood-Berry gains
3. **Time constant order**: Similar order of magnitude (factor of 2-3)
4. **Coupling structure**: x_D primarily responds to R, x_B primarily responds to S
5. **NRMSE**: Document any mismatch with explanation

## Notes for Comparison

The JAX simulator will differ from Wood-Berry because:

1. **Nonlinearity**: Wood-Berry is linearized; JAX model is nonlinear
2. **Operating point**: Wood-Berry coefficients are for a specific point
3. **Thermodynamics**: Different VLE models
4. **Column geometry**: Different number of trays

The key is verifying that the JAX simulator exhibits similar MIMO
coupling structure and responds in the correct directions to inputs.
Exact numerical agreement is not expected.

## References

1. Wood, R.K. and Berry, M.W. (1973). Original paper.
2. Skogestad, S. and Postlethwaite, I. (2005). "Multivariable Feedback Control."
3. Ogunnaike, B.A. and Ray, W.H. (1994). "Process Dynamics, Modeling, and Control."
