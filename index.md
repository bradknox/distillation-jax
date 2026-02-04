---
layout: default
title: Home
---

# Distillation JAX

JAX-based distillation column simulator with NRTL thermodynamics and Gymnasium RL environment.

## Documentation

- [Research Foundation](docs/phase1) - Dynamic model specification with all equations
- [Implementation Plan](docs/implementation_plan) - Detailed implementation roadmap
- [Handoff Documentation](HANDOFF) - Usage guide for RL practitioners and chemical engineers

## Example LaTeX Rendering

Inline math: The experiment uses a feed with composition $z_F = 0.5$ mol fraction.

Display math (material balance):

$$
\frac{dM_i}{dt} = L_{i-1} + V_{i+1} - L_i - V_i
$$

The NRTL activity coefficient model:

$$
\ln \gamma_1 = x_2^2 \left[ \tau_{21} \left( \frac{G_{21}}{x_1 + x_2 G_{21}} \right)^2 + \frac{\tau_{12} G_{12}}{(x_2 + x_1 G_{12})^2} \right]
$$
