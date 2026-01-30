# JAX-Based Distillation Column Simulator for Reinforcement Learning

## Overview

This document contains two prompts for building a high-fidelity, JAX-native distillation column simulator compatible with Gymnasium for reinforcement learning research.

**Phase 1** is a research prompt for any capable LLM with web search capabilities. It produces a structured research document.

**Phase 2** is an implementation prompt for Claude Code. It contains placeholders `[FROM PHASE 1: ...]` that should be filled in with findings from Phase 1.

-----

# PHASE 1: Literature Review & Research Plan

## Prompt for Research Phase

You are a research assistant helping to plan the development of a high-fidelity distillation column simulator. The simulator will be used for reinforcement learning research, specifically for training control policies that can transfer to real physical columns.

Your task is to produce a comprehensive research document that will inform the implementation. This document must be thorough enough that a software engineer with RL expertise but limited chemical engineering background can implement the simulator correctly.

### Research Questions to Answer

#### 1. Mathematical Modeling of Distillation Dynamics

Search for and synthesize information on the following:

**1.1 Stage-by-Stage Dynamic Models**

- What are the MESH equations (Material balance, Equilibrium, Summation, Heat balance)?
- How are they formulated for dynamic (not steady-state) simulation?
- What simplifications are commonly used for binary distillation?
- What is the typical state vector for a single tray? For the full column?

**1.2 Thermodynamic Models**

- For binary mixtures, what vapor-liquid equilibrium (VLE) models are standard?
  - Raoult’s Law (ideal)
  - Modified Raoult’s Law with activity coefficients
  - Wilson, NRTL, UNIQUAC equations
- How is relative volatility (α) defined and used?
- What are the Antoine equation parameters for common teaching mixtures (ethanol-water, methanol-water, benzene-toluene)?

**1.3 Hydraulic Models**

- What equations govern liquid holdup on trays?
- What is the Francis weir formula?
- How do vapor and liquid flow rates interact (flooding, weeping constraints)?
- What are typical tray efficiency models (Murphree efficiency)?

**1.4 Energy Balance**

- How is the reboiler modeled dynamically?
- How is the condenser modeled?
- What are typical assumptions about heat losses?

**1.5 Time Scales**

- What are typical time constants for:
  - Hydraulic dynamics (liquid level changes)?
  - Composition dynamics?
  - Thermal dynamics?
- What timestep is appropriate for numerical integration?
- What are the stiffness considerations?

#### 2. Column Configurations and Parameters

**2.1 Typical Teaching/Lab Column Specifications**

- Search for specifications of undergraduate teaching distillation columns
- What are typical values for:
  - Number of trays (5-20 for teaching columns?)
  - Feed tray location
  - Column diameter
  - Tray spacing
  - Weir height and length
  - Reboiler and condenser capacities

**2.2 Parameterization for Sim-to-Real Transfer**

- What parameters need to be fit from real column data?
- What experiments/data collection procedures are standard for column identification?
- What are typical ranges for uncertain parameters?

#### 3. Control and Operating Modes

**3.1 Standard Control Configurations**

- What are the typical manipulated variables?
  - Reflux ratio or reflux flow rate
  - Reboiler duty or vapor boilup rate
  - Distillate rate
  - Bottoms rate
  - Feed rate (if variable)
- What are the standard control structures (LV, DV, RR configurations)?
- What are typical control valve dynamics?

**3.2 Operating Regimes**

- How is startup typically performed?
- What defines steady-state operation?
- What are typical disturbances (feed composition changes, throughput changes)?
- What are the constraint boundaries (flooding, weeping, product specs)?

**3.3 Measurements and Sensors**

- What is typically measured on real columns?
- What are realistic sensor delays and noise characteristics?
- Where are temperature sensors typically placed?
- What is the delay on composition analyzers (gas chromatograph)?

#### 4. Existing Simulators and Benchmarks

**4.1 Academic Simulators**

- Search for open-source distillation simulators
- What languages/frameworks are they implemented in?
- What modeling assumptions do they make?
- Are any JAX-based or differentiable?

**4.2 Commercial Simulators**

- What are the standard commercial tools (Aspen, HYSYS)?
- What validation data is available from these?

**4.3 Benchmark Problems**

- Are there standard benchmark distillation control problems in the literature?
- What test cases are used for validating dynamic simulators?

#### 5. Validation Approaches

**5.1 Qualitative Validation**

- What dynamic behaviors should a valid simulator exhibit?
  - Response to step changes in manipulated variables
  - Startup transients
  - Approach to steady state
- What are known failure modes of incorrect models?

**5.2 Quantitative Validation**

- What experimental datasets are publicly available?
- What accuracy metrics are standard (% error on compositions, temperatures)?
- What is “good enough” accuracy for control system design?

**5.3 Physical Consistency Checks**

- Mass balance closure
- Energy balance closure
- Thermodynamic consistency
- Constraint satisfaction (no negative compositions, physically realizable flows)

#### 6. RL-Specific Considerations

**6.1 Reward Function Design**

- What are the control objectives for distillation?
  - Product purity specifications
  - Energy minimization
  - Throughput maximization
  - Constraint satisfaction
- How are these typically weighted in practice?

**6.2 Safety Constraints**

- What operating constraints must never be violated?
- How are these handled in practice (hard vs soft constraints)?

**6.3 Sim-to-Real Transfer**

- What domain randomization approaches are used for process control?
- What parameters should be randomized during training?
- What adaptation approaches are used when deploying to real columns?

### Output Format

Structure your research document with the following sections:

```
# Distillation Column Simulator: Research Foundation

## Executive Summary
[2-3 paragraphs summarizing key findings and recommendations]

## 1. Dynamic Model Specification
### 1.1 State Variables
[Complete specification of state vector with physical units]

### 1.2 MESH Equations for Binary Distillation
[Full mathematical formulation with all equations numbered]

### 1.3 Thermodynamic Closure
[VLE equations, recommended models for different mixtures]

### 1.4 Hydraulic Equations
[Liquid holdup, flow relationships]

### 1.5 Boundary Conditions
[Reboiler and condenser models]

## 2. Recommended Default Parameters
### 2.1 Column Geometry
[Table of default parameters with physical justification]

### 2.2 Thermodynamic Parameters
[Antoine coefficients, activity coefficient parameters for default mixture]

### 2.3 Operating Conditions
[Typical ranges for all inputs]

## 3. Numerical Considerations
### 3.1 Stiffness Analysis
[Discussion of time scales and integration requirements]

### 3.2 Recommended Integration Scheme
[Specific recommendations for JAX implementation]

### 3.3 Timestep Selection
[Guidance on choosing simulation timestep]

## 4. Control Interface Specification
### 4.1 Action Space
[Complete specification of manipulated variables]

### 4.2 Observation Space
[Complete specification of measurements]

### 4.3 Constraint Specification
[All operating constraints with numerical bounds]

## 5. Validation Protocol
### 5.1 Unit Tests
[Specific tests for individual components]

### 5.2 Integration Tests
[Tests for coupled behavior]

### 5.3 Benchmark Comparisons
[Specific published results to reproduce]

### 5.4 Physical Consistency Checks
[Automated checks to include]

## 6. Key References
[Annotated bibliography of essential sources]

## Appendix A: Complete Equation Set
[All equations in implementable form]

## Appendix B: Parameter Tables
[All parameters with units, typical values, and uncertainty ranges]
```

### Search Strategy

Use web search to find:

1. Academic papers on dynamic distillation modeling (terms: “dynamic distillation simulation”, “distillation column dynamics”, “MESH equations dynamic”)
1. Textbook references (terms: “Skogestad distillation control”, “Luyben distillation”, “Seader separation processes”)
1. Open-source implementations (terms: “distillation simulator github”, “open source process simulation”)
1. Experimental data (terms: “distillation column experimental data”, “pilot plant distillation data”)
1. RL for process control (terms: “reinforcement learning distillation”, “RL chemical process control”)

Be thorough. The implementation will depend entirely on the accuracy and completeness of this research document.

-----

# PHASE 2: Claude Code Implementation Prompt

## Prompt for Implementation Phase

You are building a high-fidelity, JAX-native distillation column simulator for reinforcement learning research. This simulator must be:

1. **Fully JIT-compilable and vmappable** for GPU-accelerated parallel simulation
1. **Gymnasium-compatible** with standard RL interfaces
1. **Physically accurate** based on established chemical engineering models
1. **Parameterizable** to fit any specific binary distillation column
1. **Extensible** to support future enhancements (multi-component, partial observability, etc.)

### Research Foundation

---

# Distillation Column Simulator: Research Foundation

**Phase 1 Handoff Document (complete and self-contained)**

## Executive Summary

Your goal (train an agent in simulation that transfers to a real column) is primarily a **model fidelity and uncertainty-management** problem, not an “RL algorithm” problem. The best modeling option for transfer is a **dynamic equilibrium-stage (MESH) model with energy balance and realistic tray hydraulics**, plus **non-ideal vapor–liquid equilibrium (VLE)** via an activity-coefficient model (NRTL), and **realistic sensors/actuators** (delays, noise, saturations). This combination is widely used in rigorous dynamic column models and directly targets the phenomena that practitioners say matter for control-relevant dynamics, especially the “initial response” and operability limits. ([Sigurd Skogestad][1])

A key practical takeaway from the distillation dynamics literature is that **tray hydraulics (liquid holdup, a hydraulic time constant, and a vapor–liquid coupling term)** strongly shape the short-time response that affects pressure/level/temperature stabilization loops, and therefore affects what a learned policy experiences during transients. A well-known study reports typical ranges of **hydraulic time constant 0.5–15 s** and vapor coupling parameter **j in roughly −5 to +5** (dimensionless in their linearized formulation). ([Sigurd Skogestad][1]) This is exactly the kind of “small” effect that can break sim-to-real if omitted, because RL will exploit the simulator’s transient quirks.

Finally, for transfer you should treat the simulator as a **calibratable model class**: you will (i) implement the physics, (ii) fit the uncertain parameters from step tests and steady-state data, and (iii) train with **domain randomization** over the remaining uncertainty (and sensor/actuator imperfections), so the learned policy is robust rather than brittle.

---

## 1. Dynamic Model Specification

### 1.1 State Variables (with units)

Assume a binary mixture: **light component** (index 1) and **heavy component** (index 2). A column has **N trays**, a **total condenser + reflux drum** at the top, and a **kettle reboiler** at the bottom.

A high-fidelity, control-oriented state for tray *i* (i = 1…N) is:

* (M_i) — **liquid holdup** on tray *i* [mol]
* (x_i) — **liquid mole fraction of light component** in tray holdup [mol fraction]
* (U_i) — **internal energy** of the tray control volume [J] (or equivalently (T_i) as a state; see §1.2/§1.3)

Recommended top/bottom vessels:

* Reflux drum (accumulator): (M_D) [mol], (x_D) [mol fraction], (U_D) [J]
* Reboiler: (M_B) [mol], (x_B) [mol fraction], (U_B) [J]

Optional but recommended for best transfer (especially for “initial response”):

* A hydraulic coupling parameterization that represents how **liquid outflow responds to holdup and vapor flow** (see §1.4). ([Sigurd Skogestad][1])

**Algebraic (non-state) variables** computed each step:

* (L_i) — liquid flow leaving tray *i* downward [mol/s]
* (V_i) — vapor flow leaving tray *i* upward [mol/s]
* (y_i) — vapor mole fraction of light component leaving tray *i* [mol fraction]
* (P_i) — tray pressure [Pa] or [bar] (often modeled as fixed profile or fixed top + pressure drops)

---

### 1.2 MESH Equations for Binary Distillation (dynamic form; implementable)

This is the standard **MESH** structure:
**M**aterial balances (total + component), **E**quilibrium (VLE), **S**ummation, **H**eat (energy).

Number trays from top to bottom: tray 1 is top tray below the condenser, tray N is above the reboiler.

#### Notation for flows (consistent indexing)

* Liquid flows downward: (L_{i}) leaves tray *i* and enters tray *i+1*
* Vapor flows upward: (V_{i}) leaves tray *i* and enters tray *i−1*
* Thus vapor entering tray *i* from below is (V_{i+1}) with composition (y_{i+1}).
* Liquid entering tray *i* from above is (L_{i-1}) with composition (x_{i-1}).

Let the feed enter tray *f* at total molar rate (F) [mol/s], overall light fraction (z_F), and **quality** (q\in[0,1]) (fraction of feed that enters as liquid). Then:

* Liquid feed part: (F_L = qF) enters liquid phase
* Vapor feed part: (F_V = (1-q)F) enters vapor phase

(If you want higher fidelity, compute q from feed enthalpy via a flash; but quality is a workable parameter.)

---

#### 1.2.1 Total material balance on tray i (Equation 1)

For i = 1…N, with feed only if i = f:

[
\frac{dM_i}{dt} = L_{i-1} + \mathbb{1}*{i=f}F_L + V*{i+1} + \mathbb{1}_{i=f}F_V - L_i - V_i
\tag{1}
]

If you assume vapor holdup is negligible (common in many control-oriented models), then (M_i) is liquid holdup only and Eq. (1) simplifies by dropping vapor holdup dynamics but retaining vapor flow terms as throughputs. (Wittgens & Skogestad discuss both rigorous holdup accounting and simplified forms.) ([Sigurd Skogestad][1])

---

#### 1.2.2 Component material balance (light component) on tray i (Equation 2)

[
\frac{d(M_i x_i)}{dt} = L_{i-1}x_{i-1} + \mathbb{1}*{i=f}F_L z_F + V*{i+1}y_{i+1} + \mathbb{1}_{i=f}F_V z_F - L_i x_i - V_i y_i
\tag{2}
]

This is the core composition propagation equation used for dynamic tray models. ([Sigurd Skogestad][1])

---

#### 1.2.3 Energy balance on tray i (Equation 3)

A rigorous formulation uses internal energy as the differential state. Wittgens & Skogestad write tray energy balances in terms of internal energy with inlet/outlet enthalpy flows. ([Sigurd Skogestad][1]) A practical implementable form is:

[
\frac{dU_i}{dt} =
L_{i-1} h_L(x_{i-1},T_{i-1})

* \mathbb{1}_{i=f}F_L h_F^{(L)}
* V_{i+1} h_V(y_{i+1},T_{i+1})
* \mathbb{1}_{i=f}F_V h_F^{(V)}

- L_i h_L(x_i,T_i)
- V_i h_V(y_i,T_i)

* Q_i
  \tag{3}
  ]

Where:

* (h_L(x,T)) is **molar liquid enthalpy** [J/mol]
* (h_V(y,T)) is **molar vapor enthalpy** [J/mol]
* (Q_i) is external heat to tray i [W]; typically (Q_i=0) for trays (heat losses handled separately if desired)

**Implementation note:** you can either:

1. integrate (U_i) and solve for (T_i) each step via an “energy inversion” (e.g., Newton solve on (U_i - U(x_i,T_i)=0)), or
2. integrate (T_i) directly using an effective heat capacity model (simpler numerically, but be consistent).

Wittgens’ rigorous stage model uses internal energy states and equilibrium flash assumptions. ([Sigurd Skogestad][1])

---

#### 1.2.4 Equilibrium (VLE) closure (Equation 4)

For each tray, compute equilibrium K-values:

[
K_k = \frac{\gamma_k(x,T) , P_k^{sat}(T)}{P}
\quad\text{for }k\in{1,2}
\tag{4}
]

Assuming ideal vapor phase (often acceptable for low pressures; refine later if needed).

For a binary mixture, vapor composition in equilibrium with liquid is:

[
y_1^{*} = \frac{K_1 x}{K_1 x + K_2(1-x)}
,\qquad y_2^{*}=1-y_1^{*}
\tag{5}
]

---

#### 1.2.5 Murphree vapor efficiency (recommended non-equilibrium correction) (Equation 6)

Real trays are not perfectly equilibrated. A widely used, calibratable correction is **Murphree vapor efficiency** (E_{M,i}\in(0,1]). Define vapor entering tray i from below as (y_{in}=y_{i+1}). Then:

[
y_i = y_{in} + E_{M,i}\bigl(y_i^{*} - y_{in}\bigr)
\tag{6}
]

This is attractive for sim-to-real because (E_{M,i}) is **identifiable from data** (e.g., from steady-state composition/temperature profiles and known reflux/boilup conditions) and can be randomized across plausible ranges.

---

#### 1.2.6 Summation (Equation 7)

Binary summations are enforced implicitly:

[
x_2 = 1-x_1,\qquad y_2 = 1-y_1
\tag{7}
]

---

### 1.3 Thermodynamic Closure (recommended models + concrete default parameters)

#### 1.3.1 Vapor pressure: Antoine equation (Equation 8)

Use the NIST Antoine form (as presented on NIST Chemistry WebBook pages):

[
\log_{10}!\bigl(P^{sat}[\mathrm{bar}]\bigr)= A - \frac{B}{T[\mathrm{K}] + C}
\tag{8}
]

**Default component Antoine parameters (examples commonly used in teaching/lab columns):**

* **Methanol** (valid 288.10–356.83 K): (A=5.20409,; B=1581.341,; C=-33.50) ([NIST WebBook][2])
* **Water** (valid 344.00–373.00 K): (A=5.08354,; B=1663.125,; C=-45.622) ([NIST WebBook][3])
* **Ethanol** (valid 292.77–366.63 K): (A=5.24677,; B=1598.673,; C=-46.424) ([NIST WebBook][4])
* **Benzene** (valid 287.70–354.07 K): (A=4.01814,; B=1203.835,; C=-53.226)
* **Toluene** (valid 308.52–384.66 K): (A=4.07827,; B=1343.943,; C=-53.773) ([NIST WebBook][5])

**Implementation note:** enforce the valid temperature ranges during testing; outside-range extrapolation can destabilize the simulator. Clamp or switch coefficient sets if you extend ranges.

---

#### 1.3.2 Activity coefficients (γ): recommended hierarchy

1. **Ideal Raoult’s law:** (\gamma_1=\gamma_2=1).
   Use for near-ideal systems (many hydrocarbon pairs) or as an initial baseline.

2. **Non-ideal (recommended for sim-to-real): NRTL model.**
   NRTL (Non-Random Two-Liquid) is widely used, stable for many polar mixtures, and parameterized with a small number of constants.

**NRTL equations (binary form; Equation 9–11)**

Let (\tau_{12}, \tau_{21}) be dimensionless interaction parameters and (\alpha) be the non-randomness parameter (commonly 0.1–0.3; IUPAC example uses 0.1). Define:

[
G_{12}=\exp(-\alpha \tau_{12}),\qquad G_{21}=\exp(-\alpha \tau_{21})
\tag{9}
]

For a binary mixture, the activity coefficients are:

[
\ln \gamma_1 = x_2^2\left[\tau_{21}\left(\frac{G_{21}}{x_1 + x_2 G_{21}}\right)^2

* \tau_{12}\frac{G_{12}}{(x_2 + x_1 G_{12})^2}\right]
  \tag{10}
  ]

[
\ln \gamma_2 = x_1^2\left[\tau_{12}\left(\frac{G_{12}}{x_2 + x_1 G_{12}}\right)^2

* \tau_{21}\frac{G_{21}}{(x_1 + x_2 G_{21})^2}\right]
  \tag{11}
  ]

(These are standard NRTL binary expressions; implement carefully and test against published VLE points.)

**Temperature dependence:** a practical common form is:

[
\tau_{12}(T)=A_{12}+\frac{B_{12}}{T},\qquad
\tau_{21}(T)=A_{21}+\frac{B_{21}}{T}
\tag{12}
]

---

#### 1.3.3 Default non-ideal mixture parameters (high-value for teaching columns)

Armfield’s UOP3 teaching column explicitly lists **methanol–water** among standard mixtures. ([Armfield][6]) For methanol + water, an IUPAC supporting-information example specifies **NRTL with α = 0.1** and (\tau) parameters of the form (A + B/T): ([iupac.org][7])

* Components: 1 = methanol, 2 = water
* (\alpha = 0.1)
* (\tau_{12}(T)= 9.23811 + (-2432.61)/T)
* (\tau_{21}(T)= -5.70743 + (1538.74)/T)
  with (T) in Kelvin. ([iupac.org][7])

(These are “dimensionless interaction parameters” as presented in the IUPAC supporting info. Implement exactly as in Eq. 12.)

---

#### 1.3.4 Relative volatility α (definition; optional simplification)

Relative volatility between light (1) and heavy (2):

[
\alpha_{rel} = \frac{K_1}{K_2}
\tag{13}
]

In simplified teaching models, (\alpha_{rel}) is sometimes treated as constant, yielding:

[
y_1^{*}=\frac{\alpha_{rel} x}{1 + (\alpha_{rel}-1)x}
\tag{14}
]

**Recommendation for your project:** do **not** use constant relative volatility as the main model if you want sim-to-real; keep it only as a debugging/benchmark mode.

---

#### 1.3.5 Bubble-point / dew-point calculations (algorithm)

You will often need tray temperature consistent with composition at pressure.

**Bubble-point condition (Equation 15):**

[
P = x_1\gamma_1(x,T)P_1^{sat}(T) + x_2\gamma_2(x,T)P_2^{sat}(T)
\tag{15}
]

**Algorithm (robust bracketing + Newton):**

1. Choose a temperature bracket ([T_{low},T_{high}]) such that (f(T)=\sum x_k\gamma_k P_k^{sat}(T)-P) changes sign.
2. Use bisection for a few iterations to guarantee bracketing.
3. Switch to Newton (or secant) using derivative approximated by finite difference; clamp steps to remain within bracket.

This approach is stable in JAX (pure functional; no side effects) and avoids divergence that can crash long RL runs.

---

#### 1.3.6 Enthalpy models (sufficient for control-relevant fidelity; extensible)

A pragmatic starting point (that supports energy balances and is calibratable) is:

Liquid enthalpy:
[
h_L(x,T)= x,c_{p,L,1}(T-T_{ref}) + (1-x),c_{p,L,2}(T-T_{ref})
\tag{16}
]

Vapor enthalpy:
[
h_V(y,T)= y\left[\Delta h_{vap,1} + c_{p,V,1}(T-T_{ref})\right]
+(1-y)\left[\Delta h_{vap,2} + c_{p,V,2}(T-T_{ref})\right]
\tag{17}
]

Where:

* (c_p) are molar heat capacities [J/mol/K]
* (\Delta h_{vap}) are heats of vaporization [J/mol] at a reference (or temperature-dependent if you later upgrade)

**Why this is acceptable for transfer (initially):** you can fit effective (c_p) and (\Delta h_{vap}) to match measured temperature dynamics and steady-state heat duties. If you later need more accuracy, you can replace these with DIPPR/NIST correlations without changing the simulator architecture.

---

### 1.4 Hydraulic Equations (holdup, weirs, constraints, and vapor–liquid coupling)

A simulator intended for control transfer should include at least:

* liquid holdup dynamics,
* liquid outflow dependence on holdup,
* flooding/weeping constraints,
* and (ideally) coupling between vapor flow and liquid flow.

#### 1.4.1 Tray hydraulics: why it matters

Wittgens & Skogestad explicitly argue that simplified models that neglect hydraulics and energy balance are often criticized by practitioners, and that realistic hydraulics enables operability and control studies. ([Sigurd Skogestad][1])

#### 1.4.2 Linearized “control-relevant” tray hydraulic law (recommended; Equation 18)

A widely used approximation is:

[
dL_{out} = \frac{1}{\tau_L} dM + j, dV_{in}
\tag{18}
]

Where:

* (\tau_L) is a **hydraulic time constant** [s]
* (j) captures the **initial effect of vapor flow on liquid flow** (dimensionless in this deviation form)

Typical ranges reported: (\tau_L \approx 0.5) to (15) s and (j\in[-5,5]). ([Sigurd Skogestad][1])

**Implementation pattern:** represent (L_i) (or (L_{out,i})) as an auxiliary dynamic variable with its own first-order response to the “static” weir prediction, plus the vapor coupling term. This improves realism of short-time transients without requiring a full computational-fluid model.

#### 1.4.3 Weir/overflow relation (for “static” liquid outflow)

Use a sharp-crested weir relation (often referred to as a Francis-type correlation). NPTEL uses a standard weir form in the context of tray design calculations. ([NPTEL][8])

A practical implementable approach:

1. compute a clear liquid height over the weir (h_{ow}) [m] from tray holdup (via geometry and density),
2. compute volumetric liquid flow:
   [
   Q_L = C_w, L_w, h_{ow}^{3/2}
   \tag{19}
   ]
3. convert to molar flow:
   [
   L = Q_L \frac{\rho_L}{\overline{MW}}
   \tag{20}
   ]

Where:

* (L_w) is weir length [m]
* (\rho_L) is liquid density [kg/m³]
* (\overline{MW}) is mixture molecular weight [kg/mol]
* (C_w) is a weir coefficient (set by tray design; calibrate)

If you want to avoid “deriving” (h_{ow}) from geometry at first, you can parameterize (L) as a monotone function of holdup and fit parameters from step tests; but for best fidelity, implement geometry.

#### 1.4.4 Flooding constraint (Fair correlation) (Equation 21)

NPTEL gives Fair’s correlation for flooding gas velocity through net area: ([NPTEL][8])

[
U_{n,f} = C_{sbf}\left(\frac{\sigma}{20}\right)^{0.2}
\left(\frac{\rho_L-\rho_V}{\rho_V}\right)^{0.5}
\tag{21}
]

Where:

* (U_{n,f}) = flooding superficial gas velocity [m/s]
* (C_{sbf}) = capacity parameter (depends on tray spacing and flow parameter)
* (\sigma) = surface tension [mN/m]
* (\rho_L,\rho_V) = liquid and vapor densities

Design guidance: operate at ~80–85% of flooding for non-foaming liquids. ([NPTEL][8])

**Simulator use:** treat flooding as a hard constraint (terminate episode) or as a steep penalty once (U_n/U_{n,f}) exceeds ~0.85–0.9.

#### 1.4.5 Weeping constraint (weep-point velocity) (Equation 22)

NPTEL provides a weeping criterion via a minimum vapor velocity at the weep point: ([NPTEL][8])

[
U_{min} = \frac{K_2 - 0.9(25.4 - d_h)}{\sqrt{\rho_V}}
\tag{22}
]

Where:

* (d_h) is hole diameter [mm]
* (K_2) is a tray-specific coefficient (from design correlations)

**Simulator use:** penalize operation where (U_n < U_{min}) (loss of contacting efficiency, possible “dumping”).

#### 1.4.6 Tray efficiency dependency

Tray efficiency depends on flow regimes (flooding/weeping) and design; Armfield explicitly lists experiments measuring “column efficiency as a function of boil-up rate” under total reflux. ([Armfield][6])

**Practical sim-to-real strategy:** model (E_{M,i}) as:

* nominal tray efficiency (fit from data),
* degraded as flooding/weeping are approached (smooth function),
* and randomized during training within uncertainty bands.

---

### 1.5 Boundary Conditions (condenser, reflux drum, reboiler)

#### 1.5.1 Total condenser + reflux drum

Armfield’s UOP3 includes an overhead condenser and reflux tank, with reflux control from 0–100%. ([Armfield][6])

Model a **total condenser**: all incoming vapor condenses to liquid in the drum.

Let vapor from tray 1 enter condenser at (V_1, y_1). Then condensed liquid composition is approximately (x_D \approx y_1) (for total condensation).

Reflux drum holdup dynamics:
[
\frac{dM_D}{dt} = V_1 - (R + D)
\tag{23}
]

Component balance:
[
\frac{d(M_D x_D)}{dt} = V_1 y_1 - (R + D)x_D
\tag{24}
]

Energy balance:
[
\frac{dU_D}{dt} = V_1 h_V(y_1,T_1) - (R+D)h_L(x_D,T_D) - Q_C
\tag{25}
]

Where (Q_C) [W] is condenser duty (heat removed). Often you will specify condenser outlet temperature or pressure and solve for required (Q_C); for RL, (Q_C) is usually not a manipulated variable.

Reflux to tray 1: (L_0 = R), composition (x_0=x_D), temperature (T_0=T_D).

#### 1.5.2 Reboiler (kettle type)

Reboiler holdup dynamics:
[
\frac{dM_B}{dt} = L_N - (V_{N+1} + B)
\tag{26}
]

Component balance:
[
\frac{d(M_B x_B)}{dt} = L_N x_N - V_{N+1} y_B - B x_B
\tag{27}
]

Energy balance:
[
\frac{dU_B}{dt} = L_N h_L(x_N,T_N) - V_{N+1}h_V(y_B,T_B) - B h_L(x_B,T_B) + Q_R
\tag{28}
]

Where:

* (Q_R) [W] is reboiler duty (a primary manipulated variable)
* (y_B) is vapor composition in equilibrium with reboiler liquid (use Eq. 5)

---

## 2. Recommended Default Parameters

### 2.1 Column Geometry (default: Armfield-like lab column)

Armfield UOP3 plate column: **50 mm diameter**, **8 sieve plates**, **downcomers**, and a temperature sensor at each plate with very small sheath diameter for rapid response. ([Armfield][6])

A good default configuration (explicitly intended as “teaching column” scale) is:

| Parameter                  |    Symbol |         Default | Units | Notes / justification                                                              |
| -------------------------- | --------: | --------------: | ----- | ---------------------------------------------------------------------------------- |
| Number of trays            |       (N) |               8 | –     | UOP3 has 8 sieve plates ([Armfield][6])                                            |
| Column internal diameter   |     (D_c) |            0.05 | m     | UOP3 plate column is 50 mm diameter ([Armfield][6])                                |
| Tray spacing               |       (S) |       0.20–0.30 | m     | Not stated on page; treat as configurable; affects flooding correlation and holdup |
| Weir height                |     (h_w) |       0.01–0.03 | m     | Typical lab tray weirs; treat as configurable and fit if known                     |
| Weir length                |     (L_w) | 0.6–0.9 × (D_c) | m     | Approximate; geometry-dependent                                                    |
| Downcomer area fraction    | (A_d/A_t) |       0.08–0.15 | –     | Typical tray design range; fit if known                                            |
| Hole diameter (sieve tray) |     (d_h) |             2–5 | mm    | Needed for weeping constraint                                                      |
| Operating pressure range   |       (P) |         0.2–1.0 | bar   | UOP3 supports reduced pressure down to 200 mbar ([Armfield][6])                    |

For sim-to-real you should replace defaults with the real column’s design drawings when available.

---

### 2.2 Thermodynamic Parameters (default mixture: methanol–water with NRTL)

**Why this default:** methanol–water is explicitly listed as a standard mixture for the lab column platform. ([Armfield][6]) It is non-ideal, so it forces the simulator architecture to support activity coefficients (needed for transfer across many real mixtures).

**Antoine coefficients:** see §1.3.1 for methanol and water. ([NIST WebBook][2])

**NRTL parameters (methanol=1, water=2):**

* (\alpha=0.1)
* (\tau_{12}(T)= 9.23811 - 2432.61/T)
* (\tau_{21}(T)= -5.70743 + 1538.74/T) ([iupac.org][7])

---

### 2.3 Operating Conditions (inputs; typical ranges)

Because real columns differ widely, the key is to define **safe, physically plausible bounds** and then fit them to your real equipment.

For an Armfield-like teaching column (based on listed capabilities): ([Armfield][6])

* Feed volumetric rate: 0–0.25 L/min (given for a peristaltic pump) ([Armfield][6])
  Convert to mol/s using mixture density and molecular weight.
* Reflux control: 0–100% return (interpretable as a reflux fraction or reflux ratio depending on plumbing) ([Armfield][6])
* Pressure: 0.2–1.0 bar (vacuum down to 200 mbar) ([Armfield][6])
* Column maximum operating temperature: at least 130 °C ([Armfield][6])

For RL, define action bounds in the same units and limits as the real actuators (or their setpoints).

---

## 3. Numerical Considerations

### 3.1 Stiffness analysis (time scales)

Distillation columns have **multiple time scales**:

* **Hydraulic dynamics:** seconds (liquid levels/holdup responding to flow changes). Reported hydraulic time constants can be **0.5–15 s**. ([Sigurd Skogestad][1])
* **Thermal and composition propagation:** typically slower (tens of seconds to minutes on lab columns; longer on industrial columns).
* **Control-relevant point:** if you want transfer to a real system with regulatory loops and transient constraints, the simulator must reproduce short-time hydraulic responses, not only steady state. ([Sigurd Skogestad][1])

This combination can create **stiffness** (fast and slow dynamics coupled), especially if you explicitly model holdup and energy.

### 3.2 Recommended integration scheme (JAX implementation)

For RL training you usually want a **fixed-step** integrator (for determinism, JIT compilation, and vectorization). Recommended progression:

1. Start with **Runge–Kutta 4 (RK4)** fixed-step for stability at moderate steps.
2. If stiffness causes instability, use:

   * smaller internal substeps, or
   * a semi-implicit treatment for the fastest hydraulic states (e.g., treat liquid outflow response with a stable first-order update).

### 3.3 Timestep selection (practical guidance)

Define two timesteps:

* **Simulator internal timestep** (dt_{int}): chosen to resolve hydraulics. A conservative rule is (dt_{int} \le 0.1\tau_L). With (\tau_L) as low as 0.5 s, this suggests (dt_{int}\approx 0.05) s in the worst case. ([Sigurd Skogestad][1])
* **RL environment timestep** (dt_{env}): 1–10 s (or larger), implemented by taking multiple internal steps per environment step.

This structure preserves physical fidelity while keeping RL rollouts efficient.

---

## 4. Control Interface Specification

### 4.1 Action space (best option for sim-to-real)

For transfer, the most realistic and safest interface is usually **supervisory control**: the RL policy sets **setpoints** (or setpoint biases) for existing regulatory loops, rather than directly commanding valves.

A strong default action vector:

1. (RR_{sp}) — reflux ratio setpoint (or reflux flow setpoint)
2. (Q_{R,sp}) — reboiler duty setpoint (or boilup setpoint)

And keep two internal level controllers (conventional proportional–integral loops) to regulate:

* reflux drum level via distillate flow (D)
* reboiler/sump level via bottoms flow (B)

This reflects common practice: levels are regulated locally for safety, while composition/temperature objectives are handled by higher-level control.

**Actuator dynamics:** represent each manipulated flow/duty as a first-order lag with saturation:
[
u_{actual}(t+dt)=u_{actual}(t)+\frac{dt}{\tau_u}\left(\text{clip}(u_{cmd},u_{min},u_{max})-u_{actual}(t)\right)
\tag{29}
]

### 4.2 Observation space (realistic sensors)

For an Armfield-like lab column, temperatures on each plate are explicitly measured with fast-response sensors. ([Armfield][6])

Recommended observation vector:

* Tray temperatures (T_1,\dots,T_N) (or a subset if you want partial observability)
* Reflux drum level (or holdup) (M_D)
* Reboiler level (or holdup) (M_B)
* Column pressure drop (Armfield lists a differential manometer top-to-bottom) ([Armfield][6])
* Key flows: reflux, distillate, bottoms, feed, boilup (as measured or inferred)

**Sensor dynamics:** model temperature/level sensors with a first-order filter and additive noise. Armfield highlights rapid dynamic response of plate temperature sensors (very small sheath). ([Armfield][6])

### 4.3 Constraint specification (numerical bounds)

Include at least:

* **Flooding constraint:** terminate or steep-penalize when (U_n/U_{n,f}\gtrsim 0.85–0.9). ([NPTEL][8])
* **Weeping constraint:** penalize when (U_n < U_{min}). ([NPTEL][8])
* **Level constraints:** (M_D) and (M_B) must stay within safe bounds (avoid dry-out/overflow).
* **Composition bounds:** enforce (x_i,y_i\in[0,1]) and nonnegative holdups.
* **Thermal bounds:** enforce max temperature consistent with equipment (Armfield notes at least 130 °C maximum inside column). ([Armfield][6])

---

## 5. Validation Protocol

### 5.1 Unit tests (module-level)

Thermodynamics:

* Antoine vapor pressures reproduce NIST values within a small tolerance across the stated temperature ranges. ([NIST WebBook][2])
* Bubble point solver satisfies Eq. (15) to a tight residual tolerance.
* NRTL activity coefficients match published reference VLE points for methanol–water at a few temperatures (spot checks). ([iupac.org][7])

Hydraulics:

* Flooding velocity calculation reproduces NPTEL example calculations for given densities/surface tension. ([NPTEL][8])
* Weeping threshold behaves correctly as a function of hole diameter and vapor density. ([NPTEL][8])
* Linear hydraulics law uses plausible (\tau_L) and (j) ranges and produces reasonable initial responses. ([Sigurd Skogestad][1])

### 5.2 Integration tests (column-level)

* **Mass balance closure:** total moles in column + products track net inputs (tight tolerance).
* **Energy balance closure:** heat input/removal reconciles internal energy change + enthalpy flows (within tolerance; allow small numerical drift).
* **Monotonic temperature profile:** qualitatively correct (for typical binary separations, temperature increases down the column at steady operation).
* **Step responses:** increase reflux should (generally) increase top purity; increase reboiler duty should increase vapor flow and strengthen separation (directional tests).

### 5.3 Benchmark comparisons (practical targets)

Armfield lists experiments you can reproduce qualitatively and (if you have access to the unit) quantitatively:

* pressure drop vs boil-up,
* efficiency vs boil-up at total reflux,
* response of top composition under constant reflux ratio,
* mass balance across the system. ([Armfield][6])

Even if you do not use Armfield hardware, these define realistic validation behaviors for a teaching-scale column.

### 5.4 Physical consistency checks (automated “always-on”)

At every step:

* (0\le x_i \le 1), (0\le y_i \le 1)
* (M_i, M_D, M_B \ge 0)
* No NaN/Inf
* Flooding/weeping indicators computed and recorded
* Enforce action/actuator saturations

---

## 6. Key References (annotated)

1. **Wittgens & Skogestad (2000)**, *Evaluation of Dynamic Models of Distillation Columns with Emphasis on the Initial Response.*
   Core reference for why hydraulics matters, and for a rigorous stage model structure and control-relevant hydraulic parameters ((\tau_L), (j)). ([Sigurd Skogestad][1])

2. **NPTEL Chemical Engineering Design II, Module 7 (tray design)**
   Provides implementable flooding and weeping correlations (Fair correlation; weep-point velocity). ([NPTEL][8])

3. **NIST Chemistry WebBook** (Antoine vapor pressure parameters)
   Source for Antoine coefficients with explicit temperature ranges. ([NIST WebBook][2])

4. **Armfield UOP3 Distillation Columns** (teaching column specs and instrumentation)
   Provides a concrete lab-scale configuration: 50 mm diameter, 8 sieve plates, per-plate temperature sensors with rapid response, reflux 0–100%, reduced pressure operation. ([Armfield][6])

5. **IUPAC supporting info (example VLE parameterization)**
   Provides NRTL parameters for methanol–water (and another pair) in a directly implementable form. ([iupac.org][7])

---

## Appendix A: Complete Equation Set (ready to implement)

For tray i = 1…N:

1. Total holdup:
   [
   \frac{dM_i}{dt} = L_{i-1} + \mathbb{1}*{i=f}qF + V*{i+1} + \mathbb{1}_{i=f}(1-q)F - L_i - V_i
   ]

2. Light component:
   [
   \frac{d(M_i x_i)}{dt} = L_{i-1}x_{i-1} + \mathbb{1}*{i=f}qF z_F + V*{i+1}y_{i+1} + \mathbb{1}_{i=f}(1-q)F z_F - L_i x_i - V_i y_i
   ]

3. Energy:
   [
   \frac{dU_i}{dt} =
   L_{i-1} h_L(x_{i-1},T_{i-1})

* \mathbb{1}_{i=f}qF h_F^{(L)}
* V_{i+1} h_V(y_{i+1},T_{i+1})
* \mathbb{1}_{i=f}(1-q)F h_F^{(V)}

- L_i h_L(x_i,T_i)
- V_i h_V(y_i,T_i)
  ]

4. Vapor pressure (Antoine):
   [
   \log_{10}!\bigl(P_k^{sat}[\mathrm{bar}]\bigr)= A_k - \frac{B_k}{T[\mathrm{K}] + C_k}
   ]

5. NRTL:
   [
   \tau_{12}=A_{12}+B_{12}/T,\quad \tau_{21}=A_{21}+B_{21}/T,\quad
   G_{12}=e^{-\alpha\tau_{12}},\quad G_{21}=e^{-\alpha\tau_{21}}
   ]
   [
   (\ln\gamma_1,\ln\gamma_2)\ \text{from Eqs. (10–11)}
   ]

6. K-values:
   [
   K_k=\gamma_k P_k^{sat}/P
   ]

7. Equilibrium vapor composition:
   [
   y_1^{*}=\frac{K_1 x}{K_1 x + K_2(1-x)}
   ]

8. Murphree efficiency:
   [
   y_i = y_{i+1} + E_{M,i}(y_i^{*}-y_{i+1})
   ]

9. Hydraulic coupling (recommended):
   [
   dL_{out} = \frac{1}{\tau_L} dM + j, dV_{in}
   ]

Top condenser + reflux drum:

[
\frac{dM_D}{dt} = V_1 - (R + D),\quad
\frac{d(M_D x_D)}{dt} = V_1 y_1 - (R + D)x_D
]

Bottom reboiler:

[
\frac{dM_B}{dt} = L_N - (V_{N+1}+B),\quad
\frac{d(M_B x_B)}{dt} = L_N x_N - V_{N+1} y_B - B x_B
]

---

## Appendix B: Parameter Tables (defaults + uncertainty ranges for domain randomization)

**Hydraulics (control-relevant):**

* (\tau_L): 0.5–15 s ([Sigurd Skogestad][1])
* (j): −5 to +5 ([Sigurd Skogestad][1])
* (E_{M,i}): start 0.5–0.9 (fit later; randomize per tray)

**Teaching column (Armfield-like):**

* (D_c = 0.05) m, (N=8), downcomers, per-tray temperature sensors, reflux 0–100%, pressure down to 0.2 bar ([Armfield][6])

**Methanol–water NRTL:**

* (\alpha=0.1)
* (\tau_{12}(T)= 9.23811 - 2432.61/T)
* (\tau_{21}(T)= -5.70743 + 1538.74/T) ([iupac.org][7])

**Flooding/weeping constraints:**

* Flooding velocity from Eq. (21) (Fair correlation) ([NPTEL][8])
* Weep-point from Eq. (22) ([NPTEL][8])

---

# Next step recommendation (what to do after handing this to Phase 2)

Proceed to Phase 2 implementation exactly as your plan outlines, **but require two non-negotiables from the implementer**:

1. **Implement energy balance + non-ideal VLE (NRTL) from day 1**, using methanol–water as the default mixture (because you can cite parameters and it stresses the thermo stack). ([iupac.org][7])
2. **Implement hydraulic initial-response realism** via (\tau_L) and (j) (Eq. 18), and enforce flooding/weeping constraints (Eqs. 21–22). ([Sigurd Skogestad][1])

If either is skipped, you should assume sim-to-real will likely fail for transient behaviors, because RL will exploit the simulator’s incorrect short-time dynamics.

---

[1]: https://skoge.folk.ntnu.no/publications/2000/Wittgens2000/Wittgens2000.pdf "https://skoge.folk.ntnu.no/publications/2000/Wittgens2000/Wittgens2000.pdf"
[2]: https://webbook.nist.gov/cgi/cbook.cgi?ID=C67561&Mask=4 "https://webbook.nist.gov/cgi/cbook.cgi?ID=C67561&Mask=4"
[3]: https://webbook.nist.gov/cgi/cbook.cgi?ID=C71432&Mask=4&Plot=on&Type=ANTOINE "https://webbook.nist.gov/cgi/cbook.cgi?ID=C71432&Mask=4&Plot=on&Type=ANTOINE"
[4]: https://webbook.nist.gov/cgi/cbook.cgi?ID=C64175&Mask=4&Plot=on&Type=ANTOINE "https://webbook.nist.gov/cgi/cbook.cgi?ID=C64175&Mask=4&Plot=on&Type=ANTOINE"
[5]: https://webbook.nist.gov/cgi/cbook.cgi?ID=C108883&Mask=4&Plot=on&Type=ANTOINE "https://webbook.nist.gov/cgi/cbook.cgi?ID=C108883&Mask=4&Plot=on&Type=ANTOINE"
[6]: https://armfield.co.uk/product/uop3-distillation-columns/ "https://armfield.co.uk/product/uop3-distillation-columns/"
[7]: https://iupac.org/wp-content/uploads/2017/09/2011-037-2-100_SupportingInfo1-VLE.pdf "https://iupac.org/wp-content/uploads/2017/09/2011-037-2-100_SupportingInfo1-VLE.pdf"
[8]: https://archive.nptel.ac.in/content/storage2/courses/103103027/pdf/mod7.pdf "https://archive.nptel.ac.in/content/storage2/courses/103103027/pdf/mod7.pdf"

### Project Structure

Create the following directory structure:

```
jax_distillation/
├── README.md
├── pyproject.toml
├── jax_distillation/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── thermodynamics.py      # VLE, activity coefficients, enthalpy
│   │   ├── hydraulics.py          # Tray hydraulics, holdup, flows
│   │   ├── mass_balance.py        # Component material balances
│   │   ├── energy_balance.py      # Heat balances
│   │   ├── integration.py         # ODE integration utilities
│   │   └── types.py               # JAX-compatible dataclasses for state
│   ├── column/
│   │   ├── __init__.py
│   │   ├── tray.py                # Single tray dynamics
│   │   ├── reboiler.py            # Reboiler model
│   │   ├── condenser.py           # Condenser model
│   │   ├── column.py              # Full column assembly
│   │   └── config.py              # Column configuration dataclass
│   ├── env/
│   │   ├── __init__.py
│   │   ├── base_env.py            # Core Gymnasium environment
│   │   ├── wrappers.py            # Observation/action wrappers
│   │   ├── rewards.py             # Reward function components
│   │   └── spaces.py              # Action/observation space definitions
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── units.py               # Unit conversion utilities
│   │   ├── visualization.py       # Plotting utilities
│   │   └── data_loading.py        # For loading real column data
│   └── validation/
│       ├── __init__.py
│       ├── steady_state.py        # Steady-state validation tests
│       ├── dynamic_response.py    # Step response tests
│       ├── conservation.py        # Mass/energy balance checks
│       └── benchmarks.py          # Literature benchmark comparisons
├── tests/
│   ├── __init__.py
│   ├── test_thermodynamics.py
│   ├── test_hydraulics.py
│   ├── test_tray.py
│   ├── test_column.py
│   ├── test_env.py
│   └── test_integration.py
├── examples/
│   ├── basic_simulation.py        # Simple forward simulation
│   ├── step_response.py           # Step response analysis
│   ├── vectorized_sim.py          # Parallel simulation demo
│   └── rl_training.py             # Basic RL training example
├── notebooks/
│   ├── 01_thermodynamics.ipynb    # Interactive thermo exploration
│   ├── 02_single_tray.ipynb       # Single tray dynamics
│   ├── 03_full_column.ipynb       # Full column behavior
│   └── 04_validation.ipynb        # Validation analysis
└── data/
    ├── mixtures/                   # Thermodynamic data for mixtures
    │   ├── ethanol_water.json
    │   └── benzene_toluene.json
    └── columns/                    # Example column configurations
        └── teaching_column.json
```

### Core Design Principles

#### 1. JAX-Native State Representation

Use JAX-compatible dataclasses (via `flax.struct` or `chex.dataclass`) for all state:

```python
import chex
import jax.numpy as jnp

@chex.dataclass
class TrayState:
    """State of a single tray."""
    liquid_holdup: float      # mol
    liquid_composition: float # mol fraction of light component
    temperature: float        # K
    
@chex.dataclass  
class ColumnState:
    """Full column state."""
    tray_states: TrayState    # Batched over n_trays
    reboiler_state: TrayState
    condenser_state: TrayState
    reflux_drum_level: float  # mol
    sump_level: float         # mol
```

#### 2. Pure Functional Transition

The core dynamics must be a pure function suitable for `jax.jit`:

```python
def column_step(
    state: ColumnState,
    action: Action,
    params: ColumnParams,
    dt: float
) -> ColumnState:
    """
    Single timestep transition function.
    
    This is THE core function. Everything flows through here.
    Must be pure (no side effects) and JIT-compilable.
    """
    # 1. Compute flows from current state
    flows = compute_flows(state, action, params)
    
    # 2. Compute derivatives (MESH equations)
    derivatives = compute_derivatives(state, flows, params)
    
    # 3. Integrate forward
    new_state = integrate_step(state, derivatives, dt, params)
    
    # 4. Apply physical constraints (non-negative, bounded)
    new_state = apply_constraints(new_state, params)
    
    return new_state
```

#### 3. Vectorization Strategy

Design for `vmap` from the start:

```python
# Single column step
new_state = column_step(state, action, params, dt)

# Vectorized over batch of columns (e.g., for parallel RL)
batched_step = jax.vmap(column_step, in_axes=(0, 0, None, None))
new_states = batched_step(states, actions, params, dt)

# Vectorized over parameter variations (domain randomization)
param_varied_step = jax.vmap(column_step, in_axes=(0, 0, 0, None))
```

#### 4. Gymnasium Interface

```python
class DistillationColumnEnv(gymnasium.Env):
    """
    Gymnasium environment wrapping the JAX simulator.
    
    Handles:
    - JAX/NumPy conversion at boundaries
    - Episode management
    - Reward computation
    - Observation construction
    """
    
    def __init__(self, config: ColumnConfig):
        self.config = config
        self._params = config.to_params()
        self._state = None
        
        # Define spaces based on config
        self.action_space = self._build_action_space()
        self.observation_space = self._build_observation_space()
        
    def reset(self, seed=None, options=None):
        """Reset to initial or specified state."""
        super().reset(seed=seed)
        self._state = self._get_initial_state(options)
        return self._get_obs(), self._get_info()
    
    def step(self, action):
        """Execute one timestep."""
        # Convert action to JAX
        jax_action = self._process_action(action)
        
        # Core JAX transition
        self._state = column_step(
            self._state, 
            jax_action, 
            self._params, 
            self.config.dt
        )
        
        # Compute outputs
        obs = self._get_obs()
        reward = self._compute_reward()
        terminated = self._check_termination()
        truncated = False  # Continuing task
        info = self._get_info()
        
        return obs, reward, terminated, truncated, info
```

### Implementation Order

Follow this order to build incrementally with testing at each stage:

#### Stage 1: Thermodynamics Module (Day 1)

Implement in `core/thermodynamics.py`:

1. **Antoine equation** for vapor pressure
- Antoine equation (NIST form):
log10(P^sat[bar]) = A - B/(T[K] + C).
Default Antoine parameters (examples):
- Methanol: A=5.20409, B=1581.341, C=-33.50 (288.10–356.83 K).
- Water: A=5.08354, B=1663.125, C=-45.622 (344.00–373.00 K).
- Ethanol: A=5.24677, B=1598.673, C=-46.424 (292.77–366.63 K).
- Benzene: A=4.01814, B=1203.835, C=-53.226.
- Toluene: A=4.07827, B=1343.943, C=-53.773.
Clamp or warn outside coefficient validity ranges.
1. **Bubble point calculation** given liquid composition
- Bubble point condition: P = x1*gamma1*P1^sat(T) + x2*gamma2*P2^sat(T).
Algorithm: bracket temperature, do a few bisection steps, then switch to Newton/secant with finite-diff derivative; clamp to bracket to avoid divergence.
1. **Vapor-liquid equilibrium**
- Raoult’s Law (ideal) as baseline
- Activity coefficient models (Wilson) for non-ideal
- K-values: Kk = gamma_k(x,T) * Pk^sat(T) / P.
Binary equilibrium vapor: y1* = (K1*x)/(K1*x + K2*(1-x)).
NRTL (binary):
tau_12=A12 + B12/T, tau_21=A21 + B21/T, G12=exp(-alpha*tau_12), G21=exp(-alpha*tau_21).
ln(gamma1)=x2^2[ tau21*(G21/(x1+x2*G21))^2 + tau12*G12/(x2+x1*G12)^2 ].
ln(gamma2)=x1^2[ tau12*(G12/(x2+x1*G12))^2 + tau21*G21/(x1+x2*G21)^2 ].
1. **Enthalpy calculations**
- Liquid enthalpy vs temperature and composition
- Vapor enthalpy vs temperature and composition
- Heat of vaporization
- Liquid enthalpy: h_L(x,T)= x*cp_L1*(T-Tref) + (1-x)*cp_L2*(T-Tref).
Vapor enthalpy: h_V(y,T)= y*(dhvap1 + cp_V1*(T-Tref)) + (1-y)*(dhvap2 + cp_V2*(T-Tref)).

**Tests for Stage 1:**

- Antoine equation matches published vapor pressures
- Bubble point converges for known mixtures
- VLE matches published xy diagrams
- Enthalpy calculations are thermodynamically consistent

#### Stage 2: Hydraulics Module (Day 1-2)

Implement in `core/hydraulics.py`:

1. **Liquid holdup model**
- Francis weir formula
- Use a static weir relation for liquid outflow as a function of clear liquid height: Q_L = Cw*L_w*h_ow^(3/2).
Convert to molar flow: L = Q_L * rho_L / MW.
Optionally include hydraulic coupling: dL_out = (1/tau_L) dM + j dV_in (tau_L ~ 0.5–15 s, j in [-5,5]).
1. **Liquid flow rate** from tray to tray
- Liquid downflow from weir: Q_L = Cw*L_w*h_ow^(3/2), L = Q_L * rho_L / MW.
Include dynamic first-order lag on L_out with tau_L and vapor coupling j if modeling initial response.
1. **Vapor flow rate** through column
- Pressure drop considerations
- Start with specified/controlled boilup or pressure-driven vapor flow.
Constrain via flooding correlation (Fair) and weeping threshold; treat pressure-drop correlation as a parameterized function if needed.
1. **Operating limits**
- Flooding correlation
- Weeping correlation
- Flooding (Fair): U_n,f = C_sbf*(sigma/20)^0.2*((rho_L-rho_V)/rho_V)^0.5.
Weeping: U_min = (K2 - 0.9*(25.4 - d_h)) / sqrt(rho_V).

**Tests for Stage 2:**

- Francis weir matches textbook examples
- Flow rates are physically reasonable
- Operating limits trigger appropriately

#### Stage 3: Single Tray Dynamics (Day 2)

Implement in `column/tray.py`:

1. **Material balance** (component)
- d(M*x)/dt = L_in*x_in - L_out*x_out + V_in*y_in - V_out*y_out
- Component balance: d(M_i x_i)/dt = L_{i-1}x_{i-1} + 1_{i=f}F_L z_F + V_{i+1}y_{i+1} + 1_{i=f}F_V z_F - L_i x_i - V_i y_i.
Murphree vapor efficiency: y_i = y_{in} + E_{M,i}*(y_i* - y_{in}), with y_in = y_{i+1}.
1. **Total material balance**
- dM/dt = L_in - L_out + V_in - V_out
- Total balance: dM_i/dt = L_{i-1} + 1_{i=f}F_L + V_{i+1} + 1_{i=f}F_V - L_i - V_i.
1. **Energy balance**
- d(M*h)/dt = L_in*h_in - L_out*h_out + V_in*H_in - V_out*H_out
- Energy balance: dU_i/dt = L_{i-1}h_L(x_{i-1},T_{i-1}) + 1_{i=f}F_L h_F^L + V_{i+1}h_V(y_{i+1},T_{i+1}) + 1_{i=f}F_V h_F^V - L_i h_L(x_i,T_i) - V_i h_V(y_i,T_i) + Q_i.
1. **Tray step function**
   
   ```python
   def tray_step(tray_state, inflows, params, dt) -> TrayState:
       """Single tray transition."""
   ```

**Tests for Stage 3:**

- Mass is conserved
- Energy is conserved
- Steady state matches analytical solutions
- Dynamic response has correct time constant

#### Stage 4: Reboiler and Condenser (Day 2-3)

Implement in `column/reboiler.py` and `column/condenser.py`:

1. **Reboiler model**
- Partial reboiler (kettle type)
- Reboiler: dM_B/dt = L_N - (V_{N+1} + B).
d(M_B x_B)/dt = L_N x_N - V_{N+1} y_B - B x_B.
dU_B/dt = L_N h_L(x_N,T_N) - V_{N+1} h_V(y_B,T_B) - B h_L(x_B,T_B) + Q_R.
1. **Condenser model**
- Total condenser with reflux drum
- Total condenser + drum: dM_D/dt = V_1 - (R + D).
d(M_D x_D)/dt = V_1 y_1 - (R + D) x_D.
dU_D/dt = V_1 h_V(y_1,T_1) - (R + D) h_L(x_D,T_D) - Q_C.
1. **Level control** (implicit or explicit)
- Reflux drum level dynamics
- Sump level dynamics

**Tests for Stage 4:**

- Reboiler reaches specified vapor rate
- Condenser achieves total condensation
- Level dynamics are stable

#### Stage 5: Column Assembly (Day 3)

Implement in `column/column.py`:

1. **Stack trays** with appropriate indexing
1. **Connect reboiler** at bottom
1. **Connect condenser** at top
1. **Feed tray** handling
1. **Full column step function**

```python
def column_step(state: ColumnState, action: Action, params: ColumnParams, dt: float) -> ColumnState:
    """
    Full column transition.
    
    This assembles:
    - Condenser dynamics
    - Tray-by-tray dynamics (vectorized with scan/vmap)
    - Reboiler dynamics
    - Feed injection
    """
```

**Tests for Stage 5:**

- Column reaches expected steady state from startup
- Overall mass balance closes
- Overall energy balance closes
- Temperature profile is monotonic

#### Stage 6: Gymnasium Environment (Day 3-4)

Implement in `env/`:

1. **Base environment** with full state observation
1. **Action processing** (scaling, clipping)
1. **Observation construction**
1. **Reward functions** (configurable)
1. **Termination conditions**

**Tests for Stage 6:**

- Environment passes Gymnasium API check
- Actions have expected effects
- Rewards align with objectives

#### Stage 7: Validation Suite (Day 4-5)

Implement in `validation/`:

1. **Steady-state validation**
- Compare to analytical solutions for total reflux
- Compare to published simulation results
- Armfield-style teaching column experiments: pressure drop vs boil-up; tray efficiency vs boil-up at total reflux; top composition response under constant reflux ratio; overall mass balance checks.
1. **Dynamic response validation**
- Step response characteristics
- Time constants match expectations
- Directional step tests: increase reflux ratio -> higher top purity and lower top temperature; increase reboiler duty -> higher vapor flow and stronger separation (until constraints).
1. **Conservation checks**
- Continuous mass balance monitoring
- Energy balance closure
1. **Literature benchmarks**
- Compare to teaching-column benchmarks (pressure drop vs boil-up, efficiency vs boil-up, step response of top composition).

#### Stage 8: Examples and Documentation (Day 5)

1. **Basic simulation example**
1. **Vectorized simulation demo**
1. **RL training example** with stable-baselines3 or PureJaxRL
1. **Comprehensive README**
1. **Docstrings** throughout

### Specific Implementation Notes

#### Numerical Integration

For the ODE integration within `column_step`:

```python
# Option 1: Simple Euler (baseline, may need small dt)
new_state = state + dt * derivatives

# Option 2: RK4 (more stable, 4x cost)
# Implement if Euler is insufficient

# Option 3: Use Diffrax for adaptive stepping
# Only if fixed-step methods fail

# Start with Euler, dt=1s, and validate stability
```

#### Handling Stiffness

Hydraulic dynamics can be fast (tau_L ~ 0.5–15 s), composition/thermal slower; use fixed-step RK4 with internal substeps; pick dt_int <= 0.1*tau_L and take multiple internal steps per env step.

If the system is stiff:

- Consider implicit methods
- Consider splitting fast/slow dynamics
- Document the tradeoffs

#### Configuration System

```python
@chex.dataclass
class ColumnConfig:
    """User-facing configuration."""
    # Column geometry
    n_trays: int = 10
    feed_tray: int = 5
    column_diameter: float = 0.5  # m
    weir_height: float = 0.05    # m
    tray_spacing: float = 0.6    # m
    
    # Mixture properties
    mixture: str = "ethanol_water"
    
    # Operating conditions  
    feed_rate: float = 100.0     # mol/s
    feed_composition: float = 0.5  # mol fraction
    feed_temperature: float = 350.0  # K
    
    # Simulation
    dt: float = 10.0  # seconds
    
    def to_params(self) -> ColumnParams:
        """Convert to JAX-compatible parameters."""
        ...
```

#### Extensibility Hooks

Design for future extensions:

```python
# Multi-component: Change composition from float to array
liquid_composition: jnp.ndarray  # shape (n_components-1,)

# Partial observability: Add observation function
def observe(state: ColumnState, sensor_config: SensorConfig) -> Observation:
    """Extract observable quantities with noise/delay."""
    
# Different column configurations: Use protocol/interface
class ColumnModel(Protocol):
    def step(self, state, action, params, dt) -> State: ...
```

### Validation Checklist

Before considering the simulator complete, verify:

**Physical Correctness:**

- [ ] Column reaches expected steady state compositions
- [ ] Temperature profile is qualitatively correct (monotonic)
- [ ] Response to reflux ratio change has correct direction
- [ ] Response to reboiler duty change has correct direction
- [ ] Startup transient is reasonable
- [ ] Mass balance error < 0.1% at all times
- [ ] Energy balance error < 1% at all times

**Numerical Stability:**

- [ ] No NaN/Inf values during normal operation
- [ ] Stable for at least 10,000 timesteps
- [ ] No drift in conserved quantities
- [ ] Reasonable with 10-second timestep

**RL Compatibility:**

- [ ] Passes `gymnasium.utils.env_checker.check_env()`
- [ ] Vectorized version produces same results
- [ ] Can complete 1000 episodes without error
- [ ] JIT compilation succeeds
- [ ] vmap over batch dimension works

**Performance:**

- [ ] Single step < 1ms (CPU)
- [ ] Vectorized 1000 columns < 100ms (GPU)
- [ ] JIT compilation < 30s

**Documentation:**

- [ ] All public functions have docstrings
- [ ] README explains installation and basic usage
- [ ] At least one working example
- [ ] Validation results documented

### Dependencies

```toml
[project]
name = "jax-distillation"
version = "0.1.0"
dependencies = [
    "jax>=0.4.20",
    "jaxlib>=0.4.20", 
    "chex>=0.1.8",
    "gymnasium>=0.29.0",
    "numpy>=1.24.0",
    "matplotlib>=3.7.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-xdist>=3.0.0",
    "black>=23.0.0",
    "ruff>=0.1.0",
]
cuda = [
    "jax[cuda12_pip]>=0.4.20",
]
```

### Key References

- Wittgens & Skogestad (2000): dynamic models, hydraulics importance, tau_L and j ranges.
- NPTEL (tray design): flooding and weeping correlations.
- NIST WebBook: Antoine parameters and validity ranges.
- Armfield UOP3: lab column specs and instrumentation.
- IUPAC supporting info: NRTL parameters for methanol–water.

### Getting Started

After receiving this prompt, proceed as follows:

1. Create the project structure
1. Implement Stage 1 (thermodynamics) with tests
1. Run tests, fix issues
1. Proceed to Stage 2
1. Continue incrementally through all stages
1. Run full validation suite
1. Create examples and documentation

Ask clarifying questions if any part of the research document is ambiguous or incomplete. The physics must be correct—when in doubt, search for additional references rather than guessing.

-----

## Notes for Using These Prompts

### Phase 1 Execution

1. Give the Phase 1 prompt to Claude (or another capable LLM) with web search enabled
1. Allow it to search extensively—this may take multiple interactions
1. Review the output for completeness before proceeding
1. The quality of Phase 2 depends entirely on the thoroughness of Phase 1

### Phase 2 Execution

1. Copy the Phase 1 output into the marked placeholders in the Phase 2 prompt
1. Give the complete prompt to Claude Code
1. Claude Code should proceed incrementally, testing at each stage
1. Be prepared to provide additional guidance if:
- Numerical instabilities arise
- Validation tests fail
- Research document has gaps

### Expected Outcomes

A successful implementation will provide:

- A JAX-native simulator running 1000+ parallel columns on GPU
- Gymnasium-compatible interface for standard RL libraries
- Validated dynamics matching chemical engineering expectations
- Clear path to sim-to-real transfer through parameterization
- Extensible architecture for future enhancements



-----

# PHASE 3: Public-Benchmark Validation and Fit-Readiness for Sim-to-Real Distillation RL


## Prompt for Codex / Claude Code

You are working in an existing repository produced in “Phase 2” that already contains a JAX-native distillation column simulator and a Gymnasium-compatible environment. Your task is to **make the project credible and “fit-ready” for arbitrary real tray columns (within the model class)** using only **publicly available benchmarks, public property data, and public (or publicly described) datasets**.

### High-level goal

Implement a **Public Benchmark Validation and Fitting Readiness Pack** that:

1. **Verifies** the code solves its equations correctly (numerical correctness, conservation, invariants).
2. **Validates** the simulator against respected **public benchmark column models** and **public reference data** (not plant data).
3. Produces a single **Credibility Report** artifact (Markdown + plots; optional PDF) that a distillation expert would recognize as serious (even if it is not yet plant-validated).
4. Implements the **full data → reconciliation → state estimation → parameter estimation** pipeline so that when plant data is later provided, the simulator can be fit in a principled, auditable way.

### Non-negotiable constraints

* Do **not** change the core physics model semantics unless you are fixing a clear bug; prioritize backward compatibility.
* The core simulator step must remain **JAX-jittable** and **vmap-friendly** (pure functional state transition).
* All benchmark/fit tooling may use NumPy/Pandas/SciPy as needed (does not need to be jittable), but must be deterministic and reproducible.
* Do not require proprietary software to run the validation suite. If a benchmark has MATLAB reference code, provide an Octave-compatible pathway or provide a way to generate “golden” reference outputs once and store them as data files.

---

# Deliverables

## 1) A new “credibility” subsystem in the repo

Add a new top-level package directory:

```
jax_distillation/
  validation_pack/
    __init__.py
    data_sources/
      __init__.py
      download.py
      registry.py
      checksums.py
      licenses.md
    benchmarks/
      __init__.py
      skogestad_cola/
        __init__.py
        cola_spec.md
        cola_loader.py
        cola_reference_runner.py
        cola_config_builder.py
        cola_metrics.py
      wood_berry/
        __init__.py
        wood_berry_spec.md
        wood_berry_model.py
        wood_berry_reference_data.py
        wood_berry_metrics.py
      debutanizer_delay/
        __init__.py
        delay_spec.md
        dataset_loader.py
        delay_wrapper.py
        delay_metrics.py
    thermo_validation/
      __init__.py
      nist_points.py
      test_antoine_against_nist.py
      test_bubble_point.py
      test_vle_consistency.py
    verification/
      __init__.py
      mass_energy_closure.py
      timestep_convergence.py
      invariants.py
      stiffness_probes.py
    fitting/
      __init__.py
      units.py
      reconciliation.py
      state_estimation.py
      parameter_estimation.py
      identifiability.py
      reporting.py
    reports/
      __init__.py
      credibility_report.py
      templates/
        credibility_report.md.jinja
```

Add top-level scripts:

```
scripts/
  download_public_benchmarks.py
  run_public_validation.py
  run_fit_pipeline_demo.py
  build_credibility_report.py
```

Add tests:

```
tests/
  validation_pack/
    test_verification_mass_energy.py
    test_verification_convergence.py
    test_benchmark_cola.py
    test_benchmark_wood_berry.py
    test_debutanizer_delay_wrapper.py
    test_nist_thermo.py
    test_fit_pipeline_smoke.py
```

Update README with a new section:

* “Public-data validation and fitting readiness”
* “How to run the credibility report”
* “What is validated vs what is not (no plant data yet)”

---

# Required public references and how to integrate them

You must implement the following four validation steps (these are mandatory), plus additional steps described later.

## Step 1 (Mandatory): Reproduce Skogestad “Column A” benchmark behavior

### What to do

* Implement a benchmark runner that configures the simulator to match the **Skogestad Column A** model assumptions and parameters.
* Run a standard set of scenarios:

  * steady-state initialization to the benchmark nominal operating point
  * step changes in key inputs (reflux/boilup or equivalent, depending on the benchmark’s definition)
  * compare trajectories of key outputs (top/bottom composition proxies or internal composition states; and temperature profiles if benchmark provides them)

### How to obtain benchmark data

Write a downloader and spec file that references the public Skogestad Column A materials and any associated code/files.

Put URLs in code blocks only:

```text
Skogestad Column A page (public):
https://skoge.folk.ntnu.no/book/matlab_m/cola/cola.html
```

### Implementation requirements

* Provide a `cola_loader.py` that can:

  1. download the reference files (if present),
  2. parse whatever structured parameters are available,
  3. build a `ColumnConfig` for your simulator that matches those assumptions.

If the reference includes MATLAB scripts:

* Provide `cola_reference_runner.py` that can run in GNU Octave to generate reference trajectories, and save them as CSV/NPZ in `data/third_party/skogestad_cola/`.
* If Octave execution is not feasible in CI, commit *only* small precomputed reference trajectories (short time horizons) and provide a script to regenerate them.

### Metrics

Implement at least:

* normalized root mean square error (NRMSE) between simulator outputs and reference
* steady-state error on key outputs
* qualitative checks: correct sign/direction of response and settling behavior

### Acceptance criteria

* On the published nominal scenarios, achieve:

  * NRMSE < 10% for the main reported outputs, or explain clearly (in report) why mismatch is expected due to differing assumptions and how to align them.
  * Correct sign/direction and approximate time-scale agreement.
* If mismatch remains, the report must diagnose likely causes (thermo closure mismatch, efficiency assumptions, hydraulic modeling mismatch, etc.) and show at least one alignment experiment (e.g., turning off hydraulics to match the benchmark’s simpler dynamics if required).

---

## Step 2 (Mandatory): Match Wood–Berry benchmark step responses (control-facing validation)

### What to do

Wood–Berry is a classic **linear MIMO distillation benchmark** used in process control. You must:

* Implement the Wood–Berry transfer function model (or equivalent state-space form).
* Create a harness that:

  1. runs Wood–Berry linear model step responses,
  2. runs your simulator under a comparable operating point and comparable input perturbations,
  3. compares the input–output step responses.

### Data acquisition

Do not hard-code coefficients from memory. Instead:

* Use `download_public_benchmarks.py` to fetch the canonical Wood–Berry model coefficients from reliable public sources.
* Store the model parameters in a versioned JSON file under `data/third_party/wood_berry/wood_berry.json`.

Provide a source locator in a code block:

```text
Wood–Berry benchmark (search target):
"Wood Berry distillation column transfer function model"
```

### Metrics

* Compare step response curves using:

  * NRMSE on each output channel
  * gain (final value change) error
  * dominant time constant estimation error (fit a first/second order approximation and compare)

### Acceptance criteria

* Your simulator should match the *qualitative MIMO coupling structure* (signs of gains and relative magnitudes).
* Quantitative mismatch is allowed, but only if the report explains which modeling assumptions differ (e.g., your nonlinear stage model vs. Wood–Berry linearized approximation).

---

## Step 3 (Mandatory): Debutanizer-style delayed composition observation scenario

### What to do

Industrial columns often have composition measured via gas chromatography with **long delays** (tens of minutes). You must implement a realistic observation wrapper that introduces:

* fixed or variable **dead time** (transport + analysis)
* **sample-and-hold** measurement updates (analyzer cycle time)
* additive noise and occasional missing measurements (optional but recommended)

This wrapper must work with your Gymnasium environment and must be usable for RL training.

### Dataset use (public)

Attempt to integrate a publicly available debutanizer dataset used in soft-sensor literature. Implementer must:

* locate the dataset (often circulated with papers or repositories),
* download it via `download_public_benchmarks.py`,
* implement `dataset_loader.py` that standardizes it into a common schema:

  * timestamp, input tags (flows, temperatures), delayed composition label (if present)

If the dataset cannot be located with a stable public URL:

* Still implement the delay wrapper and provide a “synthetic delayed analyzer” demonstration using your simulator’s true composition as the latent variable.

Provide a source locator in a code block:

```text
Debutanizer dataset search targets:
"debutanizer column dataset 2394 samples"
"debutanizer soft sensor dataset gas chromatograph delay"
```

### Acceptance criteria

* Wrapper passes unit tests:

  * correct delay behavior (measurement appears after the configured dead time)
  * correct sample-and-hold schedule
  * deterministic under fixed RNG seed
* Provide a demo script that shows the delayed measurement vs. true composition over time.

---

## Step 4 (Mandatory): Thermodynamics validation against NIST public reference data

### What to do

You must validate the thermodynamics module using **public reference data**:

* Validate Antoine vapor pressure calculation against NIST reference points for at least:

  * methanol, water, ethanol, benzene, toluene
* Validate bubble-point solver:

  * for selected compositions at 1 atm (or a chosen pressure), confirm the residual of the bubble-point equation is small
* Validate VLE mapping:

  * sanity: y is bounded in [0,1], monotonic in x for ideal-ish cases, and behaves sensibly at limits

### How to obtain data

Use NIST Chemistry WebBook. Provide locator:

```text
NIST Chemistry WebBook (Antoine parameters and vapor pressure points):
https://webbook.nist.gov/chemistry/
```

Implementation details:

* Create `nist_points.py` that contains a small set of **hard-coded reference points** (component, temperature, expected Psat) derived from NIST pages.
* Do not scrape live NIST at test time.
* Include the reference points in the repository with a clear note “derived from NIST Chemistry WebBook on <date>” in comments.

### Acceptance criteria

* Antoine vapor pressure values match reference points within a tight tolerance (e.g., <1–2% relative error) across the chosen points.
* Bubble point residual < 1e-4 bar (or comparable small value) for test cases.

---

# Additional required work (do not stop after the 4 steps)

The 4 steps above are necessary but not sufficient for “fit readiness.” Implement the items below as part of the same effort.

## 5) Verification suite: prove the simulator solves its own equations correctly

Implement in `validation_pack/verification/`:

### 5.1 Mass balance closure tests

* For closed system scenarios: no feed, no products, no heat exchange; show total moles conserved (up to numerical error).
* For open system: integrate net in/out and compare to change in total holdup.

### 5.2 Energy balance closure tests

* Similar structure: compare change in internal energy plus enthalpy flows to heat duties and losses.

### 5.3 Timestep convergence

* Run the same scenario with dt, dt/2, dt/4 and show convergence of key outputs.

### 5.4 Invariants and physical bounds

* No negative holdups
* compositions in [0,1]
* no NaN/Inf under long rollouts (e.g., 50,000 internal steps)

### Acceptance criteria

* Mass closure error: < 0.1% of typical holdup over a long run (or a clearly justified tolerance)
* Energy closure error: < 1% of cumulative duty over a long run (or justified)
* Convergence plots included in credibility report

## 6) Hydraulics constraint plausibility checks

Even if your hydraulics are simplified, you must have:

* indicators for flooding/weeping or at least conservative proxies
* tests that demonstrate the indicators trigger in the expected direction when vapor/liquid loading increases/decreases
* a “near-constraint sweep” experiment: sweep boilup and reflux and show where constraints activate

Document clearly what is modeled and what is approximated.

## 7) A full “fit pipeline” that is runnable without plant data

This is the core “fit readiness” deliverable: the pipeline must run end-to-end on public/synthetic benchmark data.

### 7.1 Data schema and unit handling

Create a canonical schema used across all datasets:

```python
class TimeSeriesData(TypedDict):
    t: np.ndarray  # seconds
    u: Dict[str, np.ndarray]  # manipulated variables (setpoints or flows)
    y: Dict[str, np.ndarray]  # measured outputs (temperatures, pressures, levels, compositions)
    meta: Dict[str, Any]      # units, tags, delays, notes
```

Implement unit conversion helpers in `fitting/units.py` and require every dataset loader to provide explicit units.

### 7.2 Data reconciliation (mass/energy)

Implement `reconciliation.py`:

* constrained least squares reconciliation:

  * decision variables: adjusted measurements (flows, temperatures, compositions)
  * constraints: mass balance (total + component), basic bounds
  * objective: weighted deviation from raw measurements (weights from sensor noise assumptions)

Use SciPy optimization (or a small quadratic program solver) and keep it deterministic.

### 7.3 State estimation

Implement `state_estimation.py`:

* Provide at least one robust estimator that works with delayed and partial observations.
* Recommended approach: Moving Horizon Estimation (MHE) as nonlinear least squares over a sliding window:

  * decision: sequence of latent states
  * objective: measurement residuals + process residuals
  * constraints: physical bounds

If MHE is too heavy initially, implement an Extended Kalman Filter (EKF) with a delayed-measurement buffer. Either way, produce uncertainty estimates.

### 7.4 Parameter estimation

Implement `parameter_estimation.py`:

* Optimize a parameter vector θ that includes:

  * tray efficiency parameters (global or sectional)
  * hydraulic parameters (e.g., holdup scale factors, time constants)
  * heat-loss coefficients
  * sensor bias/scale (optional but recommended)
* Objective: match measured trajectories (temperatures, levels, delayed compositions if available).
* Provide:

  * gradient-based optimizer using JAX autodiff (preferred) or SciPy if easier
  * regularization / priors to keep parameters physically plausible
  * cross-validation: fit on one segment, evaluate on a held-out segment

### 7.5 Identifiability analysis

Implement `identifiability.py`:

* local sensitivity analysis:

  * compute ∂outputs/∂parameters along a trajectory
  * rank parameters by influence
* warn if parameters are unidentifiable given observation set (e.g., strong collinearity)

### 7.6 Fit report

Implement `fitting/reporting.py` that outputs:

* fitted parameters with units and plausible ranges
* fit metrics (NRMSE per signal, drift, constraint violations)
* residual diagnostics (autocorrelation if possible)

### Acceptance criteria

* `run_fit_pipeline_demo.py` runs end-to-end on:

  1. a benchmark-generated dataset (from Skogestad Column A simulation), and
  2. a delayed-observation dataset (debutanizer-style wrapper around simulator truth),
     producing a fit report and showing improved trajectory match after fitting.

## 8) Produce a single Credibility Report artifact

Implement `reports/credibility_report.py`:

* A script that runs all validations and assembles:

  * verification results (conservation, convergence, invariants)
  * benchmark comparisons (ColA, Wood–Berry)
  * delayed observation demo
  * thermo validation results (NIST points)
  * fitting pipeline demo results
  * known limitations / unvalidated assumptions

Output:

* `artifacts/credibility_report.md`
* `artifacts/plots/*.png`
* Optional: `artifacts/credibility_report.pdf` if you add a simple Markdown→PDF path (optional dependency)

Include a section titled:

* “Validated with public benchmarks”
* “Not yet validated with plant data”
* “What must be done when plant data is available”

---

# Project hygiene requirements

## Dependency updates

Update `pyproject.toml` to include (as needed):

* `pandas`
* `scipy`
* `jaxopt` (recommended for gradient-based fitting)
  Keep optional dependencies separated (`dev`, `fit`, `reporting`).

## Dataset licensing and reproducibility

* Add `validation_pack/data_sources/licenses.md` listing each downloaded artifact and its license or usage notes.
* Do not commit large third-party datasets if licensing is unclear. Instead:

  * commit a downloader script and checksums,
  * commit only small “smoke test” subsets that are clearly redistributable, or generate synthetic tests.

## CI friendliness

* Unit tests must run without downloading large assets by default.
* Provide markers:

  * `pytest -m "not slow"` for regular CI
  * `pytest -m slow` for full benchmark runs that download assets

---

# Definition of done (hard acceptance checklist)

You are finished only when all items below are true:

1. `python scripts/download_public_benchmarks.py` downloads/organizes benchmark resources (or provides clear fallback behavior) and records checksums.
2. `python scripts/run_public_validation.py` runs:

   * Skogestad Column A benchmark comparison,
   * Wood–Berry step response comparison,
   * debutanizer delay wrapper demo,
   * NIST thermo tests,
     and writes results to `artifacts/`.
3. `python scripts/run_fit_pipeline_demo.py` runs end-to-end reconciliation → state estimation → parameter estimation on at least one public/synthetic dataset and outputs a fit report.
4. `python scripts/build_credibility_report.py` produces a single coherent report (Markdown + plots) that:

   * clearly states what is validated and what is not,
   * includes quantitative metrics,
   * includes convergence and conservation evidence,
   * includes the benchmark comparisons and delayed-observation demo.
5. All non-slow unit tests pass in a clean environment.

---

# Guidance on handling ambiguity (do not ask the user)

If anything is underspecified (for example, exact Wood–Berry coefficients or exact Column A reference signals), you must:

* search public sources,
* document what you used and why,
* store parameters in versioned data files with provenance notes,
* and write tests that verify the parameters are loaded and used consistently.

Do not ask the user for clarifications. Resolve by research + explicit documentation.

---

# Notes on scope (what this does and does not claim)

This validation pack must make the simulator **credible and fit-ready** using public data, but it must also be honest: it should **not** claim plant-level predictive accuracy for a particular industrial column without plant data. The report must frame this clearly and propose the next plant-data validation steps as “future work.”

---

If you follow this prompt precisely, the repository will be in a state where a distillation expert can review it seriously, and where you can start fitting to any compatible real column as soon as you obtain time-series data from that column.
