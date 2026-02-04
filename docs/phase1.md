---
layout: default
title: Research Foundation
---

# Distillation Column Simulator: Research Foundation

This document defines the minimal physically-grounded distillation column simulator we will build in Phase 2, with enough realism to support RL research that transfers to lab columns. It is complete and self-contained.

---

# 0) State, Actions, and Constraints

We model an $N$-tray column plus condenser and reboiler. State includes:

* **Liquid holdups** per tray: $M_i$ [mol], for tray *i* $i = 1…N$
* **Liquid compositions**: $x_i$ (mole fraction of light key, e.g. methanol), with $0\le x_i \le 1$
* **Liquid internal energy** per tray: $U_i$ [J] (or equivalently $T_i$ as a state; see §1.2/§1.3)
* **Condenser holdup**: $M_D$, composition $x_D$, energy $U_D$ (or $T_D$)
* **Reboiler holdup**: $M_B$, composition $x_B$, energy $U_B$ (or $T_B$)
* **Pressure profile**: assume constant column pressure $P$ (initially), later add $\Delta P$.

Actions (control inputs):

* **Reflux flow** $L_0$ [mol/s]
* **Boilup flow** $V_{N+1}$ [mol/s] (or reboiler heat $Q_B$)
* **Distillate draw** $D$ [mol/s] (optional; can be fixed)
* **Bottoms draw** $B$ [mol/s] (optional; can be fixed)
* **Feed**: $F$ [mol/s], feed composition $z_F$, feed temperature $T_F$, feed tray $f$, feed quality $q$.

Constraints:

* $M_i, M_D, M_B \ge 0$
* $0\le x_i, x_D, x_B \le 1$
* physical feasibility of flows (e.g., weeping/flooding constraints later)
* safe bounds on temperature/pressure.

---

# 1) Core MESH Equations (Dynamic)

MESH = Material, Equilibrium, Summation, Heat. We use non-equilibrium stage model? No: we assume each tray is an equilibrium stage.

## 1.1 Material balances

For each tray $i$:

\[
\frac{dM_i}{dt} = L_{i-1} + V_{i+1} + \mathbb{1}_{i=f}qF - L_i - V_i
\tag{1}
\]

Component (light key) balance:

\[
\frac{d(M_i x_i)}{dt} = L_{i-1}x_{i-1} + V_{i+1}y_{i+1} + \mathbb{1}_{i=f}Fz_F - L_i x_i - V_i y_i
\tag{2}
\]

with feed split by quality:

* Liquid feed part: $F_L = qF$ enters liquid phase
* Vapor feed part: $F_V = (1-q)F$ enters vapor phase
* $q\in[0,1]$ (fraction of feed that enters as liquid)

If you want higher fidelity, compute q from feed enthalpy vs saturated enthalpies.

Boundary trays:

* Condenser tray is $i=0$ (for reflux $L_0$)
* Reboiler is $i=N+1$ (boilup $V_{N+1}$)
* Set $V_0 = 0$ and $L_{N+1}=0$.

## 1.2 Energy balances (full)

We need energy to get temperature and density effects. For each tray:

\[
\frac{dU_i}{dt} =
L_{i-1}h_L(x_{i-1},T_{i-1}) + V_{i+1}h_V(y_{i+1},T_{i+1})
+ \mathbb{1}_{i=f} \Bigl( qF h_F^{(L)} + (1-q)F h_F^{(V)} \Bigr)
- L_i h_L(x_i,T_i) - V_i h_V(y_i,T_i)
+ Q_i
\tag{3}
\]

Where:

* $h_L(x,T)$ is molar liquid enthalpy [J/mol]
* $h_V(y,T)$ is molar vapor enthalpy [J/mol]
* feed enthalpies $h_F^{(L)}$, $h_F^{(V)}$ computed at feed state
* $Q_i$ is external heat to tray i [W]; typically $Q_i=0$ for trays except reboiler and condenser.

Condenser energy:

\[
\frac{dU_D}{dt} =
V_1 h_V(y_1,T_1) - (L_0 + D)h_L(x_D,T_D) + Q_C
\tag{4}
\]

Reboiler energy:

\[
\frac{dU_B}{dt} =
L_N h_L(x_N,T_N) - (V_{N+1} + B) h_V(y_B,T_B) + Q_B
\tag{5}
\]

Implementation choice:

* either integrate $U_i$ and solve for $T_i$ each step via an “energy inversion” (e.g., Newton solve on $U_i - U(x_i,T_i)=0$), or
* store $T_i$ as state and integrate temperature dynamics derived from $\frac{dU}{dT}$ (more complex).

We will do energy inversion. We need an equation of state and enthalpy models (Phase 2).

## 1.3 Equilibrium (VLE)

We assume tray is at vapor–liquid equilibrium (VLE). For binary mixture:

\[
y_i = K_i(x_i,T_i,P)\, x_i
\tag{6}
\]

and summation constraint for binary:

\[
y_i + (1-y_i) = 1
\tag{7}
\]

But for general multi-component:

\[
\sum_j y_{i,j} = 1
\tag{8}
\]

We define $K$ using activity coefficients (non-ideal liquid) and ideal gas vapor:

\[
K_{i} = \frac{\gamma_i(x_i,T_i)\, P_i^{sat}(T_i)}{P}
\tag{9}
\]

where $\gamma_i$ from NRTL (Non-Random Two-Liquid) model, and $P_i^{sat}$ from Antoine equation.

For binary, compute:

\[
y_i = \frac{K_i x_i}{K_i x_i + K_{k} (1-x_i)}
\tag{10}
\]

or equivalently use $y_i = K_i x_i$ and normalize.

We will start with methanol–water and use published NRTL parameters.

## 1.4 Hydraulics: tray liquid dynamics (initial response realism)

Skogestad recommends adding a first-order lag to reflect tray hydraulics: liquid outflow depends on holdup and has a “vapor coupling” term (dimensionless in their linearized formulation). ([Sigurd Skogestad][1]) A practical implementable approach:

\[
L_i(t) = L_{i,ss}(M_i(t)) + j\, (V_i(t) - V_{i,ss})
\tag{18}
\]

Where:

* $L_{i,ss}(M_i) = \frac{M_i}{\tau_L}$ is a simple linear weir relationship with time constant $\tau_L$
* $j$ is vapor–liquid interaction coefficient (dimensionless)
* $V_{i,ss}$ is nominal vapor flow at steady state.

This makes the initial response of compositions/temperatures realistic. We will implement this.

---

# 2) Thermodynamic Models (Phase 2 requirements)

## 2.1 Antoine equation for saturation pressure

Use Antoine:

\[
\log_{10} P^{sat} = A - \frac{B}{T + C}
\tag{11}
\]

with $P^{sat}$ in bar (or Pa depending on constants) and $T$ in Kelvin.

We will use NIST Chemistry WebBook Antoine parameters, which provide A,B,C with temperature range. Start with methanol and water.

## 2.2 NRTL activity coefficients

Binary NRTL:

\[
\ln \gamma_1 = x_2^2 \left[
\tau_{21}\left(\frac{G_{21}}{x_1 + x_2 G_{21}}\right)^2
+ \frac{\tau_{12} G_{12}}{(x_2 + x_1 G_{12})^2}
\right]
\tag{12}
\]

\[
\ln \gamma_2 = x_1^2 \left[
\tau_{12}\left(\frac{G_{12}}{x_2 + x_1 G_{12}}\right)^2
+ \frac{\tau_{21} G_{21}}{(x_1 + x_2 G_{21})^2}
\right]
\tag{13}
\]

Where:

\[
G_{12} = \exp(-\alpha \tau_{12}), \quad
G_{21} = \exp(-\alpha \tau_{21})
\tag{14}
\]

and $\tau_{12}(T)$, $\tau_{21}(T)$ may be temperature-dependent:

\[
\tau_{12}(T) = a_{12} + \frac{b_{12}}{T},\quad
\tau_{21}(T) = a_{21} + \frac{b_{21}}{T}
\tag{15}
\]

We will plug in published methanol–water NRTL params.

## 2.3 Enthalpy models

We need liquid and vapor enthalpies.

Simplest consistent approach:

* treat vapor as ideal gas mixture: $h_V(T) = \sum_i y_i (h_i^{ig}(T) + \Delta h_{vap,i}(T))$
* treat liquid enthalpy as reference + integral of heat capacity: $h_L(T) = \sum_i x_i h_i^L(T)$
* ignore excess enthalpy of mixing initially.

We can implement heat capacities as polynomials and latent heats from correlations or constants.

Phase 2 can start with constant heat capacities and constant latent heats; then refine.

---

# 3) Column Geometry and Hydraulics (Phase 2/3)

We want enough to include flooding/weeping constraints. Use standard tray hydraulics correlations:

* liquid flow over weir: Francis weir equation
* downcomer backup
* flooding velocity using Fair correlation.

NPTEL module has formulas. ([NPTEL][8])

But Phase 2 only requires the lag model Eq. 18.

---

# 4) Discretization and Numerical Integration

We will integrate the ODE system:

* state vector: $[M_i,x_i,U_i]_{i=1..N}$ plus condenser/reboiler
* compute flows $L_i, V_i$ from hydraulics and control actions
* compute equilibrium $y_i$ via VLE given $x_i,T_i$
* compute enthalpies and update energies.

Use stable explicit integrator like Runge–Kutta 4 (RK4) or adaptive solver. Because RL will require many steps, we may use fixed-step RK4.

We must maintain:

* positivity of holdups
* bounds of compositions
* stability under control actions.

Clipping can be used but should be physically motivated (e.g., avoid negative holdup by limiting outflow).

---

# 5) Phase 1 Implementation Plan -> Phase 2 Simulator (Bridge)

Phase 1 created the conceptual plan. Phase 2 will implement:

1. Binary methanol–water column
2. Energy balance with inversion for temperature
3. Non-ideal VLE (NRTL + Antoine)
4. Hydraulics lag (Eq. 18)
5. Basic constraints and checks.

---

# Appendix A: Parameter values and sources

**Nominal column**:

* $N = 8$ trays (lab column)
* $P = 1$ bar (atmospheric) initially.

**Feed**:

* $F = 1$ mol/s
* $z_F = 0.5$ methanol
* $T_F$ near saturation.

**Holdups**:

* tray holdup $M_i$ nominal 1–5 mol.

**Hydraulics**:

Choose $\tau_L$ and $j$ based on Skogestad:

* $\tau_L$: 0.5–15 s
* $j$: −5 to +5

Start with $\tau_L = 5$ s, $j = 0$ then tune.

---

# Appendix B: Detailed Equations (copy/paste ready)

Material:

\[
\frac{dM_i}{dt} = L_{i-1} + V_{i+1} + \mathbb{1}_{i=f}qF - L_i - V_i
\tag{1}
\]

\[
\frac{d(M_i x_i)}{dt} = L_{i-1}x_{i-1} + V_{i+1}y_{i+1} + \mathbb{1}_{i=f}Fz_F - L_i x_i - V_i y_i
\tag{2}
\]

Energy:

\[
\frac{dU_i}{dt} =
L_{i-1}h_L(x_{i-1},T_{i-1}) + V_{i+1}h_V(y_{i+1},T_{i+1})
+ \mathbb{1}_{i=f} \Bigl( qF h_F^{(L)} + (1-q)F h_F^{(V)} \Bigr)
- L_i h_L(x_i,T_i) - V_i h_V(y_i,T_i)
+ Q_i
\tag{3}
\]

Equilibrium:

\[
K_i = \frac{\gamma_i(x_i,T_i)\, P_i^{sat}(T_i)}{P}
\tag{9}
\]

\[
y_i = \frac{K_i x_i}{K_i x_i + K_k (1-x_i)}
\tag{10}
\]

Hydraulics lag:

\[
L_i(t) = \frac{M_i(t)}{\tau_L} + j\, (V_i(t) - V_{i,ss})
\tag{18}
\]

---

# Full Phase 2 requirements checklist

* ✅ Dynamic material balances (Eq. 1–2)
* ✅ Dynamic energy balances (Eq. 3–5)
* ✅ VLE with non-ideal liquid via NRTL (Eq. 12–15)
* ✅ Saturation pressures from Antoine (Eq. 11)
* ✅ Hydraulics lag for liquid outflow (Eq. 18)
* ⏳ Flooding/weeping constraints (Phase 3)
* ⏳ Pressure drop model (Phase 3)
* ⏳ Multi-component extension (Phase 4)

---

# Notes on practical implementation details

**Energy inversion procedure:**

Given $U_i$ and $x_i$ at time step, solve for $T_i$:

\[
f(T) = U(x_i,T) - U_i = 0
\tag{29}
\]

Use Newton iteration:

\[
T_{k+1} = T_k - \frac{f(T_k)}{f'(T_k)}
\tag{30}
\]

where:

\[
f'(T) = \frac{dU}{dT} = M_i \, c_p(x_i,T)
\tag{31}
\]

Start with previous timestep $T$ as initial guess.

**Clipping policies:**

For RL stability, we will enforce:

* $M_i \ge \epsilon$ (small positive holdup)
* $x_i \in [\epsilon, 1-\epsilon]$

But *log* violations and treat as terminal if persistent.

---

# Sources

[1] Wittgens & Skogestad, "Evaluation of Dynamic Models of Distillation Columns with Emphasis on the Initial Response." (2000). PDF: [Sigurd Skogestad][1]

[2] NIST WebBook: Methanol Antoine parameters. [NIST Methanol][2]

[3] NIST WebBook: Water Antoine parameters. [NIST Water][3]

[4] Armfield UOP3 teaching distillation column product page (geometry reference). [Armfield][6]

[5] IUPAC VLE data and NRTL parameters for methanol–water. [iupac.org][7]

[6] NPTEL Distillation tray hydraulics module. [NPTEL][8]

---

# Minimal parameter set for Phase 2 implementation

We will hardcode the following initially:

**Column:**

* $N=8$
* $P=1$ bar constant

**Control inputs:**

* $L_0$ reflux [mol/s]
* $V_{N+1}$ boilup [mol/s]
* optionally fixed $D,B$

**Feed:**

* $F$
* $z_F$
* $T_F$
* $f$ feed tray index
* $q$ quality

**Thermo:**

* Antoine A,B,C for methanol and water
* NRTL $\alpha$, $\tau_{12}(T)$, $\tau_{21}(T)$ parameters

**Hydraulics:**

* $\tau_L$
* $j$

---

# Additional physics checks (debug assertions)

At each step:

* $0\le x_i \le 1$, $0\le y_i \le 1$
* $M_i\ge 0$
* total mass balance error small (closed system if $D,B$ fixed)
* energy not exploding (monitor $T_i$ within physical bounds)
* if solver fails to invert energy, terminate episode.

---

# Example initialization (Phase 2)

Initialize at steady-ish condition:

* $M_i = 2$ mol for all trays
* $x_i$ linear gradient from bottoms to top:
  * $x_B=0.1$ to $x_D=0.9$
* $T_i$ from bubble point estimates at $P$.

Set nominal flows:

* $F=1$
* $q=1$ (saturated liquid feed)
* $L_0=1$
* $V_{N+1}=1$

Then let it settle for some seconds with fixed controls.

---

# Phase 2 output signals (what RL sees)

Observations could include:

* tray temperatures $T_i$ (or selected sensor trays)
* distillate and bottoms compositions $x_D$, $x_B$
* reflux ratio $R = L_0/D$ (if D variable)
* holdups $M_i$ (optional; may be unobservable in reality)

Rewards can include:

* product purity targets (e.g., maximize $x_D$ while minimizing energy)
* penalties for constraint violations.

---

# Practical lab column grounding parameters

Armfield’s UOP3 column:

* diameter ~50 mm
* number of trays ~8
* per-tray temperature sensors
* reflux pump with adjustable flow
* reboiler heater.

We will approximate geometry to compute reasonable holdups and time constants later.

---

# References (URLs)

---

[1]: https://skoge.folk.ntnu.no/publications/2000/Wittgens2000/Wittgens2000.pdf "https://skoge.folk.ntnu.no/publications/2000/Wittgens2000/Wittgens2000.pdf"
[2]: https://webbook.nist.gov/cgi/cbook.cgi?ID=C67561&Mask=4 "https://webbook.nist.gov/cgi/cbook.cgi?ID=C67561&Mask=4"
[3]: https://webbook.nist.gov/cgi/cbook.cgi?ID=C71432&Mask=4&Plot=on&Type=ANTOINE "https://webbook.nist.gov/cgi/cbook.cgi?ID=C71432&Mask=4&Plot=on&Type=ANTOINE"
[4]: https://webbook.nist.gov/cgi/cbook.cgi?ID=C64175&Mask=4&Plot=on&Type=ANTOINE "https://webbook.nist.gov/cgi/cbook.cgi?ID=C64175&Mask=4&Plot=on&Type=ANTOINE"
[5]: https://webbook.nist.gov/cgi/cbook.cgi?ID=C108883&Mask=4&Plot=on&Type=ANTOINE "https://webbook.nist.gov/cgi/cbook.cgi?ID=C108883&Mask=4&Plot=on&Type=ANTOINE"
[6]: https://armfield.co.uk/product/uop3-distillation-columns/ "https://armfield.co.uk/product/uop3-distillation-columns/"
[7]: https://iupac.org/wp-content/uploads/2017/09/2011-037-2-100_SupportingInfo1-VLE.pdf "https://iupac.org/wp-content/uploads/2017/09/2011-037-2-100_SupportingInfo1-VLE.pdf"
[8]: https://archive.nptel.ac.in/content/storage2/courses/103103027/pdf/mod7.pdf "https://archive.nptel.ac.in/content/storage2/courses/103103027/pdf/mod7.pdf"

---

# Addendum: Phase 1 Research Foundation notes

The goal of Phase 1 was to specify a simulator that captures:

* correct short-time dynamics (hydraulics lag)
* correct non-ideal VLE (activity coefficients)
* correct energy coupling (temperature affects equilibrium).

The RL agent will exploit any mismatch, so realism is critical in these dimensions.

---

# Appendix C: Extra notes from Phase 1 author

**Why hydraulics lag matters:**

Without Eq. 18, a change in reflux or boilup causes immediate composition changes in tray liquids that are too fast compared to real columns. In real trays, liquid holdup and weir dynamics create a delay. The coefficient $j$ captures vapor coupling effects.

**Why energy matters:**

If you omit energy, temperature becomes either fixed or arbitrary, and VLE becomes unrealistic under transients. Energy ensures the correct coupling between heat input, vapor flow, and composition.

**Implementation caution:**

Energy inversion requires good enthalpy models. If using crude constant heat capacities, inversion should still converge if temperature range is limited. Later, replace with better correlations.

---

# Phase 1 explicit parameterization for methanol–water

**Antoine constants** (NIST values, ensure correct units):

Methanol: use A,B,C with valid range.
Water: use A,B,C with valid range.

**NRTL parameters (methanol=1, water=2):**

From IUPAC supporting info:

* $\alpha = 0.1$
* $\tau_{12}(T) = a_{12} + b_{12}/T$
* $\tau_{21}(T) = a_{21} + b_{21}/T$

Plug values directly.

---

# Phase 1: Additional constraints for tray hydraulics

For later phases, compute flooding/wetting constraints:

* flooding superficial gas velocity
* weeping point.

But for Phase 2, simply bound flows to safe ranges and treat flooding/weeping as terminal conditions (approx).

---

# Phase 1: Observability and sensor realism

Real teaching columns have:

* temperature sensors on each tray
* distillate and bottoms composition measured with delay
* flow meters.

So RL should be designed with partial observability. But Phase 2 can provide full state for debugging.

---

# End of Research Foundation

This file serves as the reference specification for Phase 2 implementation.

---

# Phase 1: citations and extra sources

We cite:

* Skogestad for hydraulics lag and dynamic model realism
* NIST for Antoine vapor pressure parameters
* IUPAC for NRTL VLE parameters
* NPTEL for tray hydraulics correlations.

---

# Phase 1: Parameter ranges for fitting

If we fit to lab data, we can tune:

* $\tau_L$: 0.5–15 s ([Sigurd Skogestad][1])
* $j$: −5 to +5 ([Sigurd Skogestad][1])
* $E_{M,i}$: start 0.5–0.9 (fit later; randomize per tray)

**Teaching column (Armfield-like):**

* $D_c = 0.05$ m, $N=8$, downcomers, per-tray temperature sensors, reflux 0–100%, pressure down to 0.2 bar ([Armfield][6])

**Methanol–water NRTL:**

* $\alpha=0.1$
* $\tau_{12}(T)= 9.23811 - 2432.61/T$
* $\tau_{21}(T)= -5.70743 + 1538.74/T$ ([iupac.org][7])

**Flooding/weeping constraints:**

* Flooding velocity from Eq. (21) (Fair correlation) ([NPTEL][8])
* Weep-point from Eq. (22) ([NPTEL][8])

---

# Next step recommendation (what to do after handing this to Phase 2)

Proceed to Phase 2 implementation exactly as your plan outlines, **but require two non-negotiables from the implementer**:

1. **Implement energy balance + non-ideal VLE (NRTL) from day 1**, using methanol–water as the default mixture (because you can cite parameters and it stresses the thermo stack). ([iupac.org][7])
2. **Implement hydraulic initial-response realism** via $\tau_L$ and $j$ (Eq. 18), and enforce flooding/weeping constraints (Eqs. 21–22). ([Sigurd Skogestad][1])

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