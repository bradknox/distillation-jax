---
layout: default
title: Research Foundation
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

A high-fidelity, control-oriented state for tray *i* $i = 1…N$ is:

* $M_i$ — **liquid holdup** on tray *i* [mol]
* $x_i$ — **liquid mole fraction of light component** in tray holdup [mol fraction]
* $U_i$ — **internal energy** of the tray control volume [J] (or equivalently $T_i$ as a state; see §1.2/§1.3)

Recommended top/bottom vessels:

* Reflux drum (accumulator): $M_D$ [mol], $x_D$ [mol fraction], $U_D$ [J]
* Reboiler: $M_B$ [mol], $x_B$ [mol fraction], $U_B$ [J]

Optional but recommended for best transfer (especially for “initial response”):

* A hydraulic coupling parameterization that represents how **liquid outflow responds to holdup and vapor flow** (see §1.4). ([Sigurd Skogestad][1])

**Algebraic (non-state) variables** computed each step:

* $L_i$ — liquid flow leaving tray *i* downward [mol/s]
* $V_i$ — vapor flow leaving tray *i* upward [mol/s]
* $y_i$ — vapor mole fraction of light component leaving tray *i* [mol fraction]
* $P_i$ — tray pressure [Pa] or [bar] (often modeled as fixed profile or fixed top + pressure drops)

---

### 1.2 MESH Equations for Binary Distillation (dynamic form; implementable)

This is the standard **MESH** structure:
**M**aterial balances (total + component), **E**quilibrium (VLE), **S**ummation, **H**eat (energy).

Number trays from top to bottom: tray 1 is top tray below the condenser, tray N is above the reboiler.

#### Notation for flows (consistent indexing)

* Liquid flows downward: $L_{i}$ leaves tray *i* and enters tray *i+1*
* Vapor flows upward: $V_{i}$ leaves tray *i* and enters tray *i−1*
* Thus vapor entering tray *i* from below is $V_{i+1}$ with composition $y_{i+1}$.
* Liquid entering tray *i* from above is $L_{i-1}$ with composition $x_{i-1}$.

Let the feed enter tray *f* at total molar rate $F$ [mol/s], overall light fraction $z_F$, and **quality** $q\in[0,1]$ (fraction of feed that enters as liquid). Then:

* Liquid feed part: $F_L = qF$ enters liquid phase
* Vapor feed part: $F_V = (1-q)F$ enters vapor phase

(If you want higher fidelity, compute q from feed enthalpy via a flash; but quality is a workable parameter.)

---

#### 1.2.1 Total material balance on tray i (Equation 1)

For i = 1…N, with feed only if i = f:

\[
\frac{dM_i}{dt} = L_{i-1} + \mathbb{1}*{i=f}F_L + V*{i+1} + \mathbb{1}_{i=f}F_V - L_i - V_i
\tag{1}
\]

If you assume vapor holdup is negligible (common in many control-oriented models), then $M_i$ is liquid holdup only and Eq. (1) simplifies by dropping vapor holdup dynamics but retaining vapor flow terms as throughputs. (Wittgens & Skogestad discuss both rigorous holdup accounting and simplified forms.) ([Sigurd Skogestad][1])

---

#### 1.2.2 Component material balance (light component) on tray i (Equation 2)

\[
\frac{d(M_i x_i)}{dt} = L_{i-1}x_{i-1} + \mathbb{1}*{i=f}F_L z_F + V*{i+1}y_{i+1} + \mathbb{1}_{i=f}F_V z_F - L_i x_i - V_i y_i
\tag{2}
\]

This is the core composition propagation equation used for dynamic tray models. ([Sigurd Skogestad][1])

---

#### 1.2.3 Energy balance on tray i (Equation 3)

A rigorous formulation uses internal energy as the differential state. Wittgens & Skogestad write tray energy balances in terms of internal energy with inlet/outlet enthalpy flows. ([Sigurd Skogestad][1]) A practical implementable form is:

\[
\frac{dU_i}{dt} =
L_{i-1} h_L(x_{i-1},T_{i-1})

* \mathbb{1}_{i=f}F_L h_F^{(L)}
* V_{i+1} h_V(y_{i+1},T_{i+1})
* \mathbb{1}_{i=f}F_V h_F^{(V)}

- L_i h_L(x_i,T_i)
- V_i h_V(y_i,T_i)

* Q_i
  \tag{3}
  \]

Where:

* $h_L(x,T)$ is **molar liquid enthalpy** [J/mol]
* $h_V(y,T)$ is **molar vapor enthalpy** [J/mol]
* $Q_i$ is external heat to tray i [W]; typically $Q_i=0$ for trays (heat losses handled separately if desired)

**Implementation note:** you can either:

1. integrate $U_i$ and solve for $T_i$ each step via an "energy inversion" (e.g., Newton solve on $U_i - U(x_i,T_i)=0$), or
2. integrate $T_i$ directly using an effective heat capacity model (simpler numerically, but be consistent).

Wittgens’ rigorous stage model uses internal energy states and equilibrium flash assumptions. ([Sigurd Skogestad][1])

---

#### 1.2.4 Equilibrium (VLE) closure (Equation 4)

For each tray, compute equilibrium K-values:

\[
K_k = \frac{\gamma_k(x,T) , P_k^{sat}(T)}{P}
\quad\text{for }k\in{1,2}
\tag{4}
\]

Assuming ideal vapor phase (often acceptable for low pressures; refine later if needed).

For a binary mixture, vapor composition in equilibrium with liquid is:

\[
y_1^{*} = \frac{K_1 x}{K_1 x + K_2(1-x)}
,\qquad y_2^{*}=1-y_1^{*}
\tag{5}
\]

---

#### 1.2.5 Murphree vapor efficiency (recommended non-equilibrium correction) (Equation 6)

Real trays are not perfectly equilibrated. A widely used, calibratable correction is **Murphree vapor efficiency** $E_{M,i}\in(0,1]$. Define vapor entering tray i from below as $y_{in}=y_{i+1}$. Then:

\[
y_i = y_{in} + E_{M,i}\bigl(y_i^{*} - y_{in}\bigr)
\tag{6}
\]

This is attractive for sim-to-real because $E_{M,i}$ is **identifiable from data** (e.g., from steady-state composition/temperature profiles and known reflux/boilup conditions) and can be randomized across plausible ranges.

---

#### 1.2.6 Summation (Equation 7)

Binary summations are enforced implicitly:

\[
x_2 = 1-x_1,\qquad y_2 = 1-y_1
\tag{7}
\]

---

### 1.3 Thermodynamic Closure (recommended models + concrete default parameters)

#### 1.3.1 Vapor pressure: Antoine equation (Equation 8)

Use the NIST Antoine form (as presented on NIST Chemistry WebBook pages):

\[
\log_{10}!\bigl(P^{sat}[\mathrm{bar}]\bigr)= A - \frac{B}{T[\mathrm{K}] + C}
\tag{8}
\]

**Default component Antoine parameters (examples commonly used in teaching/lab columns):**

* **Methanol** (valid 288.10–356.83 K): $A=5.20409,; B=1581.341,; C=-33.50$ ([NIST WebBook][2])
* **Water** (valid 344.00–373.00 K): $A=5.08354,; B=1663.125,; C=-45.622$ ([NIST WebBook][3])
* **Ethanol** (valid 292.77–366.63 K): $A=5.24677,; B=1598.673,; C=-46.424$ ([NIST WebBook][4])
* **Benzene** (valid 287.70–354.07 K): $A=4.01814,; B=1203.835,; C=-53.226$
* **Toluene** (valid 308.52–384.66 K): $A=4.07827,; B=1343.943,; C=-53.773$ ([NIST WebBook][5])

**Implementation note:** enforce the valid temperature ranges during testing; outside-range extrapolation can destabilize the simulator. Clamp or switch coefficient sets if you extend ranges.

---

#### 1.3.2 Activity coefficients (γ): recommended hierarchy

1. **Ideal Raoult’s law:** $\gamma_1=\gamma_2=1$.
   Use for near-ideal systems (many hydrocarbon pairs) or as an initial baseline.

2. **Non-ideal (recommended for sim-to-real): NRTL model.**
   NRTL (Non-Random Two-Liquid) is widely used, stable for many polar mixtures, and parameterized with a small number of constants.

**NRTL equations (binary form; Equation 9–11)**

Let $\tau_{12}, \tau_{21}$ be dimensionless interaction parameters and $\alpha$ be the non-randomness parameter (commonly 0.1–0.3; IUPAC example uses 0.1). Define:

\[
G_{12}=\exp(-\alpha \tau_{12}),\qquad G_{21}=\exp(-\alpha \tau_{21})
\tag{9}
\]

For a binary mixture, the activity coefficients are:

\[
\ln \gamma_1 = x_2^2\left[\tau_{21}\left(\frac{G_{21}}{x_1 + x_2 G_{21}}\right)^2

* \tau_{12}\frac{G_{12}}{(x_2 + x_1 G_{12})^2}\right]
  \tag{10}
  \]

\[
\ln \gamma_2 = x_1^2\left[\tau_{12}\left(\frac{G_{12}}{x_2 + x_1 G_{12}}\right)^2

* \tau_{21}\frac{G_{21}}{(x_1 + x_2 G_{21})^2}\right]
  \tag{11}
  \]

(These are standard NRTL binary expressions; implement carefully and test against published VLE points.)

**Temperature dependence:** a practical common form is:

\[
\tau_{12}(T)=A_{12}+\frac{B_{12}}{T},\qquad
\tau_{21}(T)=A_{21}+\frac{B_{21}}{T}
\tag{12}
\]

---

#### 1.3.3 Default non-ideal mixture parameters (high-value for teaching columns)

Armfield’s UOP3 teaching column explicitly lists **methanol–water** among standard mixtures. ([Armfield][6]) For methanol + water, an IUPAC supporting-information example specifies **NRTL with α = 0.1** and $\tau$ parameters of the form (A + B/T): ([iupac.org][7])

* Components: 1 = methanol, 2 = water
* $\alpha = 0.1$
* (\tau_{12}(T)= 9.23811 + (-2432.61)/T)
* (\tau_{21}(T)= -5.70743 + (1538.74)/T)
  with (T) in Kelvin. ([iupac.org][7])

(These are “dimensionless interaction parameters” as presented in the IUPAC supporting info. Implement exactly as in Eq. 12.)

---

#### 1.3.4 Relative volatility α (definition; optional simplification)

Relative volatility between light (1) and heavy (2):

\[
\alpha_{rel} = \frac{K_1}{K_2}
\tag{13}
\]

In simplified teaching models, $\alpha_{rel}$ is sometimes treated as constant, yielding:

\[
y_1^{*}=\frac{\alpha_{rel} x}{1 + (\alpha_{rel}-1)x}
\tag{14}
\]

**Recommendation for your project:** do **not** use constant relative volatility as the main model if you want sim-to-real; keep it only as a debugging/benchmark mode.

---

#### 1.3.5 Bubble-point / dew-point calculations (algorithm)

You will often need tray temperature consistent with composition at pressure.

**Bubble-point condition (Equation 15):**

\[
P = x_1\gamma_1(x,T)P_1^{sat}(T) + x_2\gamma_2(x,T)P_2^{sat}(T)
\tag{15}
\]

**Algorithm (robust bracketing + Newton):**

1. Choose a temperature bracket ([T_{low},T_{high}]) such that (f(T)=\sum x_k\gamma_k P_k^{sat}(T)-P) changes sign.
2. Use bisection for a few iterations to guarantee bracketing.
3. Switch to Newton (or secant) using derivative approximated by finite difference; clamp steps to remain within bracket.

This approach is stable in JAX (pure functional; no side effects) and avoids divergence that can crash long RL runs.

---

#### 1.3.6 Enthalpy models (sufficient for control-relevant fidelity; extensible)

A pragmatic starting point (that supports energy balances and is calibratable) is:

Liquid enthalpy:
\[
h_L(x,T)= x,c_{p,L,1}(T-T_{ref}) + (1-x),c_{p,L,2}(T-T_{ref})
\tag{16}
\]

Vapor enthalpy:
\[
h_V(y,T)= y\left[\Delta h_{vap,1} + c_{p,V,1}(T-T_{ref})\right]
+(1-y)\left[\Delta h_{vap,2} + c_{p,V,2}(T-T_{ref})\right]
\tag{17}
\]

Where:

* $c_p$ are molar heat capacities [J/mol/K]
* $\Delta h_{vap}$ are heats of vaporization [J/mol] at a reference (or temperature-dependent if you later upgrade)

**Why this is acceptable for transfer (initially):** you can fit effective $c_p$ and $\Delta h_{vap}$ to match measured temperature dynamics and steady-state heat duties. If you later need more accuracy, you can replace these with DIPPR/NIST correlations without changing the simulator architecture.

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

\[
dL_{out} = \frac{1}{\tau_L} dM + j, dV_{in}
\tag{18}
\]

Where:

* $\tau_L$ is a **hydraulic time constant** [s]
* (j) captures the **initial effect of vapor flow on liquid flow** (dimensionless in this deviation form)

Typical ranges reported: $\tau_L \approx 0.5$ to (15) s and $j\in[-5,5]$. ([Sigurd Skogestad][1])

**Implementation pattern:** represent $L_i$ (or $L_{out,i}$) as an auxiliary dynamic variable with its own first-order response to the “static” weir prediction, plus the vapor coupling term. This improves realism of short-time transients without requiring a full computational-fluid model.

#### 1.4.3 Weir/overflow relation (for “static” liquid outflow)

Use a sharp-crested weir relation (often referred to as a Francis-type correlation). NPTEL uses a standard weir form in the context of tray design calculations. ([NPTEL][8])

A practical implementable approach:

1. compute a clear liquid height over the weir $h_{ow}$ [m] from tray holdup (via geometry and density),
2. compute volumetric liquid flow:
   \[
   Q_L = C_w, L_w, h_{ow}^{3/2}
   \tag{19}
   \]
3. convert to molar flow:
   \[
   L = Q_L \frac{\rho_L}{\overline{MW}}
   \tag{20}
   \]

Where:

* $L_w$ is weir length [m]
* $\rho_L$ is liquid density [kg/m³]
* $\overline{MW}$ is mixture molecular weight [kg/mol]
* $C_w$ is a weir coefficient (set by tray design; calibrate)

If you want to avoid “deriving” $h_{ow}$ from geometry at first, you can parameterize (L) as a monotone function of holdup and fit parameters from step tests; but for best fidelity, implement geometry.

#### 1.4.4 Flooding constraint (Fair correlation) (Equation 21)

NPTEL gives Fair’s correlation for flooding gas velocity through net area: ([NPTEL][8])

\[
U_{n,f} = C_{sbf}\left(\frac{\sigma}{20}\right)^{0.2}
\left(\frac{\rho_L-\rho_V}{\rho_V}\right)^{0.5}
\tag{21}
\]

Where:

* $U_{n,f}$ = flooding superficial gas velocity [m/s]
* $C_{sbf}$ = capacity parameter (depends on tray spacing and flow parameter)
* $\sigma$ = surface tension [mN/m]
* $\rho_L,\rho_V$ = liquid and vapor densities

Design guidance: operate at ~80–85% of flooding for non-foaming liquids. ([NPTEL][8])

**Simulator use:** treat flooding as a hard constraint (terminate episode) or as a steep penalty once $U_n/U_{n,f}$ exceeds ~0.85–0.9.

#### 1.4.5 Weeping constraint (weep-point velocity) (Equation 22)

NPTEL provides a weeping criterion via a minimum vapor velocity at the weep point: ([NPTEL][8])

\[
U_{min} = \frac{K_2 - 0.9(25.4 - d_h)}{\sqrt{\rho_V}}
\tag{22}
\]

Where:

* $d_h$ is hole diameter [mm]
* $K_2$ is a tray-specific coefficient (from design correlations)

**Simulator use:** penalize operation where $U_n < U_{min}$ (loss of contacting efficiency, possible “dumping”).

#### 1.4.6 Tray efficiency dependency

Tray efficiency depends on flow regimes (flooding/weeping) and design; Armfield explicitly lists experiments measuring “column efficiency as a function of boil-up rate” under total reflux. ([Armfield][6])

**Practical sim-to-real strategy:** model $E_{M,i}$ as:

* nominal tray efficiency (fit from data),
* degraded as flooding/weeping are approached (smooth function),
* and randomized during training within uncertainty bands.

---

### 1.5 Boundary Conditions (condenser, reflux drum, reboiler)

#### 1.5.1 Total condenser + reflux drum

Armfield’s UOP3 includes an overhead condenser and reflux tank, with reflux control from 0–100%. ([Armfield][6])

Model a **total condenser**: all incoming vapor condenses to liquid in the drum.

Let vapor from tray 1 enter condenser at $V_1, y_1$. Then condensed liquid composition is approximately $x_D \approx y_1$ (for total condensation).

Reflux drum holdup dynamics:
\[
\frac{dM_D}{dt} = V_1 - (R + D)
\tag{23}
\]

Component balance:
\[
\frac{d(M_D x_D)}{dt} = V_1 y_1 - (R + D)x_D
\tag{24}
\]

Energy balance:
\[
\frac{dU_D}{dt} = V_1 h_V(y_1,T_1) - (R+D)h_L(x_D,T_D) - Q_C
\tag{25}
\]

Where $Q_C$ [W] is condenser duty (heat removed). Often you will specify condenser outlet temperature or pressure and solve for required $Q_C$; for RL, $Q_C$ is usually not a manipulated variable.

Reflux to tray 1: $L_0 = R$, composition $x_0=x_D$, temperature $T_0=T_D$.

#### 1.5.2 Reboiler (kettle type)

Reboiler holdup dynamics:
\[
\frac{dM_B}{dt} = L_N - (V_{N+1} + B)
\tag{26}
\]

Component balance:
\[
\frac{d(M_B x_B)}{dt} = L_N x_N - V_{N+1} y_B - B x_B
\tag{27}
\]

Energy balance:
\[
\frac{dU_B}{dt} = L_N h_L(x_N,T_N) - V_{N+1}h_V(y_B,T_B) - B h_L(x_B,T_B) + Q_R
\tag{28}
\]

Where:

* $Q_R$ [W] is reboiler duty (a primary manipulated variable)
* $y_B$ is vapor composition in equilibrium with reboiler liquid (use Eq. 5)

---

## 2. Recommended Default Parameters

### 2.1 Column Geometry (default: Armfield-like lab column)

Armfield UOP3 plate column: **50 mm diameter**, **8 sieve plates**, **downcomers**, and a temperature sensor at each plate with very small sheath diameter for rapid response. ([Armfield][6])

A good default configuration (explicitly intended as “teaching column” scale) is:

| Parameter                  |    Symbol |         Default | Units | Notes / justification                                                              |
| -------------------------- | --------: | --------------: | ----- | ---------------------------------------------------------------------------------- |
| Number of trays            |       (N) |               8 | –     | UOP3 has 8 sieve plates ([Armfield][6])                                            |
| Column internal diameter   |     $D_c$ |            0.05 | m     | UOP3 plate column is 50 mm diameter ([Armfield][6])                                |
| Tray spacing               |       (S) |       0.20–0.30 | m     | Not stated on page; treat as configurable; affects flooding correlation and holdup |
| Weir height                |     $h_w$ |       0.01–0.03 | m     | Typical lab tray weirs; treat as configurable and fit if known                     |
| Weir length                |     $L_w$ | 0.6–0.9 × $D_c$ | m     | Approximate; geometry-dependent                                                    |
| Downcomer area fraction    | $A_d/A_t$ |       0.08–0.15 | –     | Typical tray design range; fit if known                                            |
| Hole diameter (sieve tray) |     $d_h$ |             2–5 | mm    | Needed for weeping constraint                                                      |
| Operating pressure range   |       (P) |         0.2–1.0 | bar   | UOP3 supports reduced pressure down to 200 mbar ([Armfield][6])                    |

For sim-to-real you should replace defaults with the real column’s design drawings when available.

---

### 2.2 Thermodynamic Parameters (default mixture: methanol–water with NRTL)

**Why this default:** methanol–water is explicitly listed as a standard mixture for the lab column platform. ([Armfield][6]) It is non-ideal, so it forces the simulator architecture to support activity coefficients (needed for transfer across many real mixtures).

**Antoine coefficients:** see §1.3.1 for methanol and water. ([NIST WebBook][2])

**NRTL parameters $methanol=1, water=2$:**

* $\alpha=0.1$
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

* **Simulator internal timestep** $dt_{int}$: chosen to resolve hydraulics. A conservative rule is $dt_{int} \le 0.1\tau_L$. With $\tau_L$ as low as 0.5 s, this suggests $dt_{int}\approx 0.05$ s in the worst case. ([Sigurd Skogestad][1])
* **RL environment timestep** $dt_{env}$: 1–10 s (or larger), implemented by taking multiple internal steps per environment step.

This structure preserves physical fidelity while keeping RL rollouts efficient.

---

## 4. Control Interface Specification

### 4.1 Action space (best option for sim-to-real)

For transfer, the most realistic and safest interface is usually **supervisory control**: the RL policy sets **setpoints** (or setpoint biases) for existing regulatory loops, rather than directly commanding valves.

A strong default action vector:

1. $RR_{sp}$ — reflux ratio setpoint (or reflux flow setpoint)
2. $Q_{R,sp}$ — reboiler duty setpoint (or boilup setpoint)

And keep two internal level controllers (conventional proportional–integral loops) to regulate:

* reflux drum level via distillate flow (D)
* reboiler/sump level via bottoms flow (B)

This reflects common practice: levels are regulated locally for safety, while composition/temperature objectives are handled by higher-level control.

**Actuator dynamics:** represent each manipulated flow/duty as a first-order lag with saturation:
\[
u_{actual}(t+dt)=u_{actual}(t)+\frac{dt}{\tau_u}\left(\text{clip}(u_{cmd},u_{min},u_{max})-u_{actual}(t)\right)
\tag{29}
\]

### 4.2 Observation space (realistic sensors)

For an Armfield-like lab column, temperatures on each plate are explicitly measured with fast-response sensors. ([Armfield][6])

Recommended observation vector:

* Tray temperatures $T_1,\dots,T_N$ (or a subset if you want partial observability)
* Reflux drum level (or holdup) $M_D$
* Reboiler level (or holdup) $M_B$
* Column pressure drop (Armfield lists a differential manometer top-to-bottom) ([Armfield][6])
* Key flows: reflux, distillate, bottoms, feed, boilup (as measured or inferred)

**Sensor dynamics:** model temperature/level sensors with a first-order filter and additive noise. Armfield highlights rapid dynamic response of plate temperature sensors (very small sheath). ([Armfield][6])

### 4.3 Constraint specification (numerical bounds)

Include at least:

* **Flooding constraint:** terminate or steep-penalize when $U_n/U_{n,f}\gtrsim 0.85–0.9$. ([NPTEL][8])
* **Weeping constraint:** penalize when $U_n < U_{min}$. ([NPTEL][8])
* **Level constraints:** $M_D$ and $M_B$ must stay within safe bounds (avoid dry-out/overflow).
* **Composition bounds:** enforce $x_i,y_i\in[0,1]$ and nonnegative holdups.
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
* Linear hydraulics law uses plausible $\tau_L$ and (j) ranges and produces reasonable initial responses. ([Sigurd Skogestad][1])

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

* $0\le x_i \le 1$, $0\le y_i \le 1$
* $M_i, M_D, M_B \ge 0$
* No NaN/Inf
* Flooding/weeping indicators computed and recorded
* Enforce action/actuator saturations

---

## 6. Key References (annotated)

1. **Wittgens & Skogestad (2000)**, *Evaluation of Dynamic Models of Distillation Columns with Emphasis on the Initial Response.*
   Core reference for why hydraulics matters, and for a rigorous stage model structure and control-relevant hydraulic parameters ($\tau_L$, (j)). ([Sigurd Skogestad][1])

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
   \[
   \frac{dM_i}{dt} = L_{i-1} + \mathbb{1}*{i=f}qF + V*{i+1} + \mathbb{1}_{i=f}(1-q)F - L_i - V_i
   \]

2. Light component:
   \[
   \frac{d$M_i x_i$}{dt} = L_{i-1}x_{i-1} + \mathbb{1}*{i=f}qF z_F + V*{i+1}y_{i+1} + \mathbb{1}_{i=f}(1-q)F z_F - L_i x_i - V_i y_i
   \]

3. Energy:
   \[
   \frac{dU_i}{dt} =
   L_{i-1} h_L$x_{i-1},T_{i-1}$

* \mathbb{1}_{i=f}qF h_F^{(L)}
* V_{i+1} h_V$y_{i+1},T_{i+1}$
* \mathbb{1}_{i=f}(1-q)F h_F^{(V)}

- L_i h_L$x_i,T_i$
- V_i h_V$y_i,T_i$
  \]

4. Vapor pressure (Antoine):
   \[
   \log_{10}!\bigl$P_k^{sat}[\mathrm{bar}]\bigr$= A_k - \frac{B_k}{T[\mathrm{K}] + C_k}
   \]

5. NRTL:
   \[
   \tau_{12}=A_{12}+B_{12}/T,\quad \tau_{21}=A_{21}+B_{21}/T,\quad
   G_{12}=e^{-\alpha\tau_{12}},\quad G_{21}=e^{-\alpha\tau_{21}}
   \]
   \[
   $\ln\gamma_1,\ln\gamma_2$\ \text{from Eqs. (10–11)}
   \]

6. K-values:
   \[
   K_k=\gamma_k P_k^{sat}/P
   \]

7. Equilibrium vapor composition:
   \[
   y_1^{*}=\frac{K_1 x}{K_1 x + K_2(1-x)}
   \]

8. Murphree efficiency:
   \[
   y_i = y_{i+1} + E_{M,i}$y_i^{*}-y_{i+1}$
   \]

9. Hydraulic coupling (recommended):
   \[
   dL_{out} = \frac{1}{\tau_L} dM + j, dV_{in}
   \]

Top condenser + reflux drum:

\[
\frac{dM_D}{dt} = V_1 - (R + D),\quad
\frac{d(M_D x_D)}{dt} = V_1 y_1 - (R + D)x_D
\]

Bottom reboiler:

\[
\frac{dM_B}{dt} = L_N - (V_{N+1}+B),\quad
\frac{d(M_B x_B)}{dt} = L_N x_N - V_{N+1} y_B - B x_B
\]

---

## Appendix B: Parameter Tables (defaults + uncertainty ranges for domain randomization)

**Hydraulics (control-relevant):**

* $\tau_L$: 0.5–15 s ([Sigurd Skogestad][1])
* (j): −5 to +5 ([Sigurd Skogestad][1])
* $E_{M,i}$: start 0.5–0.9 (fit later; randomize per tray)

**Teaching column (Armfield-like):**

* $D_c = 0.05$ m, $N=8$, downcomers, per-tray temperature sensors, reflux 0–100%, pressure down to 0.2 bar ([Armfield][6])

**Methanol–water NRTL:**

* $\alpha=0.1$
* (\tau_{12}(T)= 9.23811 - 2432.61/T)
* (\tau_{21}(T)= -5.70743 + 1538.74/T) ([iupac.org][7])

**Flooding/weeping constraints:**

* Flooding velocity from Eq. (21) (Fair correlation) ([NPTEL][8])
* Weep-point from Eq. (22) ([NPTEL][8])

---

# Next step recommendation (what to do after handing this to Phase 2)

Proceed to Phase 2 implementation exactly as your plan outlines, **but require two non-negotiables from the implementer**:

1. **Implement energy balance + non-ideal VLE (NRTL) from day 1**, using methanol–water as the default mixture (because you can cite parameters and it stresses the thermo stack). ([iupac.org][7])
2. **Implement hydraulic initial-response realism** via $\tau_L$ and (j) (Eq. 18), and enforce flooding/weeping constraints (Eqs. 21–22). ([Sigurd Skogestad][1])

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
