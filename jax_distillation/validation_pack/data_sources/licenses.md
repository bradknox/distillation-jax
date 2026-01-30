# Data Sources and Licenses

This document details the public data sources used for benchmark validation
in the JAX distillation simulator, along with their licensing information.

## Overview

All data used in the validation pack comes from publicly available sources
and is properly attributed. No proprietary data or software is required.

---

## 1. NIST Chemistry WebBook

**Source:** https://webbook.nist.gov/chemistry/

**License:** Public Domain (US Government Work)

**Usage:** Thermodynamic reference data including:
- Antoine equation coefficients for vapor pressure
- Heat of vaporization values
- Heat capacity data

**Citation:**
> Linstrom, P.J. and Mallard, W.G. (Eds.), NIST Chemistry WebBook,
> NIST Standard Reference Database Number 69, National Institute of
> Standards and Technology, Gaithersburg MD, 20899.
> https://doi.org/10.18434/T4D303

---

## 2. Skogestad Column A (COLA) Benchmark

**Source:** https://skoge.folk.ntnu.no/book/matlab_m/cola/cola.html

**License:** Academic/Fair Use

**Usage:** Column configuration and operating parameters for a standard
40-tray binary distillation benchmark. Parameters are published in
academic literature and freely available for research purposes.

**Citation:**
> Skogestad, S. (2007). "The dos and don'ts of distillation column control."
> Chemical Engineering Research and Design, 85(1), 13-23.
> https://doi.org/10.1205/cherd06133

**Additional Reference:**
> Skogestad, S. and Postlethwaite, I. (2005). "Multivariable Feedback Control:
> Analysis and Design." John Wiley & Sons, 2nd edition.

---

## 3. Wood-Berry Distillation Model

**Source:** Published academic paper

**License:** Fair Use (Published Constants)

**Usage:** Classic 2x2 MIMO transfer function model for distillation control
benchmarking. The transfer function coefficients are widely published
constants from the original 1973 paper.

**Citation:**
> Wood, R.K. and Berry, M.W. (1973). "Terminal composition control of a
> binary distillation column." Chemical Engineering Science, 28(9), 1707-1717.
> https://doi.org/10.1016/0009-2509(73)80002-0

---

## 4. Debutanizer Column Characteristics

**Source:** Published literature

**License:** Academic/Fair Use

**Usage:** The debutanizer benchmark is characterized by significant measurement
delay due to gas chromatograph analysis time (~15-30 minutes). We implement
a delay wrapper based on published characteristics rather than using
proprietary plant data.

**Citation:**
> Fortuna, L., Graziani, S., Rizzo, A., and Xibilia, M.G. (2007).
> "Soft Sensors for Monitoring and Control of Industrial Processes."
> Springer-Verlag London. ISBN 978-1-84628-479-3.

---

## Fair Use Statement

This validation pack uses published academic constants, reference data from
public government sources, and synthesized test cases based on published
characteristics. All usage falls within fair use for academic and research
purposes:

1. **Transformative use:** Data is used to validate simulation accuracy,
   not as the primary product
2. **Non-commercial:** Intended for research and educational purposes
3. **Attribution:** All sources are properly cited
4. **No market impact:** Use of published constants does not substitute for
   or compete with the original publications

---

## Licensing Notes

- **NIST data:** Explicitly public domain as a US government work
- **Academic papers:** Constants and equations from published papers are
  used under fair use principles for research validation
- **No proprietary data:** This validation pack does not include or require
  any proprietary plant data or commercial software

---

## Contact

For questions about data licensing or attribution, please open an issue
in the project repository.
