# MDMC

The main goal of MDMC is to refine interatomic force fields, using neutron
scattering experiment results as reference. This is achieved by running
molecular dynamics (MD) simulations iteratively for different force field
parameters and minimising the difference between the calculated and
experimentally observed properties.

As of 2026, the MD simulations in MDMC are run using [OpenMM](https://openmm.org) and
the optimisation is driven by [CMA-ES](https://github.com/CMA-ES/pycma).
MDMC's own code includes calculators for both pair distribution function (PDF) and
the quasielastic neutron scattering results as S(q,w). The same and many other
observables can also be calculated using [MDANSE](https://github.com/ISISNeutronMuon/MDANSE).

[![codecov](https://codecov.io/gh/MDMCproject/MDMCv0.2_pilot/branch/master/graph/badge.svg?token=Ysd1yn7alI)](https://codecov.io/gh/MDMCproject/MDMCv0.2_pilot)
