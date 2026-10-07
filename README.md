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

# Installation instructions

## Set up a virtual environment

We encourage installing MDMC in a Python virtual environment.
A basic way of creating one in your current working directory
is by using Python's venv module:
```
python -m venv mdmc_venv
```

When you want to use the environment, activate it by running
```
source mdmc_venv/bin/activate
```
or
```
mdmc_venv\Scripts\activate.bat
```

There are other popular tools for creating and using
virtual environments, including conda and uv. If you are familiar
with them, you are welcome to use them instead.

## Get the code

Clone the repository and install the software by running
```
git clone https://github.com/ISISNeutronMuon/MDMC
cd MDMC
pip install .
```




# Getting started