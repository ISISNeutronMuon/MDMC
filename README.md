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

Instead of using git, you can also ownload the code from
the [MDMC Github repository](https://github.com/ISISNeutronMuon/MDMC)
as a zip file, then install using pip same as above.

# Getting started

## Tutorials

For a step-by-step introduction to MDMC and its workflow, you can try out
the Jupyter notebooks located in `doc/tutorials`. They contain Python code
together with the text explaining the role of each code section.

## Examples

The `examples` directory contains Python scripts designed for specific
optimisation tasks. Depending on the kind of system you want to simulate,
you can try one of the example scripts and use them as the starting point
for creating your own.

#### Argon

Simple example of a monoatomic gas. There are no chemical bonds or
electrostatic interactions. The reference data are the results
of a QENS experiment on argon-36, which has 0 incoherent scattering
cross-section.

- argon-openmm.py - a basic refinement using MDMC.
- argon-openmm-ensembles.py - the equilibration is done in multiple run with different ensembles.
- argon-openmm-mdanse.py - the S(Q,w) is calculated using MDANSE instead of MDMC's own implementation

#### Water

Simulations of water using TIP3P and TIP4P potentials. The reference data
are the QENS results for supercooled water at T=263K. This simulation
includes small molecules with bonds and angles, and atoms with non-zero
electric charge.

- water-openmm.py - a refinement using a TIP4P water model.
- water-openmm-charge.py - uses TIP3P as a starting point, and allows electric charge to be refined.
- water-openmm-mdanse.py - the S(Q,w) calculation is performed using MDANSE

#### Methanol

These examples do not result in a refinement, but illustrate how a molecule
can be defined in MDMC.

- methanol-openmm.py - runs a simulation of a system containing methanol molecules
- methanol-water-from-file.py - loads a system from a PDB file and runs an MD simulation using
  OPLS force field. Also saves the trajectory as an H5MD file.

#### Paracetamol

Example of loading a molecule from a CIF file.

- paracetamol-openmm.py - loads the molecule definition from a CIF file and runs an MD simulation
  of a system containing several copies of the molecule

#### Benzene

Refinement example using benzene molecules. Currently does not include
valid experimental results for benzene.

- benzene-mdanse.py - loads a benzene molecule structure from a PDB file; bond angles and
  dihedral angles are added manually. Runs a refinement based on QENS results calculated
  using MDANSE.

#### Strontium titanate

Refinement is performed for a solid system. Reference data are simulated, produced from an
MD simulation performed using VASP and analysed using MDANSE.

- sto-dos.py - refinement aiming to reproduce the vibrational density of states of the reference system.
- srtio3-openmm-mdanse.py - This simulation uses chemical bonds which span across the entire simulation
  box and form an infinite network within the periodic boundary of the system. Also, it uses two
  observables that are included in the fitting at each step. The refinement aims to reproduce both
  the reference pair distribution function (PDF) and vibrational density of states (DOS).
