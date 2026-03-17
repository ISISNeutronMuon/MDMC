"""
An example MDMC script for optimizing Lennard Jones parameters for liquid Ar.
For info on syntax see the MDMC docs, including the jupyter notebook tutorials.
A copy of the data fitting against is assumed to be located in
../doc/tutorials/data/Well_s_q_omega_Ar_data.xml
"""

import numpy as np
import os
# Change the number of threads depending on the number of physical cores on
# your computer as it was tested for LAMMPS
os.environ["OMP_NUM_THREADS"] = "4"

from MDMC.control import Control
from MDMC.MD import Atom, LennardJones, Simulation, Universe
from MDMC.MD.interactions import Dispersion
from MDMC.trajectory_analysis.observables.mdanse_observable import MDANSEObservable, get_default_mdanse_settings

# Build universe with density 0.0176 atoms per AA^-3
density = 0.0176
# This means cubic universe of side:
# 23.0668 A will contain 216 Ar atoms
# 26.911 A will contain 343 Ar atoms
# 30.7553 A will contain 512 Ar atoms
# 38.4441 A will contain 1000 Ar atoms
universe = Universe(dimensions=38.4441)
Ar = Atom('Ar', charge=0.)
# Calculating number of Ar atoms needed to obtain density
n_ar_atoms = int(density * np.prod(universe.dimensions))
print(n_ar_atoms)
universe.fill(Ar, num_struc_units=(n_ar_atoms))

""" # Above an universe of non-interacting argon atoms was created. Below
# specify how these atoms will interact
Ar_dispersion = Dispersion(universe,
                           (Ar.atom_type, Ar.atom_type),
                           cutoff=8.,
                           vdw_tail_correction=True,
                           function=LennardJones(1.0243, 3.36))

# MD Engine setup. time_step of 10 fs is somewhat high, but for argon OK-ish.
# If time_step is descreased by a factor consider increasing traj_step by the
# same factor.
simulation = Simulation(universe,
                        engine="lammps",
                        time_step=10.18893,
                        temperature=120.,
                        traj_step=15) """

# Setup refinement of the force field parameters

# exp_datasets is a list of dictionaries with one dictionary per experimental
# dataset
exp_datasets = [{'file_name':'../doc/tutorials/data/Well_s_q_omega_Ar_data.xml',
                 'type':'SQw',
                 'reader':'xml_SQw',
                 'weight':1.,
                 'resolution':None}]

start_params = get_default_mdanse_settings("DensityOfStates")
print(start_params)

""" fit_parameters = universe.parameters

# Specify how the refinement is going to be controlled
control = Control(simulation=simulation,
                  exp_datasets=exp_datasets,
                  fit_parameters=fit_parameters,
                  MD_steps=570)

# Energy Minimization and equilibration
control.minimize(n_steps=5000)
control.equilibrate(n_steps=50000)

# Run the refinement, i.e. refine the FF parameters against the data.
# n_steps = 3 is too small, but a good choice to first test this script
control.refine(n_steps=3)
 """