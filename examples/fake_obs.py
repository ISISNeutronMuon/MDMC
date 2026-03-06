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
from MDMC.MD import Parameter, Simulation, Universe

universe = Universe(dimensions=10.0)
universe.parameters = [Parameter(name="centre_x", value=1.0, constraints=(0.0,10.0)),
                       Parameter(name="centre_y", value=1.0, constraints=(0.0,10.0)),
                       Parameter(name="width_x", value=1.0, constraints=(0.0,10.0)),
                       Parameter(name="width_y", value=1.0, constraints=(0.0,10.0))]

# MD Engine setup. time_step of 10 fs is somewhat high, but for argon OK-ish.
# If time_step is descreased by a factor consider increasing traj_step by the
# same factor.
simulation = Simulation(universe,
                        engine="null_engine",
                        time_step=1,
                        temperature=1,
                        traj_step=1)

# Setup refinement of the force field parameters

# exp_datasets is a list of dictionaries with one dictionary per experimental
# dataset
exp_datasets = [{'file_name':'../doc/tutorials/data/Well_s_q_omega_Ar_data.xml',
                 'type':'gauss2D',
                 'reader':'xml_SQw',
                 'weight':1.,
                 'resolution':None}]

fit_parameters = universe.parameters

# Specify how the refinement is going to be controlled
control = Control(simulation=simulation,
                  exp_datasets=exp_datasets,
                  fit_parameters=fit_parameters,
                  MD_steps=5,
                  equilibration_steps=5)

# Run the refinement, i.e. refine the FF parameters against the data.
# n_steps = 3 is too small, but a good choice to first test this script
control.refine(n_steps=300)
