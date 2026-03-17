"""
An example MDMC script for optimizing Lennard Jones parameters for liquid Ar.
For info on syntax see the MDMC docs, including the jupyter notebook tutorials.
A copy of the data fitting against is assumed to be located in
../doc/tutorials/data/Well_s_q_omega_Ar_data.xml
"""

import copy

import numpy as np

from MDMC.control import Control
from MDMC.MD import Atom, LennardJones, Simulation, Universe
from MDMC.MD.interactions import Dispersion
from MDMC.readers.observables.csv_reader import csv_reader
from MDMC.refinement.FoM.FoM_abs import ObservablePair
from MDMC.trajectory_analysis.observables.mdanse_observable import (
                 MDANSEObservable,
                 get_default_mdanse_settings,
)

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

# Above an universe of non-interacting argon atoms was created. Below
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
                        engine="openmm",
                        time_step=10.18893,
                        temperature=120.,
                        traj_step=15,
                        openmm_platform="OpenCL")

# Setup refinement of the force field parameters

# exp_datasets is a list of dictionaries with one dictionary per experimental
# dataset
exp_datasets = [{'file_name':'argon_dos_as_text.csv',
                 'type':'MDANSE',
                 'reader':'csv_reader',
                 'weight':1.,
                 'resolution':None}]

start_params = get_default_mdanse_settings("DensityOfStates")

data_parser = csv_reader('argon_dos_as_text.csv')
data_parser.parse()

exp_observable = MDANSEObservable(mdanse_job_type="DensityOfStates")
exp_observable.read_from_file(data_parser)
md_observable = MDANSEObservable(mdanse_job_type="DensityOfStates")
md_observable.origin = 'MD'
md_observable.independent_variables = copy.deepcopy(
    exp_observable.independent_variables)

observable_pair = ObservablePair(exp_obs=exp_observable,
                                 MD_obs=md_observable,
                                 weight=1.0,
                                 rescale_factor=1.0,
                                 auto_scale=False)

fit_parameters = universe.parameters

# Specify how the refinement is going to be controlled
control = Control(simulation=simulation,
                  exp_datasets=exp_datasets,
                  fit_parameters=fit_parameters,
                  observable_pairs= [observable_pair],
                  MD_steps=5700,
                  equilibration_steps=8000,
                  cont_slicing=True)

# Energy Minimization and equilibration
control.minimize(n_steps=5000)
control.equilibrate(n_steps=5000)

# Run the refinement, i.e. refine the FF parameters against the data.
# n_steps = 3 is too small, but a good choice to first test this script
control.refine(n_steps=300)
