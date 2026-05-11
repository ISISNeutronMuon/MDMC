"""
An example MDMC script for optimizing Lennard Jones parameters for liquid Ar.
For info on syntax see the MDMC docs, including the jupyter notebook tutorials.
A copy of the data fitting against is assumed to be located in
../doc/tutorials/data/Well_s_q_omega_Ar_data.xml
"""

import copy

import numpy as np

from MDMC.control import Control
from MDMC.MD import Atom, NonBonded, Simulation, Universe
from MDMC.MD.interactions import NonBondedForce
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
NonBondedForce(
    universe,
    Ar.atom_type,
    cutoff=10.0,
    ewald=1e-6,
    function=NonBonded(charge=0.0, epsilon=1.0, sigma=3.0)
)

# MD Engine setup. time_step of 10 fs is somewhat high, but for argon OK-ish.
# If time_step is descreased by a factor consider increasing traj_step by the
# same factor.
simulation = Simulation(universe,
                        engine="openmm",
                        time_step=5.0,
                        temperature=120.,
                        traj_step=20,
                        openmm_platform="OpenCL")

# Setup refinement of the force field parameters

# exp_datasets is a list of dictionaries with one dictionary per experimental
# dataset
exp_datasets = [{'file_name':'argon_dos_as_text.csv',
                 'type':'MDANSE',
                 'reader':'csv_reader',
                 'weight':1.,
                 'auto_scale': True,
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
fit_parameters['sigma'].constraints = [2.0, 4.0]
fit_parameters['epsilon'].constraints = [0.5, 1.5]

# Specify how the refinement is going to be controlled
control = Control(simulation=simulation,
                  exp_datasets=exp_datasets,
                  fit_parameters=fit_parameters,
                  observable_pairs= [observable_pair],
                  file_dump_frequency="best",
                  file_dump_extent="all",
                  file_dump_timestamped=False,
                  MD_steps=30000,
                  equilibration_steps=18000,
                  cont_slicing=True,
                  CMA_tolx = 1e-6,
                  conv_tol = 1e-9,
                  )

# Energy Minimization and equilibration
control.minimize(n_steps=15000)
control.equilibrate(n_steps=15000)

# Run the refinement, i.e. refine the FF parameters against the data.
# n_steps = 3 is too small, but a good choice to first test this script
control.refine(n_steps=300)
