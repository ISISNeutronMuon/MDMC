"""
An example MDMC script for optimizing Lennard Jones parameters for liquid Ar.
For info on syntax see the MDMC docs, including the jupyter notebook tutorials.
A copy of the data fitting against is assumed to be located in
../doc/tutorials/data/Well_s_q_omega_Ar_data.xml
"""

import copy

import numpy as np

from MDMC.control import Control
from MDMC.MD import Atom, Molecule, NonBonded, Simulation, Universe
from MDMC.MD.interactions import NonBondedForce
from MDMC.readers.observables.csv_reader import csv_reader
from MDMC.refinement.FoM.FoM_abs import ObservablePair
from MDMC.trajectory_analysis.observables.mdanse_observable import (
    MDANSEObservable,
    get_default_mdanse_settings,
)

universe = Universe(dimensions=3.905*6)
O1 = Atom("O", position=np.array((0.5, 0.5, 0.0)) * 3.905, charge = -0.4)
O2 = Atom("O", position=np.array((0.5, 0.0, 0.5)) * 3.905, charge = -0.4)
O3 = Atom("O", position=np.array((0.0, 0.5, 0.5)) * 3.905, charge = -0.4)
Ti = Atom("Ti", position=np.array((0.5, 0.5, 0.5)) * 3.905, charge = 0.1)
Sr = Atom("Sr", position=np.array((0.0, 0.0, 0.0)) * 3.905, charge = 1.1)
# Calculating number of Ar atoms needed to obtain density
sto_unit = Molecule(position=(0, 0, 0),
                     velocity=(0, 0, 0),
                     atoms=[O1, O2, O3, Ti, Sr],
                     name='SrTiO3')
universe.fill(sto_unit, num_struc_units=6*6*6)

# Above an universe of non-interacting argon atoms was created. Below
# specify how these atoms will interact
NonBondedForce(
    universe,
    O1.atom_type,
    cutoff=10.0,
    ewald=1e-6,
    function=NonBonded(charge=-0.4, epsilon=5.0, sigma=2.0),
)
NonBondedForce(
    universe,
    Ti.atom_type,
    cutoff=10.0,
    ewald=1e-6,
    function=NonBonded(charge=0.1, epsilon=5.0, sigma=3.0),
)
NonBondedForce(
    universe,
    Sr.atom_type,
    cutoff=10.0,
    ewald=1e-6,
    function=NonBonded(charge=1.1, epsilon=5.0, sigma=3.0),
)

# MD Engine setup. time_step of 10 fs is somewhat high, but for argon OK-ish.
# If time_step is descreased by a factor consider increasing traj_step by the
# same factor.
simulation = Simulation(
    universe,
    engine="openmm",
    time_step=1.0,
    temperature=150.0,
    traj_step=4,
    openmm_platform="OpenCL",
)

# Setup refinement of the force field parameters

# exp_datasets is a list of dictionaries with one dictionary per experimental
# dataset
exp_datasets = [
    {
        "file_name": "sto_dos_as_text.csv",
        "type": "MDANSE",
        "reader": "csv_reader",
        "weight": 1.0,
        "auto_scale": True,
        "resolution": None,
    }
]

start_params = get_default_mdanse_settings("DensityOfStates")

data_parser = csv_reader("sto_dos_as_text.csv")
data_parser.parse()

exp_observable = MDANSEObservable(mdanse_job_type="DensityOfStates")
exp_observable.read_from_file(data_parser)
md_observable = MDANSEObservable(mdanse_job_type="DensityOfStates")
md_observable.origin = "MD"
md_observable.independent_variables = copy.deepcopy(exp_observable.independent_variables)

observable_pair = ObservablePair(
    exp_obs=exp_observable, MD_obs=md_observable, weight=1.0, rescale_factor=1.0, auto_scale=False
)

fit_parameters = universe.parameters
for par_name in fit_parameters:
    if "sigma" in par_name:
        fit_parameters[par_name].constraints = [0.5, 4.5]
    if "epsilon" in par_name:
        fit_parameters[par_name].constraints = [2.0, 80.0]
    if "charge" in par_name:
        fit_parameters[par_name].fixed = True

# Specify how the refinement is going to be controlled
control = Control(
    simulation=simulation,
    exp_datasets=exp_datasets,
    fit_parameters=fit_parameters,
    observable_pairs=[observable_pair],
    file_dump_frequency="best",
    file_dump_extent="all",
    file_dump_timestamped=False,
    reset_config=True,
    MD_steps=4000,
    equilibration_steps=20000,
    cont_slicing=True,
    sigma0=10.0,
    CMA_tolx=1e-6,
    conv_tol=1e-9,
)


# Run the refinement, i.e. refine the FF parameters against the data.
# n_steps = 3 is too small, but a good choice to first test this script
control.refine(n_steps=3000)
