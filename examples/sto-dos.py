"""
A basic example showing how to run MDMC with MDANSEObservable.
The reference data used here have been generated using MDANSE
and not taken from an experiment.
The observable used here is density of states, and the simulated
system is strontium titanate (SrTiO3).
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

NonBondedForce(
    universe,
    O1.atom_type,
    cutoff=10.0,
    ewald=1e-6,
    function=NonBonded(charge=-0.4, epsilon=5.0, sigma=1.35, elements=["O"], molecules = ["SrTiO3"]),
)
NonBondedForce(
    universe,
    Ti.atom_type,
    cutoff=10.0,
    ewald=1e-6,
    function=NonBonded(charge=0.1, epsilon=15.0, sigma=1.9, elements=["Ti"], molecules = ["SrTiO3"]),
)
NonBondedForce(
    universe,
    Sr.atom_type,
    cutoff=10.0,
    ewald=1e-6,
    function=NonBonded(charge=1.1, epsilon=2.0, sigma=1.9, elements=["Sr"], molecules = ["SrTiO3"]),
)

simulation = Simulation(
    universe,
    engine="openmm",
    time_step=1.0,
    temperature=150.0,
    traj_step=4,
    openmm_platform="OpenCL",
)


# This list should not be needed for MDANSEObservable,
# but is still expected by Control.
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
data_parser.parse(scale_factor=1e5)

exp_observable = MDANSEObservable(mdanse_job_type="DensityOfStates")
exp_observable.read_from_file(data_parser)
md_observable = MDANSEObservable(mdanse_job_type="DensityOfStates")
md_observable.origin = "MD"
md_observable.independent_variables = copy.deepcopy(exp_observable.independent_variables)

observable_pair = ObservablePair(
    exp_obs=exp_observable, MD_obs=md_observable, weight=1.0, rescale_factor=1.0, auto_scale=True
)

fit_parameters = universe.parameters
for par_name in fit_parameters:
    par = fit_parameters[par_name]
    print(par.name, par.value, par.molecules)
    if "sigma" in par_name:
        fit_parameters[par_name].constraints = [0.3, 5.0]
    if "epsilon" in par_name:
        fit_parameters[par_name].constraints = [0.0, 40.0]
    if "charge" in par_name:
        if fit_parameters[par_name].value < 0.0:
            fit_parameters[par_name].constraints = [-1.2, 0.0]
        elif fit_parameters[par_name].value < 1.0:
            fit_parameters[par_name].constraints = [0.0, 1.2]
        else:
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
    file_dump_prefix="SrTiO3_MDMC_fitToDOS",
    FoM_options={"error": "none"},
    reset_config=True,
    MD_steps=4000,
    equilibration_steps=30000,
    cont_slicing=True,
    sigma0=2.0,
    CMA_popsize=16,
    CMA_tolx=1e-6,
    conv_tol=1e-6,
)


# Run the refinement, i.e. refine the FF parameters against the data.
# n_steps = 3 is too small, but a good choice to first test this script
control.refine(n_steps=3000)
