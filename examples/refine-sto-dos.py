"""
Test run of the optimised SrTiO3 parameters.
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


O_charge = -0.7235399323821826
Ti_charge = 0.515738324378003
Sr_charge = -1*(Ti_charge + 3* O_charge)
O_epsilon = 23.65147247664405
O_sigma = 1.5826646662857802
Ti_epsilon = 20.289754968386067
Ti_sigma = 3.4195673565232605
Sr_epsilon = 9.737576223098147
Sr_sigma = 3.124497231730163

universe = Universe(dimensions=3.905*6)
O1 = Atom("O", position=np.array((0.5, 0.5, 0.0)) * 3.905, charge = O_charge)
O2 = Atom("O", position=np.array((0.5, 0.0, 0.5)) * 3.905, charge = O_charge)
O3 = Atom("O", position=np.array((0.0, 0.5, 0.5)) * 3.905, charge = O_charge)
Ti = Atom("Ti", position=np.array((0.5, 0.5, 0.5)) * 3.905, charge = Ti_charge)
Sr = Atom("Sr", position=np.array((0.0, 0.0, 0.0)) * 3.905, charge = Sr_charge)
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
    function=NonBonded(charge=O_charge, epsilon=O_epsilon, sigma=O_sigma, elements=["O"], molecules = ["SrTiO3"]),
)
NonBondedForce(
    universe,
    Ti.atom_type,
    cutoff=10.0,
    ewald=1e-6,
    function=NonBonded(charge=Ti_charge, epsilon=Ti_epsilon, sigma=Ti_sigma, elements=["Ti"], molecules = ["SrTiO3"]),
)
NonBondedForce(
    universe,
    Sr.atom_type,
    cutoff=10.0,
    ewald=1e-6,
    function=NonBonded(charge=Sr_charge, epsilon=Sr_epsilon, sigma=Sr_sigma, elements=["Sr"], molecules = ["SrTiO3"]),
)

# MD Engine setup. time_step of 10 fs is somewhat high, but for argon OK-ish.
# If time_step is descreased by a factor consider increasing traj_step by the
# same factor.
simulation = Simulation(
    universe,
    engine="openmm",
    time_step=1.0,
    temperature=80.0,
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

data_parser = csv_reader("sto_dft_dos_80K.csv")
data_parser.parse(scale_factor=1e5)

exp_observable = MDANSEObservable(mdanse_job_type="DensityOfStates")
exp_observable.read_from_file(data_parser)
md_observable = MDANSEObservable(mdanse_job_type="DensityOfStates")
md_observable.origin = "MD"
md_observable.independent_variables = copy.deepcopy(exp_observable.independent_variables)

observable_pair_dos = ObservablePair(
    exp_obs=exp_observable, MD_obs=md_observable, weight=1.0, rescale_factor=1.0, auto_scale=True
)


start_params_OO = get_default_mdanse_settings("DensityOfStates")

data_parser_OO = csv_reader("sto_dft_dos_80K_OO.csv")
data_parser_OO.parse(scale_factor=1e5)

exp_observable_OO = MDANSEObservable(mdanse_job_type="DensityOfStates")
exp_observable_OO.read_from_file(data_parser)
md_observable_OO = MDANSEObservable(mdanse_job_type="DensityOfStates")
md_observable_OO.origin = "MD"
md_observable_OO.independent_variables = copy.deepcopy(exp_observable.independent_variables)

md_observable_OO.set_parameters({
    "instrument_resolution": ['gaussian', {'mu': 0.0, 'sigma': 0.6452}],
    "atom_selection": """{"0": {"function_name": "select_all", "operation_type": "union"}, "1": {"function_name": "select_dummy", "operation_type": "difference"}, "2": {"function_name": "select_atoms", "atom_types": ["O"], "operation_type": "intersection"}}"""
})

observable_pair_dos_OO = ObservablePair(
    exp_obs=exp_observable_OO, MD_obs=md_observable_OO, weight=1.0, rescale_factor=1.0, auto_scale=True
)

start_params_pdf = get_default_mdanse_settings("PairDistributionFunction")

data_parser_pdf = csv_reader("sto_dft_pdf_80K_fine.csv")
data_parser_pdf.parse()

exp_observable_pdf = MDANSEObservable(mdanse_job_type="PairDistributionFunction")
exp_observable_pdf.read_from_file(data_parser_pdf)
md_observable_pdf = MDANSEObservable(mdanse_job_type="PairDistributionFunction")
md_observable_pdf.origin = "MD"
md_observable_pdf.independent_variables = copy.deepcopy(exp_observable_pdf.independent_variables)

observable_pair_pdf = ObservablePair(
    exp_obs=exp_observable_pdf, MD_obs=md_observable_pdf, weight=2.0, rescale_factor=1.0, auto_scale=False
)

md_observable_pdf.set_parameters({"r_values":[0.0,0.77,0.002],
                                  "frames":[5,1000,10]})

start_params_pdf_OO = get_default_mdanse_settings("PairDistributionFunction")

data_parser_pdf_OO = csv_reader("sto_dft_pdf_80K_OO_fine.csv")
data_parser_pdf_OO.parse()

exp_observable_pdf_OO = MDANSEObservable(mdanse_job_type="PairDistributionFunction")
exp_observable_pdf_OO.read_from_file(data_parser_pdf)
md_observable_pdf_OO = MDANSEObservable(mdanse_job_type="PairDistributionFunction")
md_observable_pdf_OO.origin = "MD"
md_observable_pdf_OO.independent_variables = copy.deepcopy(exp_observable_pdf.independent_variables)

observable_pair_pdf_OO = ObservablePair(
    exp_obs=exp_observable_pdf_OO, MD_obs=md_observable_pdf_OO, weight=2.0, rescale_factor=1.0, auto_scale=False
)

md_observable_pdf_OO.set_parameters({"r_values":[0.0,0.77,0.002],
                                  "frames":[5,1000,10],
                                  "atom_selection": """{"0": {"function_name": "select_all", "operation_type": "union"}, "1": {"function_name": "select_dummy", "operation_type": "difference"}, "2": {"function_name": "select_atoms", "atom_types": ["O"], "operation_type": "intersection"}}"""})

fit_parameters = universe.parameters
for par_name in fit_parameters:
    par = fit_parameters[par_name]
    print(par.name, par.value, par.molecules)
    if "sigma" in par_name:
        fit_parameters[par_name].constraints = [1.1, 5.0]
    if "epsilon" in par_name:
        fit_parameters[par_name].constraints = [5.0, 65.0]
    if "charge" in par_name:
        if fit_parameters[par_name].value < -0.5:
            fit_parameters[par_name].constraints = [-2.0, -0.5]
        elif fit_parameters[par_name].value < 1.0:
            fit_parameters[par_name].constraints = [0.4, 4.0]
        else:
            fit_parameters[par_name].fixed = True

# Specify how the refinement is going to be controlled
control = Control(
    simulation=simulation,
    exp_datasets=exp_datasets,
    fit_parameters=fit_parameters,
    observable_pairs=[observable_pair_dos, observable_pair_dos_OO, observable_pair_pdf, observable_pair_pdf_OO],
    file_dump_frequency="best",
    file_dump_extent="all",
    file_dump_timestamped=False,
    file_dump_prefix="SrTiO3_4obs",
    FoM_options={"error": "none"},
    reset_config=True,
    MD_steps=5000,
    equilibration_steps=40000,
    cont_slicing=True,
    sigma0=0.08,
    CMA_popsize=16,
    CMA_tolx=1e-6,
    conv_tol=1e-4,
)


# Run the refinement, i.e. refine the FF parameters against the data.
# n_steps = 3 is too small, but a good choice to first test this script
control.refine(n_steps=2400)
