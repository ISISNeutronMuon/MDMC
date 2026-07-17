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
O1 = Atom("O", position=np.array((0.5, 0.5, 0.0)) * 3.905, charge = -0.492)
O2 = Atom("O", position=np.array((0.5, 0.0, 0.5)) * 3.905, charge = -0.492)
O3 = Atom("O", position=np.array((0.0, 0.5, 0.5)) * 3.905, charge = -0.492)
Ti = Atom("Ti", position=np.array((0.5, 0.5, 0.5)) * 3.905, charge = -0.079)
Sr = Atom("Sr", position=np.array((0.0, 0.0, 0.0)) * 3.905, charge = 1.555)
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
    function=NonBonded(charge=-0.492, epsilon=24.483, sigma=1.0136, elements=["O"], molecules = ["SrTiO3"]),
)
NonBondedForce(
    universe,
    Ti.atom_type,
    cutoff=10.0,
    ewald=1e-6,
    function=NonBonded(charge=-0.079, epsilon=19.676, sigma=3.05, elements=["Ti"], molecules = ["SrTiO3"]),
)
NonBondedForce(
    universe,
    Sr.atom_type,
    cutoff=10.0,
    ewald=1e-6,
    function=NonBonded(charge=1.555, epsilon=10.195, sigma=3.295, elements=["Sr"], molecules = ["SrTiO3"]),
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

start_params_pdf = get_default_mdanse_settings("PairDistributionFunction")

data_parser_pdf = csv_reader("sto_dft_pdf_80K.csv")
data_parser_pdf.parse()

exp_observable_pdf = MDANSEObservable(mdanse_job_type="PairDistributionFunction")
exp_observable_pdf.read_from_file(data_parser_pdf)
md_observable_pdf = MDANSEObservable(mdanse_job_type="PairDistributionFunction")
md_observable_pdf.origin = "MD"
md_observable_pdf.independent_variables = copy.deepcopy(exp_observable_pdf.independent_variables)

observable_pair_pdf = ObservablePair(
    exp_obs=exp_observable_pdf, MD_obs=md_observable_pdf, weight=4.0, rescale_factor=1.0, auto_scale=False
)
print(start_params_pdf)

md_observable_pdf.set_parameters({"r_values":[0.0,0.77,0.01],
                                  "frames":[50,1000,30]})

fit_parameters = universe.parameters
for par_name in fit_parameters:
    par = fit_parameters[par_name]
    print(par.name, par.value, par.molecules)
    if "sigma" in par_name:
        fit_parameters[par_name].constraints = [0.1, 5.0]
    if "epsilon" in par_name:
        fit_parameters[par_name].constraints = [0.01, 45.0]
    if "charge" in par_name:
        if fit_parameters[par_name].value < -0.1:
            fit_parameters[par_name].constraints = [-2.0, 0.05]
        elif fit_parameters[par_name].value < 0.6:
            fit_parameters[par_name].constraints = [-1.0, 1.2]
        else:
            fit_parameters[par_name].fixed = True

# Specify how the refinement is going to be controlled
control = Control(
    simulation=simulation,
    exp_datasets=exp_datasets,
    fit_parameters=fit_parameters,
    observable_pairs=[observable_pair_dos, observable_pair_pdf],
    file_dump_frequency="best",
    file_dump_extent="all",
    file_dump_timestamped=False,
    file_dump_prefix="SrTiO3_MDMC_DOS_PDF_fromVASP",
    FoM_options={"error": "none"},
    reset_config=True,
    MD_steps=4000,
    equilibration_steps=30000,
    cont_slicing=True,
    sigma0=0.04,
    CMA_popsize=16,
    CMA_tolx=1e-6,
    conv_tol=1e-4,
)


# Run the refinement, i.e. refine the FF parameters against the data.
# n_steps = 3 is too small, but a good choice to first test this script
control.refine(n_steps=2400)
