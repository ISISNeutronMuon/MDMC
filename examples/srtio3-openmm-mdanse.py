"""
A test case for a solid system.

This scripts builds a cubic SrTiO3 supercell and defines bonds
between nearest neighbour pairs of Ti and O, resulting in a
network of bonds that extends indefinitely in 3 dimensions.
The engine parameters had to be adjusted to handle this correctly.

The reference datasets are not taken from measurements, but calculated
from an ab-initio MD simulation using VASP 4.6.
"""

import copy

import numpy as np

from MDMC.control import Control
from MDMC.MD import Atom, HarmonicPotential, Molecule, NonBonded, Simulation, Universe
from MDMC.MD.interactions import Bond, NonBondedForce
from MDMC.readers.observables.csv_reader import csv_reader
from MDMC.refinement.FoM.FoM_abs import ObservablePair
from MDMC.trajectory_analysis.observables.mdanse_observable import (
    MDANSEObservable,
    get_default_mdanse_settings,
)

BOX_EDGE = 3.905*6

O_charge = -1.3
Ti_charge = 0.9
Sr_charge = -1*(Ti_charge + 3* O_charge)
O_epsilon = 15
O_sigma = 1.05
Ti_epsilon = 10.0
Ti_sigma = 2.5
Sr_epsilon = 14.0
Sr_sigma = 2.9
TiO_bond_length = 1.85
TiO_bond_strength = 15.0

universe = Universe(dimensions=BOX_EDGE)
O1 = Atom("O", position=np.array((0.5, 0.5, 0.0)) * 3.905, charge = O_charge)
O2 = Atom("O", position=np.array((0.5, 0.0, 0.5)) * 3.905, charge = O_charge)
O3 = Atom("O", position=np.array((0.0, 0.5, 0.5)) * 3.905, charge = O_charge)
Ti = Atom("Ti", position=np.array((0.5, 0.5, 0.5)) * 3.905, charge = Ti_charge)
Sr = Atom("Sr", position=np.array((0.0, 0.0, 0.0)) * 3.905, charge = Sr_charge)
# Fortunately, the fill method translates the molecules without rotating.
sto_unit = Molecule(position=(0, 0, 0),
                     velocity=(0, 0, 0),
                     atoms=[O1, O2, O3, Ti, Sr],
                     name='SrTiO3')
universe.fill(sto_unit, num_struc_units=6*6*6)

# Now the system is the cubic structure of SrTiO3 in a 6x6x6 supercell.

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

# A custom-made section which creates bonds that span across the MDMC "molecules"

ti_o_bonds = []

finished = False

for atom in universe.atoms:
    if atom.element.symbol == "Ti":
        ti_pos = atom.position
        for other_atom in universe.atoms:
            if other_atom.element.symbol == "O":
                distance = np.abs(ti_pos - other_atom.position)
                correct = all((distance < 2.1) | (distance > 21.2))
                if correct:
                    ti_o_bonds.append((atom, other_atom))

tio_bond = Bond(*ti_o_bonds)
tio_bond.function = HarmonicPotential(TiO_bond_length,
                                      TiO_bond_strength,
                                      interaction_type = "bond")


simulation = Simulation(
    universe,
    engine="openmm",
    time_step=1.0,
    temperature=80.0,
    traj_step=4,
    openmm_platform="OpenCL",
    openmm_nonbonded_scaling=[
        [0.0, 1.0, 0.0],
        [0.5, 1.0, 0.5],
        [1.0, 1.0, 1.0],
    ],
    openmm_bonds_use_pbc=True,
    openmm_exceptions_use_pbc=True,
)

# Setup refinement of the force field parameters

start_params = get_default_mdanse_settings("DensityOfStates")

data_parser = csv_reader("sto_dft_dos_80K.csv")
data_parser.parse(scale_factor=1e5)

exp_observable = MDANSEObservable(mdanse_job_type="DensityOfStates")
exp_observable.read_from_file(data_parser)
md_observable = MDANSEObservable(mdanse_job_type="DensityOfStates")
md_observable.origin = "MD"
md_observable.independent_variables = copy.deepcopy(exp_observable.independent_variables)
md_observable.set_parameters({
    "instrument_resolution": ['gaussian', {'mu': 0.0, 'sigma': 0.6452}],
})

observable_pair_dos = ObservablePair(
    exp_obs=exp_observable, MD_obs=md_observable, weight=1.0, rescale_factor=1.0, auto_scale=True
)


data_parser_pdf = csv_reader("sto_dft_pdf_80K_fine.csv")
data_parser_pdf.parse()

exp_observable_pdf = MDANSEObservable(mdanse_job_type="PairDistributionFunction")
exp_observable_pdf.read_from_file(data_parser_pdf)
md_observable_pdf = MDANSEObservable(mdanse_job_type="PairDistributionFunction")
md_observable_pdf.origin = "MD"
md_observable_pdf.independent_variables = copy.deepcopy(exp_observable_pdf.independent_variables)
md_observable_pdf.set_parameters({"r_values":[0.0,0.77,0.002],
                                  "frames":[5,1000,10]})

observable_pair_pdf = ObservablePair(
    exp_obs=exp_observable_pdf, MD_obs=md_observable_pdf, weight=2.0, rescale_factor=1.0, auto_scale=False
)


fit_parameters = universe.parameters
for par_name in fit_parameters:
    par = fit_parameters[par_name]
    if "sigma" in par_name:
        fit_parameters[par_name].constraints = [0.1, 5.0]
    if "epsilon" in par_name:
        fit_parameters[par_name].constraints = [0.1, 65.0]
    if "charge" in par_name:
        if fit_parameters[par_name].value < -0.5:
            fit_parameters[par_name].constraints = [-2.0, -0.1]
        elif fit_parameters[par_name].value < 1.0:
            fit_parameters[par_name].constraints = [0.4, 4.0]
        else:
            fit_parameters[par_name].fixed = True
    if "equilibrium" in par_name:
        fit_parameters[par_name].constraints = [1.6, 2.2]
    if "potential" in par_name:
        fit_parameters[par_name].constraints = [1.0, 45.0]


control = Control(
    simulation=simulation,
    exp_datasets={},
    fit_parameters=fit_parameters,
    observable_pairs=[observable_pair_dos, observable_pair_pdf], # two observables are used here.
    file_dump_frequency="best",
    file_dump_extent="all",
    file_dump_timestamped=False,
    file_dump_prefix="SrTiO3_4obs",
    FoM_options={"error": "none"},
    reset_config=True,
    MD_steps=5000,
    equilibration_steps=40000,
    cont_slicing=True,
    sigma0=0.25,
    CMA_tolx=1e-6,
    conv_tol=1e-4,
)

control.refine(n_steps=2400)
