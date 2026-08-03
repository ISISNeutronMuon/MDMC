"""
An example MDMC script for optimizing Lennard Jones parameters for liquid Ar.
For info on syntax see the MDMC docs, including the jupyter notebook tutorials.
A copy of the data fitting against is assumed to be located in
../doc/tutorials/data/Well_s_q_omega_Ar_data.xml
"""

import os

# Currently MDMC uses OMP_NUM_THREADS to control the number of processes
# in the sqw calculation
os.environ["OMP_NUM_THREADS"] = "4"

import copy

from openmm import unit

from MDMC.control import Control
from MDMC.MD import Atom, NonBonded, Simulation, Universe
from MDMC.MD.interactions import NonBondedForce
from MDMC.refinement.FoM.FoM_abs import ObservablePair
from MDMC.trajectory_analysis.observables.sqw import SQw


# Build universe with density 0.0176 atoms per AA^-3
density = 0.0176
# This means cubic universe of side:
# 23.0668 A will contain 216 Ar atoms
# 26.911 A will contain 343 Ar atoms
# 30.7553 A will contain 512 Ar atoms
# 38.4441 A will contain 1000 Ar atoms
universe = Universe(dimensions=38.4441)
Ar = Atom("Ar[36]", charge=0.0)
# Calculating number of Ar atoms needed to obtain density
universe.fill(Ar, num_density=density)

# Above a universe of non-interacting argon atoms was created. Below
# specify how these atoms will interact
NonBondedForce(
    universe,
    Ar.atom_type,
    cutoff=10.0,
    ewald=1e-6,
    function=NonBonded(charge=0.0, epsilon=1.0, sigma=3.0),
)

# MD Engine setup. time_step of 10 fs is somewhat high, but for argon OK-ish.
# If time_step is descreased by a factor consider increasing traj_step by the
# same factor.
simulation = Simulation(
    universe,
    engine="openmm",
    time_step=10.18893,
    temperature=120.0,
    traj_step=15,
    openmm_platform="OpenCL",
    openmm_ensembles=[
        # equilibration stage 1 NVT to equilibrate the temperature quickly
        # using a large friction coefficient
        {
            "integrator": "LangevinMiddle",
            "frictionCoeff": 10.0 / unit.picoseconds,
            "n_step": 10000
        },
        # equilibration stage 2 now equilibrate the cell volume
        {
            "integrator": "LangevinMiddle",
            "frictionCoeff": 1.0 / unit.picoseconds,
            "barostat": {
                "barostat": "MonteCarlo",
                # https://journals.aps.org/pra/pdf/10.1103/PhysRevA.31.3391
                # table 2 measurement (a) 2.01 MPa
                "defaultPressure": 20.1 * unit.bar
            },
            "n_step": 10000
        },
        # equilibration stage 3 NVT again at the now equilibrated cell volume
        {
            "integrator": "LangevinMiddle",
            "frictionCoeff": 1.0 / unit.picoseconds,
            "n_step": 10000
        },
        # equilibration stage 4 NVE equilibration to prepare for production
        {
            "integrator": "Verlet",
            "n_step": 10000
        },
        # production NVE, n_steps not specified here since this is
        # determined by MDMC from the expt data
        {
            "integrator": "Verlet",
        },
    ]
)

# Energy Minimization and equilibration
# since we specified n_steps in the equilibration stage above in
# openmm_ensembles n_steps below will be ignored
simulation.run(n_steps=10000, equilibration=True)

# Setup refinement of the force field parameters

# exp_datasets is a list of dictionaries with one dictionary per experimental
# dataset
exp_datasets = [
    {
        "file_name": "../doc/tutorials/data/Well_s_q_omega_Ar_data.xml",
        "type": "SQw",
        "reader": "xml_SQw",
        "weight": 1.0,
        "resolution": None,
        "cont_slicing": True,
    }
]

exp_observable = SQw()
exp_observable.read_from_file("xml_SQw", "../doc/tutorials/data/Well_s_q_omega_Ar_data.xml")
md_observable = SQw()
md_observable.origin = "MD"
for obs in {exp_observable, md_observable}:
    obs.name = "SQw"
md_observable.independent_variables = copy.deepcopy(exp_observable.independent_variables)

observable_pair = ObservablePair(
    exp_obs=exp_observable, MD_obs=md_observable, weight=1.0, rescale_factor=1.0, auto_scale=True
)

fit_parameters = universe.parameters
fit_parameters["sigma"].constraints = [2.0, 4.0]
fit_parameters["epsilon"].constraints = [0.5, 1.5]

# Specify how the refinement is going to be controlled
control = Control(
    simulation=simulation,
    exp_datasets=exp_datasets,
    fit_parameters=fit_parameters,
    observable_pairs=[observable_pair],
    reset_config=True,
    file_dump_frequency="every",
    file_dump_extent="all",
    # since we specified n_steps in the equilibration stage above in
    # openmm_ensembles n_steps below will be ignored
    equilibration_steps=10000,
    MD_steps=16000,
    FoM_options={"error": "none"},
)

# Run the refinement, i.e. refine the FF parameters against the data.
control.refine(n_steps=1000)
