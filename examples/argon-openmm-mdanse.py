"""
An example MDMC script for optimizing Lennard Jones parameters for liquid Ar.

We perform the calculation of S(Q,w) using MDANSE instead of the
MDMC internal implementation.

A copy of the data fitting against is assumed to be located in
../doc/tutorials/data/Well_s_q_omega_Ar_data.xml
"""

import copy

import numpy as np

from MDMC.control import Control
from MDMC.MD import Atom, NonBonded, Simulation, Universe
from MDMC.MD.interactions import NonBondedForce
from MDMC.readers.observables.xml_SQw import XML_SQw
from MDMC.refinement.FoM.FoM_abs import ObservablePair
from MDMC.trajectory_analysis.observables.mdanse_observable import (
    MDANSEObservable,
    create_mdanse_resolution,
    get_default_mdanse_settings,
    MDANSE_RESOLUTION_FUNCTIONS,
)


def run_everything():
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
        function=NonBonded(charge=0.0, epsilon=1.0, sigma=3.0),
    )

    # MD Engine setup. time_step of 10 fs is somewhat high, but for argon OK-ish.
    # If time_step is descreased by a factor consider increasing traj_step by the
    # same factor.
    simulation = Simulation(
        universe,
        engine="openmm",
        time_step=5.0,
        temperature=120.0,
        traj_step=15,
        openmm_platform="OpenCL",
    )

    # exp_datasets is a list of dictionaries with one dictionary per experimental
    # dataset
    exp_datasets = [
        {
            "file_name": "../doc/tutorials/data/Well_s_q_omega_Ar_data.xml",
            "type": "MDANSE",
            "reader": "xml_SQw",
            "weight": 1.0,
            "resolution": None,
            "cont_slicing": True,
        }
    ]

    start_params = get_default_mdanse_settings("SQw")
    print(f"Available resolution functions: {MDANSE_RESOLUTION_FUNCTIONS}")
    mdanse_resolution = create_mdanse_resolution(
        exp_datasets[0]["resolution"],
    )

    data_parser = XML_SQw("../doc/tutorials/data/Well_s_q_omega_Ar_data.xml")

    exp_observable = MDANSEObservable(mdanse_job_type="SQw")
    exp_observable.read_from_file(data_parser)
    md_observable = MDANSEObservable(mdanse_job_type="SQw")
    md_observable.origin = "MD"
    md_observable.independent_variables = copy.deepcopy(exp_observable.independent_variables)

    observable_pair = ObservablePair(
        exp_obs=exp_observable,
        MD_obs=md_observable,
        weight=1.0,
        rescale_factor=1.0,
        auto_scale=True,
    )

    fit_parameters = universe.parameters
    fit_parameters["sigma"].constraints = [2.0, 4.0]
    fit_parameters["epsilon"].constraints = [0.5, 1.5]

    job_settings = md_observable.initial_parameters()

    new_settings = {
        "running_mode": ("multicore", 8),
        "instrument_resolution": mdanse_resolution,
    }
    md_observable.set_parameters(new_settings)

    # Specify how the refinement is going to be controlled
    control = Control(
        simulation=simulation,
        exp_datasets=exp_datasets,
        fit_parameters=fit_parameters,
        observable_pairs=[observable_pair],
        equilibration_steps=9000,
        MD_steps=4800,
        cont_slicing=True,
        file_dump_extent="all",
        file_dump_frequency="best",
        FoM_options={"error": "none"},
        conv_tol=1e-6,
    )
    # Energy Minimization and equilibration
    control.minimize(n_steps=15000)
    control.equilibrate(n_steps=15000)

    # Run the refinement, i.e. refine the FF parameters against the data.
    control.refine(n_steps=200)


if __name__ == "__main__":
    run_everything()
