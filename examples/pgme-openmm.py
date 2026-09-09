"""
Prototype script for working with I(q,t) data.
The simulation at this stage does not match the data.
"""

import os

# Currently MDMC uses OMP_NUM_THREADS to control the number of processes
# in the sqw calculation
os.environ["OMP_NUM_THREADS"] = "4"

import copy

from MDMC.control import Control
from MDMC.MD import Atom, NonBonded, Simulation, Universe
from MDMC.MD.interactions import NonBondedForce
from MDMC.readers.observables.text_IQt import text_IQt
from MDMC.refinement.FoM.FoM_abs import ObservablePair
from MDMC.trajectory_analysis.observables.mdanse_observable import (
    MDANSEObservable,
    create_mdanse_resolution,
    get_default_mdanse_settings,
    MDANSE_RESOLUTION_FUNCTIONS,
)


def normalise_to_first_value(input_2D_array):
    return input_2D_array / input_2D_array[:1, :]


if __name__ == "__main__":
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
        time_step=5,
        temperature=120.0,
        traj_step=300,
        openmm_platform="OpenCL",
    )

    # Energy Minimization and equilibration
    simulation.run(n_steps=30000, equilibration=True)

    # Setup refinement of the force field parameters

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

    data_parser = text_IQt(
        [
            ["../doc/tutorials/data/irs40979_fqt__spec_1.asc", 0.444],
            ["../doc/tutorials/data/irs40979_fqt__spec_3.asc", 0.617],
            ["../doc/tutorials/data/irs40979_fqt__spec_5.asc", 0.784],
            ["../doc/tutorials/data/irs40979_fqt__spec_7.asc", 0.954],
            ["../doc/tutorials/data/irs40979_fqt__spec_9.asc", 1.191],
        ]
    )
    data_parser.parse(axis1_limits=(0.0, 0.15))

    exp_observable = MDANSEObservable(mdanse_job_type="SQw")
    exp_observable.read_from_file(data_parser)
    md_observable = MDANSEObservable(mdanse_job_type="SQw", pick_dataset="/ndsf/f(q,t)/total")
    md_observable.origin = "MD"
    md_observable.independent_variables = copy.deepcopy(exp_observable.independent_variables)

    print(exp_observable.independent_variables)
    print(exp_observable.dependent_variables)

    observable_pair = ObservablePair(
        exp_obs=exp_observable,
        MD_obs=md_observable,
        weight=1.0,
        rescale_factor=1.0,
        auto_scale=False,
    )
    observable_pair.postprocessing_function = normalise_to_first_value

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
        cont_slicing=True,
        file_dump_frequency="best",
        file_dump_extent="all",
        equilibration_steps=60000,
        MD_steps=60000,
        FoM_options={"error": "none"},
    )

    # Run the refinement, i.e. refine the FF parameters against the data.
    control.refine(n_steps=1000)
