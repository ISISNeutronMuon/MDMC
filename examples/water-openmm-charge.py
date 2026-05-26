import copy

from MDMC.MD import *
from MDMC.MD.force_fields.three_site_water import ThreeSiteWater, add_three_site_water_ff
from MDMC.control import Control
from MDMC.readers.observables.LAMPSQw import LAMPSQw
from MDMC.refinement.FoM.FoM_abs import ObservablePair
from MDMC.trajectory_analysis.observables.mdanse_observable import (
    MDANSEObservable,
    create_mdanse_resolution,
    get_default_mdanse_settings,
)


# Currently MDMC uses OMP_NUM_THREADS to control the number of processes
# in the sqw calculation
# Build universe
# Cubic universe of side:
# 18.6270199 A is 216 water molecules
# 21.731523217 is 343 water molecules
# 24.83602653 is 512 water molecules
def run_everything():
    universe = Universe(dimensions=24.83602653)
    universe.fill(ThreeSiteWater(model_name="TIP3P", name="H2O"), num_density=0.03356718472021752)
    add_three_site_water_ff(universe, cutoff=10.0, ewald=1e-4, model_name="TIP3P")

    # Setup refinement
    QENS = [
        {
            "file_name": "../doc/tutorials/data/263K05Awat_LAMP",
            "type": "SQw",
            "reader": "LAMPSQw",
            "weight": 1.0,
            "auto_scale": True,
            "use_FFT": True,
            "resolution": {"file": "../doc/tutorials/data/262p7K0A5van_LAMP"},
            "cont_slicing": True,
        }
    ]

    start_params = get_default_mdanse_settings("SQw")

    data_parser = LAMPSQw("../doc/tutorials/data/263K05Awat_LAMP")

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

    job_settings = md_observable.initial_parameters()
    mdanse_resolution = create_mdanse_resolution(
        13.6,
    )

    new_settings = {
        "running_mode": ("multicore", 8),
        "q_vectors": (
            "SphericalLatticeQVectors",
            {
                "shells": [1.0, 21.0, 2.0],
                "width": 1.0,
                "n_samples": 500000,
                "n_vectors": 120,
                "seed": 0,
                "force_equal_weights": False,
            },
        ),
        "instrument_resolution": mdanse_resolution,
    }
    md_observable.set_parameters(new_settings)

    # only refit the LJ parameters on oxygen
    for p in universe.parameters.as_array:
        if p.parameter_name == "TIP3P-O-nonbonded_epsilon":
            p.constraints = [0.4, 0.8]
        elif p.parameter_name == "TIP3P-O-nonbonded_sigma":
            p.constraints = [2.5, 3.5]
        elif p.parameter_name == "TIP3P-H-nonbonded_charge":
            p.constraints = [0.0, 0.9]
        else:
            p.fixed = True

    for name, axis in exp_observable.independent_variables.items():
        print(f"Experiment {name}: {axis}")

    TIME_STEP_FS = 1.0
    FRAME_STEP = 40
    CORR_FRAMES = 4000
    TOTAL_FRAMES = 400000

    for output_axis in md_observable.predict_output(
        time_step=TIME_STEP_FS,
        frame_step=FRAME_STEP,
        total_frames=TOTAL_FRAMES,
        correlation_frames=CORR_FRAMES,
        universe=universe,
    ):
        print(f"Predicted MD {output_axis[0]}: {output_axis[1]} {output_axis[2]}")

    # MD Engine setup
    # NOTE: the temperatures of the measured data sets are:
    # B Halle / ILL data: 263K; S Howells / ISIS data: 280K
    # The below simulation object is for the ISIS data
    simulation = Simulation(
        universe,
        engine="openmm",
        time_step=TIME_STEP_FS,
        temperature=280,
        traj_step=FRAME_STEP,
        openmm_platform="OpenCL",
    )

    md_observable.set_parameters({"correlation_frames": CORR_FRAMES})

    control = Control(
        simulation=simulation,
        exp_datasets=QENS,
        observable_pairs=[observable_pair],
        fit_parameters=universe.parameters,
        reset_config=True,
        equilibration_steps=60000,
        minimizer_type="CMAES",
        MD_steps=TOTAL_FRAMES,
        energy_resolution=13.6,
        cont_slicing=True,
        file_dump_extent="all_obs",
        file_dump_frequency="best",
        file_dump_prefix="water_with_charges",
        FoM_options={"error": "none"},
        conv_tol=1e-6,
    )

    control.minimize(n_steps=15000)
    control.equilibrate(n_steps=45000)

    # Run refinement
    control.refine(n_steps=200)


if __name__ == "__main__":
    run_everything()
