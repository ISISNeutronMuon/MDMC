import copy
import os

from MDMC.MD import *
from MDMC.MD.force_fields.four_site_water import FourSiteWater, add_four_site_water_ff
from MDMC.control import Control
from MDMC.refinement.FoM.FoM_abs import ObservablePair
from MDMC.trajectory_analysis.observables.sqw import SQw

# Currently MDMC uses OMP_NUM_THREADS to control the number of processes
# in the sqw calculation
os.environ["OMP_NUM_THREADS"] = "4"

# Build universe
# Cubic universe of side:
# 18.6270199 A is 216 water molecules
# 21.731523217 is 343 water molecules
# 24.83602653 is 512 water molecules
universe = Universe(dimensions=24.83602653)
universe.fill(FourSiteWater(model_name="TIP4P-Ewald"), num_density=0.03356718472021752)
add_four_site_water_ff(universe, cutoff=12.0, ewald=1e-4, model_name="TIP4P-Ewald")

# MD Engine setup
# NOTE: the temperatures of the measured data sets are:
# B Halle / ILL data: 263K; S Howells / ISIS data: 280K
# The below simulation object is for the ISIS data
simulation = Simulation(
    universe,
    engine="openmm",
    time_step=1.0,
    temperature=280,
    traj_step=10,
    openmm_platform="OpenCL",
    # default precision on CUDA and OpenCL is single
    openmm_properties={"Precision": "mixed"},
)

simulation.run(n_steps=30000, equilibration=True)


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

exp_observable = SQw()
exp_observable.read_from_file("LAMPSQw", "../doc/tutorials/data/263K05Awat_LAMP")
md_observable = SQw()
md_observable.origin = "MD"
for obs in {exp_observable, md_observable}:
    obs.name = "SQw"
md_observable.independent_variables = copy.deepcopy(exp_observable.independent_variables)

observable_pair = ObservablePair(
    exp_obs=exp_observable, MD_obs=md_observable, weight=1.0, rescale_factor=1.0, auto_scale=True
)

# only refit the LJ parameters on oxygen and the HOH angle
for p in universe.parameters.as_array:
    if p.parameter_name == "TIP4P-Ewald-O-nonbonded_epsilon":
        p.constraints = [0.5, 0.8]
    elif p.parameter_name == "TIP4P-Ewald-O-nonbonded_sigma":
        p.constraints = [2.95, 3.35]
    elif p.parameter_name == 'TIP4P-Ewald-HOH-harmonicangle_equilibrium_state':
        p.constraints = [80, 120]
    else:
        p.fixed = True


control = Control(
    simulation=simulation,
    exp_datasets=QENS,
    fit_parameters=universe.parameters,
    observable_pairs=[observable_pair],
    reset_config=True,
    equilibration_steps=30000,
    minimizer_type="CMAES",
    MD_steps=424620,
    energy_resolution=13.6,
    FoM_options={"error": "none"},
)

# Run refinement
control.refine(n_steps=1000)
