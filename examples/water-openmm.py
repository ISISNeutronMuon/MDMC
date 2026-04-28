import os

from MDMC.MD import *
from MDMC.MD.force_fields.TIP3P import TIP3PMol, add_tip3p_ff
from MDMC.control import Control
# Currently MDMC uses OMP_NUM_THREADS to control the number of processes
# in the sqw calculation
os.environ["OMP_NUM_THREADS"] = "4"

# Build universe
# Cubic universe of side:
# 18.6270199 A is 216 water molecules
# 21.731523217 is 343 water molecules
# 24.83602653 is 512 water molecules
universe = Universe(dimensions=21.75)
universe.fill(TIP3PMol(constrained=False), num_density=0.03356718472021752)
add_tip3p_ff(universe, cutoff=10.0, ewald=1e-6, constrained=False)

# MD Engine setup
# NOTE: the temperatures of the measured data sets are:
# B Halle / ILL data: 263K; S Howells / ISIS data: 280K
# The below simulation object is for the ISIS data
simulation = Simulation(
    universe,
    engine="openmm",
    time_step=1.033916924,
    temperature=280,
    traj_step=100,
    openmm_platform="OpenCL"
)

simulation.run(n_steps=30000, equilibration=True)


# Setup refinement
QENS = [{
    'file_name':'../doc/tutorials/data/263K05Awat_LAMP',
    'type':'SQw',
    'reader':'LAMPSQw',
    'weight':1.,
    'auto_scale':True,
    'use_FFT': True,
    'filter': {
        'abs': 0.1,              # All values below 0.1 to be removed
        'rel': 0.1,              # All values less than 10% of maximum to be removed.
        'use_magnitude': False,  # Use raw value (negative values will be removed)
        'warn_threshold': 0.1,   # Warn if more than 10% of values are removed by this filter
    },
    'resolution':{'file': '../doc/tutorials/data/262p7K0A5van_LAMP'},
}]


# Fit parameters is a set(?) of all unique fit parameters in the universe which can then be filtered.
print(universe.parameters)
for p in universe.parameters.as_array:
    if p.type == 'charge':
        p.fixed = True


fit_parameters = universe.parameters
control = Control(
    simulation=simulation,
    exp_datasets=QENS,
    fit_parameters=fit_parameters,
    reset_config=True,
    equilibration_steps=30000,
    minimizer_type="CMAES",
    MD_steps=804000,
    energy_resolution=13.6,
    FoM_options={'error': 'none'}
)

# Run refinement
control.refine(n_steps=1000)