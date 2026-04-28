"""
An example MDMC script for optimizing spce parameters for water at 263 K

Water data from ILL (IN5, 263K) provided by Bertil Halle. Ref: J. Chem. Phys. 134, 144508 (2011)
Water data from ISIS (IRIS, 280K) provided by Spencer Howells
"""

from MDMC.MD.interactions import Bond, BondAngle

from MDMC.control import Control
from MDMC.MD import *

# Build universe
# Cubic universe of side:
# 18.6270199 A is 216 water molecules
# 21.731523217 is 343 water molecules
# 24.83602653 is 512 water molecules
universe = Universe(dimensions=21.75)
H1 = Atom('H')
H2 = Atom('H', position=(0., 1.63298, 0.))
O = Atom('O', position=(0., 0.81649, 0.57736))
H_coulombic = Coulombic(atoms=[H1, H2], cutoff=10.)
O_coulombic = Coulombic(atoms=O, cutoff=10.)
water_mol = Molecule(position=(0, 0, 0),
                     velocity=(0, 0, 0),
                     atoms=[H1, H2, O],
                     interactions=[Bond((H1, O), (H2, O), constrained=True),
                                   BondAngle(H1, O, H2, constrained=True)],
                     name='water')
shake = Shake(1e-4, 100)
universe.constraint_algorithm = shake
e_solver = PPPM(accuracy=1e-5)
universe.electrostatic_solver = e_solver
universe.fill(water_mol, num_density=0.03356718472021752)
O_dispersion = Dispersion(universe, (O.atom_type, O.atom_type), cutoff=10.,
                          vdw_tail_correction=True)
universe.add_force_field('SPCE')

# MD Engine setup
# NOTE: the temperatures of the measured data sets are:
# B Halle / ILL data: 263K; S Howells / ISIS data: 280K
# The below simulation object is for the ISIS data
simulation = Simulation(universe,
                        engine="lammps",
                        time_step=1.033916924,
                        temperature=280.,
                        traj_step=4000)

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
for p in universe.parameters.as_array:
    if p.type != 'epsilon':
        p.fixed = True

fit_parameters = universe.parameters
control = Control(simulation=simulation,
                  exp_datasets=QENS,
                  fit_parameters=fit_parameters,
                  MC_norm=1,
                  minimizer_type="CMAES",
                  MD_steps=804000,
                  energy_resolution=13.6)

# Energy Minimization and equilibration
control.minimize(n_steps=5000)
control.equilibrate(n_steps=25000)

# Run refinement
control.refine(n_steps=1000)
