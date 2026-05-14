"""
An example MDMC script for optimizing spce parameters for water at 263 K

Water data from ILL (IN5, 263K) provided by Bertil Halle. Ref: J. Chem. Phys. 134, 144508 (2011)
Water data from ISIS (IRIS, 280K) provided by Spencer Howells
"""

import os

from MDMC.MD.interactions import Bond, BondAngle

from MDMC.control import Control
from MDMC.MD import *
from tests.test_data import data

# Build universe
# Cubic universe of side:
# 18.6270199 A is 216 water molecules
# 21.731523217 is 343 water molecules
# 24.83602653 is 512 water molecules
os.environ["OMP_NUM_THREADS"] = "4"

universe = Universe(dimensions=24.83602653)
H1 = Atom('H')
H2 = Atom('H', position=(0., 1.63298, 0.))
O = Atom('O', position=(0., 0.81649, 0.57736))
H_coulombic = Coulombic(atoms=[H1, H2], cutoff=10.)
O_coulombic = Coulombic(atoms=O, cutoff=10.)
water_mol = Molecule(position=(0, 0, 0),
                     velocity=(0, 0, 0),
                     atoms=[H1, H2, O],
                     interactions=[Bond((H1, O), (H2, O), constrained=True),
                                   BondAngle(H1, O, H2)],
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
                        engine="dlpoly",
                        time_step=1.033916924,
                        temperature=280.,
                        traj_step=4000,
                        numprocs=2)

# Setup refinement

# in general exp_datasets is a list of dictionaries with one dictionary per experimental dataset
# below are 2 separate objects for the 2 datasets as they were measured at different temperatures
exp_dataset_ILL = [{'file_name':'../doc/tutorials/data/263K05Awat_LAMP',
                 'type':'SQw',
                 'reader':'LAMPSQw',
                 'auto_scale':'minimise_fom',
                 'weight':1.}]
exp_dataset_ISIS = [{'file_name':'../doc/tutorials/data/IRIS_26176_water_data.dat',
                 'type':'SQw',
                 'reader':'MantidSQw',
                 'auto_scale':'minimise_fom',
                 'weight':1.,
                 'resolution':'../doc/tutorials/data/IRIS_26173_water_data_resolution.dat'}]

# Fit parameters is a set(?) of all unique fit parameters in the universe which can then be filtered.
for p in universe.parameters.as_array:
    if p.type != 'epsilon':
        p.fixed = True

fit_parameters = universe.parameters
control = Control(simulation=simulation,
                  exp_datasets=exp_dataset_ISIS,
                  fit_parameters=fit_parameters,
                  MC_norm=1,
                  minimizer_type="CMAES",
                  MD_steps=804000,
                  energy_resolution=13.6)

# Energy Minimization and equilibration
control.minimize(n_steps=5000,output_log='minim-water.log',work_dir='minim-water')
control.equilibrate(n_steps=25000,output_log='equil-water.log',work_dir='equil-water')

# Run refinement
control.refine(n_steps=3)
