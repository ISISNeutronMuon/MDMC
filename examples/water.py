"""
An example MDMC script for optimizing spce parameters for water at 263 K
"""

import numpy as np
from scipy.interpolate import interp2d

from MDMC.control import Control
from MDMC.MD import *
from tests.test_data import data

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
simulation = Simulation(universe,
                        engine="lammps",
                        time_step=1.057564,
                        temperature=263.,
                        traj_step=1000)

# Energy Minimization and equilibration
simulation.minimize(n_steps=5000)
simulation.run(n_steps=25000, equilibration=True)

# Setup refinement

# exp_datasets is a list of dictionaries with one dictionary per experimental dataset
exp_datasets = [{'file_name':data.READER_DATA['LAMPSQw'],
                 'type':'SQw',
                 'reader':'LAMPSQw',
                 'weight':1.}]

# Fit parameters is a set(?) of all unique fit parameters in the universe which can then be filtered.
for p in universe.parameters:
    if p.name != 'epsilon':
        p.fixed = True

fit_parameters = set([p for p in universe.parameters if p.fixed is False])
control = Control(simulation=simulation,
                  exp_datasets=exp_datasets,
                  fit_parameters=fit_parameters,
                  MC_norm=1,
                  minimizer_type="MMC",
                  MD_steps=208000,
                  energy_resolution=13.6)

# Bertil Halle water data is non-symmetric, and has a non-rectangular grid with
# a non-uniform E step.
# To account for this, a limited E is used and undefined errors are set to zero
# for the purposes of interpolation.
# This should really be performed before the data is read into control - the
# final step is a reflection of this as the MD observable is changed to match
# the new independent variables of the experimental observable
# So that the MD simulation size can be minimized, the Q min is increased and
# the Q resolution is reduced.
exp_obs = control.observable_pairs[0].exp_obs
Q_slice = slice(6, len(exp_obs.Q), 2)
Q = exp_obs.Q[Q_slice]
E_range = (exp_obs.E >=0)
E = exp_obs.E[E_range]
SQw = np.array([Sw[E_range] for Sw in exp_obs.SQw[Q_slice]])
SQw_err = np.array([Sw_err[E_range] for Sw_err
                    in exp_obs.SQw_err[Q_slice]])
SQw_fun = interp2d(E, Q, SQw)
SQw_err_zero = SQw_err
SQw_err_zero[SQw_err == np.float('inf')] = 0
SQw_err_fun = interp2d(E, Q, SQw_err_zero)
# Use the largest step size from the E data for the uniform step size
E_step = max([E[i] - E[i-1] for i in np.arange(len(E) - 1) + 1])
# Currently forced to start from E = 0. due to limitations of SQw calculation
E_uniform = np.arange(0., E[-1] - E[0], E_step )
SQw_uniform = SQw_fun(E_uniform, Q)
SQw_err_uniform = SQw_err_fun(E_uniform, Q)
SQw_err_uniform[SQw_err_uniform == 0.] = np.float('inf')
control.observable_pairs[0].exp_obs.independent_variables = {'E':E_uniform,
                                                             'Q':Q}
control.observable_pairs[0].exp_obs._dependent_variables = {'SQw':SQw_uniform}
control.observable_pairs[0].exp_obs._errors = {'SQw':SQw_err_uniform}
control.observable_pairs[0].MD_obs.independent_variables = {'E':E_uniform,
                                                            'Q':Q}

# Run refinement
control.refine(n_steps=0)
