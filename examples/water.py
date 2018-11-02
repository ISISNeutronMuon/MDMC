"""
An example MDMC script for optimizing spce parameters for water at 263 K
"""

import numpy as np
from scipy.interpolate import interp2d

import MDMC.MD.simulation as sim
import MDMC.MD.structural_units as su
import MDMC.MD.force_fields as ff
from MDMC.control.control import MDMCControl

from tests.test_data import data

# Build universe
# Cubic universe of side 9.32 A is 27 water molecules, 24.86 is 512 water molecules
side = 21.75
universe = sim.Universe(dimensions=(side, side, side),
                        shape=sim.Shape.orthorhombic)
H1 = su.Atom('H', mass=1.008)
H2 = su.Atom('H', position=(1.51390, 0., 0.), mass=1.008)
O = su.Atom('O', position=(0.75695, 0., 0.58588), mass=16.000)
water_mol = su.Molecule(position=(0, 0, 0),
                        velocity=(0, 0, 0),
                        atoms=[H1, H2, O],
                        interactions=[su.Bond(H1, O),
                                      su.Bond(H2, O),
                                      su.Dispersion(O),
                                      su.BondAngle(atoms=[H1, O, H2])],
                        name='water')
universe.fill(water_mol, force_field=ff.SPCE, num_density=0.0333679)

# MD Engine setup
md_engine = sim.Simulation(universe,
                              engine="mmtk",
                              time_step=1,
                              temperature=263.,
                              integrator='velocity_verlet',
                              lj_options=12,
                              es_options='ewald',
                              minimizer='steepest_descent',
                              traj_step=1057,
                              rigid=True)

# Energy Minimization and equilibration
md_engine.minimize(n_steps=5000)
md_engine.run(n_steps=5000, equilibration=True)

# Setup refinement

# exp_datasets is a list of dictionaries with one dictionary per experimental dataset
exp_datasets = [{'file_name':data.READER_DATA['LAMPSQw'],
                 'type':'SQw',
                 'reader':'LAMPSQw',
                 'weight':1.}]

# Fit parameters is a set(?) of all unique fit parameters in the universe which can then be filtered.
fit_params = universe.parameters
control = MDMCControl(MD_engine=md_engine,
                      exp_datasets=exp_datasets,
                      fit_params=fit_params,
                      MC_norm=1,
                      minimizer_type="MMC",
                      MD_steps=108929,
                      t_resolution=114.)

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
E_uniform = np.arange(E[0], E[-1], E_step)
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
control.refine(n_steps=1)
