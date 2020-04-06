"""
An example MDMC script for optimizing spce parameters for water at 263 K
"""

import numpy as np
from scipy.interpolate import interp2d

from MDMC.control import Control
from MDMC.MD import Atom, Dispersion, LennardJones, Simulation, Universe

# Build universe
# Cubic universe of side:
# 23.0668 A is 216 Ar atoms
# 26.911 A is 343 Ar atoms
# 30.7553 A is 512 Ar atoms
# 38.4441 A is 1000 Ar atoms
universe = Universe(dimensions=38.4441)
Ar = Atom('Ar', charge=0.)
n_units_xyz = universe.dimensions // (1. / 0.0176) ** (1 / 3.)
print(n_units_xyz)
universe.fill(Ar, num_struc_units=int(np.product(n_units_xyz)))
Ar_dispersion = Dispersion(universe,
                           (Ar.atom_type, Ar.atom_type),
                           cutoff=8.,
                           vdw_tail_correction=True,
                           function=LennardJones(1.0243, 3.36))

# MD Engine setup
simulation = Simulation(universe,
                        engine="lammps",
                        time_step=10.0,#10.340088,#9.40008,
                        temperature=120.,
                        traj_step=25)

# Energy Minimization and equilibration
simulation.minimize(n_steps=5000)
print("Minimization Complete")
simulation.run(n_steps=50000, equilibration=True)
print("Equilibration Complete")

# Setup refinement

# exp_datasets is a list of dictionaries with one dictionary per experimental dataset
exp_datasets = [{'file_name':'Well_s_q_omega_Ag_data.xml',
                 'type':'SQw',
                 'reader':'xml_SQw',
                 'weight':1.}]

# Fit parameters is a set(?) of all unique fit parameters in the universe which can then be filtered.
fit_params = set([p for p in universe.parameters if p.fixed is False])
# 3000 MD steps are required for a single trajectory
control = Control(simulation=simulation,
                  exp_datasets=exp_datasets,
                  fit_params=fit_params,
                  MC_norm=1,
                  minimizer_type="MMC",
                  reset_config=False,
                  MD_steps=60000,
                  t_resolution=467.)

# Make E and Q uniform
exp_obs = control.observable_pairs[0].exp_obs
Q = exp_obs.Q
# Well Argon file has incorrectly labelled f as w, so correct for this by
# multiplying E by 2pi
E = exp_obs.E * 2 * np.pi
SQw = exp_obs.SQw
SQw_err = exp_obs.SQw_err
SQw_fun = interp2d(Q, E, SQw)
SQw_err_zero = SQw_err
SQw_err_zero[SQw_err == np.float('inf')] = 0
SQw_err_fun = interp2d(Q, E, SQw_err_zero)
# Use the largest step size from the E data for the uniform step size
E_step = max([E[i] - E[i-1] for i in np.arange(len(E) - 1) + 1]) / 2
Q_step = min([Q[i] - Q[i-1] for i in np.arange(len(Q) - 1) + 1])
# Currently forced to start from E = 0. due to limitations of SQw calculation
E_uniform = np.arange(E[0], E[-1], E_step )
Q_uniform = np.arange(Q[0], Q[-1], Q_step )
SQw_uniform = np.transpose(SQw_fun(Q_uniform, E_uniform))
SQw_err_uniform = np.transpose(SQw_err_fun(Q_uniform, E_uniform))
SQw_err_uniform[SQw_err_uniform == 0.] = np.float('inf')
control.observable_pairs[0].exp_obs.independent_variables = {'E':E_uniform,
                                                             'Q':Q_uniform}
control.observable_pairs[0].exp_obs._dependent_variables = {'SQw':SQw_uniform}
control.observable_pairs[0].exp_obs._errors = {'SQw':SQw_err_uniform}
control.observable_pairs[0].MD_obs.independent_variables = {'E':E_uniform,
                                                            'Q':Q_uniform}
# Run refinement
control.refine(n_steps=100)
