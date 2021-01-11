"""
An example MDMC script for optimizing Lennard Jones parameters for liquid Ar.
For info on syntax see the MDMC docs, including the jupyter notebook tutorials.
A copy of the data fitting against is assumed to be located in
../doc/tutorials/data/Well_s_q_omega_Ar_data.xml
"""

import numpy as np
import os
# Change the number of threads depending on the number of physical cores on
# your computer as it was tested for LAMMPS
os.environ["OMP_NUM_THREADS"] = "4"
from scipy.interpolate import interp2d

from MDMC.control import Control
from MDMC.MD import Atom, Dispersion, LennardJones, Simulation, Universe

# Build universe with density 0.0176 atoms per AA^-3
density = 0.0176
# This means cubic universe of side:
# 23.0668 A will contain 216 Ar atoms
# 26.911 A will contain 343 Ar atoms
# 30.7553 A will contain 512 Ar atoms
# 38.4441 A will contain 1000 Ar atoms
universe = Universe(dimensions=38.4441)
Ar = Atom('Ar', charge=0.)
# Calculating number of Ar atoms needed to obtain density
n_ar_atoms = int(density * np.product(universe.dimensions))
print(n_ar_atoms)
universe.fill(Ar, num_struc_units=(n_ar_atoms))

# Above an universe of non-interacting argon atoms was created. Below
# specify how these atoms will interact
Ar_dispersion = Dispersion(universe,
                           (Ar.atom_type, Ar.atom_type),
                           cutoff=8.,
                           vdw_tail_correction=True,
                           function=LennardJones(1.0243, 3.36))

# MD Engine setup. time_step of 10 fs is somewhat high, but for argon OK-ish.
# If time_step is descreased by a factor consider increasing traj_step by the
# same factor.
simulation = Simulation(universe,
                        engine="lammps",
                        time_step=10.0,
                        temperature=120.,
                        traj_step=25)

# Energy Minimization and equilibration
simulation.minimize(n_steps=5000)
print("Minimization Complete")
simulation.run(n_steps=50000, equilibration=True)
print("Equilibration Complete")

# Setup refinement of the force field parameters

# exp_datasets is a list of dictionaries with one dictionary per experimental
# dataset
exp_datasets = [{'file_name':'../doc/tutorials/data/Well_s_q_omega_Ar_data.xml',
                 'type':'SQw',
                 'reader':'xml_SQw',
                 'weight':1.}]

# Fit parameters is the set of all unique fit parameters in the universe
# which are not fixed
fit_params = set([p for p in universe.parameters if p.fixed is False])

# Specify how the refinement is going to be controlled
control = Control(simulation=simulation,
                  exp_datasets=exp_datasets,
                  fit_params=fit_params,
                  MD_steps=10000,
                  t_resolution=200.)

# Hack the input data onto a uniform grid, i.e. make E and Q uniform
# (alternatively, create a new versions of the Well_s_q_omega_Ar_data.xml data
# on uniform grids. As of the writing MDMC requires data to be on
# a uniform grid)
exp_obs = control.observable_pairs[0].exp_obs
Q = exp_obs.Q
# Well's Argon file seem to have incorrectly labelled f as w, so correct for
# this by multiplying E by 2pi
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
# Currently forced to start from E = 0.0 due to limitations of SQw calculation
E_uniform = np.arange(E[0], E[-1], E_step)
Q_uniform = np.arange(Q[0], Q[-1], Q_step)
SQw_uniform = np.transpose(SQw_fun(Q_uniform, E_uniform))
SQw_err_uniform = np.transpose(SQw_err_fun(Q_uniform, E_uniform))
SQw_err_uniform[SQw_err_uniform == 0.] = np.float('inf')
# copy the hacked the E, Q and SQw value back to the control.observable
control.observable_pairs[0].exp_obs.independent_variables = {'E':E_uniform,
                                                             'Q':Q_uniform}
control.observable_pairs[0].exp_obs._dependent_variables = {'SQw':SQw_uniform}
control.observable_pairs[0].exp_obs._errors = {'SQw':SQw_err_uniform}
control.observable_pairs[0].MD_obs.independent_variables = {'E':E_uniform,
                                                            'Q':Q_uniform}

# Run the refinement, i.e. refine the FF parameters against the data.
# n_steps = 3 is too small, but a good choice to first test this script
control.refine(n_steps=3)
