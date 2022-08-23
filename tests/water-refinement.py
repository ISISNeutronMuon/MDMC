import os
os.environ["OMP_NUM_THREADS"] = "4"
import pyfftw
pyfftw.config.NUM_THREADS = 4
pyfftw.config.PLANNER_EFFORT = 'FFTW_MEASURE'
# Import the Universe class
from MDMC.MD import Universe
# Initialise a Universe with dimensions in Ang
from MDMC.MD import *
from MDMC.common.time_keeper import TimeKeeper

universe = Universe(dimensions=21.75, constraint_algorithm=Shake(1e-4, 100), electrostatic_solver=PPPM(accuracy=1e-5))
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
universe.fill(water_mol, num_density=0.03356718472021752)
O_dispersion = Dispersion(universe, [O.atom_type, O.atom_type], cutoff=10., vdw_tail_correction=True)
universe.add_force_field('SPCE')

simulation = Simulation(universe, engine='lammps', time_step=1., temperature=300.,
                        pressure=101325., traj_step=10, thermostat='nose',
                        barostat='nose', t_damp=100, p_damp=1000)

simulation.minimize(1000)

simulation.run(1000, equilibration=True)

simulation.run(2000)

# Dataset from: Johan Qvist et al, J. Chem. Phys. 134, 144508 (2011)
QENS = {'file_name':'/workspaces/MDMCv0.2_pilot/doc/tutorials/data/263K05Awat_LAMP',
        'type':'SQw',
        'reader':'LAMPSQw',
        'weight':1.,
        'auto_scale':True,
        'use_FFT':False,
        'resolution':{'file': '/workspaces/MDMCv0.2_pilot/doc/tutorials/data/262p7K0A5van_LAMP'}}


exp_datasets = [QENS]

n_diffraction = {'file_name':'/workspaces/MDMCv0.2_pilot/doc/tutorials/data/water_PDF',
                 'type':'PDF',
                 'reader':'ASCII',
                 'weight':1.,
                 'auto_scale':True,
                 'resolution': {'gaussian': 84}}
two_exp_datasets = [QENS, n_diffraction]

universe = simulation.universe
fit_parameters = universe.parameters
print(fit_parameters['epsilon'])

fit_parameters['charge'][0].set_tie(fit_parameters['charge'][1], ' * - 2')
print(fit_parameters['charge'][0])

fit_parameters['epsilon'].constraints = (0.6, 0.7)
print(fit_parameters['epsilon'])

fit_parameters['equilibrium_state'][0].fixed = True
print(fit_parameters['equilibrium_state'][0])

from MDMC.refinement.FoM import ChiSquared_experror
error = ChiSquared_experror.ChiSquaredExpError

FoM_options = {'error':'exp', 'norm':'data_points', 'cont_slicing':False}

# Assuming a Universe called universe and a Simulation called simulation have been created
from MDMC.control import Control

control = Control(simulation=simulation,
                  exp_datasets=exp_datasets,
                  fit_parameters=fit_parameters,
                  MC_norm=1.0,
                  minimizer_type="MMC",
                  FoM_options = FoM_options,
                  MD_steps=424620,
                  equilibration_steps=1000,
                  results_filename='results_output_filename.csv')

# So that the MD simulation size can be minimized, the Q min is increased and
# the Q resolution is reduced.
import numpy as np
exp_obs = control.observable_pairs[0].exp_obs
Q_slice = slice(6, len(exp_obs.Q), 2)
Q = exp_obs.Q[Q_slice]
E = exp_obs.E
# copy the updated Q values back to the control.observable
control.observable_pairs[0].exp_obs.independent_variables = {'E':E, 'Q':Q}
control.observable_pairs[0].MD_obs.independent_variables = {'E':E, 'Q':Q}

exp_obs = control.observable_pairs[0].exp_obs
#help(exp_obs.resolution)
resolution_function = exp_obs.resolution.resolution_function
# Note that for multidimensional resolution functions, the innermost independent variable should be passed first
t = np.linspace(0, 1e5)
resolution_array = resolution_function(t, Q)

control.refine(3)

tk = TimeKeeper()
print(f"Total time = {tk.total_time()}")
timing_stats = tk.summarise_results()
sorted_stats = sorted(timing_stats, key = lambda x: x[2], reverse = True)
print("## Function     NumberOfCalls    TotalTime")
for ll in sorted_stats:
    print(f"## {ll[0]}   {ll[1]}   {ll[2]}")
