"""
An example MDMC script for optimizing Lennard Jones parameters for liquid Ar.
For info on syntax see the MDMC docs, including the jupyter notebook tutorials.
A copy of the data fitting against is assumed to be located in
../doc/tutorials/data/Well_s_q_omega_Ar_data.xml
"""

from MDMC.control import Control
from MDMC.MD import Atom, LennardJones, Simulation, Universe
from MDMC.MD.interactions import Dispersion

# Build universe with density 0.0176 atoms per AA^-3
density = 0.0176
# This means cubic universe of side:
# 23.0668 A will contain 216 Ar atoms
# 26.911 A will contain 343 Ar atoms
# 30.7553 A will contain 512 Ar atoms
# 38.4441 A will contain 1000 Ar atoms
universe = Universe(dimensions=38.4441)
Ar = Atom('Ar[36]', charge=0.)
# Calculating number of Ar atoms needed to obtain density
universe.fill(Ar, num_density=density)

# Above an universe of non-interacting argon atoms was created. Below
# specify how these atoms will interact
Ar_dispersion = Dispersion(universe,  # the universe our interaction applies to
                           (Ar.atom_type, Ar.atom_type),  # the types of atoms to which it applies (only one type here!)
                           cutoff=8.,  # the cutoff distance
                           function=LennardJones(epsilon=1.0, sigma=3.0))

# MD Engine setup. time_step of 10 fs is somewhat high, but for argon OK-ish.
# If time_step is descreased by a factor consider increasing traj_step by the
# same factor.
simulation = Simulation(universe,
                        engine="openmm",
                        time_step=10.18893,
                        temperature=120.,
                        traj_step=15)

# Energy Minimization and equilibration
simulation.minimize(n_steps=2000)
simulation.run(n_steps=20000, equilibration=True)

# Setup refinement of the force field parameters

# exp_datasets is a list of dictionaries with one dictionary per experimental
# dataset
exp_datasets = [{'file_name':'../doc/tutorials/data/Well_s_q_omega_Ar_data.xml',
                 'type':'SQw',
                 'reader':'xml_SQw',
                 'weight':1.,
                 'resolution':None,
                 'use_FFT': True,
                 'auto_scale': False,
                 'rescale_factor': 18.5,
                 'cont_slicing': True}]


fit_parameters = universe.parameters
fit_parameters['sigma'].constraints = [2.0, 4.0]
fit_parameters['epsilon'].constraints = [0.5, 1.5]

# Specify how the refinement is going to be controlled
control = Control(simulation=simulation,
                  exp_datasets=exp_datasets,
                  fit_parameters=fit_parameters,
                  minimizer_type="GPO",
                  reset_config=True,
                  equilibration_steps=20000,
                  MD_steps=16000,
                  FoM_options={'error': 'none'})

# Run the refinement, i.e. refine the FF parameters against the data.
control.refine(n_steps=1000)