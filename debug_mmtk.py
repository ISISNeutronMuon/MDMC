# Example: Create a cubic water box
#
# This example requires only trivial modification to perform solvatation
# of another molecule. Simply put the molecule into the universe before
# the water molecule is added, and put its bounding sphere on the list
# of excluded regions. And don't forget to calculate the final box
# size (real_size) correctly.
#

from MMTK import *
from MMTK.ForceFields import SPCEFF
from MMTK.Trajectory import Trajectory, TrajectoryOutput, LogOutput
from MMTK.Minimization import SteepestDescentMinimizer
from MMTK.Dynamics import VelocityVerletIntegrator, VelocityScaler, \
                          TranslationRemover
from MMTK.Random import randomPointInBox

import numpy as np

from MDMC.src.MD.engine_facades import mmtk
import MDMC.src.trajectory_analysis.observables.exp_obs_factory as eof

# Set the number of molecules and the temperature
n_molecules = 3000
temperature = 300.*Units.K

# Calculate the size of the box
density = 1.*Units.g/(Units.cm)**3
number_density = density/Molecule('water').mass()
real_size = (n_molecules/number_density)**(1./3.)

# Create the universe with a larger initial size
current_size = 1.*real_size
world = CubicPeriodicUniverse(current_size,
                              SPCEFF.SPCEForceField(1.2*Units.nm,{'method': 'ewald'}))

# Add solvent molecules at random positions, avoiding the neighbourhood
# of previously placed molecules
excluded_regions = []
for i in range(n_molecules):
    m = Molecule('water', position = randomPointInBox(current_size))
    while 1:
        s = m.boundingSphere()
        collision = 0
        for region in excluded_regions:
            if s.intersectWith(region) is not None:
                collision = 1
                break
        if not collision:
            break
        m.translateTo(randomPointInBox(current_size))
    world.addObject(m)
    excluded_regions.append(s)

# Reduce energy
minimizer = SteepestDescentMinimizer(world, step_size = 0.05*Units.Ang)
minimizer(steps = 100)

# Set velocities and equilibrate for a while
world.initializeVelocitiesToTemperature(temperature)
integrator = VelocityVerletIntegrator(world,
                                      actions=[VelocityScaler(300., 10.),
                                               TranslationRemover()])
integrator(steps = 200)
#save(world, 'water'+`n_molecules`+'_spce.intermediate.setup')

# Final equilibration
n_steps = 1000
trajectory = Trajectory(world, 'water'+`n_molecules`+'_'+`n_steps`+'steps_spce'+'.nc', 'w', 'Final equilibration')
integrator(steps = n_steps,
           actions = [TrajectoryOutput(trajectory, ("time", "energy",
                                                    "thermodynamic",
                                                    "configuration"),
                                       0, None, 10),LogOutput('water'+`n_molecules`+'_'+`n_steps`+'steps_spce'+'.log',('time','energy'),0,None,100)])

MDMC_traj = mmtk.convert_trajectory(trajectory)
SQw = eof.ExperimentalObservableFactory.create_observable('SQw')
n_Q = 10
Q_vectors = [2 * np.pi * i / world.cellParameters() for i in range(1,n_Q+1)]
SQw.calculate_from_MD(MDMC_traj, Q_vectors = Q_vectors, cell = world.cellParameters)




trajectory.close()

# Save final system
save(world, 'water'+`n_molecules`+'.setup')
