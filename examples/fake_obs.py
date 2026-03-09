"""
A test script which attempts to run an optimisation without actually running
any MD simulations. This is only used to test the MDMC minimisers.
At the moment it is attempting to fit a 2D array created by multiplying
two Gaussian functions.

Currently, the solution is hardcoded. The minimiser should produce:
centre_x=5.0
centre_y=4.0
width_x=3.3
width_y=2.1
"""

from MDMC.control import Control
from MDMC.MD import Parameter, Simulation, Universe

universe = Universe(dimensions=10.0)
# Here we specify the initial values or bounds for the refinement.
universe.parameters = [Parameter(name="centre_x", value=1.0, constraints=(0.1,10.0)),
                       Parameter(name="centre_y", value=1.0, constraints=(0.1,10.0)),
                       Parameter(name="width_x", value=1.0, constraints=(0.1,10.0)),
                       Parameter(name="width_y", value=1.0, constraints=(0.1,10.0))]

simulation = Simulation(universe,
                        engine="null_engine", # does not run any MD.
                        time_step=1,
                        temperature=1,
                        traj_step=1)

# The file still gets loaded, but is not used.
exp_datasets = [{'file_name':'../doc/tutorials/data/Well_s_q_omega_Ar_data.xml',
                 'type':'gauss2D',
                 'reader':'xml_SQw',
                 'weight':1.,
                 'resolution':None}]

fit_parameters = universe.parameters

# The input values are less important now.
control = Control(simulation=simulation,
                  exp_datasets=exp_datasets,
                  fit_parameters=fit_parameters,
                  MD_steps=5,
                  equilibration_steps=5)

control.refine(n_steps=300)
