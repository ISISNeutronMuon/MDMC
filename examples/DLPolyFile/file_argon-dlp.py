#!/usr/bin/env python3

"""
An example MDMC script for optimizing Lennard Jones parameters for liquid Ar.
For info on syntax see the MDMC docs, including the jupyter notebook tutorials.
A copy of the data fitting against is assumed to be located in
../doc/tutorials/data/Well_s_q_omega_Ar_data.xml
"""

from pathlib import Path
import os
# Change the number of threads depending on the number of physical cores on
# your computer as it was tested for LAMMPS

# This variable has been added to the script to allow testing
# on machines where, for whatever reason, the user only enabled
# a single CPU core in Docker. This way it is still possible
# to test the MPI functionality, as MPI will launch multiple
# processes, even though not enough slots are available to
# run them. Of course, we should not expect any performance gain
# in such a case.
# vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
os.environ["OMPI_MCA_rmaps_base_oversubscribe"]="true"
#
# This disables the vader BTL in OpenMPI, which is necessary,
# since vader BTL is not allowed by Docker.
# vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
os.environ["OMPI_MCA_btl"] = "^vader"
os.environ["OMP_NUM_THREADS"] = "4"

from MDMC.control import Control
from MDMC.MD.engine_facades.dlpoly_file_engine import DLPolyFileSimulation

# MD Engine setup. time_step of 10 fs is somewhat high, but for argon OK-ish.
# If time_step is descreased by a factor consider increasing traj_step by the
# same factor.
simulation = DLPolyFileSimulation(control="argon.control",
                                  config="argon.config",
                                  field="argon.field",
                                  time_step=10.18893/2,
                                  traj_step=30,
                                  numprocs=4)

# simulation.run(n_steps=10000, equilibration=False)
# print(simulation.trajectory)

# Dataset
exp_datasets = [{'file_name': Path(__file__).parent / '../../doc/tutorials/data/Well_s_q_omega_Ar_data.xml',
                 'type': 'SQw',
                 'reader': 'xml_SQw',
                 'weight': 1.,
                 'auto_scale': True,
                 'resolution': None}]

fit_parameters = simulation.parameters

# Specify how the refinement is going to be controlled
control = Control(simulation=simulation,
                  exp_datasets=exp_datasets,
                  fit_parameters=fit_parameters,
                  equilibration_steps = 1000,
                  MD_steps=1140)

# Energy Minimization and equilibration
control.minimize(n_steps=10, output_log='minim.log', work_dir='minim')
control.equilibrate(n_steps=1000, output_log='equilibration.log', work_dir='equil')

# Run the refinement, i.e. refine the FF parameters against the data.
# n_steps = 3 is too small, but a good choice to first test this script
# simulation.time_step = 0.0015283423720166564 / simulation.traj_step

control.refine(n_steps=10)
