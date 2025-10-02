"""
An example MDMC script for optimizing Lennard Jones parameters for liquid Ar.
For info on syntax see the MDMC docs, including the jupyter notebook tutorials.
A copy of the data fitting against is assumed to be located in
../doc/tutorials/data/Well_s_q_omega_Ar_data.xml
"""
from pathlib import Path

from MDMC.control import Control
from MDMC.MD.engine_facades.lammps_file_demo import LAMMPSFullFileSimulation

fact = 8

simulation = LAMMPSFullFileSimulation(
    struct_file="argon.data",
    run_script="run_lammps.lmp",
    minim_script="minim_lammps.lmp",
    equil_script="equil_lammps.lmp",
    traj_step=15*fact,
    time_step=10.18893/fact,
    extra_files={"_pot_file": "argon_pot.lmp"},
)

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
                  equilibration_steps=1000,
                  MD_steps=570*fact)

# Energy Minimization and equilibration
control.minimize(n_steps=10000, output_log='minim.log', work_dir='minim')
control.equilibrate(n_steps=1000, output_log='equilibration.log', work_dir='equil')

# Run the refinement, i.e. refine the FF parameters against the data.
# n_steps = 3 is too small, but a good choice to first test this script
# simulation.time_step = 0.0015283423720166564 / simulation.traj_step

control.refine(n_steps=10)
