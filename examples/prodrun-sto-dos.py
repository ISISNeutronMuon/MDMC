"""
Test run of the optimised SrTiO3 parameters.
"""

import numpy as np

from MDMC.exporters.trajectories.H5MD_build import write_H5MD
from MDMC.MD import Atom, Molecule, NonBonded, Simulation, Universe
from MDMC.MD.interactions import NonBondedForce


O_charge = -0.7235399323821826
Ti_charge = 0.515738324378003
Sr_charge = -1*(Ti_charge + 3* O_charge)
O_epsilon = 23.65147247664405
O_sigma = 1.5826646662857802
Ti_epsilon = 20.289754968386067
Ti_sigma = 3.4195673565232605
Sr_epsilon = 9.737576223098147
Sr_sigma = 3.124497231730163

universe = Universe(dimensions=3.905*6)
O1 = Atom("O", position=np.array((0.5, 0.5, 0.0)) * 3.905, charge = O_charge)
O2 = Atom("O", position=np.array((0.5, 0.0, 0.5)) * 3.905, charge = O_charge)
O3 = Atom("O", position=np.array((0.0, 0.5, 0.5)) * 3.905, charge = O_charge)
Ti = Atom("Ti", position=np.array((0.5, 0.5, 0.5)) * 3.905, charge = Ti_charge)
Sr = Atom("Sr", position=np.array((0.0, 0.0, 0.0)) * 3.905, charge = Sr_charge)
# Calculating number of Ar atoms needed to obtain density
sto_unit = Molecule(position=(0, 0, 0),
                     velocity=(0, 0, 0),
                     atoms=[O1, O2, O3, Ti, Sr],
                     name='SrTiO3')
universe.fill(sto_unit, num_struc_units=6*6*6)

# Above an universe of non-interacting argon atoms was created. Below
# specify how these atoms will interact
NonBondedForce(
    universe,
    O1.atom_type,
    cutoff=10.0,
    ewald=1e-6,
    function=NonBonded(charge=O_charge, epsilon=O_epsilon, sigma=O_sigma, elements=["O"], molecules = ["SrTiO3"]),
)
NonBondedForce(
    universe,
    Ti.atom_type,
    cutoff=10.0,
    ewald=1e-6,
    function=NonBonded(charge=Ti_charge, epsilon=Ti_epsilon, sigma=Ti_sigma, elements=["Ti"], molecules = ["SrTiO3"]),
)
NonBondedForce(
    universe,
    Sr.atom_type,
    cutoff=10.0,
    ewald=1e-6,
    function=NonBonded(charge=Sr_charge, epsilon=Sr_epsilon, sigma=Sr_sigma, elements=["Sr"], molecules = ["SrTiO3"]),
)

# MD Engine setup. time_step of 10 fs is somewhat high, but for argon OK-ish.
# If time_step is descreased by a factor consider increasing traj_step by the
# same factor.
simulation = Simulation(
    universe,
    engine="openmm",
    time_step=1.0,
    temperature=80.0,
    traj_step=4,
    openmm_platform="OpenCL",
)

simulation.run(n_steps=80000, equilibration=True)

simulation.run(n_steps=16000)

write_H5MD(simulation.trajectory,
           "SrTiO3_T80K_wrong_test.h5",
           timestamp="",
           creator_name="MDMC",
           creator_email="place@hold.er")

