"""
An example MDMC script for optimizing spce parameters for water at 263 K
"""

import MDMC.MD.simulation as sim
import MDMC.MD.structural_units as su
import MDMC.MD.force_fields as ff
import MDMCrefinement.minimizer as minim
from MDMC.control import MDMCControl

from tests.test_data import data

# Build universe
universe = sim.Universe(dimensions=(10., 10., 10.), shape=sim.Shape.orthorhombic)
H1 = su.Atom('H', mass=1.008)
H2 = su.Atom('H', position=(0.151390, 0., 0.), mass=1.008)
O = su.Atom('O', position=(0.075695, 0., 0.058588), mass=16.000)
water_mol = su.Molecule(position=(0, 0, 0),
                        velocity=(0, 0, 0),
                        atoms=[H1, H2, O],
                        interactions=[su.Bond(H1, O),
                                      su.Bond(H2, O),
                                      su.Dispersion(O),
                                      su.BondAngle(atoms=[H1, O, H2])],
                        name='water')
universe.fill(water_mol,force_field=ff.SPCE,num_density=0.0333679)

# MD Engine setup
md_engine = sim.NVESimulation(universe, engine="mmtk", time_step=1, temperature=263,
    integrator='velocity_verlet', lj_options=1.2, es_options={'method':'ewald'})

# Setup and run refinement
# exp_datasets is a list of dictionaries with one dictionary per experimental dataset
exp_datasets = [{'file_name':data.LAMP_SQW_FILE, 'type':'SQw', 'reader':'LAMPSQw', 'weight':1.}]
# Fit parameters is a set(?) of all unique fit parameters in the universe which can then be filtered.
fit_params = universe.parameters
control = MDMCControl(MD_engine=md_engine, exp_datasets=exp_datasets,
    fit_params=fit_params, minimizer_type = "MMC")
control.refine(n_steps=100)
