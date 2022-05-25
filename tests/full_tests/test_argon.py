"""
Tests a full refinement process with various simulation and minimizer types.

Only runs if pytest is invoked with the --argon option.
"""

import numpy as np
import pytest
from pytest_cases import fixture_ref, parametrize

from MDMC.MD.simulation import Simulation, Universe
from MDMC.MD.structures import Atom
from MDMC.MD.interactions import Dispersion
from MDMC.MD.interaction_functions import LennardJones
from MDMC.control import Control

from tests.test_data import data

@pytest.fixture
def Ar():
    return Atom('Ar', charge=0., cutoff=10.)

@pytest.fixture
def universe(Ar):
    universe = Universe(dimensions=38.4441)
    # Calculating number of Ar atoms needed to obtain density
    n_ar_atoms = int(0.0176 * np.product(universe.dimensions))
    universe.fill(Ar, num_struc_units=(n_ar_atoms))

    return universe

@pytest.fixture
def Ar_dispersion(universe, Ar):
    return Dispersion(universe,
                      (Ar.atom_type, Ar.atom_type),
                      cutoff=8.,
                      vdw_tail_correction=True,
                      function=LennardJones(1.0243, 3.36))

@pytest.fixture
def lammps_simulation(universe):
    simulation = Simulation(universe,
                            engine="lammps",
                            time_step=10.18893,
                            temperature=120.,
                            traj_step=15)

    simulation.minimize(n_steps=5000)
    simulation.run(n_steps=50000, equilibration=True)

    return simulation


@pytest.mark.argon
@parametrize("simulation", [fixture_ref(lammps_simulation)])
@pytest.mark.parametrize("minimizer", ['MMC'])
def test_argon_system(universe, simulation, minimizer):
    """Full test of a refinement on an Argon system."""

    exp_datasets = [{'file_name':data.READER_DATA['xml_SQw'],
                     'type':'SQw',
                     'reader':'xml_SQw',
                     'weight':1.,
                     'resolution':None}]

    fit_parameters = universe.parameters

    control = Control(simulation=simulation,
                      minimizer_type=minimizer,
                      exp_datasets=exp_datasets,
                      fit_parameters=fit_parameters,
                      MD_steps=570)

    control.refine(n_steps=1)