"""Tests for running full DLPoly simulations"""

import numpy as np
import pytest

pytestmark = pytest.mark.dlpoly
pytest.importorskip("dlpoly")

from MDMC.control import Control
from MDMC.MD.interaction_functions import LennardJones
from MDMC.MD.interactions import Dispersion
from MDMC.MD.simulation import Simulation, Universe
from MDMC.MD.structures import Atom

from tests.test_data import data


@pytest.fixture
def universe():
    """ Universe with argon ready """
    # Build universe with density 0.0176 atoms per AA^-3
    density = 0.0176

    universe = Universe(dimensions=23.0668)
    Ar = Atom('Ar', charge=0.)

    # Calculating number of Ar atoms needed to obtain density
    n_ar_atoms = int(density * np.prod(universe.dimensions))

    universe.fill(Ar, num_struc_units=n_ar_atoms)

    Ar_dispersion = Dispersion(universe,
                               (Ar.atom_type, Ar.atom_type),
                               cutoff=8.0,
                               vdw_tail_correction=True,
                               function=LennardJones(1.0243, 3.36))

    return universe


@pytest.fixture
def simulation(universe):
    """ Simulation for argon """
    return Simulation(universe,
                      engine="dlpoly",
                      time_step=10.18893/2,
                      temperature=120.,
                      traj_step=30,
                      numprocs=1,
                      density_variance=1.4)


@pytest.fixture
def exp_datasets():
    """ Experimental dataset for argon """
    return [{'file_name': data.READER_DATA['xml_SQw'],
             'type': 'SQw',
             'reader': 'xml_SQw',
             'weight': 1.,
             'auto_scale': True,
             'resolution': None}]


@pytest.fixture
def control(simulation, universe, exp_datasets):
    """ Control for argon """

    return Control(simulation=simulation,
                   exp_datasets=exp_datasets,
                   fit_parameters=universe.parameters,
                   equilibration_steps=1000,
                   MD_steps=1140)


def test_minimize(tmp_path, control):
    """ Test that minimize runs and minimises the energy """

    orig_conf = control.simulation.engine.dlpoly.config.atoms

    control.minimize(n_steps=10,
                     output_log=tmp_path / 'minim.log',
                     work_dir=tmp_path)

    new_conf = control.simulation.engine.dlpoly.config.atoms

    assert orig_conf != new_conf


def test_run(tmp_path, control):
    """ Test that refine starts a DLP calculation """

    orig_conf = control.simulation.engine.dlpoly.config.atoms

    control.simulation.engine.dlpoly.workdir = tmp_path
    control.simulation.engine.dlpoly.control.io_file_output = tmp_path / "run.log"
    control.refine(n_steps=1)

    new_conf = control.simulation.engine.dlpoly.config.atoms

    assert orig_conf != new_conf


def test_equil(tmp_path, control):
    """ Test that equilibrate runs an equilibration phase """

    orig_conf = control.simulation.engine.dlpoly.config.atoms

    control.equilibrate(n_steps=10,
                        output_log=tmp_path / 'equilibration.log',
                        work_dir=tmp_path,
                        debug=True)

    new_conf = control.simulation.engine.dlpoly.config.atoms

    assert orig_conf != new_conf


@pytest.mark.mpi
def test_minimize_mpi(tmp_path, control):
    """ Test that minimize runs in MPI and minimises the energy """

    orig_conf = control.simulation.engine.dlpoly

    control.minimize(n_steps=10,
                     output_log=tmp_path / 'minim.log',
                     work_dir=tmp_path,
                     numProcs=4)

    new_conf = control.simulation.engine.dlpoly.config.atoms

    assert orig_conf != new_conf
