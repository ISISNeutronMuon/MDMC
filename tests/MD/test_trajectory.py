"""
Tests for creating Structure, BoundingBox, and Coulombic objects
and setting their attributes.
"""


import numpy as np
import pytest

pytestmark = [pytest.mark.lammps]
pytest.importorskip("lammps")

from MDMC.MD.interactions import Bond, BondAngle, Coulombic, Dispersion
from MDMC.MD.simulation import Universe, Shake, PPPM, Simulation
from MDMC.trajectory_analysis.compact_trajectory import CompactTrajectory
from MDMC.MD.structures import (Atom, Molecule)


NUMBER_OF_STEPS = 2000


@pytest.fixture(scope='module')
def water_trajectory():
    """Runs a short simulation of water using LAMMPS.
    The main features of the simulation are:
    - it has a finite number of steps
    - it has a finite number of atoms
    - it has 3 atom types, but only 2 chemical elements

    Yields:
        a CompactTrajectory produced by LAMMPSEngine.
    """
    # Build universe
    # Cubic universe of side:
    # 24.83602653 is 512 water molecules
    universe = Universe(dimensions=24.836)
    H1 = Atom('H', name = 'H1')
    H2 = Atom('H', position=(0., 1.63298, 0.), name = 'H2')
    O = Atom('O', position=(0., 0.81649, 0.57736))
    H_coulombic = Coulombic(atoms=[H1, H2], cutoff=10.)
    O_coulombic = Coulombic(atoms=O, cutoff=10.)
    water_mol = Molecule(position=(0, 0, 0),
                        velocity=(0, 0, 0),
                        atoms=[H1, H2, O],
                        interactions=[Bond((H1, O), (H2, O), constrained=True),
                                    BondAngle(H1, O, H2, constrained=True)],
                        name='water')
    shake = Shake(1e-4, 100)
    universe.constraint_algorithm = shake
    e_solver = PPPM(accuracy=1e-5)
    universe.electrostatic_solver = e_solver
    universe.fill(water_mol, num_density=0.03356718472021752)
    O_dispersion = Dispersion(universe, (O.atom_type, O.atom_type), cutoff=10.,
                            vdw_tail_correction=True)
    universe.add_force_field('SPCE')

    simulation = Simulation(universe,
                            engine="lammps",
                            time_step=0.5,
                            temperature=280.,
                            traj_step=1)

    # Energy Minimization and equilibration
    simulation.minimize(n_steps=2000)
    simulation.run(n_steps=2000, equilibration=True)
    simulation.run(n_steps=NUMBER_OF_STEPS)
    traj = simulation.trajectory

    yield traj
    simulation.engine.lmp.close()


def test_empty_trajectory():
    """Test that we can create an empty trajectory,
    and it really is empty.
    """
    traj = CompactTrajectory()
    assert len(traj) == 0

def test_number_of_atoms(water_trajectory):
    """Test that the number of atoms in the trajectory
    is the same as in the simulation.

    Arguments:
        water_trajectory -- The CompactTrajectory (fixture)
    """
    traj = water_trajectory
    assert traj.n_atoms == 1536

def test_name_masking(water_trajectory):
    """Test that the number of atoms in the trajectory
    is the same as in the simulation.

    Arguments:
        water_trajectory -- The CompactTrajectory (fixture)
    """
    traj = water_trajectory
    assert traj.time is traj.times
    assert traj.position is traj.positions
    assert traj.velocity is traj.velocities

def test_number_of_elements(water_trajectory):
    """Test that the number of atoms in the trajectory
    is correct for each chemical element.

    Arguments:
        water_trajectory -- The CompactTrajectory (fixture)
    """
    traj = water_trajectory
    element_array = np.array(traj.element_list)
    assert np.sum(element_array == 'O') == 512
    assert np.sum(element_array == 'H') == 1024

def test_lammps_trajectory_length(water_trajectory):
    """
    Test that the number of the simulation frames in the
    trajectory is correct.

    A LAMMPS 'run 0' command generates a trajectory
    with length 1. For this reason the trajectory
    has to be 1 step longer than the number of steps.

    Arguments:
        water_trajectory -- The CompactTrajectory (fixture)
    """
    traj = water_trajectory
    assert len(traj) == NUMBER_OF_STEPS + 1

def test_lammps_trajectory_slicing(water_trajectory):
    """
    Check that the time array is sliced correctly,
    and the last element of the slice is the same
    and element 80 of the original trajectory.

    Arguments:
        water_trajectory -- The CompactTrajectory (fixture)
    """
    traj = water_trajectory
    sliced = traj[5:81:5]
    assert sliced.time[-1] == traj.time[80]

def test_trajectory_identity(water_trajectory):
    """Test that a subtrajectory containing all the frames
    is identical to the original trajectory.

    Arguments:
        water_trajectory -- The CompactTrajectory (fixture)
    """
    traj = water_trajectory
    identical_slice = traj[traj.first_index : traj.last_index + 1: 1]
    assert traj == identical_slice

def test_trajectory_identity_filter_by_element(water_trajectory):
    """Test that a subtrajectory containing all the chemical elements
    is identical to the original trajectory.

    Arguments:
        water_trajectory -- The CompactTrajectory (fixture)
    """
    subtraj = water_trajectory.filter_by_element(['H', 'O'])
    assert subtraj == water_trajectory

def test_trajectory_filter_by_element(water_trajectory):
    """Test that filtering by chemical element returns
    the correct number of atoms.

    Arguments:
        water_trajectory -- The CompactTrajectory (fixture)
    """
    subtraj = water_trajectory.filter_by_element(['H'])
    assert subtraj.n_atoms == 1024
    assert len(subtraj.element_set) == 1

def test_trajectory_filter_by_type(water_trajectory):
    """Test that filtering by atom type returns
    the correct number of atoms.

    Arguments:
        water_trajectory -- The CompactTrajectory (fixture)
    """
    subtraj = water_trajectory.filter_by_type([1])
    assert subtraj.n_atoms == 512
    assert len(subtraj.element_set) == 1

def test_trajectory_identity_two_filters(water_trajectory):
    """Test that filtering by atom types 1 and 2 (H1, H2),
    and by chemical element H produces the same subtrajectory.

    Arguments:
        water_trajectory -- The CompactTrajectory (fixture)
    """
    subtraj1 = water_trajectory.filter_by_type([1, 2])
    subtraj2 = water_trajectory.filter_by_element(['H'])
    assert subtraj1 == subtraj2

def test_trajectory_infinite_for_loop(water_trajectory):
    """Test that the iteration over the CompactTrajectory
    elements does not run over an infinite number of
    0-lenght CompactTrajectories.

    Arguments:
        water_trajectory -- The CompactTrajectory (fixture)
    """
    correct_loop_length = len(water_trajectory)
    iterator = 0
    for _ in water_trajectory:
        iterator += 1
        if iterator > correct_loop_length:
            raise RuntimeError("Infinite loop in CompactTrajectory!")
    assert correct_loop_length == iterator
