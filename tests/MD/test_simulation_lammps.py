"""Tests for setting up and running MDMC using LAMMPS

AUTHOR :    Thomas Farmer        START DATE :    11/02/2019, 16:21:38"""

from copy import deepcopy

import pytest

import MDMC.MD.engine_facades.lammps_engine as lmp
from MDMC.MD.simulation import ConstraintAlgorithm, Shake, Rattle
from MDMC.MD.structural_units import Atom, Bond, BondAngle


N_ATOMS = 10

@pytest.fixture
def atom():

    """
    Returns:
    A single H atom
    """

    return Atom('H')

@pytest.fixture
def atoms(atom):

    """
    Returns:
    A list of atoms
    """

    return [deepcopy(atom) for _ in range(N_ATOMS)]

@pytest.fixture
def bonds(atoms):

    """
    Returns:
    A list of bonds
    """

    return [Bond(atoms[i], atoms[i+1]) for i in range(0, len(atoms), 2)]

@pytest.fixture
def constrained_bonds(bonds):

    """
    Returns:
    A list of constrained bonds
    """

    for bond in bonds:
        bond.constrained = True

    return bonds

@pytest.fixture
def angles(atoms):

    """
    Returns:
    A list of bond angles
    """

    return [BondAngle(atoms[i], atoms[i+1], atoms[i+2]) for i
            in range(0, len(atoms)-2, 3)]

@pytest.fixture
def constrained_angles(angles):

    """
    Returns:
    A list of constrained bond angles
    """

    for angle in angles:
        angle.constrained = True
    return angles


@pytest.fixture
def bond_ID_dict(constrained_bonds):

    """
    Returns:
    A dictionary of bond: ID pairs
    """

    return {bond: ID for ID, bond in enumerate(constrained_bonds)}

@pytest.fixture
def angle_ID_dict(constrained_angles):

    """
    Returns:
    A dictionary of angle: ID pairs
    """

    return {angle: ID for ID, angle in enumerate(constrained_angles)}


def test_universe_dims():

    """
    Tests that creating a simulation box from an MDMC universe results in the
    correct universe dimensions
    """

    pass


def test_universe_shape():

    """
    Tests that creating a simulation box from MDMC universe results in the
    correct universe shape
    """

    pass


def test_number_elements():

    """
    Tests that creating a simulation box from an MDMC universe results in the
    correct number of elements
    """

    pass


def test_number_interaction_types():

    """
    Tests that creating a simulation box from an MDMC universe results in the
    correct number of each interaction type:

    - bond
    - angle
    - dihedral
    - improper

    DIHEDRAL AND IMPROPER ARE NOT CURRENTLY IMPLEMENTED
    """

    pass


def test_number_interactions_per_atom():

    """
    Tests that creating a simulation box from an MDMC universe results in the
    correct allowed number of interactions per atom for each interaction type:

    - bond
    - angle
    - dihedral
    - improper

    DIHEDRAL AND IMPROPER ARE NOT CURRENTLY IMPLEMENTED
    """

    pass


def test_atom_type_properties():

    """
    Tests that element and mass are assigned to each list index correspond to
    type equivalent to that index (-1 offset due to atom_type starting from 0)
    """

    pass


def test_atom_type_mass():

    """
    Tests that the mass of each atom type is set correctly in LAMMPS
    """

    pass


def test_atom_id():

    """
    Tests that atoms created in LAMMPS have the correct ID
    """

    pass


def test_atom_type():

    """
    Tests that atoms created in LAMMPS have the correct atom types
    """

    pass


def test_atom_position():

    """
    Tests that atoms created in LAMMPS have the correct position
    """

    pass


def test_atom_in_molecule():

    """
    Tests that atoms in a molecule created in LAMMPS have the correct molecule
    ID
    """

    pass


def test_partion_interactions():

    """
    Tests that MDMC universe interactions are partioned based on type
    """

    pass


def test_unsupported_interactions():

    """
    Tests that if a universe passed to LAMMPSEngine._add_topology has any
    interactions which have not been implemented in LAMMPS, NotImplementedError
    is raised
    """

    pass


def test_create_interaction_style():

    """
    Tests that all interactions are created with a hybrid style, for:

    - bond
    - angle
    - dihedral
    - improper

    DIHEDRAL AND IMPROPER ARE NOT CURRENTLY IMPLEMENTED
    """

    pass


def test_create_coulombic_fatal_error():

    """
    Tests that setting Coulombic pair coefficients does not result in a fatal
    error, where the LAMMPS Python interface causes Python to exit without
    throwing an error, presumably due to a segfault

    A more stringent test would check that the correct pair coefficients have
    been set in LAMMPS, however there is no way to check this through the Python
    interface. Therefore the minimum test of checking for a fatal error is used.
    """

    pass


def test_atom_charge_set():

    """
    Tests that atom charges are set correctly
    """

    pass


def test_atom_charges_update():

    """
    Tests that atom charges are updated correctly
    """


def test_create_bonds():

    """
    Tests that setting bond coefficients does not result in a fatal error, where
    the LAMMPS Python interface causes Python to exit without throwing an error,
    presumably due to a segfault

    A more stringent test would check that the correct bond coefficients have
    been set in LAMMPS, however there is no way to check this through the Python
    interface. Therefore the minimum test of checking for a fatal error is used.
    """

    pass


def test_update_bonds():

    """
    Tests that updating bond coefficients does not result in a fatal error,
    where the LAMMPS Python interface causes Python to exit without throwing an
    error, presumably due to a segfault

    A more stringent test would check that the correct bond coefficients have
    been set in LAMMPS, however there is no way to check this through the Python
    interface. Therefore the minimum test of checking for a fatal error is used.
    """

    pass


def test_create_angles():

    """
    Tests that setting angle coefficients does not result in a fatal error,
    where the LAMMPS Python interface causes Python to exit without throwing an
    error, presumably due to a segfault

    A more stringent test would check that the correct angle coefficients have
    been set in LAMMPS, however there is no way to check this through the Python
    interface. Therefore the minimum test of checking for a fatal error is used.
    """

    pass


def test_update_angles():

    """
    Tests that updating angle coefficients does not result in a fatal error,
    where the LAMMPS Python interface causes Python to exit without throwing an
    error, presumably due to a segfault

    A more stringent test would check that the correct angle coefficients have
    been set in LAMMPS, however there is no way to check this through the Python
    interface. Therefore the minimum test of checking for a fatal error is used.
    """

    pass


def test_set_ksapce_solver_single_solver():

    """
    Tests setting the kspace solver if the Universe has a kspace_solver
    """

    pass


def test_set_ksapce_solver_multiple_solvers():

    """
    Tests setting the kspace solver if the Universe has both an
    electrostatic_solver and a dispersive_solver and they are equal
    """

    pass


def test_set_ksapce_solver_single_solver_error():

    """
    Tests setting the kspace solver if the Universe only has an
    electrostatic_solver or a dispersive_solver, which should result in a
    TypeError
    """

    pass


def test_set_ksapce_solver_multiple_solvers_error():

    """
    Tests setting the kspace solver if the Universe has both an
    electrostatic_solver and a dispersive_solver and they are not equal
    """

    pass


@pytest.mark.parametrize('constraint, name', [(Shake, 'shake'),
                                              (Rattle, 'rattle')])
def test_parse_constraint_algorithm_name(constraint, name, constrained_bonds,
                                         bond_ID_dict):

    """
    Tests that passing different ConstraintAlgorithms produces the expected
    algorithm name for the input to LAMMPS fix

    Excluding the fix ID and and group-ID, the algorithm name is the index 0
    entry submitted to LAMMPS fix
    """

    constraint_algorithm = constraint(accuracy=1.0, max_iterations=1)
    assert name == lmp.parse_constraint(constraint_algorithm,
                                        bonds=constrained_bonds,
                                        bond_ID_dict=bond_ID_dict)[0]


def test_parse_constraint_algorithm_unimplemented(constrained_bonds,
                                                  bond_ID_dict):

    """
    Tests that passing an ConstraintAlgorithm that is not implemented raises a
    NotImplementedError
    """

    constraint_algorithm = ConstraintAlgorithm(accuracy=1.0, max_iterations=1)
    with pytest.raises(NotImplementedError):
        invalid_constraint = lmp.parse_constraint(constraint_algorithm,
                                                  bonds=constrained_bonds,
                                                  bond_ID_dict=bond_ID_dict)


@pytest.mark.parametrize('accuracy', [1.0, 1e-4, 5])
def test_parse_constraint_accuracy(accuracy, constrained_bonds,
                                   bond_ID_dict):

    """
    Tests that accuracy is correct in the input to LAMMPS fix

    Excluding the fix ID and and group-ID, the accuracy is the index 1
    entry submitted to LAMMPS fix. The accuracy must be a float.
    """

    constraint_algorithm = Shake(accuracy=accuracy, max_iterations=1)
    assert float(accuracy) == lmp.parse_constraint(constraint_algorithm,
                                                   bonds=constrained_bonds,
                                                   bond_ID_dict=bond_ID_dict)[1]


@pytest.mark.parametrize('max_iter', [1, 5.4])
def test_parse_constraint_max_iterations(max_iter, constrained_bonds,
                                         bond_ID_dict):

    """
    Tests that the max number of iterations is correct in the input to LAMMPS
    fix

    Excluding the fix ID and and group-ID, the number of max iterations is the
    index 2 entry submitted to LAMMPS fix. The number of max iterations must be
    an integer.
    """

    constraint_algorithm = Shake(accuracy=1.0, max_iterations=max_iter)
    assert int(max_iter) == lmp.parse_constraint(constraint_algorithm,
                                                 bonds=constrained_bonds,
                                                 bond_ID_dict=bond_ID_dict)[2]


def test_parse_constraint_bonds(constrained_bonds, bond_ID_dict):

    """
    Tests that the input to LAMMPS has the correct bond IDs

    Excluding the fix ID and and group-ID, the declaration of bond constraints
    (indicated by 'b') is the index 4 entry submitted to LAMMPS fix. Following
    this the IDs of all of the constrained bonds must be listed.
    """

    constraint_algorithm = Shake(accuracy=1.0, max_iterations=1)
    lmp_input = lmp.parse_constraint(constraint_algorithm,
                                     bonds=constrained_bonds,
                                     bond_ID_dict=bond_ID_dict)
    assert lmp_input[4] == 'b'
    assert sorted(lmp_input[5:]) == sorted([bond_ID_dict[bond] for bond
                                            in constrained_bonds])


def test_parse_constraint_angles(constrained_angles, angle_ID_dict):

    """
    Tests that the input to LAMMPS has the correct angle IDs

    Excluding the fix ID and and group-ID, the declaration of angle constraints
    (indicated by 'a') is the index 4 entry submitted to LAMMPS fix, if no bonds
    are included. Following this the IDs of all of the constrained angles must
    be listed.
    """

    constraint_algorithm = Shake(accuracy=1.0, max_iterations=1)
    lmp_input = lmp.parse_constraint(constraint_algorithm,
                                     angles=constrained_angles,
                                     angle_ID_dict=angle_ID_dict)
    assert lmp_input[4] == 'a'
    assert sorted(lmp_input[5:]) == sorted([angle_ID_dict[angle] for angle
                                            in constrained_angles])



def test_parse_constraint_bonds_angles(constrained_bonds, constrained_angles,
                                       bond_ID_dict, angle_ID_dict):

    """
    Tests that the input to LAMMPS has the correct bond IDs and angle IDs

    Excluding the fix ID and and group-ID, the declaration of bond constraints
    (indicated by 'b') is the index 4 entry submitted to LAMMPS fix. Following
    this the IDs of all of the constrained bonds must be listed. The index
    after this must be the declaration of angle constraints (indicated by 'a'),
    and then the IDs of all of the constrained angles must be listed.
    """

    constraint_algorithm = Shake(accuracy=1.0, max_iterations=1)
    lmp_input = lmp.parse_constraint(constraint_algorithm,
                                     bonds=constrained_bonds,
                                     bond_ID_dict=bond_ID_dict,
                                     angles=constrained_angles,
                                     angle_ID_dict=angle_ID_dict)
    assert lmp_input[4] == 'b'
    n_bonds = len(constrained_bonds)
    assert sorted(lmp_input[5:5+n_bonds]) == sorted([bond_ID_dict[bond]
                                                     for bond
                                                     in constrained_bonds])
    assert lmp_input[5+n_bonds] == 'a'
    assert sorted(lmp_input[5+n_bonds+1:]) == sorted([angle_ID_dict[angle]
                                                      for angle
                                                      in constrained_angles])


def test_parse_constraint_no_interactions(bond_ID_dict):

    """
    Tests that if neither bonds or angles are provided when parsing the
    constraint, a ValueError is raised
    """

    constraint_algorithm = Shake(accuracy=1.0, max_iterations=1)
    with pytest.raises(TypeError):
        lmp_input = lmp.parse_constraint(constraint_algorithm,
                                         bond_ID_dict=bond_ID_dict)


@pytest.mark.parametrize('arguments', [{'bonds':'constrained_bonds'},
                                       {'bonds':'constrained_bonds',
                                        'angle_ID_dict':'angle_ID_dict'},
                                       {'angles':'constrained_angles'},
                                       {'angles':'constrained_angles',
                                        'bond_ID_dict':'bond_ID_dict'}])
def test_parse_constraint_no_IDs(arguments, request):

    """
    Tests that if a dictionary corresponding to interaction types is not passed,
    a KeyError is raised

    The following combinations are tested:
    bonds, no ID dictionary
    bonds, angle ID dictionary
    angles, no ID dictionary
    angles, bond ID dictionary
    """

    # As fixtures cannot be included in parameterization, the names of the
    # fixtures are included instead - the return values of the fixtures are then
    # recovered using request.getfixturevalue
    arg_fixtures = {k:request.getfixturevalue(v) for k, v in arguments.items()}
    constraint_algorithm = Shake(accuracy=1.0, max_iterations=1)
    with pytest.raises(KeyError):
        lmp_input = lmp.parse_constraint(constraint_algorithm, **arg_fixtures)


def test_partition_single_interaction():

    """
    Tests using partition_interactions function to filter a single interaction
    name from a list
    """

    pass


def test_partition_multiple_interactions():

    """
    Tests using partition_interactions function to partition multiple
    interactions based on name
    """

    pass


def test_partition_interactions_unpartitioned():

    """
    Tests that when unpartitioned=True is passed to partition_interactions, the
    final entry returned is all interactions in input that did not have a name
    in the names argument
    """

    pass


def test_partion_interactions_return_list():

    """
    Tests that when lst=True is passed to partition_interactions, a tuple of
    lists is returned, rather than a tuple of generators
    """
