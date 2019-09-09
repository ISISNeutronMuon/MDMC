"""Tests for setting up a simulation

 AUTHOR :    Thomas Farmer        START DATE :    2018-4-30 13:05:13"""

from collections import Counter
from itertools import combinations, permutations

import numpy as np
import numpy.testing as npt
import pytest

from MDMC.MD.force_fields.ff import WaterModel
from MDMC.MD.interaction_functions import Parameter
import MDMC.MD.simulation as sim
from MDMC.MD.solvents.SPC_config import SPC216
import MDMC.MD.structural_units as su


UNIVERSE_DIMS = (10., 10., 10.)
UNIVERSE_SHAPE = sim.Shape.cubic

H1_POSITION = (0., 0., 0.)
H2_POSITION = (0.151390, 0., 0.)
O_POSITION = (0.075695, 0., 0.058588)
H_MASS = 1.008
O_MASS = 16.000
WATER_POSITION = (1., 2., 3.)
WATER_NUM_DENSITY = 0.0333679

TOLERANCE = 1
SPCE_MASS = 18.01499
SPCE_DIMS = SPC216['box_dims']
SPCE_NUM_MOL = len(SPC216['molecules'])
SPCE_DENSITY = SPCE_MASS * SPCE_NUM_MOL / np.prod(SPCE_DIMS)


@pytest.fixture
def universe():

    return sim.Universe(UNIVERSE_DIMS, UNIVERSE_SHAPE)

@pytest.fixture
def atom():

    return su.Atom('H', mass=H_MASS)

@pytest.fixture
def water_molecule(atom):

    H1 = atom
    H2 = su.Atom('H', position=H2_POSITION, mass=H_MASS)
    O = su.Atom('O', position=O_POSITION, mass=O_MASS)
    H_coulombic = su.Coulombic(atoms=[H1, H2])
    O_coulombic = su.Coulombic(atoms=O)
    water_molecule = su.Molecule(position=WATER_POSITION, atoms=[H1, H2, O],
                                 interactions=[su.Bond((H1, O), (H2, O)),
                                               su.BondAngle(H1, O, H2)],
                                 name='water')
    return water_molecule

@pytest.fixture
def water_SPCE_universe(water_molecule):

    water_universe = sim.Universe(UNIVERSE_DIMS, UNIVERSE_SHAPE)
    water_universe.fill(water_molecule, force_field='SPCE',
                        num_density=WATER_NUM_DENSITY)
    O_atom_type = next(atom.atom_type for atom in water_universe.atom_list
                       if atom.element == 'O')
    O_dispersion = su.Dispersion(water_universe, O_atom_type)
    return water_universe

@pytest.fixture
def kspace_solver():

    return sim.Ewald(accuracy=0.0001)

@pytest.fixture
def small_diatomic():

    """
    Creates molecular hydrogen (H2) with normal internuclear separation
    and therefore a small bounding box relative to the size of the universe.
    """

    return su.Molecule(atoms=[su.Atom('H', position=(0, 0, 0)),
                              su.Atom('H', position=([np.sqrt(3)] * 3))])

@pytest.fixture
def large_diatomic():

    """
    Creates molecular hydrogen with a large internuclear separation,
    orientated so that its bounding box is very large relative to
    the universe.
    """

    return su.Molecule(atoms=[su.Atom('H', position=(0, 0, 0)),
                              su.Atom('H', position=(np.array(UNIVERSE_DIMS) / 2))])

@pytest.fixture
def solvated_universe():

    uni = sim.Universe(SPCE_DIMS)
    uni.solvate(SPCE_DENSITY, tolerance=TOLERANCE)

    return uni


def test_create_universe(universe):

    assert UNIVERSE_SHAPE == universe.shape
    npt.assert_array_equal(UNIVERSE_DIMS, universe.dims)


def test_create_atom(atom):

    npt.assert_array_equal((0., 0., 0.), atom.position)
    npt.assert_array_equal((0., 0., 0.), atom.velocity)
    assert atom.element == 'H'
    assert atom.mass == 1.008


@pytest.mark.parametrize("unit, changed_attr",
                         [(atom(),
                           ['ID', 'parent', '_interactions']),
                          (water_molecule(atom()),
                           ['ID', 'parent', '_interactions', '_structure_list',
                            '_CoM_frame_positions'])
                         ]
                        )
def test_copy_structural_unit(unit, changed_attr):

    """
    Tests that structural_unit.copy copies the correct attributes and modifies
    the other attributes

    Checks that for all structural units which are not subunits (which is all
    units in this case) self.parent is self
    """

    new_position = (5., 5., 5.)
    cpy_unit = unit.copy(position=new_position)
    for attr in unit.__dict__:
        if attr == '_position':
            assert all(getattr(cpy_unit, attr) == new_position)
        elif attr in changed_attr:
            assert getattr(cpy_unit, attr) != getattr(unit, attr)
        elif attr == 'parent':
            assert attr is unit
        else:
            assert np.all(getattr(cpy_unit, attr) == getattr(unit, attr))


def test_structure_unique_ID(water_SPCE_universe):

    """
    Tests that each StructuralUnit in water_SPCE_universe has a unique ID

    Also creates copies of an atom and a molecule and tests that their IDs are
    unique
    """

    IDs = []
    for unit in list(water_SPCE_universe.structure_list):
        IDs.append(unit.ID)

    assert len(IDs) == len(set(IDs))

    cpy_atom = water_SPCE_universe.atom_list[0].copy([1., 1., 1.])
    assert cpy_atom.ID not in IDs

    cpy_molecule = water_SPCE_universe.molecule_list[0].copy([5., 5., 5.])
    assert cpy_molecule.ID not in IDs + [cpy_atom.ID]


def test_structure_parent():

    """
    Tests that a structure which is not a subunit has itself as a parent

    Tests that when an atom is added to a molecule, its parent attribute changes

    Tests that atoms of copied molecules have the correct parent
    """

    atom = su.Atom('H')
    assert atom.parent is atom
    cpy_atom = atom.copy([1., 1., 1.])

    atoms = [atom, cpy_atom]
    molecule = su.Molecule(position=WATER_POSITION, atoms=atoms,
                           interactions=[su.Bond(*atoms)],
                           name='water')
    for atom in atoms:
        assert atom.parent is molecule

    cpy_molecule = molecule.copy([5., 5., 5.])
    for atom in cpy_molecule.atom_list:
        assert atom.parent is cpy_molecule


def test_top_level_structure(water_molecule):

    """
    Tests that the top_level_structure method returns self (if not a subunit),
    or the parent which returns self

    Tests for a free atom, an atom in a molecule, a molecule
    """

    atom = su.Atom('H')
    assert atom.top_level_structure() is atom
    assert water_molecule.top_level_structure() is water_molecule

    for atom in water_molecule.atom_list:
        assert atom.top_level_structure() is water_molecule


def test_atom_list(atom):

    assert atom in atom.atom_list


def test_atom_type(atom):

    """
    Tests that atom_type can only be set if it has not previously been set
    """

    assert atom.atom_type is None
    atom.atom_type = 1
    assert atom.atom_type == 1
    with pytest.raises(AttributeError):
        atom.atom_type = 2


def test_add_atom(universe, atom):

    """
    Tests that atom is added to Universe.atom_list

    Tests that both Universe.atom_types and Atom.atom_type are updated

    Tests that atom interactions are added to Universe.interactions
    """

    atom_coulombic = su.Coulombic(atoms=atom)
    assert len(universe.atom_types) == 0
    universe.add_structural_unit(atom)
    assert atom.atom_list == universe.atom_list
    assert atom.atom_type == 1
    assert atom in universe.atom_types[1]
    assert su.Coulombic == type(universe.interactions.pop())


def test_add_molecule(universe, water_molecule):

    universe.add_structural_unit(water_molecule)
    assert water_molecule.position.all() == np.array(WATER_POSITION).all()
    assert sorted(water_molecule.atom_list) == sorted(universe.atom_list)

    water_CoM = ((np.array(H1_POSITION) * H_MASS)
                 + (np.array(H2_POSITION) * H_MASS)
                 + (np.array(O_POSITION) * O_MASS)) / (H_MASS * 2 + O_MASS)

    CoM_frame_correction = water_CoM + WATER_POSITION
    atom_pos_water_CoM_frame = [H1_POSITION - CoM_frame_correction,
                                H2_POSITION - CoM_frame_correction,
                                O_POSITION - CoM_frame_correction]

    for i in range(len(atom_pos_water_CoM_frame)):
        assert atom_pos_water_CoM_frame[i].all() ==\
                                    water_molecule.atom_list[i].position.all()

    # Test interactions have expected element lists - 1 bond angle, 2 H-O bonds,
    # 1 dispersive on O, 1 Coulombic on O, and 2 Coulombic on H

    # Add Dispersion interaction
    O_atom_type = next(atom.atom_type for atom in water_molecule.atom_list
                       if atom.element == 'O')
    O_dispersion = su.Dispersion(universe, O_atom_type)
    interaction_elements = []
    for interaction in water_molecule.interactions:
        interaction_elements.append(interaction.sorted_element_list())
    assert sorted([['H', 'H', 'O'], ['H', 'O'], ['H', 'O'], ['O', 'O'], ['O'],
                   ['H'], ['H']]) == sorted(interaction_elements)


def test_spce_water_molecule(universe, water_molecule):

    universe.add_structural_unit(water_molecule)
    # Add Dispersion interaction
    O_atom_type = next(atom.atom_type for atom in water_molecule.atom_list
                       if atom.element == 'O')
    O_dispersion = su.Dispersion(universe, O_atom_type)
    universe.add_force_field('SPCE')

    functions = [inter.function for inter in universe.interactions]
    function_names = [inter.function.name for inter in universe.interactions]

    # Test interaction functions
    assert Counter(function_names) == Counter(['Coulomb',
                                               'Coulomb',
                                               'HarmonicPotential',
                                               'HarmonicPotential',
                                               'HarmonicPotential',
                                               'LennardJones'])

    # A list of dictionaries with each dictionary containing a Parameter type
    # and the correspoding Parameter value
    params = []
    for function in functions:
        {p.name:p.value for p in function.params}

    # Test interaction parameters
    SPCEparams = [{'charge':-0.8476}, {'charge':0.4238}, {'charge':0.4238},
                  {'sigma':3.166, 'epsilon':0.6502},
                  {'equilibrium_state':1.000, 'potential_strength':4637.},
                  {'equilibrium_state':1.000, 'potential_strength':4637.},
                  {'equilibrium_state':109.47, 'potential_strength':383.}]
    for param in params:
        assert param in SPCEparams
        # Remove the instance so that multiple identical instances are tested
        SPCEparams.remove(param)


def test_spce_water_box(water_SPCE_universe):

    """
    Tests for correct number of interactions
    """

    n_molecules_xyz = np.array(UNIVERSE_DIMS) * WATER_NUM_DENSITY**(1./3.)
    n_molecules = np.prod(n_molecules_xyz.astype(int))

    assert int(n_molecules) == \
        len(water_SPCE_universe.configuration.molecule_list)

    # Universe only keeps a reference to a single copy of each
    # NonBondedInteraction. so the expected number of interactions, relative to
    # number of atoms, N, is:
    # Coulombic = 2
    # Dispersion = 1
    # Bond = 2N/3
    # BondAngle = N/3
    N = len(water_SPCE_universe.atom_list)
    assert len(water_SPCE_universe.interactions) == N + 3

    # TODO: Test for correct positions
    # water_positions = sorted([list(structural_unit.position) for structural_unit
    #                                 in water_SPCE_universe.configuration])
    # intermol_dist = np.array(UNIVERSE_DIMS) / int(n_molecules**(1./3.))
    # calc_positions = []
    # for x in np.arange(0, UNIVERSE_DIMS[0], intermol_dist[0]):
    #     for y in np.arange(0, UNIVERSE_DIMS[1], intermol_dist[1]):
    #         for z in np.arange(0, UNIVERSE_DIMS[2], intermol_dist[2]):
    #             calc_positions.append([x, y, z])
    # assert sorted(calc_positions) == water_positions


def test_universe_membership(water_SPCE_universe):

    """
    Tests that structures that have been added to a universe have that universe
    as self.universe

    Tests that structures that have not been added to a universe have
    self.universe == None

    Does not test for the effects of copying a StructuralUnit, as this is
    tested in test_copy_structural_unit
    """

    uni_false = sim.Universe(5.)
    for structure in water_SPCE_universe.structure_list:
        assert structure.universe == water_SPCE_universe
        assert structure.universe != uni_false

    atom_false = su.Atom('H')
    assert atom_false.universe is None


@pytest.mark.parametrize("unit", [atom(), water_molecule(atom())])
def test_translate(unit, universe):

    """
    Tests that the translate method changes the position of an atom, a molecule,
    and the corresponding positions in the universe which they belong to
    """

    def positions_in_universe(positions, universe):
        # List construction due to ambiguity with array in array
        uni_positions = [list(position) for position
                         in universe.configuration.atom_positions]
        for position in positions:
            assert list(position) in uni_positions

    unit_position = unit.position
    atom_positions = [atom.position for atom in unit.atom_list]
    universe.add_structural_unit(unit)
    positions_in_universe(atom_positions, universe)

    DISPLACEMENT = np.array([1.0, 1.5, -2.0])
    unit.translate(DISPLACEMENT)
    atom_positions = [atom.position for atom in unit.atom_list]
    assert np.all(unit.position == unit_position + DISPLACEMENT)
    positions_in_universe(atom_positions, universe)


def test_valid_position(atom):

    """
    Tests if StructuralUnit.valid_position returns True if an atom is either not
    in a universe or within the bounds of the universe, and False otherwise
    """

    assert atom.universe is None
    assert atom.valid_position()

    atom.position = [0., 0., 0.]
    uni = sim.Universe(5.0)
    uni.add_structural_unit(atom)
    assert atom.valid_position()

    atom.position = [3., 3., 3.]
    assert atom.valid_position()

    atom.position = [5., 5., 5.]
    assert atom.valid_position()

    lt_dims = list(set(permutations([-3., 3., 3.])))
    gt_dims = list(set(permutations([5.1, 3., 3.])))
    invalid_positions = lt_dims + gt_dims
    for position in invalid_positions:
        atom.position = position
        assert atom.valid_position() is False


@pytest.mark.parametrize("position, expected", [(None, [2., 9., 7.5]),
                                                ([5., 4., 3.], [5., 4., 3.])])
def test_molecule_position(position, expected):

    """
    Tests that a molecules position is set correctly on initialization, both
    when the position argument is passed and when it is left as default

    In the default case the position should be set to the CoM of the molecule,
    as determined by the atoms which are being added to the molecule
    """

    element_properties = {'H':{'pos':(2., 0., 0.), 'mass':1.0},
                          'Be':{'pos':(2., 10., 5.), 'mass':9.0},
                          'C':{'pos':(2., 9., 10.), 'mass':12.0}}

    mol = su.Molecule(position=position, atoms=[su.Atom(element,
                                                        position=prop['pos'],
                                                        mass=prop['mass'])
                                                for element, prop
                                                in element_properties.items()])
    assert np.all(mol.position == expected)


def test_molecule_subunit_positions(water_molecule):

    """
    Tests that the positions of atoms belonging to a molecule are set correctly
    when the molecule's position is set
    """

    rel_pos = {}
    for atom in water_molecule.atom_list:
        rel_pos[atom] = (atom.position - water_molecule.position)

    water_molecule.translate([1.2, 1.4, 1.6])
    for atom in water_molecule.atom_list:
        assert all(atom.position == water_molecule.position + rel_pos[atom])


@pytest.mark.parametrize("Int, n_atoms", [(su.Bond, [2]),
                                          (su.BondAngle, [3]),
                                          (su.DihedralAngle, [4])])
def test_bonded_interactions(Int, n_atoms, atom):

    """
    Tests that only the correct number of atoms can be used for the interaction

    Tests that atoms added to interactions are unique i.e. there are no
    duplicates
    """

    for n in n_atoms:
        atoms = []
        for _ in range(n):
            atoms.append(atom.copy([1., 1., 1.]))

        # Test correct number and values of atoms (i.e. no duplicates)
        valid_bond = Int(*atoms)
        assert len(valid_bond.atoms[-1]) == n

    # Test zero atoms
    empty_bond = Int()
    assert len(empty_bond.atoms) == 0

    # Test duplicates for bonded interactions
    atoms_duplicate = []
    n_duplicates = max(n_atoms)
    for _ in range(n_duplicates):
        atoms_duplicate.append(atom)

    with pytest.raises(ValueError):
        invalid_bond = Int(*atoms_duplicate)

    # Test incorrect number of atoms for bonded interactions
    for n in [min(n_atoms) - 1, max(n_atoms) + 1]:
        atoms = []
        for _ in range(n):
            atoms.append(atom.copy([1., 1., 1.]))
        with pytest.raises(TypeError):
            invalid_bond = Int(*atoms)


def test_universe_atom_types(water_molecule, universe):

    """
    Tests that Universe.atom_types is set correctly when atoms are added and
    when interactions are added to the atoms
    """

    C = su.Atom('C', mass=12.0107, atom_type=2)
    assert C.atom_type == 2
    C_coulombic = su.Coulombic(atoms=C)
    H1, H2, O = water_molecule.atom_list

    assert len(universe.atom_types) == 0
    universe.add_structural_unit(C)
    universe.add_structural_unit(water_molecule)

    for atom, atom_type in {C:2, H1:1, H2:1, O:3}.items():
        assert atom.atom_type == atom_type
        assert atom in universe.atom_types[atom_type]


@pytest.mark.parametrize("atom_types_init, atom_types_expected",
                         [((1, ), ((1, ), (1, ))),
                          (((1, ), (2, )), ((1, ), (2, ))),
                          ((1, 2), ((1, ), (2, ))),
                          (((1, 2), (3, )), ((1, 2), (3, ))),
                          (((1, 2), (3, 4)), ((1, 2), (3, 4)))])
def test_init_dispersion(atom_types_init, atom_types_expected,
                         water_SPCE_universe):

    """
    Tests initializing a dispersion object with:

    - 1 atom_type
    - 2 atom_types (same atom_types)
    - 2 atom_types (different atom_types)
    - 2 atom_types (different atom_types, full tuple)
    - 3 atom_types
    - 4 atom_types
    """

    # Add more atoms with interactions to universe so that there are sufficient
    # atom_types for all parameterizations
    He = su.Atom('He', mass=2.)
    He_coulombic = su.Coulombic(atoms=He)
    C = su.Atom('C', mass=12.)
    C_coulombic = su.Coulombic(atoms=C)

    for atom in [He, C]:
        water_SPCE_universe.add_structural_unit(atom)

    disp = su.Dispersion(water_SPCE_universe, *atom_types_init)
    assert disp.atom_types == atom_types_expected


def test_dispersion_cutoff(water_SPCE_universe):

    """
    Tests that Dispersion can be initialized with a cutoff, and that not
    specifying a cutoff results in a cutoff attribute set to None
    """

    cutoff_disp = su.Dispersion(water_SPCE_universe, 1, cutoff=5.0)
    assert cutoff_disp.cutoff == 5.0
    infinite_disp = su.Dispersion(water_SPCE_universe, 1)
    assert infinite_disp.cutoff is None


def test_charge_setting(water_SPCE_universe):

    """
    Tests that charges can be set from the atom.charge attribute, if the atom
    already has a Coulombic interaction
    """

    atom = water_SPCE_universe.atom_list[0]
    atom.charge = 5.0
    assert atom.charge == 5.0


def test_init_coulombic_atom_types():

    """
    Tests initializing a coulombic object with atom_types:

    - 1 atom_type
    - 2 atom_types (different)
    """

    pass


def test_init_coulombic_atoms():

    """
    Tests initializing a coulombic object with atoms
    """

    pass


def test_init_coulombic_with_charge():

    """
    Tests initializing a coulombic object with the charge keyword

    Test if the correct InteractionFunction (with the correct parameter) is
    created, and if the charge keyword renders the function keyword redundant.
    """

    pass


def test_coulombic_add_atom_types():

    """
    Tests adding atom_types to a coulombic object
    """

    pass


def test_coulombic_add_atoms():

    """
    Tests adding atoms to a coulombic object
    """

    pass


@pytest.mark.parametrize("bonded_interaction, n_atoms", [(su.Bond, 2),
                                                         (su.BondAngle, 3)])
def test_bonded_constraint_set_True(bonded_interaction, n_atoms, atom):

    """
    Tests that constraints can be applied to BondedInteractions
    """

    atoms = [atom.copy([1., 1., 1.]) for i in range(n_atoms)]
    b_i = bonded_interaction(*atoms, constrained=True)
    assert b_i.constrained


@pytest.mark.parametrize("bonded_interaction, n_atoms", [(su.Bond, 2),
                                                         (su.BondAngle, 3)])
def test_bonded_constraint_set_False(bonded_interaction, n_atoms, atom):

    """
    Tests that BondedInteractions can be unconstrained if set to False
    """

    atoms = [atom.copy([1., 1., 1.]) for i in range(n_atoms)]
    b_i = bonded_interaction(*atoms, constrained=False)
    assert b_i.constrained is False


@pytest.mark.parametrize("bonded_interaction, n_atoms", [(su.Bond, 2),
                                                         (su.BondAngle, 3)])
def test_bonded_constraint_unset(bonded_interaction, n_atoms, atom):

    """
    Tests that BondedInteractions are unconstrained if no constraint is applied
    """

    atoms = [atom.copy([1., 1., 1.]) for i in range(n_atoms)]
    b_i = bonded_interaction(*atoms)
    assert b_i.constrained == False


def test_universe_multiple_solvers(kspace_solver):

    """
    Tests that both an electrostatic_solver and a dispersive solver can be
    passed when initializing a Universe
    """

    uni = sim.Universe(UNIVERSE_DIMS, UNIVERSE_SHAPE,
                       electrostatic_solver=kspace_solver,
                       dispersive_solver=kspace_solver)
    assert uni.electrostatic_solver == kspace_solver
    assert uni.dispersive_solver == kspace_solver


def test_universe_multiple_solvers_error(kspace_solver):

    """
    Tests that if either electrostatic_solver or dispersive_solver and a
    kspace_solver are passed when initializing a Universe, a ValueError is
    raised.
    """

    with pytest.raises(ValueError):
        uni = sim.Universe(UNIVERSE_DIMS, UNIVERSE_SHAPE,
                           kspace_solver=kspace_solver,
                           electrostatic_solver=kspace_solver)
        uni = sim.Universe(UNIVERSE_DIMS, UNIVERSE_SHAPE,
                           kspace_solver=kspace_solver,
                           dispersive_solver=kspace_solver)
        uni = sim.Universe(UNIVERSE_DIMS, UNIVERSE_SHAPE,
                           kspace_solver=kspace_solver,
                           electrostatic_solver=kspace_solver,
                           dispersive_solver=kspace_solver)


def test_universe_fill_orientations(universe):

    """
    Tests that filling 2 separate Universe objects with a diatomic
    molecule of different orientations but the same internuclear
    separation results in the same number density of the Universe.
    """

    univ1 = universe
    univ2 = universe
    origin = (0, 0, 0)
    pos1 = (0, 1, 0)
    pos2 = (np.sqrt(0.5), np.sqrt(0.5), 0)
    # Check that the internuclear separation is the same.
    assert np.linalg.norm(pos1) == np.linalg.norm(pos2)
    # Build the 2 diatomics with different orientations.
    diatomic1 = su.Molecule(atoms=[su.Atom('H', position=origin),
                                   su.Atom('H', position=pos1)])
    diatomic2 = su.Molecule(atoms=[su.Atom('H', position=origin),
                                   su.Atom('H', position=pos2)])
    # Fill each respective universe.
    density = 0.567438
    univ1.fill(diatomic1, num_density=density)
    univ2.fill(diatomic2, num_density=density)
    # Test number densities.
    assert len(univ1.molecule_list) == len(univ2.molecule_list)


@pytest.mark.parametrize("uni", [sim.Universe(SPCE_DIMS * scalar)
                                 for scalar in [0.9, 1.0, 1.1]])
def test_solvate_spce_no_solute(uni):

    """
    Tests that the achieved density is within the tolerance for solvating
    with SPCE water an empty universe of dimensions smaller, equal to, and
    larger than those of the SPCE configuration box.
    """

    uni.solvate(SPCE_DENSITY, tolerance=TOLERANCE)
    actual_dens = len(uni.molecule_list) * SPCE_MASS / uni.volume
    assert SPCE_DENSITY * (100 - TOLERANCE) / 100 < actual_dens
    assert actual_dens < SPCE_DENSITY * (100 + TOLERANCE) / 100


@pytest.mark.parametrize("molecule", [small_diatomic(), large_diatomic()])
def test_solvate_spce_with_solute(molecule):

    """
    Tests that the achieved density is within the tolerance for solvating
    with SPCE water a universe containing a small diatomic molecule.

    Tests that the achieved density is within the tolerance for solvating
    with SPCE water a universe containing a large diatomic molecule.
    """

    univ = sim.Universe(SPCE_DIMS / 2)
    univ.add_structural_unit(molecule)
    univ.solvate(SPCE_DENSITY, tolerance=TOLERANCE)
    tot_mass = 0
    for mol in univ.molecule_list:
        tot_mass += mol.mass
    actual_dens = tot_mass / univ.volume
    assert SPCE_DENSITY * (100 - TOLERANCE) / 100 < actual_dens
    assert actual_dens < SPCE_DENSITY * (100 + TOLERANCE) / 100


def test_solvate_spce_no_out_of_bounds(solvated_universe):

    """
    Tests that solvating an empty universe with SPCE water results in no
    atoms of those solvent moleules being outside the universe bounds.
    """

    for atom in solvated_universe.atom_list:
        assert all(atom.position <= solvated_universe.dims)
        assert all(atom.position >= [0, 0, 0])


@pytest.mark.parametrize("molecule", [small_diatomic(), large_diatomic()])
def test_solvate_spce_no_overlap_with_solute(molecule):

    """
    Tests that solvating a universe containing different solute molecules
    with SPCE water gives no overlaps between solvent and solute molecules.
    """

    univ = sim.Universe(SPCE_DIMS / 2)
    univ.add_structural_unit(molecule)
    univ.solvate(SPCE_DENSITY, tolerance=TOLERANCE)
    solute_bounds = molecule.bounding_box
    for mol in univ.molecule_list:
        names = [atom.name for atom in mol.atom_list]
        # Only compare bounding boxes of solute and SPCE water.
        if not np.array_equal(np.sort(names), np.sort(['H', 'O', 'H'])):
            for atom in mol.atom_list:
                pos = atom.position
                assert not (all(pos > solute_bounds.min)
                            and all(pos < solute_bounds.max))


@pytest.mark.parametrize("dim_scalings", [(0.9, 1.1), (0.5, 0.7)])
def test_solvate_spce_bond_lengths(dim_scalings):

    """
    Tests that solvating 2 empty universes of different dimensions results
    in no change of the intramolecular nuclear separation lengths in the
    SPCE water solvent molecules.
    """

    # Solvate 2 universes of different dimensions.
    univ1 = sim.Universe(SPCE_DIMS * dim_scalings[0])
    univ2 = sim.Universe(SPCE_DIMS * dim_scalings[1])
    univ1.solvate(SPCE_DENSITY, tolerance=TOLERANCE)
    univ2.solvate(SPCE_DENSITY, tolerance=TOLERANCE)

    def _get_min_molecule(universe):

        min_norm = np.float('inf')
        for mol in universe.molecule_list:
            mol_norm = np.linalg.norm(mol.position)
            if mol_norm < min_norm:
                min_mol = mol
                min_norm = mol_norm
        return min_mol

    def _get_sorted_bond_lengths(molecule):

        positions = [atom.position for atom in molecule.atom_list]
        lengths = []
        for pair in combinations(positions, 2):
            lengths.append(abs(np.linalg.norm(pair[1] - pair[0])))
        lengths.sort()
        return lengths

    lengths1 = _get_sorted_bond_lengths(_get_min_molecule(univ1))
    lengths2 = _get_sorted_bond_lengths(_get_min_molecule(univ2))

    assert lengths1 == lengths2


@pytest.mark.parametrize("dim_scaling", [[1, 1, 1], [1, 2, 3], [2, 3, 1]])
def test_solvate_spce_density_perfect_dims(dim_scaling):

    """
    Tests that a perfect density (i.e. the density of SPCE water box as
    provided in GROMACS' spc216.gro at 300K) is achieved when solvating an
    empty universe with dimensions that are integer multiples of the SPCE
    water box.
    """

    univ = sim.Universe(SPCE_DIMS * np.array(dim_scaling))
    univ.solvate(SPCE_DENSITY)
    total_mass = 0
    for atom in univ.atom_list:
        total_mass += atom.mass
    assert (abs(((total_mass / univ.volume) - SPCE_DENSITY) / SPCE_DENSITY)
            < 1e-10)


def test_solvate_no_spce_wrapping_for_non_int_univ_dims():

    """
    Creates a universe with a dimension that is a non-integer multiple of the
    dimensions of the SPCE water box, and tests that a known SPCE molecule with
    an out-of-bounds atom isn't added to the universe, nor that this atom is
    wrapped around back into the universe.
    """

    # Build a universe of dimensions of the SPCE box, but cut in half
    # along the z-axis, and solvate it.
    univ_dims = SPCE_DIMS * np.array([1, 1, 0.5])
    univ = sim.Universe(univ_dims)
    univ.solvate(SPCE_DENSITY)
    # Molecule 12 from the GROMACS spc216 configuration is known to have one
    # atom that sits out of bounds of these universe dims, along the z-axis
    # only. The molecule should therefore not be added to the universe, nor
    # should the atom out-of-bounds be wrapped around.
    pos1 = np.array([17.27, 3.79, 9.39])
    pos2 = np.array([15.81, 3.31, 8.84])
    pos3 = np.array([16.67, 3., 9.25])
    # If wrapped, the wrapped position is the position of the out-of-bounds
    # atom minus the universe dimension length in the z-direction.
    wrapped_pos = pos1 - univ_dims * np.array([0, 0, 1])
    # Check that no atoms in the universe have these positions.
    for atom in univ.atom_list:
        for pos in [pos1, pos2, pos3, wrapped_pos]:
            assert all(atom.position != pos)


@pytest.mark.parametrize("solvent, params", [('SPCE',
                                              (('equilibrium_state', 1.),
                                               ('potential_strength', 383.),
                                               ('equilibrium_state', 109.47),
                                               ('potential_strength', 4637.),
                                               ('charge', 0.4238),
                                               ('charge', -0.8476),
                                               ('epsilon', 0.6502),
                                               ('sigma', 3.166))
                                             )])
def test_solvate_parameter_setting(solvated_universe, solvent, params):

    """
    Tests that the parameters of the solvent molcules are set correctly when
    the solvent has been selected from inbuilt solvents
    """

    test_parameters = [Parameter(parameter[1], name=parameter[0], unit='arb')
                       for parameter in params]
    uni_parameters = list(solvated_universe.parameters)

    # Check lists are same length, then remove all Parameters that have a
    # matching name and value, finally check list of Parameters is empty (i.e.
    # all Parameters have matched)
    assert len(test_parameters) == len(uni_parameters)
    from copy import copy
    for test_p in test_parameters:
        print('test = {0}'.format(test_p))
        for uni_p in copy(uni_parameters):
            print('uni = {0}'.format(uni_p))
            if (test_p.value == uni_p.value and test_p.name == uni_p.name):
                uni_parameters.remove(uni_p)
                break
    assert uni_parameters == []


@pytest.mark.parametrize("univ_dims, pos, expected", [(10., [10., 10., 10.],
                                                       False),
                                                      (0.1, [-7., 0, 0], True),
                                                      ([20., 15., 1.],
                                                       [21., 15., 1.], True),
                                                      (10., [0., 0., -0.0001],
                                                       True),
                                                      (10., [5, 5, 5], False)])
def test_check_out_of_bounds(univ_dims, pos, expected):

    """
    Tests whether the correct bool is returned by the function that checks
    whether a position is outside the bounds of a universe.
    """

    univ = sim.Universe(univ_dims)
    assert univ._check_out_of_bounds(np.array(pos)) == expected


def test_water_model_inheritance():

    """
    Tests that a class which inherits from WaterModel requires n_body to be
    defined. This test is required because although WaterModel specifies n_body
    as an abstractproperty, it can be made concrete as a static variable
    """

    class InvalidWaterModel(WaterModel):

        @property
        def interaction_dictionary(self):

            return 0

    with pytest.raises(TypeError):
        invalid = InvalidWaterModel()

    class ValidWaterModel(InvalidWaterModel):

        n_body = 3

    assert ValidWaterModel().n_body == 3
