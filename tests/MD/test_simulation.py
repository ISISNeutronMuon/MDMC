"""Tests for setting up a simulation"""

from collections import Counter
from itertools import combinations, permutations

import numpy as np
import numpy.testing as npt
import pytest
from pytest_cases import parametrize, fixture_ref

from MDMC.MD import interactions
from MDMC.MD.force_fields.ff import WaterModel
from MDMC.MD.interaction_functions import LennardJones
import MDMC.MD.simulation as sim
from MDMC.MD.solvents.SPC_config import SPC216
import MDMC.MD.structures as su


UNIVERSE_DIMENSIONS = (10., 10., 10.)

H1_POSITION = (0., 0., 0.)
H2_POSITION = (0.151390, 0., 0.)
O_POSITION = (0.075695, 0., 0.058588)
H_MASS = 1.00794
O_MASS = 15.9994
WATER_POSITION = (1., 2., 3.)
WATER_NUM_DENSITY = 0.0333679

TOLERANCE = 1.5
SPCE_MASS = 18.01528
SPCE_DIMENSIONS = np.array([18.6206, 18.6206, 18.6206])
SPCE_NUM_MOL = len(SPC216['molecules'])  # 216
SPCE_DENSITY = SPCE_MASS * SPCE_NUM_MOL / np.prod(SPCE_DIMENSIONS)


@pytest.fixture
def universe():

    yield sim.Universe(UNIVERSE_DIMENSIONS, verbose=False)

@pytest.fixture
def atom():

    yield su.Atom('H', mass=H_MASS)

@pytest.fixture
def water_molecule():

    H1 = su.Atom('H', mass=H_MASS)
    H2 = su.Atom('H', position=H2_POSITION, mass=H_MASS)
    O = su.Atom('O', position=O_POSITION, mass=O_MASS)
    H_coulombic = interactions.Coulombic(atoms=[H1, H2])
    O_coulombic = interactions.Coulombic(atoms=O)
    water_molecule = su.Molecule(position=WATER_POSITION, atoms=[H1, H2, O],
                                 interactions=[interactions.Bond((H1, O), (H2, O)),
                                               interactions.BondAngle(H1, O, H2)],
                                 name='water')
    yield water_molecule

@pytest.fixture
def water_SPCE_universe(water_molecule):

    water_universe = sim.Universe(UNIVERSE_DIMENSIONS, verbose=False)
    water_universe.fill(water_molecule, force_field='SPCE',
                        num_density=WATER_NUM_DENSITY)
    O_atom_type = next(atom.atom_type for atom in water_universe.atoms
                       if atom.element.symbol == 'O')
    O_dispersion = interactions.Dispersion(water_universe, (O_atom_type, O_atom_type))
    yield water_universe

@pytest.fixture
def kspace_solver():

    yield sim.Ewald(accuracy=0.0001)

@pytest.fixture
def small_diatomic():
    """
    Creates molecular hydrogen (H2) with normal internuclear separation
    and therefore a small bounding box relative to the size of the universe.
    """

    yield su.Molecule(atoms=[su.Atom('H', position=(0, 0, 0)),
                              su.Atom('H', position=([np.sqrt(3)] * 3))])

@pytest.fixture
def large_diatomic():
    """
    Creates molecular hydrogen with a large internuclear separation,
    orientated so that its bounding box is very large relative to
    the universe.
    """

    yield su.Molecule(atoms=[su.Atom('H', position=(0, 0, 0)),
                              su.Atom('H',
                                      position=(np.array(UNIVERSE_DIMENSIONS)/2)
                                     )])

@pytest.fixture
def solvated_universe():

    uni = sim.Universe(SPCE_DIMENSIONS, verbose=False)
    uni.solvate(SPCE_DENSITY, tolerance=TOLERANCE)

    yield uni

class MockSimulation(sim.Simulation):
    """
    Mock the ``Simulation`` to use the MockEngine.
    """

    def __init__(self, universe: sim.Universe, traj_step: int,
                 time_step: float = 1., engine = None, **settings):
        self.universe = universe
        self.settings = settings
        self.traj_step = traj_step
        self.time_step = time_step
        self.engine = engine

class MockEngine:
    """
    A mock engine which 'runs' for equilibration
    and returns properties according to a given function.
    """
    def __init__(self, pe_stability_point, temp_stability_point, **ignored):
        self.current_steps = 0
        self.pe_stability_point = pe_stability_point
        self.temp_stability_point = temp_stability_point
        self.rng = np.random.default_rng(seed=1234567)  # rng for noising functions

    def run(self, n_steps, **ignored):
        self.current_steps += n_steps

    def eval(self, var):
        def pe_func(x):
            if x < self.pe_stability_point:
                signal = x
            else:
                signal = self.pe_stability_point  # constant chosen so curve is continuous

            return (signal + self.rng.normal(0, 50))  # add noise

        def temp_func(x):
            if x < self.temp_stability_point:
                signal = x
            else:
                signal = self.temp_stability_point  # constant chosen so curve is continuous

            return (signal + self.rng.normal(0, 50))  # add noise

        def complex_func(x):
            """A more complicated function!"""
            if x < self.temp_stability_point/2:  # use temp stability for convenience
                signal =  x + 50 * np.sin(0.05*x)
            elif x < self.temp_stability_point:
                signal = 2*x - self.temp_stability_point/2
            else:
                signal = 3/2 * self.temp_stability_point

            return (signal + np.random.normal(0, 50))  # add noise

        if var == 'pe':
            return pe_func(self.current_steps)
        if var == 'temp':
            return temp_func(self.current_steps)
        if var == 'complex':
            return complex_func(self.current_steps)


def get_dispersions(inters):
    """
    Parameters
    ----------
    inters : list
        A list of Interaction objects

    Returns
    -------
    list
        A list of all Dispersion interactions
    """
    return list(filter(lambda x: isinstance(x, interactions.Dispersion), inters))

def test_create_universe(universe):

    npt.assert_array_equal(UNIVERSE_DIMENSIONS, universe.dimensions)

    universe_equal = sim.Universe(UNIVERSE_DIMENSIONS, verbose=False)
    universe_unequal = sim.Universe((9., 9., 9.), verbose=False)
    assert universe == universe
    assert universe == universe_equal
    assert universe != universe_unequal


def test_universe_stdout(capsys):
    # Capture stdout using pytest fixure
    univ = sim.Universe(SPCE_DIMENSIONS)
    univ.solvate(SPCE_DENSITY, tolerance=TOLERANCE)
    stdout = capsys.readouterr().out
    assert stdout == ('Universe created with:\n'
                      '  Dimensions       [18.62, 18.62, 18.62]\n'
                      '  Force field                       None\n'
                      '  Number of atoms                      0\n'
                      '\n'
                      'Force field created by solvent SPCE\n')


def test_create_atom(atom):

    npt.assert_array_equal((0., 0., 0.), atom.position)
    npt.assert_array_equal((0., 0., 0.), atom.velocity)
    assert atom.element.symbol == 'H'
    assert atom.mass == 1.00794


@parametrize("unit, changed_attr",
                         [(fixture_ref(atom),
                           ['ID', 'parent', '_interactions']),
                          (fixture_ref(water_molecule),
                           ['ID', 'parent', '_interactions', '_structure_list',
                            '_CoM_frame_positions'])
                         ]
                        )
def test_copy_structures(unit, changed_attr):
    """
    Tests that structures.copy copies the correct attributes and modifies
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


def test_copy_composite_rotation(water_molecule):
    """
    Tests that CompositeStructure.copy can have rotate passed
    """

    cpy_unit = water_molecule.copy(position=(5., 5., 5.),
                                   rotation=(90., 0., 270.))
    water_molecule.rotate(x=90., z=270.)
    position_diff = np.array([5., 5., 5.]) - np.array(WATER_POSITION)
    for original, copied in zip(water_molecule.atoms, cpy_unit.atoms):
        assert np.allclose(original.position,
                           (copied.position - position_diff),
                           5)


def test_structure_unique_ID(water_SPCE_universe):
    """
    Tests that each Structure in water_SPCE_universe has a unique ID

    Also creates copies of an atom and a molecule and tests that their IDs are
    unique
    """

    IDs = []
    for unit in list(water_SPCE_universe.structure_list):
        IDs.append(unit.ID)

    assert len(IDs) == len(set(IDs))

    cpy_atom = water_SPCE_universe.atoms[0].copy([1., 1., 1.])
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
                           interactions=[interactions.Bond(*atoms)],
                           name='water')
    for atom in atoms:
        assert atom.parent is molecule

    cpy_molecule = molecule.copy([5., 5., 5.])
    for atom in cpy_molecule.atoms:
        assert atom.parent is cpy_molecule


def test_top_level_structure(water_molecule):
    """
    Tests that the top_level_structure property returns self (if not a subunit),
    or the parent which returns self

    Tests for a free atom, an atom in a molecule, a molecule
    """

    atom = su.Atom('H')
    assert atom.top_level_structure is atom
    assert water_molecule.top_level_structure is water_molecule

    for atom in water_molecule.atoms:
        assert atom.top_level_structure is water_molecule


def test_equivalent_top_level_structures_dict(
    universe: sim.Universe, water_molecule: su.Molecule):
    """
    Test that ``Universe.equivalent_top_level_structures_dict`` correctly
    counts all equivalent structures and atoms.
    """

    H1 = su.Atom('H', mass=H_MASS)
    H2 = su.Atom('H', position=H2_POSITION, mass=H_MASS)
    O = su.Atom('O', position=O_POSITION, mass=O_MASS)
    interactions.Coulombic(atoms=[H1, H2])
    interactions.Coulombic(atoms=O)
    water_copy = su.Molecule(position=[1,1,1], atoms=[H1, H2, O],
                             interactions=[interactions.Bond((H1, O), (H2, O)),
                                           interactions.BondAngle(H1, O, H2)],
                             name='water_copy')

    atom = su.Atom('Ar', charge=0., cutoff=10.)
    interactions.Dispersion(universe=universe,
                            atom_types=(atom.atom_type, atom.atom_type),
                            cutoff=8.,
                            vdw_tail_correction=True,
                            function=LennardJones(1.0243, 3.36))

    atom_copy = su.Atom('Ar', charge=0., position=[2, 2, 2], cutoff=10.)
    interactions.Dispersion(universe=universe,
                            atom_types=(atom_copy.atom_type, atom_copy.atom_type),
                            cutoff=8.,
                            vdw_tail_correction=True,
                            function=LennardJones(1.0243, 3.36))

    # Add a Molecule and atom that was created using the same parameters,
    # but different Python objects
    universe.fill(water_molecule, num_struc_units=27)
    universe.add_structure(water_copy)
    universe.fill(atom, num_struc_units=64)
    universe.add_structure(atom_copy)

    equivalent_dict = universe.equivalent_top_level_structures_dict
    keys = list(equivalent_dict.keys())
    assert len(keys) == 2

    assert isinstance(keys[0], su.Molecule)
    assert keys[0].formula == "H2O"
    assert equivalent_dict[keys[0]] == 28

    assert isinstance(keys[1], su.Atom)
    assert keys[1].element.symbol == "Ar"
    assert equivalent_dict[keys[1]] == 65


def test_atoms(atom):

    assert atom in atom.atoms


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
    Tests that atom is added to Universe.atoms

    Tests that both Universe.atom_types and Atom.atom_type are updated

    Tests that atom interactions are added to Universe.interactions
    """

    _ = interactions.Coulombic(atoms=atom)
    assert len(universe.atom_types) == 0
    universe.add_structure(atom)
    assert atom.atoms == universe.atoms
    assert atom.atom_type == 1
    assert atom in universe.atom_types[1]
    assert interactions.Coulombic == type(universe.interactions.pop())


def test_add_molecule(universe, water_molecule):

    universe.add_structure(water_molecule)
    assert water_molecule.position.all() == np.array(WATER_POSITION).all()
    assert (sorted(water_molecule.atoms, key=id)
            == sorted(universe.atoms, key=id))

    water_CoM = ((np.array(H1_POSITION) * H_MASS)
                 + (np.array(H2_POSITION) * H_MASS)
                 + (np.array(O_POSITION) * O_MASS)) / (H_MASS * 2 + O_MASS)

    CoM_frame_correction = water_CoM + WATER_POSITION
    atom_pos_water_CoM_frame = [H1_POSITION - CoM_frame_correction,
                                H2_POSITION - CoM_frame_correction,
                                O_POSITION - CoM_frame_correction]

    for i in range(len(atom_pos_water_CoM_frame)):
        assert atom_pos_water_CoM_frame[i].all() ==\
                                    water_molecule.atoms[i].position.all()

    # Test interactions have expected element lists - 1 bond angle, 2 H-O bonds,
    # 1 dispersive on O, 1 Coulombic on O, and 2 Coulombic on H

    # Add Dispersion interaction
    O_atom_type = next(atom.atom_type for atom in water_molecule.atoms
                       if atom.element.symbol == 'O')
    O_dispersion = interactions.Dispersion(universe, (O_atom_type, O_atom_type))
    interaction_elements = []
    for interaction in water_molecule.interactions:
        interaction_elements.append(interaction.sorted_element_list())
    assert sorted([['H', 'H', 'O'], ['H', 'O'], ['H', 'O'], ['O', 'O'], ['O'],
                   ['H'], ['H']]) == sorted(interaction_elements)


def test_spce_water_molecule(universe, water_molecule):

    universe.add_structure(water_molecule)
    # Add Dispersion interaction
    O_atom_type = next(atom.atom_type for atom in water_molecule.atoms
                       if atom.element.symbol == 'O')
    O_dispersion = interactions.Dispersion(universe, (O_atom_type, O_atom_type))
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
    parameters = []
    for function in functions:
        {p.name: p.value for p in function.parameters.values()}

    # Test interaction parameters
    SPCEparameters = [{'charge':-0.8476}, {'charge':0.4238}, {'charge':0.4238},
                      {'sigma':3.166, 'epsilon':0.6502},
                      {'equilibrium_state':1.000, 'potential_strength':4637.},
                      {'equilibrium_state':1.000, 'potential_strength':4637.},
                      {'equilibrium_state':109.47, 'potential_strength':383.}]
    for parameter in parameters:
        assert parameter in SPCEparameters
        # Remove the instance so that multiple identical instances are tested
        SPCEparameters.remove(parameter)


@parametrize('structures', [fixture_ref(atom), fixture_ref(water_molecule)])
def test_add_structure_center(universe, structures):
    """
    Tests that passing center=True to universe.add_structure adds the
    structures to the center of the Universe
    """

    assert all(structures.position != universe.dimensions / 2)
    universe.add_structure(structures, center=True)
    assert all(structures.position == universe.dimensions / 2)


def test_spce_water_box(water_SPCE_universe):
    """
    Tests for correct number of interactions
    """

    n_molecules_xyz = np.array(UNIVERSE_DIMENSIONS) * WATER_NUM_DENSITY**(1./3.)
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
    N = len(water_SPCE_universe.atoms)
    assert len(water_SPCE_universe.interactions) == N + 3

    # TODO: Test for correct positions
    # water_positions = sorted([list(structures.position)
    #                           for structures
    #                           in water_SPCE_universe.configuration])
    # intermol_dist = np.array(UNIVERSE_DIMENSIONS) / int(n_molecules**(1./3.))
    # calc_positions = []
    # for x in np.arange(0, UNIVERSE_DIMENSIONS[0], intermol_dist[0]):
    #     for y in np.arange(0, UNIVERSE_DIMENSIONS[1], intermol_dist[1]):
    #         for z in np.arange(0, UNIVERSE_DIMENSIONS[2], intermol_dist[2]):
    #             calc_positions.append([x, y, z])
    # assert sorted(calc_positions) == water_positions


def test_universe_membership(water_SPCE_universe):
    """
    Tests that structures that have been added to a universe have that universe
    as self.universe

    Tests that structures that have not been added to a universe have
    self.universe == None

    Does not test for the effects of copying a Structure, as this is
    tested in test_copy_structures
    """

    uni_false = sim.Universe(5., verbose=False)
    for structure in water_SPCE_universe.structure_list:
        assert structure.universe == water_SPCE_universe
        assert structure.universe != uni_false

    atom_false = su.Atom('H')
    assert atom_false.universe is None


@parametrize("unit", [fixture_ref(atom), fixture_ref(water_molecule)])
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
    atom_positions = [atom.position for atom in unit.atoms]
    universe.add_structure(unit)
    positions_in_universe(atom_positions, universe)

    DISPLACEMENT = np.array([1.0, 1.5, -2.0])
    unit.translate(DISPLACEMENT)
    atom_positions = [atom.position for atom in unit.atoms]
    assert np.all(unit.position == unit_position + DISPLACEMENT)
    positions_in_universe(atom_positions, universe)


def test_valid_position(atom):
    """
    Tests if Structure.valid_position returns True if an atom is either not
    in a universe or within the bounds of the universe, and False otherwise
    """

    assert atom.universe is None
    assert atom.valid_position()

    atom.position = [0., 0., 0.]
    uni = sim.Universe(5.0, verbose=False)
    uni.add_structure(atom)
    assert atom.valid_position()

    atom.position = [3., 3., 3.]
    assert atom.valid_position()

    atom.position = [5., 5., 5.]
    assert atom.valid_position()

    lt_dimensions = list(set(permutations([-3., 3., 3.])))
    gt_dimensions = list(set(permutations([5.1, 3., 3.])))
    invalid_positions = lt_dimensions + gt_dimensions
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
    for atom in water_molecule.atoms:
        rel_pos[atom] = (atom.position - water_molecule.position)

    water_molecule.translate([1.2, 1.4, 1.6])
    for atom in water_molecule.atoms:
        assert all(atom.position == water_molecule.position + rel_pos[atom])


@pytest.mark.parametrize("Int, n_atoms", [(interactions.Bond, [2]),
                                          (interactions.BondAngle, [3]),
                                          (interactions.DihedralAngle, [4])])
def test_bonded_interactions(Int, n_atoms, atom):
    """
    Tests that only the correct number of atoms can be used for the interaction

    Tests that atoms added to interactions are unique i.e. there are no
    duplicate atoms
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

    # Test duplicate atoms for bonded interactions
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


@pytest.mark.parametrize("interaction, n_atoms", [(interactions.Bond, 4),
                                                  (interactions.BondAngle, 6),
                                                  (interactions.DihedralAngle, 8)])
def test_bonded_interactions_duplicate_tuples(interaction, n_atoms):
    """
    Tests that atom tuples added to BondedInteractions are unique i.e. there are
    no duplicates.

    For instance reversed atom order is equivalent for Bond, BondAngle and
    proper Dihedral e.g. (1, 2, 3, 4) and (4, 3, 2, 1) are equivalent for proper
    Dihedrals, and should raise an error if both are passed.
    """

    atoms = [su.Atom('H', position=np.array([1., 1., 1.]) * n)
             for n in range(n_atoms)]

    # Test a valid BondedInteraction.__init__ (i.e. no equivalent permutations)
    # doesn't raise an error
    subset = tuple(atoms[:n_atoms//2])
    interaction(subset, tuple(atoms[n_atoms//2:]))

    with pytest.raises(ValueError):
        interaction(subset, tuple(reversed(subset)))


def test_improper_dihedral_duplicate_tuples():
    """
    Tests that atom tuples added to improper Dihedrals are unique i.e. there are
    no duplicates.

    For instance tuples with the same first atom and any permutation of same
    other three atoms for are equivalent e.g. (1, 2, 3, 4) and (1, 3, 4, 2) are
    equivalent, and should raise an error if both are passed.
    """

    atoms = [su.Atom('H', position=np.array([1., 1., 1.]) * n)
             for n in range(8)]

    # Test a valid Dihedral.__init__(improper=True) (i.e. no equivalent
    # permutations) doesn't raise an error
    subset = tuple(atoms[:4])
    interactions.DihedralAngle(subset, tuple(atoms[4:]), improper=True)

    for permutation in permutations(subset[1:]):
        duplicates = (subset[0], ) + tuple(permutation)
        with pytest.raises(ValueError):
            interactions.DihedralAngle(subset, duplicates, improper=True)


def test_universe_atom_types(water_molecule, universe):
    """
    Tests that Universe.atom_types is set correctly when atoms are added and
    when interactions are added to the atoms
    """

    C = su.Atom('C', mass=12.0107, atom_type=2)
    assert C.atom_type == 2
    _ = interactions.Coulombic(atoms=C)
    H1, H2, O = water_molecule.atoms

    assert len(universe.atom_types) == 0
    universe.add_structure(C)
    universe.add_structure(water_molecule)

    for atom, atom_type in {C:2, H1:1, H2:1, O:3}.items():
        assert atom.atom_type == atom_type
        assert atom in universe.atom_types[atom_type]


@pytest.mark.parametrize("atom_types_init, atom_types_expected",
                         [(((1, 1), ),
                           ((1, 1), )),
                          (((1, 2), ),
                           ((1, 2), )),
                          (((1, 1), (2, 2)),
                           ((1, 1), (2, 2))),
                          (((1, 1), (1, 1)),
                           ((1, 1), )),
                          (((1, 2), (2, 1)),
                           ((1, 2), )),
                          (((2, 1), ),
                           ((1, 2), )),
                          (((2, 3), (4, 1), (1, 2)),
                           ((1, 2), (1, 4), (2, 3))),
                          (([1, 2], ),
                           ((1, 2), )),
                          ([(1, 2), [2, 3]],
                           ((1, 2), (2, 3)))
                         ])
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
    _ = interactions.Coulombic(atoms=He)
    C = su.Atom('C', mass=12.)
    _ = interactions.Coulombic(atoms=C)

    for atom in [He, C]:
        water_SPCE_universe.add_structure(atom)

    disp = interactions.Dispersion(water_SPCE_universe, *atom_types_init)
    assert disp.atom_types == atom_types_expected


@pytest.mark.parametrize("atom_types_init, error",
                         [((1), TypeError),
                          ((1, 2, 3), ValueError),
                          ((1, ), ValueError),
                          (((1, 2), (1, 2, 3)), TypeError),
                          ((1, 2, (3, 4)), TypeError),
                          ((1.0, 1.0), TypeError)])
def test_init_dispersion_atom_type_error(atom_types_init, error,
                                         water_SPCE_universe):
    """
    Tests that the appropriate errors are raised when trying to initialize
    a Dispersion interaction by passing invalid atom_types.
    """

    with pytest.raises(error):
        interactions.Dispersion(water_SPCE_universe, atom_types_init)


def test_dispersion_cutoff(water_SPCE_universe):
    """
    Tests that Dispersion can be initialized with a cutoff, and that not
    specifying a cutoff results in a cutoff attribute set to None
    """

    cutoff_disp = interactions.Dispersion(water_SPCE_universe, (1, 1), cutoff=5.0)
    assert cutoff_disp.cutoff == 5.0
    infinite_disp = interactions.Dispersion(water_SPCE_universe, (1, 1))
    assert infinite_disp.cutoff is None


def test_charge_setting(water_SPCE_universe):
    """
    Tests that charges can be set from the atom.charge attribute, if the atom
    already has a Coulombic interaction
    """

    atom = water_SPCE_universe.atoms[0]
    atom.charge = 5.0
    assert atom.charge == 5.0


@pytest.mark.parametrize("bonded_interaction, n_atoms", [(interactions.Bond, 2),
                                                         (interactions.BondAngle, 3)])
def test_bonded_constraint_set_True(bonded_interaction, n_atoms, atom):
    """
    Tests that constraints can be applied to BondedInteractions
    """

    atoms = [atom.copy([1., 1., 1.]) for _ in range(n_atoms)]
    b_i = bonded_interaction(*atoms, constrained=True)
    assert b_i.constrained


@pytest.mark.parametrize("bonded_interaction, n_atoms", [(interactions.Bond, 2),
                                                         (interactions.BondAngle, 3)])
def test_bonded_constraint_set_False(bonded_interaction, n_atoms, atom):
    """
    Tests that BondedInteractions can be unconstrained if set to False
    """

    atoms = [atom.copy([1., 1., 1.]) for _ in range(n_atoms)]
    b_i = bonded_interaction(*atoms, constrained=False)
    assert b_i.constrained is False


@pytest.mark.parametrize("bonded_interaction, n_atoms", [(interactions.Bond, 2),
                                                         (interactions.BondAngle, 3)])
def test_bonded_constraint_unset(bonded_interaction, n_atoms, atom):
    """
    Tests that BondedInteractions are unconstrained if no constraint is applied
    """

    atoms = [atom.copy([1., 1., 1.]) for _ in range(n_atoms)]
    b_i = bonded_interaction(*atoms)
    assert b_i.constrained == False


def test_universe_multiple_solvers(kspace_solver):
    """
    Tests that both an electrostatic_solver and a dispersive solver can be
    passed when initializing a Universe
    """

    uni = sim.Universe(UNIVERSE_DIMENSIONS,
                       electrostatic_solver=kspace_solver,
                       dispersive_solver=kspace_solver,
                       verbose=False)
    assert uni.electrostatic_solver == kspace_solver
    assert uni.dispersive_solver == kspace_solver


def test_universe_multiple_solvers_error(kspace_solver):
    """
    Tests that if either electrostatic_solver or dispersive_solver and a
    kspace_solver are passed when initializing a Universe, a ValueError is
    raised.
    """

    with pytest.raises(ValueError):
        _ = sim.Universe(UNIVERSE_DIMENSIONS,
                           kspace_solver=kspace_solver,
                           electrostatic_solver=kspace_solver,
                       verbose=False)
        _ = sim.Universe(UNIVERSE_DIMENSIONS,
                           kspace_solver=kspace_solver,
                           dispersive_solver=kspace_solver,
                       verbose=False)
        _ = sim.Universe(UNIVERSE_DIMENSIONS,
                           kspace_solver=kspace_solver,
                           electrostatic_solver=kspace_solver,
                           dispersive_solver=kspace_solver,
                       verbose=False)


def test_universe_fill_orientations(universe):
    """
    Tests that filling 2 separate Universe objects with a diatomic
    molecule of different orientations but the same internuclear
    separation results in the same number density of the Universe.
    """

    univ1 = universe
    univ2 = sim.Universe(UNIVERSE_DIMENSIONS, verbose=False)
    origin = (0, 0, 0)
    pos1 = (0, 1, 0)
    pos2 = (np.sqrt(0.5), np.sqrt(0.5), 0)

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


@pytest.mark.parametrize('parameter', ['num_density', 'num_struc_units'])
def test_universe_fill_no_out_of_bounds(universe, water_molecule, parameter):
    """
    Tests that filling the universe with a Structure results in
    no molecules being added outside the bounds of the universe.

    Parametrized to test for both cases where either num_density or
    num_struc_units is passed as the parameter.
    """

    if parameter == 'num_density':
        universe.fill(water_molecule, num_density=3.14)
    else:
        universe.fill(water_molecule, num_struc_units=567)

    # Define a tolerance to allow for rounding errors
    tolerance = 1e-15
    for atom in universe.atoms:
        assert all(atom.position > [0, 0, 0] - np.array([tolerance] * 3))
        assert all(atom.position < universe.dimensions)


@pytest.mark.parametrize('num_density', [3.14, 0.6, 1.0])
def test_universe_fill_equivalence(universe, num_density, water_molecule):
    """
    Tests that specifying either num_density or manually calling
    add_structure fills the universe and results in no difference in the
    actual number density achieved (other than rounding down to a cube number
    as fill does) or the types of atoms.
    """

    num_strucs = num_density * np.prod(universe.dimensions)
    num_strucs_rounded = int(np.cbrt(num_strucs)) ** 3
    universe_manual = sim.Universe(universe.dimensions, verbose=False)
    universe.fill(water_molecule, num_density=num_density)
    for i in range(num_strucs_rounded):
        universe_manual.add_structure(water_molecule)

    assert len(universe.atoms) == len(universe_manual.atoms)

    for u in [universe, universe_manual]:
        for atom in u.atoms:
            if atom.element.symbol == 'H':
                assert atom.atom_type == 1
            elif atom.element.symbol == 'O':
                assert atom.atom_type == 2


@pytest.mark.parametrize("num_density, num_struc_units", ([None, None],
                                                          [3.14, 100]))
def test_universe_fill_num_density_num_struc_error(num_density, num_struc_units,
                                                   universe, water_molecule):
    """
    Tests that the appropriate error is raised when passing both or neither
    num_density and num_struc_units as parameters.
    """

    with pytest.raises(ValueError):
        if num_density and num_struc_units:
            universe.fill(water_molecule, num_density=num_density,
                          num_struc_units=num_struc_units)
        else:
            universe.fill(water_molecule)
    with pytest.raises(ValueError) as exc:
        universe.fill(water_molecule, num_density=3.14, num_struc_units=100)
        assert exc.value.message == 'Cannot pass both'


@pytest.mark.parametrize("uni", [sim.Universe(SPCE_DIMENSIONS * scalar, verbose=False)
                                 for scalar in [0.5, 1.02]])
def test_solvate_spce_no_solute(uni):
    """
    Tests that the achieved density is within the tolerance for solvating
    with SPCE water an empty universe of dimensions smaller, and
    larger than those of the SPCE configuration box, equal is in another test.
    """

    uni.solvate(SPCE_DENSITY, tolerance=TOLERANCE)
    actual_dens = len(uni.molecule_list) * SPCE_MASS / uni.volume
    assert SPCE_DENSITY * (100 - TOLERANCE) / 100 < actual_dens and \
           actual_dens < SPCE_DENSITY * (100 + TOLERANCE) / 100

def test_solvation_fail():
    """
    Tests that a ValueError is raised when the universe has not been solvated
    to within tolerance after a given number of iterations.
    """
    uni = sim.Universe(SPCE_DIMENSIONS * 0.3, verbose=False)
    with pytest.raises(ValueError):
        uni.solvate(SPCE_DENSITY, tolerance=TOLERANCE, max_iterations=2)

@parametrize("molecule", [fixture_ref(small_diatomic), fixture_ref(large_diatomic)])
def test_solvate_spce_with_solute(molecule):
    """
    Tests that the achieved density is within the tolerance for solvating
    with SPCE water a universe containing a small diatomic molecule.

    Tests that the achieved density is within the tolerance for solvating
    with SPCE water a universe containing a large diatomic molecule.
    """

    univ = sim.Universe(SPCE_DIMENSIONS / 2, verbose=False)
    univ.add_structure(molecule)
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

    for atom in solvated_universe.atoms:
        assert all(atom.position <= solvated_universe.dimensions)
        assert all(atom.position >= [0, 0, 0])


@parametrize("molecule", [fixture_ref(small_diatomic), fixture_ref(large_diatomic)])
def test_solvate_spce_no_overlap_with_solute(molecule):
    """
    Tests that solvating a universe containing different solute molecules
    with SPCE water gives no overlaps between solvent and solute molecules.
    """

    univ = sim.Universe(SPCE_DIMENSIONS / 2, verbose=False)
    univ.add_structure(molecule)
    univ.solvate(SPCE_DENSITY, tolerance=TOLERANCE)
    solute_bounds = molecule.bounding_box
    for mol in univ.molecule_list:
        names = [atom.name for atom in mol.atoms]
        # Only compare bounding boxes of solute and SPCE water.
        if not np.array_equal(np.sort(names), np.sort(['H', 'O', 'H'])):
            for atom in mol.atoms:
                pos = atom.position
                assert not (all(pos > solute_bounds.min)
                            and all(pos < solute_bounds.max))



def test_solvate_spce_bond_lengths():
    """
    Tests that solvating 2 empty universes of different dimensions results
    in no change of the intramolecular nuclear separation lengths in the
    SPCE water solvent molecules.
    """

    # Solvate 2 universes of different dimensions.
    univ1 = sim.Universe(SPCE_DIMENSIONS*0.5, verbose=False)
    univ2 = sim.Universe(SPCE_DIMENSIONS*0.7, verbose=False)
    univ1.solvate(SPCE_DENSITY, tolerance=TOLERANCE)
    univ2.solvate(SPCE_DENSITY, tolerance=TOLERANCE)

    def _get_min_molecule(universe):

        min_norm = float('inf')
        for mol in universe.molecule_list:
            mol_norm = np.linalg.norm(mol.position)
            if mol_norm < min_norm:
                min_mol = mol
                min_norm = mol_norm
        return min_mol

    def _get_sorted_bond_lengths(molecule):

        positions = [atom.position for atom in molecule.atoms]
        lengths = []
        for pair in combinations(positions, 2):
            lengths.append(abs(np.linalg.norm(pair[1] - pair[0])))
        lengths.sort()
        return lengths

    lengths1 = _get_sorted_bond_lengths(_get_min_molecule(univ1))
    lengths2 = _get_sorted_bond_lengths(_get_min_molecule(univ2))

    assert lengths1 == lengths2


@pytest.mark.parametrize('dim_scaling', [[1, 1, 1], [2, 3, 1]])
def test_solvate_spce_density_perfect_dimensions(dim_scaling):
    """
    Tests that a perfect density (i.e. the density of SPCE water box as
    provided in GROMACS' spc216.gro at 300K) is achieved when solvating an
    empty universe with dimensions that are integer multiples of the SPCE
    water box.
    """

    univ = sim.Universe(SPCE_DIMENSIONS * np.array(dim_scaling), verbose=False)
    univ.solvate(SPCE_DENSITY)
    total_mass = 0
    for atom in univ.atoms:
        total_mass += atom.mass
    assert (abs(((total_mass / univ.volume) - SPCE_DENSITY) / SPCE_DENSITY)
            < 1e-10)


def test_solvate_no_spce_wrapping_for_non_int_univ_dimensions():
    """
    Creates a universe with a dimension that is a non-integer multiple of the
    dimensions of the SPCE water box, and tests that a known SPCE molecule with
    an out-of-bounds atom isn't added to the universe, nor that this atom is
    wrapped around back into the universe.
    """

    # Build a universe of dimensions of the SPCE box, but cut in half
    # along the z-axis, and solvate it.
    univ_dimensions = SPCE_DIMENSIONS * np.array([1, 1, 0.5])
    univ = sim.Universe(univ_dimensions, verbose=False)
    univ.solvate(SPCE_DENSITY)
    # Molecule 12 from the GROMACS spc216 configuration is known to have one
    # atom that sits out of bounds of these universe dimensions, along the
    # z-axis only. The molecule should therefore not be added to the universe,
    # nor should the atom out-of-bounds be wrapped around.
    pos1 = np.array([17.27, 3.79, 9.39])
    pos2 = np.array([15.81, 3.31, 8.84])
    pos3 = np.array([16.67, 3., 9.25])
    # If wrapped, the wrapped position is the position of the out-of-bounds
    # atom minus the universe dimension length in the z-direction.
    wrapped_pos = pos1 - univ_dimensions * np.array([0, 0, 1])
    # Check that no atoms in the universe have these positions.
    for atom in univ.atoms:
        for pos in [pos1, pos2, pos3, wrapped_pos]:
            assert all(atom.position != pos)


@pytest.mark.parametrize("solvent, parameters", [('SPCE',
                                                 (('equilibrium_state', 1.),
                                                  ('potential_strength',
                                                   4637.),
                                                  ('potential_strength', 383.),
                                                  ('equilibrium_state',
                                                   109.47),
                                                  ('charge', 0.4238),
                                                  ('charge', -0.8476),
                                                  ('epsilon', 0.6502),
                                                  ('sigma', 3.166)))])
def test_solvate_parameter_setting(solvated_universe, solvent, parameters):
    """
    Tests that the parameters of the solvent molecules are set correctly when
    the solvent has been selected from inbuilt solvents
    """

    uni_parameters = solvated_universe.parameters

    assert len(parameters) == len(uni_parameters)
    assert set(parameters) == {(p.type, p.value.real)
                               for p in list(uni_parameters.values())}


@pytest.mark.parametrize("density, tolerance", [(0.7, 20.),
                                                (0.602707, 1.)])
def test_solvate_solvated_universe_same_density(density, tolerance,
                                                solvated_universe):
    """
    Tests that if a previously solvated universe is solvated with the same
    density, there is no change in the solvent_density or the number or atoms

    Parametrizations test densities where the solvent_density is within the
    solvate tolerance
    """

    solvent_density = solvated_universe.solvent_density
    solvated_universe.solvate(density=density, tolerance=tolerance)
    assert solvent_density == solvated_universe.solvent_density


@pytest.mark.parametrize("density, tolerance", [(0.7, 1.),
                                                (6.03, 0.0001)])
def test_solvate_solvated_universe_different_density(density, tolerance,
                                                     solvated_universe):
    """
    Tests that if a previously solvated universe is solvated with a different
    density, a ValueError is raised

    Tested for densities that are both to high and too low (outside of the
    solvate tolerance)
    """

    with pytest.raises(ValueError):
        solvated_universe.solvate(density=density, tolerance=tolerance)


@pytest.mark.parametrize("univ_dimensions, pos, expected",
                         [(10., [10., 10., 10.], False),
                          (0.1, [-7., 0, 0], True),
                          ([20., 15., 1.], [21., 15., 1.], True),
                          (10., [0., 0., -0.0001], True)])
def test_check_out_of_bounds(univ_dimensions, pos, expected):
    """
    Tests whether the correct bool is returned by the function that checks
    whether a position is outside the bounds of a universe.
    """

    univ = sim.Universe(univ_dimensions, verbose=False)
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

@pytest.mark.parametrize("structures, expected",
                         [([su.Atom('H', mass=1.0)],
                           0.001),
                          ([su.Atom('N', mass=15.0)],
                           0.015),
                          ([su.Molecule(atoms=[su.Atom('He', mass=2.0),
                                               su.Atom('Na', mass=21.0)])],
                           0.023),
                          ([su.Atom('H', mass=1.0), su.Atom('Na', mass=21.0)],
                           0.022)])
def test_universe_density(structures, expected, universe):
    """
    Tests that the density property of Universe is correct
    """

    assert universe.density == 0.
    for structure in structures:
        universe.add_structure(structure)
    assert universe.density == expected

@pytest.mark.parametrize("dimensions, expected",
                         [(-1., ValueError), ([0., 1., 2.], ValueError),
                          (1, TypeError), ((1., 2.), ValueError),
                          ([1., 2., 3., 4.], ValueError), ("1.", TypeError)])
def test_universe_universe_dimensions_setting(dimensions, expected):
    """
    Tests that setting incorrect `Universe.dimensions` raises the expected errors.
    """
    with pytest.raises(expected):
        sim.Universe(dimensions, verbose=False)

def test_add_force_field_dispersions_bool(universe):

    """
    Tests that the correct Dispersion interactions are created when True is
    passed as add_dispersions to add_force_field. Tests that the correct number
    of dispersions are created, and that these have the correct atom_types.
    """

    # Create some suitable atoms for OPLSAA
    atoms = [su.Atom('S', name='26', atom_type=1),
             su.Atom('H', position=(1., 1., 1.), name='7', atom_type=2),
             su.Atom('N', position=(2., 2., 2.), name='204', atom_type=3)]
    for atom in atoms:
        universe.add_structure(atom)
    #pylint: disable=len-as-condition
    assert len(get_dispersions(universe.nonbonded_interactions)) == 0

    universe.add_force_field('OPLSAA', add_dispersions=True)
    dispersions = get_dispersions(universe.nonbonded_interactions)
    assert len(dispersions) == 3
    atom_types = [disp.atom_types for disp in dispersions]

    assert sorted(atom_types) == [((1, 1), ),
                                  ((2, 2), ),
                                  ((3, 3), )]


def test_add_force_field_dispersions_atoms(universe, water_molecule):

    """
    Tests that the correct Dispersion interactions are created when a list of
    atoms is passed as add_dispersions to add_force_field. Tests that the
    correct number of dispersions are created, and that these have the correct
    atom_types.
    """

    universe.add_structure(water_molecule)
    #pylint: disable=len-as-condition
    assert len(get_dispersions(universe.nonbonded_interactions)) == 0
    O_atoms = su.filter_atoms_element(water_molecule.atoms, 'O')
    universe.add_force_field('SPCE', add_dispersions=O_atoms)
    dispersions = get_dispersions(universe.nonbonded_interactions)
    assert len(dispersions) == 1
    atom_types = [disp.atom_types for disp in dispersions]

    assert atom_types == [((O_atoms[0].atom_type, O_atoms[0].atom_type), )]

@pytest.mark.parametrize('pe_stability_point, temp_stability_point',
                         [(2500, 2500),
                          (2500, 5000),
                          (1000, 10000),
                          (0, 1000)])
@pytest.mark.parametrize('variables', [['pe', 'temp'], ['complex']])
def test_auto_equilibrate(universe, variables, pe_stability_point, temp_stability_point):
    simulation = MockSimulation(universe, 1, 1,
                                MockEngine(pe_stability_point, temp_stability_point))

    eq_steps, _ = simulation.auto_equilibrate(variables=variables)
    # assert that it doesn't under-equilibrate
    assert eq_steps >= max(pe_stability_point, temp_stability_point)
    # assert that it doesn't over-equilibrate
    assert eq_steps < 2 * max(pe_stability_point, temp_stability_point)
