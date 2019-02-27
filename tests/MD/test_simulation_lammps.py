"""Tests for setting up and running MDMC using LAMMPS

AUTHOR :    Thomas Farmer        START DATE :    11/02/2019, 16:21:38"""

from collections import Counter
from copy import deepcopy

import numpy as np
import pytest

from MDMC.common import units
import MDMC.MD.engine_facades.lammps_engine as lmp
from MDMC.MD.interaction_functions import HarmonicPotential, LennardJones, \
    Coulomb
from MDMC.MD.simulation import ConstraintAlgorithm, Rattle, Shake, Universe, \
    Ewald, PPPM, KSpaceSolver
from MDMC.MD.structural_units import Atom, Bond, BondAngle, Coulombic, \
    Dispersion, NonBondedInteraction
from MDMC.trajectory_analysis.trajectory import Trajectory


UNIVERSE_DIM = 50.0
N_ATOMS = 10
COULOMBIC_CUTOFF = 8.0

@pytest.fixture
def empty_universe():

    """
    Returns:
    A empty Universe object
    """

    return Universe(dimensions=UNIVERSE_DIM)

@pytest.fixture
def atoms():

    """
    Returns:
    A list of atoms with 4 different atom_types

    Ordering of atoms is to enable ease of comparison with atoms added to
    LAMMPS, as this is done ordered by atom_type, rather than necessary the
    order which atoms appear in universe.atom_list
    """

    symbols = ['C', 'H', 'N', 'O']
    masses = [12.011, 1.008, 14.007, 16.000]
    elements = symbols * (N_ATOMS / 4)
    elements[len(elements):N_ATOMS] = symbols[:N_ATOMS-len(elements)]
    # Sorted so that atoms of same type are grouped
    elements = sorted(elements)
    atom_types = {symbol: n+1 for n, symbol in enumerate(symbols)}
    atom_masses = {symbol: mass for symbol, mass in zip(symbols, masses)}

    return [Atom(element, position=np.array([0.5 * i]*3),
                 atom_type=atom_types[element], mass=atom_masses[element])
            for i, element in enumerate(elements)]

@pytest.fixture
def atom_pair(atoms):

    """
    Returns:
    A tuple of two atoms from the atoms fixture
    """

    return tuple(atoms[:2])

@pytest.fixture
def universe_interactions(empty_universe, atoms):

    """
    Returns:
    A tuple of (universe, bonds, angles, coulombics, dispersions) where universe
    is a Universe object with atoms and interactions, bonds is a list of Bond
    objects, angles is a list of BondAngle objects, coulombics is a list of
    Coulombic objects, and dispersions is a list of Dispersion objects.
    """

    for atom in atoms:
        empty_universe.add_structural_unit(atom)

    # Create InteractionFunctions for bonds, angles and dispersive interactions
    bond1_harmonic = HarmonicPotential((1.0, 'Ang'), (2.0, 'kJ'))
    bond2_harmonic = HarmonicPotential((2.0, 'Ang'), (4.0, 'kJ'))
    angle_harmonic = HarmonicPotential((1.0, 'Ang'), (2.0, 'kJ'))

    # Create 2 bonds for some atoms, and one angle, coulombic and dispersive
    # interaction
    bond1_atoms = [(atoms[i], atoms[i+1]) for i in range(0, len(atoms)-1, 2)]
    bond2_atoms = [(atoms[i], atoms[i+2]) for i in range(0, len(atoms)-2, 3)]
    bonds = [Bond(*bond1_atoms, function=bond1_harmonic),
             Bond(*bond2_atoms, function=bond2_harmonic)]
    angles = [BondAngle(*[(atoms[i], atoms[i+1], atoms[i+2]) for i
                          in range(0, len(atoms)-2, 3)],
                        function=angle_harmonic)]
    coulombics, dispersions = [], []
    for type in empty_universe.atom_types:
        coulombics.append(Coulombic(empty_universe, type,
                                    function=Coulomb((-1.0+type*0.5, 'e')),
                                    cutoff=COULOMBIC_CUTOFF))
        dispersions.append(Dispersion(empty_universe, type,
                                      function=LennardJones((type*0.1,
                                                             'kJ / mol'),
                                                            (type*1.0,
                                                             'Ang')),
                                      cutoff=10.0,
                                      vdw_tail_correction=True))
    return (empty_universe, bonds, angles, coulombics, dispersions)

@pytest.fixture
def universe(universe_interactions):

    """
    Returns:
    A Universe object with atoms, bonds, bond angles, coulombic and dispersion
    interactions
    """

    return universe_interactions[0]

@pytest.fixture
def bonds(universe_interactions):

    """
    Returns:
    A list of bonds
    """

    return universe_interactions[1]

@pytest.fixture
def angles(universe_interactions):

    """
    Returns:
    A list of bond angles
    """

    return universe_interactions[2]

@pytest.fixture
def coulombics(universe_interactions):

    """
    Returns:
    A list of coulombic interactions
    """

    return universe_interactions[3]

@pytest.fixture
def dispersions(universe_interactions):

    """
    Returns:
    A list of dispersion interactions
    """

    return universe_interactions[4]

@pytest.fixture
def interactions(bonds, angles, coulombics, dispersions):

    """
    Returns:
    A list of bond, angle, coulombic and dispersion interactions
    """

    return bonds + angles + coulombics + dispersions

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

@pytest.fixture
def lammps_engine_box(universe):

    """
    Returns:
    A LAMMPSEngine where the simulation box has been setup
    """

    lammps_engine = lmp.LAMMPSEngine()
    lammps_engine._init_lammps()
    lammps_engine._init_attributes(universe)
    lammps_engine._define_simulation_box(universe)

    return lammps_engine

@pytest.fixture
def lammps_engine_config(lammps_engine_box):

    """
    Returns:
    A LAMMPSEngine where the atomic configuration has been added
    """

    lammps_engine_box._build_configuration(lammps_engine_box.universe)
    return lammps_engine_box

@pytest.fixture
def lammps_engine_topology(lammps_engine_config):

    """
    Returns:
    A LAMMPSEngine where the atomic configuration and the topology have been
    added
    """

    lammps_engine_config._add_topology(lammps_engine_config.universe)
    return lammps_engine_config


@pytest.fixture
def lammps_engine_setup(lammps_engine_topology):

    """
    Returns:
    A LAMMPSEngine where the atomic configuration and the topology have been
    added, and the simulation parameters have been set. This LAMMPSEngine is
    ready to run a simulation.
    """

    # Simulation setup requires the traj_step attribute to be set. All other
    # attributes that are required are set to defaults.
    lammps_engine_topology.setup_simulation(traj_step=1, time_step=0.2)

    return lammps_engine_topology


def test_universe_dims(lammps_engine_box):

    """
    Tests that creating a simulation box from an MDMC universe results in the
    correct universe dimensions

    Lower dimensions should be 0.0
    Upper dimensions should be equal to the MDMC universe dimensions
    """

    assert 0.0 == lammps_engine_box.system_state.xlo \
               == lammps_engine_box.system_state.ylo \
               == lammps_engine_box.system_state.zlo

    assert UNIVERSE_DIM == lammps_engine_box.system_state.xhi \
                        == lammps_engine_box.system_state.yhi \
                        == lammps_engine_box.system_state.zhi


def test_number_atom_types(lammps_engine_box):

    """
    Tests that creating a simulation box from an MDMC universe results in the
    correct number of atom types
    """

    assert lammps_engine_box.system_state.ntypes == 4


def test_number_atoms(lammps_engine_config, atoms):

    """
    Tests that the correct number of atoms has been added to LAMMPS
    """

    assert lammps_engine_config.system_state.natoms == len(atoms)


def test_number_interaction_types(lammps_engine_box, bonds, angles):

    """
    Tests that creating a simulation box from an MDMC universe results in the
    correct number of each interaction type:

    - bond
    - angle
    - dihedral
    - improper

    DIHEDRAL AND IMPROPER ARE NOT CURRENTLY IMPLEMENTED
    """

    assert lammps_engine_box.system_state.nbondtypes == 2
    # LAMMPS versions <= 20190104 have a bug which incorrectly assigns the
    # number of angle types, so only test this if using a more recent version
    if lammps_engine_box.lmp.lmp.version() > 20190104:
        assert lammps_engine_box.system_state.nangletypes == 1


def test_number_interactions(lammps_engine_topology, bonds, angles):

    """
    Tests that creating a simulation box from an MDMC universe results in the
    correct allowed number of interactions per atom for each interaction type:

    - bond
    - angle
    - dihedral
    - improper

    DIHEDRAL AND IMPROPER ARE NOT CURRENTLY IMPLEMENTED
    """

    assert lammps_engine_topology.system_state.nbonds == sum([len(b) for
                                                              b in bonds])
    assert lammps_engine_topology.system_state.nangles == sum([len(a) for
                                                               a in angles])


def test_atom_type_properties(lammps_engine_config, universe):

    """
    Tests that element and mass are assigned to each list index corresponding to
    atom type equivalent to that index (-1 offset due to atom_type starting from
    1)
    """

    for atom_type, atoms in universe.atom_types.items():
        assert (lammps_engine_config.atom_type_properties[atom_type - 1]
                == (atoms[0].element, atoms[0].mass))


def test_atom_type_mass(lammps_engine_config, universe):

    """
    Tests that the mass of each atom type is set correctly in LAMMPS
    """

    for i in range(len(universe.atom_list)):
        assert (lammps_engine_config.lmp.atoms[i].mass
                == universe.atom_list[i].mass)


def test_atom_ID(lammps_engine_config, universe):

    """
    Tests that atoms created in LAMMPS have the correct ID
    """

    # Atom IDs in universe are offset by some integer related to the number of
    # time the atoms fixture is called. If this offset is subtracted, the IDs
    # should agree exactly with the LAMMPS atom IDs
    offset = universe.atom_list[0].ID - 1
    for i in range(len(universe.atom_list)):
        assert (lammps_engine_config.lmp.atoms[i].id
                == universe.atom_list[i].ID - offset)


def test_atom_type(lammps_engine_config, universe):

    """
    Tests that atoms created in LAMMPS have the correct atom types
    """

    for i in range(len(universe.atom_list)):
        assert (lammps_engine_config.lmp.atoms[i].type
                == universe.atom_list[i].atom_type)


def test_atom_position(lammps_engine_config, universe):

    """
    Tests that atoms created in LAMMPS have the correct position
    """

    for i in range(len(universe.atom_list)):
        assert (np.array(lammps_engine_config.lmp.atoms[i].position)
                == universe.atom_list[i].position).all()


def test_unimplemented_interactions(lammps_engine_config, universe):

    """
    Tests that if a universe passed to LAMMPSEngine._add_topology has any
    interactions which have not been implemented in LAMMPS, NotImplementedError
    is raised
    """

    # Add unimplemented interaction type to universe
    class Unimplemented(Dispersion): pass
    unimplemented_interaction = Unimplemented(universe, 1)

    # Create LAMMPS topology from universe, raising NotImplementedError
    with pytest.raises(NotImplementedError):
        lammps_engine_config._add_topology(universe)

@pytest.mark.parametrize('interactions, expected',
                         [('bonds', 'harmonic'),
                          ('angles', 'harmonic')])
def test_parse_bonded_styles(interactions, expected, request):

    """
    Tests that the return from parse_bonded_styles is the correct input for
    creating a LAMMPS bond_style or angle_style

    The parameters should be modified whenever a new bonded style is
    implemented
    """

    # As fixtures cannot be included in parameterization, the names of the
    # fixtures are included instead - the return values of the fixtures are then
    # recovered using request.getfixturevalue
    interactions = request.getfixturevalue(interactions)
    # Test the first interaction in each list of interactions
    assert lmp.parse_bonded_styles(interactions[0]) == expected


@pytest.mark.parametrize('interactions, expected, solver_attr',
                         [('dispersions', ['lj/cut', 10.], None),
                          ('coulombics', ['coul/cut', 8.], None),
                          ('dispersions', ['lj/long', 10.], 'kspace_solver'),
                          ('coulombics', ['coul/long', 8.], 'kspace_solver'),
                          ('dispersions', ['lj/long', 10.], 'dispersive_solver'),
                          ('coulombics', ['coul/cut', 8.], 'dispersive_solver'),
                          ('dispersions', ['lj/cut', 10.],
                           'electrostatic_solver'),
                          ('coulombics', ['coul/long', 8.],
                           'electrostatic_solver')
                         ])
def test_parse_nonbonded_styles(interactions, expected, solver_attr, universe,
                                request):

    """
    Tests that the return from parse_nonbonded_styles is the correct input for
    creating a LAMMPS pair style

    The pair style is modified if a solver is provided:
    - kspace_solver modifies both lj and coul
    - dispersive_solver modifies lj
    - coulombic_solver modifies coul

    The parameters should be modified whenever a new nonbonded style is
    implemented
    """

    # As fixtures cannot be included in parameterization, the names of the
    # fixtures are included instead - the return values of the fixtures are then
    # recovered using request.getfixturevalue
    interactions = request.getfixturevalue(interactions)
    # If a solver_attr is specified, add a PPPM solver to this attribute
    if solver_attr:
        setattr(universe, solver_attr, PPPM(accuracy=1e-4))
    # Test the first interaction in each list of interactions
    assert lmp.parse_nonbonded_styles(interactions[0]) == expected


@pytest.mark.parametrize('interaction, arguments, parser',
                         [(Bond, ['atom_pair'], 'parse_bonded_styles'),
                          (Dispersion, ['universe', 1], 'parse_nonbonded_styles')
                         ])
def test_parse_unimplemented_styles(interaction, arguments, parser, request):

    """
    Tests that parsing both bonded and nonbonded interactions with an
    unimplemented function name raises a NotImplementedError
    """

    # As fixtures cannot be included in parameterization, the names of the
    # fixtures are included instead - the return values of the fixtures are then
    # recovered using request.getfixturevalue
    # type checking enables arguments which are not dependent on fixtures (e.g.
    # atom_type)
    for i in range(len(arguments)):
        if isinstance(arguments[i], str):
            arguments[i] = request.getfixturevalue(arguments[i])

    # Add interaction without defining InteractionFunction
    undefined_interaction_function = interaction(*arguments)

    with pytest.raises(NotImplementedError):
        # Pass undefined_interaction_function as an argument to parser
        getattr(lmp, parser)(undefined_interaction_function)


@pytest.mark.parametrize('system_attr, expected',
                         [('bond_style', 'hybrid'),
                          ('angle_style', 'hybrid'),
                          ('style', 'hybrid/overlay')])
def test_create_interaction_style(lammps_engine_topology, system_attr,
                                  expected):

    """
    Tests that all interactions are created with a hybrid style, for:

    - bond
    - angle
    - dihedral
    - improper
    - nonbonded interactions

    DIHEDRAL AND IMPROPER ARE NOT CURRENTLY IMPLEMENTED
    """

    assert getattr(lammps_engine_topology.system_state, system_attr) == expected


def test_create_topology_fatal_error(lammps_engine_config, universe):

    """
    Tests that creating the topology (setting pair styles and coefficients,
    bond styles and coefficients, and angle styles and coefficients) does not
    result in a fatal error, where the LAMMPS Python interface causes Python to
    exit without throwing an error, presumably due to a segfault

    A more stringent test would check that all of the correct coefficients have
    been set in LAMMPS, however there is no way to check this through the Python
    interface. Therefore the minimum test of checking for a fatal error is used.
    """

    # No asserts as failure is indicated by segfault i.e. pytest crashes
    lammps_engine_config._add_topology(lammps_engine_config.universe)


def test_atom_charge_set(lammps_engine_topology, universe):

    """
    Tests that atom charges are set correctly
    """

    for i in range(len(universe.atom_list)):
        assert (lammps_engine_topology.lmp.atoms[i].charge
                == universe.atom_list[i].charge)


def test_atom_charges_update(lammps_engine_topology, universe):

    """
    Tests that atom charges are updated correctly

    Change the charges on the atoms in the universe and test if LAMMPS charges
    update after LAMMPEngine._update_charges is called
    """

    # Change charges and update LAMMPSEngine
    for atom in universe.atom_list:
        atom.charge *= 2.
    lammps_engine_topology._update_charges()

    for i in range(len(universe.atom_list)):
        assert (lammps_engine_topology.lmp.atoms[i].charge
                == universe.atom_list[i].charge)


@pytest.mark.parametrize('interaction_fixture, update_method',
                         [('bonds', '_update_bonds'),
                          ('angles', '_update_angles'),
                          ('dispersions', '_update_dispersions')])
def test_update_individual_interactions(lammps_engine_topology,
                                        interaction_fixture, update_method,
                                        request):

    """
    Tests that updating each individual interaction does not result in a fatal
    error, where the LAMMPS Python interface causes Python to exit without
    throwing an error, presumably due to a segfault

    A more stringent test would check that the correct coefficients for each
    interation have been set in LAMMPS, however there is no way to check this
    through the Python interface. Therefore the minimum test of checking for a
    fatal error is used.
    """

    # As fixtures cannot be included in parameterization, the names of the
    # fixtures are included instead - the return values of the fixtures are then
    # recovered using request.getfixturevalue
    interactions = request.getfixturevalue(interaction_fixture)

    # Scale all parameters for all interactions
    for interaction in interactions:
        for param in interaction.params:
            param.value *= 2

    getattr(lammps_engine_topology, update_method)(interactions)


def test_update_all_interactions(lammps_engine_topology, interactions):

    """
    Tests that updating all interactions does not result in a fatal error, where
    the LAMMPS Python interface causes Python to exit without throwing an error,
    presumably due to a segfault

    A more stringent test would check that the correct coefficients for each
    interation have been set in LAMMPS, however there is no way to check this
    through the Python interface. Therefore the minimum test of checking for a
    fatal error is used.
    """

    # Scale all parameters for all interactions
    for interaction in interactions:
        for param in interaction.params:
            param.value *= 2

    lammps_engine_topology.update_parameters()


@pytest.mark.parametrize('mix', ['GEOMETRIC',
                                 'geometric',
                                 'arithmetic',
                                 'SIXTHPOWER'])
def test_mixing(lammps_engine_config, mix):

    """
    Tests that applying different nonbonded interaction mixing styles does not
    result in a fatal error, where the LAMMPS Python interface causes Python to
    exit without throwing an error, presumably due to a segfault

    A more stringent test would check the that values of pair_modify have been
    set in LAMMPS, however there is no way to check this through the Python
    interface. Therefore the minimum test of checking for a fatal error is used.
    """

    lammps_engine_config._add_topology(lammps_engine_config.universe,
                                       nonbonded_mix=mix)


@pytest.mark.parametrize('mix', ['geometrix',
                                 'equal'])
def test_mixing_unimplemented(lammps_engine_config, mix):

    """
    Tests that applying different nonbonded interaction mixing styles does not
    result in a fatal error, where the LAMMPS Python interface causes Python to
    exit without throwing an error, presumably due to a segfault

    A more stringent test would check the that values of pair_modify have been
    set in LAMMPS, however there is no way to check this through the Python
    interface. Therefore the minimum test of checking for a fatal error is used.
    """

    with pytest.raises(ValueError):
        lammps_engine_config._add_topology(lammps_engine_config.universe,
                                           nonbonded_mix=mix)


@pytest.mark.parametrize('solver_cls, accuracy, expected', [(PPPM, 0.001,
                                                             ['pppm', 0.001]),
                                                            (Ewald, 1e-05,
                                                             ['ewald', 1e-05])])
def test_parse_kspace_solver(solver_cls, accuracy, expected):

    """
    Tests that parsing the kspace solver returns the correct input for LAMMPS
    kspace_style command
    """

    solver = solver_cls(accuracy=accuracy)
    assert lmp.parse_kspace_solver(solver) == expected


def test_parse_kspace_solver_unimplemented():

    """
    Tests that parsing an unimplemented kspace solver raises a
    NotImplementedError
    """

    solver = KSpaceSolver(accuracy=0.0001)
    with pytest.raises(NotImplementedError):
        unimplemented_solver = lmp.parse_kspace_solver(solver)


@pytest.mark.parametrize('solver_cls, style', [(PPPM, 'pppm'),
                                               (Ewald, 'ewald')])
def test_set_kspace_solver_styles(lammps_engine_config, universe, dispersions,
                                  solver_cls, style):

    """
    Tests setting the kspace solver if the Universe has a kspace_solver
    """

    # Create a kspace solver and add it to the universe of a LAMMPSEngine which
    # has not had the topology created. Then create topology to set kspace style
    # in LAMMPS.
    solver = solver_cls(accuracy=0.0001)
    universe.kspace_solver = solver
    # LAMMPS requires a single cutoff for LJ and coulombic long range
    # interactions (i.e. kspace calculations), so change the cutoff for the
    # Dispersion interactions
    for dispersion in dispersions:
        dispersion.cutoff = COULOMBIC_CUTOFF
    lammps_engine_config._add_topology(lammps_engine_config.universe)
    assert lammps_engine_config.system_state.kspace_style == style


@pytest.mark.parametrize('solver_cls, style', [(PPPM, 'pppm'),
                                               (Ewald, 'ewald')])
def test_set_kspace_solver_different_cutoffs(lammps_engine_config, universe,
                                             dispersions, solver_cls, style):

    """
    Tests that if cutoffs for dispersion and coulombic interaction are different
    it results in a ValueError
    """

    # Create a kspace solver and add it to the universe of a LAMMPSEngine which
    # has not had the topology created. Then create topology to set kspace style
    # in LAMMPS.
    solver = solver_cls(accuracy=0.0001)
    universe.kspace_solver = solver
    # Set cutoffs for dispersion interactions to be different to cutoffs for
    # coulombic interactions
    for dispersion in dispersions:
        dispersion.cutoff = COULOMBIC_CUTOFF + 2.0
    with pytest.raises(ValueError):
        lammps_engine_config._add_topology(lammps_engine_config.universe)


@pytest.mark.parametrize('solver_attr, expected',
                         [('kspace_solver', 'pppm'),
                          ('electrostatic_solver', 'pppm'),
                          ('dispersive_solver', TypeError)])
def test_set_kspace_solver_single_solver_error(lammps_engine_config, universe,
                                               dispersions, solver_attr,
                                               expected):

    """
    Tests setting the kspace solver with the different solver attributes that
    exist for a universe (kspace_solver, electrostatic_solver,
    dispersive_solver)

    kspace_solver and electrostatic_solver are valid single solvers for LAMMPS,
    however dispersive_solver must raise a TypeError
    """

    # Create a solver and add it to the universe as either a kspace_solver,
    # electrostatic_solver or a dispersive_solver. Then create topology to set
    # kspace style in LAMMPS.
    solver = PPPM(accuracy=0.0001)
    setattr(universe, solver_attr, solver)
    # LAMMPS requires a single cutoff for LJ and coulombic long range
    # interactions (i.e. kspace calculations), so change the cutoff for the
    # Dispersion interactions
    for dispersion in dispersions:
        dispersion.cutoff = COULOMBIC_CUTOFF
    if expected is TypeError:
        with pytest.raises(expected):
            lammps_engine_config._add_topology(lammps_engine_config.universe)
    else:
        lammps_engine_config._add_topology(lammps_engine_config.universe)
        assert lammps_engine_config.system_state.kspace_style == expected


def test_set_kspace_solver_multiple_solvers(lammps_engine_config, universe,
                                            dispersions):

    """
    Tests setting the kspace solver if the Universe has both an
    electrostatic_solver and a dispersion_solver and they are equal
    """

    # Create a kspace solver and add it to the universe as both an
    # electrostatic_solver and a dispersive_solver. Then create topology to set
    # kspace style in LAMMPS.
    solver = PPPM(accuracy=0.0001)
    universe.electrostatic_solver = solver
    universe.dispersive_solver = solver
    # LAMMPS requires a single cutoff for LJ and coulombic long range
    # interactions (i.e. kspace calculations), so change the cutoff for the
    # Dispersion interactions
    for dispersion in dispersions:
        dispersion.cutoff = COULOMBIC_CUTOFF
    lammps_engine_config._add_topology(lammps_engine_config.universe)
    assert lammps_engine_config.system_state.kspace_style == 'pppm'


def test_set_kspace_solver_multiple_solvers_error(lammps_engine_config,
                                                  universe,
                                                  dispersions):

    """
    Tests setting the kspace solver if the Universe has both an
    electrostatic_solver and a dispersion_solver and they are not equal
    """

    # Create different kspace solvers for universe's electrostatic_solver and
    # dispersive_solvers. Then create topology to set kspace style in LAMMPS.
    universe.electrostatic_solver = PPPM(accuracy=0.0001)
    universe.dispersive_solver = PPPM(accuracy=0.0005)
    # LAMMPS requires a single cutoff for LJ and coulombic long range
    # interactions (i.e. kspace calculations), so change the cutoff for the
    # Dispersion interactions
    for dispersion in dispersions:
        dispersion.cutoff = COULOMBIC_CUTOFF
    with pytest.raises(TypeError):
        lammps_engine_config._add_topology(lammps_engine_config.universe)


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
    constraint, a TypeError is raised
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


@pytest.mark.parametrize('temperature', [150., 300., 450.])
def test_initialize_velocities(lammps_engine_topology, temperature):

    """
    Test that the velocities have been set correctly

    Initialize the velocities by setting the temperature. Set the ensemble to
    NVE and run for 0 steps. Test if the 0 step temperature is as expected.
    """

    lammps_engine_topology.temperature = temperature

    # NVE ensemble used because it is the simplest to apply and velocity
    # initialization is ensemble independent. It is applied directly through the
    # LAMMPS interface i.e. by calling fix.
    lammps_engine_topology.lmp.fix('integrate', 'all', 'nve')
    lammps_engine_topology.lmp.run(0)
    assert lammps_engine_topology.lmp.runs[0][0].Temp[0] == temperature


@pytest.mark.parametrize('skin, neighbor_steps', [(1, 2),
                                                  (1., 2.),
                                                  (1., 2),
                                                  (3., 100)])
def test_set_neighbor_list_parameters(lammps_engine_topology, skin,
                                      neighbor_steps):

    """
    Tests that setting neighbor list parameters does not result in a fatal
    error, where the LAMMPS Python interface causes Python to exit without
    throwing an error, presumably due to a segfault

    A more stringent test would check that the neighbor list parameters have
    been set in LAMMPS, however there is no way to check this through the Python
    interface. Therefore the minimum test of checking for a fatal error is used.
    """

    lammps_engine_topology.skin = skin
    lammps_engine_topology.neighbor_steps = neighbor_steps


@pytest.mark.parametrize('momentum_steps, expected_names',
                         [({'lin_momentum_steps':5},
                           ['RemoveLinearMomentum']),
                          ({'ang_momentum_steps':10},
                           ['RemoveAngularMomentum']),
                          ({'lin_momentum_steps':20, 'ang_momentum_steps':20},
                           ['RemoveMomentum']),
                          ({'lin_momentum_steps':15, 'ang_momentum_steps':20},
                           ['RemoveLinearMomentum', 'RemoveAngularMomentum'])])
def test_remove_momentum(lammps_engine_topology, momentum_steps,
                         expected_names):

    """
    Tests that linear and/or angular momentum remover fixes are correctly
    created
    """

    # Set momentum_step attributes and apply fixes. If a ValueError is expected,
    # test for it here.
    for attr, steps in momentum_steps.items():
        setattr(lammps_engine_topology, attr, steps)

    lammps_engine_topology._set_momentum_removers()

    # The fix styles of all momentum removers should be 'momentum'. There
    # should be one fix with this fix style.
    assert (Counter(lammps_engine_topology.fix_styles)['momentum']
            == len(expected_names))

    # The name of the fix is defined by whether linear and/or angular
    # momentum is removed
    for name in expected_names:
        assert name in lammps_engine_topology.fix_names


@pytest.mark.parametrize('thermostat, styles, attributes',
                         [(None, ['nve'], {}),
                          ('nose', ['nvt'],
                           {'temperature':400., 'time_step':2., 't_damp':100}),
                          ('berendsen', ['temp/berendsen'],
                           {'temperature':400., 'time_step':2, 't_damp':100}),
                          ('langevin', ['langevin'],
                           {'temperature':400., 'time_step':2, 't_damp':100}),
                          ('rescale', ['nve', 'temp/rescale'],
                           {'temperature':100., 't_fraction':0.5,
                            't_window':10., 'rescale_step':100})])
def test_apply_thermostat(lammps_engine_topology, thermostat, styles,
                          attributes):

    """
    Tests that applying a thermostat results in the correct fix being applying
    to LAMMPS
    """

    # Set the attributes required for the specified thermostat
    for attr, value in attributes.items():
        setattr(lammps_engine_topology, attr, value)

    # Add the thermostat
    lammps_engine_topology.thermostat = thermostat

    # Test that the fix styles returned from the LAMMPS wrapper fixes attribute
    # are correct
    assert styles == lammps_engine_topology.fix_styles


@pytest.mark.parametrize('barostat, styles',
                         [(None, ['nve']),
                          ('berendsen', ['press/berendsen']),
                          ('nose', ['nph'])])
def test_apply_barostat(lammps_engine_topology, barostat, styles):

    """
    Tests that applying a barostat results in the correct fix being applied to
    LAMMPS
    """

    # Set the attributes required for all barostats and add the barostat
    lammps_engine_topology.pressure = 10.
    lammps_engine_topology.time_step = 2.0
    lammps_engine_topology.p_damp = 1000
    lammps_engine_topology.barostat = barostat

    # Test that the fix styles returned from the LAMMPS wrapper fixes attribute
    # are correct
    assert styles == lammps_engine_topology.fix_styles


@pytest.mark.parametrize('thermostat, barostat, styles, attributes',
                         [(None, None, ['nve'], {}),
                          ('nose', 'nose', ['npt'],
                           {'temperature':400., 't_damp':100, 'pressure':10.,
                            'p_damp':1000}),
                          ('berendsen', 'nose', ['temp/berendsen', 'nph'],
                           {'temperature':400., 't_damp':100, 'pressure':10.,
                            'p_damp':1000}),
                          ('langevin', 'nose', ['langevin', 'nph'],
                           {'temperature':400., 't_damp':100, 'pressure':10.,
                            'p_damp':1000}),
                          ('rescale', 'nose', ['nve', 'temp/rescale', 'nph'],
                           {'temperature':400., 't_fraction':.5, 't_window':10.,
                            'rescale_step':100, 'pressure':10., 'p_damp':1000}),
                          ('nose', 'berendsen', ['nvt', 'press/berendsen'],
                           {'temperature':400., 't_damp':100, 'pressure':10.,
                            'p_damp':1000}),
                          ('berendsen', 'berendsen', ['temp/berendsen',
                                                      'press/berendsen'],
                           {'temperature':400., 't_damp':100, 'pressure':10.,
                            'p_damp':1000}),
                          ('langevin', 'berendsen', ['langevin',
                                                     'press/berendsen'],
                           {'temperature':400., 't_damp':100, 'pressure':10.,
                            'p_damp':1000}),
                          ('rescale', 'berendsen', ['nve', 'temp/rescale',
                                                    'press/berendsen'],
                           {'temperature':400., 't_fraction':.5, 't_window':10.,
                            'rescale_step':100, 'pressure':10., 'p_damp':1000})]
                        )
def test_apply_thermostat_barostat(lammps_engine_topology, thermostat, barostat,
                                   styles, attributes):

    """
    Tests that applying both a thermostat and a barostat results in the correct
    fixes being applied to LAMMPS
    """

    # The time_step must be assigned first
    lammps_engine_topology.time_step = 2.0

    # Set the attributes required by each thermostat/barostat pair
    for attr, value in attributes.items():
        setattr(lammps_engine_topology, attr, value)

    # Add the thermostat and barostat
    lammps_engine_topology.thermostat = thermostat
    lammps_engine_topology.barostat = barostat

    # Test that the fix styles returned from the LAMMPS wrapper fixes attribute
    # are correct
    assert styles == lammps_engine_topology.fix_styles


@pytest.mark.parametrize('n_steps', [1, 5, 10])
def test_trajectory_output(lammps_engine_setup, n_steps):

    """
    Tests if a trajectory file of the correct length has been created by LAMMPS
    wrapper
    """

    # lammps_engine_simulation is setup to output trajectory every step. Run for
    # a total of n_steps
    lammps_engine_setup.run(n_steps)

    n_atoms = lammps_engine_setup.system_state.natoms
    n_lines = (n_atoms + 9) * ((n_steps / lammps_engine_setup.traj_step) + 1)
    assert len(lammps_engine_setup.trajectory_file.readlines()) == n_lines


def test_save_config(lammps_engine_topology, universe):

    """
    Tests that the LAMMPS configuration is correctly saved, by checking the
    positions, mass and charge of the LAMMPS wrapper atoms attribute
    """

    lammps_engine_topology.save_config()
    # Positions should be the same as those of the MDMC universe atoms, which
    # are also ordered by ID
    for i in range(len(universe.atom_list)):
        assert (np.array(lammps_engine_topology.saved_config[i][:3])
                == universe.atom_list[i].position).all()


def test_reset_config(lammps_engine_setup):

    """
    Tests that the reset_config method correctly changes the positions of the
    LAMMPS wrapper atoms back to the saved values

    To do this the config is saved, a short simulation is run, and the config
    is reset
    """

    lammps_engine_setup.save_config()
    lammps_engine_setup.lmp.run(100)

    n_atoms = lammps_engine_setup.system_state.natoms
    # Ensure that the atoms have moved from their starting positions - see atoms
    # fixture for what the starting positions are
    for i in range(n_atoms):
        assert (np.array(lammps_engine_setup.lmp.atoms[i].position)
                != np.array([0.5 * i]*3)).all()

    lammps_engine_setup.reset_config()
    for i in range(n_atoms):
        assert (np.array(lammps_engine_setup.lmp.atoms[i].position)
                == np.array([0.5 * i]*3)).all()


def test_convert_trajectory_output(lammps_engine_setup):

    """
    Tests that converting a trajectory results in an MDMC Trajectory object

    This does not test the correctness of the converted trajectory, purely that
    a trajectory can be converted with the correct type. The correctness of
    the trajectory conversion is covered by a system test.
    """

    lammps_engine_setup.run(10)
    assert isinstance(lammps_engine_setup.convert_trajectory(), Trajectory)


@pytest.mark.parametrize('args',
                         [{'n_steps':1000},
                          {'n_steps':1000, 'etol':0., 'ftol':1.e-8,
                           'maxeval':1000},
                          {'n_steps':5000, 'ftol':1.e-8, 'maxeval':500}])
def test_minimize(args, lammps_engine_setup):

    """
    Tests that the potential energy has been minimized

    This does not test that the minimization reduces the potential energy into a
    local minima, just that the potential energy of the system reduces

    Parameterization tests for both default and non-default minimization
    arguments
    """

    # LAMMPS needs to run for 0 steps to calculate energies - run directly using
    # LAMMPS wrapper run so that any bugs in LAMMPSEngine.run do not affect test
    lammps_engine_setup.lmp.run(0)
    start_energy = lammps_engine_setup.lmp.eval('pe')
    lammps_engine_setup.minimize(**args)
    assert lammps_engine_setup.lmp.eval('pe') < start_energy


@pytest.mark.parametrize('thermostat, barostat, add_args',
                         [(None, None, {}),
                          ('nose', None, {}),
                          ('nose', 'nose', {'pressure':1.0})])
def test_setup_simulation_run(lammps_engine_topology, thermostat, barostat,
                              add_args):

    """
    Tests that the simulation setup can run an NVE, NVT and NPT simulation with
    the default attribute values
    """

    # Simulation setup requires the traj_step attribute to be set, even though
    # it is not being used in this test
    # add_args is a dictionary of additional arguments that are required for the
    # specific ensemble
    lammps_engine_topology.setup_simulation(traj_step=1, thermostat=thermostat,
                                            barostat=barostat, **add_args)

    N_STEPS = 20
    lammps_engine_topology.lmp.run(20)

    # Test that the largest step number in the LAMMPS wrapper runs attribute
    # (which records ThermoData from the previous run) is correct
    assert max(lammps_engine_topology.lmp.runs[0][0].Step) == N_STEPS


@pytest.mark.parametrize('value', [1.0, 2.0])
def test_convert_mdmc_base_units_identity(value):

    """
    Tests converting MDMC base units to LAMMPS base units, where the units are
    the same
    """

    for unit in units.SYSTEM.values():
        if unit.components['numerator'][0] == unit \
            and unit in lmp.SYSTEM.values():
            assert lmp.convert_unit(value, unit) == value


@pytest.mark.parametrize('value', [1.0, 2.0])
def test_convert_lammps_base_units_identity(value):

    """
    Tests converting LAMMPS base units to MDMC base units, where the units are
    the same

    The same units are converted as in test_convert_mdmc_base_units_identity,
    except they are being converted from LAMMPS to MDMC
    """

    for unit in lmp.SYSTEM.values():
        if unit.components['numerator'][0] == unit \
            and unit in units.SYSTEM.values():
            assert lmp.convert_unit(value, unit, to_LAMMPS=False) == value


@pytest.mark.parametrize('mdmc_unit, lmp_value', [(units.Unit('Pa'),
                                                   1 / 101325.),
                                                  (units.Unit('kJ'),
                                                   1 / 4.184),
                                                  (units.Unit('amu'),
                                                   1 / 1.660539040e-30)])
def test_convert_mdmc_base_units(mdmc_unit, lmp_value):

    """
    Tests converting MDMC base units to LAMMPS base units, where the units are
    not the same in the two systems
    """

    assert np.isclose(lmp.convert_unit(1., mdmc_unit), lmp_value)


@pytest.mark.parametrize('lmp_unit, mdmc_value', [(units.Unit('atm'), 101325.)])
def test_convert_lammps_base_units(lmp_unit, mdmc_value):

    """
    Tests converting LAMMPS base units to MDMC base units, where the units are
    not the same in the two systems
    """

    assert np.isclose(lmp.convert_unit(1., lmp_unit, to_LAMMPS=False),
                      mdmc_value)


@pytest.mark.parametrize('mdmc_unit, lmp_value',
                         [(units.Unit('kJ') / units.Unit('mol'), 1 / 4.184),
                          (units.Unit('Pa') * units.Unit('fs'), 1. / 101325),
                          (units.Unit('amu') ** 2, 1 / (1.660539040e-30 ** 2)),
                          (units.Unit('amu') ** -1, 1.660539040e-30),
                          (units.SYSTEM['ENERGY'], 1 / 4.184),
                          (units.SYSTEM['FORCE'], 1 / 4.184)])
def test_convert_mdmc_compound_units(mdmc_unit, lmp_value):

    """
    Tests converting between MDMC compound units (units made up of multiple base
    units)
    """

    assert np.isclose(lmp.convert_unit(1., mdmc_unit), lmp_value)


def test_convert_mdmc_angular_potential_strength():

    """
    Tests converting into LAMMPS angular potential strength units for harmonic
    bond angles, which uses radians as the unit of angle, rather than degrees
    """

    mdmc_unit = units.SYSTEM['ENERGY'] / units.SYSTEM['ANGLE'] ** 2
    lmp_value = 784.6095482819655
    assert np.isclose(lmp.convert_unit(1., mdmc_unit), lmp_value)

@pytest.mark.parametrize('lmp_unit, mdmc_value',
                         [(units.Unit('kcal') / units.Unit('mol'), 4.184),
                          (units.Unit('atm') * units.Unit('fs'), 101325.),
                          (lmp.SYSTEM['MASS'] ** 2, (1.660539040e-30 **2)),
                          (lmp.SYSTEM['MASS'] ** -1, 1. / 1.660539040e-30),
                          (lmp.SYSTEM['ENERGY'], 4.184),
                          (lmp.SYSTEM['FORCE'], 4.184)])
def test_convert_lammps_compound_units(lmp_unit, mdmc_value):

    """
    Tests converting between MDMC compound units (units made up of multiple base
    units)
    """

    assert np.isclose(lmp.convert_unit(1., lmp_unit, to_LAMMPS=False),
                      mdmc_value)


def test_convert_mdmc_compound_equivalence():

    """
    Tests that converting an MDMC compound unit produces the same answer as
    performing the conversions individually
    """

    p = units.SYSTEM['PRESSURE']
    e = units.SYSTEM['ENERGY']

    assert lmp.convert_unit(1., p / e) == (lmp.convert_unit(1., p)
                                           / lmp.convert_unit(1., e))


def test_partition_single_interaction(interactions, bonds):

    """
    Tests using partition_interactions function to filter a single interaction
    name from a list
    """

    assert bonds == list(lmp.partition_interactions(interactions, ['Bond'])[0])


def test_partition_multiple_interactions(interactions, bonds, angles,
                                         coulombics):

    """
    Tests using partition_interactions function to partition multiple
    interactions based on name
    """

    p_bonds, p_angles, p_coulombics = lmp.partition_interactions(interactions,
                                                                 ['Bond',
                                                                  'BondAngle',
                                                                  'Coulombic'])
    assert list(p_bonds) == bonds
    assert list(p_angles) == angles
    assert list(p_coulombics) == coulombics


def test_partition_interactions_unpartitioned(interactions, bonds, angles,
                                              coulombics, dispersions):

    """
    Tests that when unpartitioned=True is passed to partition_interactions, the
    final entry returned is all interactions in input that did not have a name
    in the names argument
    """

    _, _, _, p_dispersions = lmp.partition_interactions(interactions,
                                                        ['Bond',
                                                         'BondAngle',
                                                         'Coulombic'],
                                                        unpartitioned=True)
    assert list(p_dispersions) == dispersions


def test_partion_interactions_return_list(interactions, bonds, angles):

    """
    Tests that when lst=True is passed to partition_interactions, a tuple of
    lists is returned, rather than a tuple of generators
    """

    assert (bonds, angles) == lmp.partition_interactions(interactions,
                                                         ['Bond',
                                                          'BondAngle'],
                                                         lst=True)
