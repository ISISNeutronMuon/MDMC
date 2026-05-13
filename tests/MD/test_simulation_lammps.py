"""Tests for setting up and running MDMC using LAMMPS"""

from collections import Counter

import numpy as np
import pytest
from numpy.testing import assert_allclose

import MDMC.MD.engine_facades.lammps_engine as lmp_eng
from MDMC.common import units
from MDMC.MD.constraints import Rattle, Shake
from MDMC.MD.interaction_functions import (
    Buckingham,
    Coulomb,
    HarmonicPotential,
    LennardJones,
    Periodic,
)
from MDMC.MD.interactions import Bond, BondAngle, Coulombic, DihedralAngle, Dispersion
from MDMC.MD.kspace_solvers import PPPM, Ewald, KSpaceSolver
from MDMC.MD.simulation import ConstraintAlgorithm, Simulation, Universe
from MDMC.MD.structures import Atom
from MDMC.trajectory_analysis.compact_trajectory import CompactTrajectory

pytestmark = [pytest.mark.lammps]

CUTOFF = 3.14
COUL_CUTOFF = 8.0
DISP_CUTOFF = 10.0
N_ATOMS = 10
UNIVERSE_DIM = 50.0
CONST = units.CODATA[units.CODATA_VERSION]


@pytest.fixture
def empty_universe():

    """
    Returns:
    A empty Universe object
    """

    return Universe(dimensions=UNIVERSE_DIM, verbose=False)

@pytest.fixture
def atoms():

    """
    Returns:
    A list of atoms with 4 different atom_types

    Ordering of atoms is to enable ease of comparison with atoms added to
    LAMMPS, as this is done ordered by atom_type, rather than necessary the
    order which atoms appear in universe.atoms
    """

    symbols = ['C', 'H', 'N', 'O']
    masses = [12.011, 1.008, 14.007, 16.000]
    elements = symbols * (N_ATOMS // 4)
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
        empty_universe.add_structure(atom)

    # Create InteractionFunctions for bonds, angles, dihedrals and dispersive
    # interactions
    bond1_harmonic = HarmonicPotential(1.0, 2.0, interaction_type='bond')
    bond2_harmonic = HarmonicPotential(2.0, 4.0, interaction_type='bond')
    angle_harmonic = HarmonicPotential(1.0, 0.0005, interaction_type='angle')
    proper_periodic = Periodic(1.0, 1, 90.,
                               2.0, 2, 180.,
                               0.1, 3, -90.,
                               0.5, 4, -45.)
    improper_harmonic = HarmonicPotential(1.0, 0.0002,
                                          interaction_type='improper')

    # Create 2 bonds for some atoms, and one angle, coulombic and dispersive
    # interaction
    bond1_atoms = [(atoms[i], atoms[i+1]) for i in range(0, len(atoms)-1, 2)]
    bond2_atoms = [(atoms[i], atoms[i+2]) for i in range(0, len(atoms)-2, 3)]
    bonds = [Bond(*bond1_atoms, function=bond1_harmonic),
             Bond(*bond2_atoms, function=bond2_harmonic)]

    angles = [BondAngle(*zip(atoms[0::3], atoms[1::3], atoms[2::3]),
                        function=angle_harmonic)]

    propers = [DihedralAngle(tuple(atom for atom in atoms[:4]),
                             function=proper_periodic, improper=False)]
    impropers = [DihedralAngle(tuple(atom for atom in atoms[:4]),
                               function=improper_harmonic, improper=True)]
    coulombics, dispersions = [], []
    for type in empty_universe.atom_types:
        coulombics.append(Coulombic(empty_universe, atom_types=type,
                                    function=Coulomb(-1.0+type*0.5),
                                    cutoff=COUL_CUTOFF))
        dispersions.append(Dispersion(empty_universe, (type, type),
                                      function=Buckingham(type * 0.1,
                                                          type * 1.0,
                                                          type * 2.0),
                                      cutoff=DISP_CUTOFF,
                                      vdw_tail_correction=True))
        dispersions.append(Dispersion(empty_universe, (type, type),
                                      function=LennardJones(type*0.1,
                                                            type*1.0),
                                      cutoff=DISP_CUTOFF,
                                      vdw_tail_correction=True))

    return (empty_universe, bonds, angles, propers, impropers, coulombics,
            dispersions)

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
def propers(universe_interactions):

    """
    Returns:
    A list of proper dihedrals
    """

    return universe_interactions[3]

@pytest.fixture
def impropers(universe_interactions):

    """
    Returns:
    A list of improper dihedrals
    """

    return universe_interactions[4]

@pytest.fixture
def coulombics(universe_interactions):

    """
    Returns:
    A list of coulombic interactions
    """

    return universe_interactions[5]

@pytest.fixture
def dispersions(universe_interactions):

    """
    Returns:
    A list of dispersion interactions
    """

    return universe_interactions[6]

@pytest.fixture
def interactions(bonds, angles, propers, impropers, coulombics, dispersions):

    """
    Returns:
    A list of bond, angle, coulombic and dispersion interactions
    """

    return bonds + angles + propers + impropers + coulombics + dispersions

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
def lammps_universe(universe):

    """
    Returns:
    A LAMMPSUniverse where the atomic configuration and the topology have been
    added
    """

    lammps_universe = lmp_eng.LAMMPSUniverse(universe)
    return lammps_universe

@pytest.fixture
def lammps_simulation(universe):

    """
    Returns:
    A LAMMPSSimulation where the simulation parameters have been set. The
    PyLammps wrapper belonging to this LAMMPSSimulation does not have an atomic
    configuration or topology, and so it not ready to run LAMMPS.
    """

    # Simulation setup requires the traj_step attribute to be set. All other
    # attributes that are required are set to defaults.
    lammps_simulation = lmp_eng.LAMMPSSimulation(universe, traj_step=10)
    return lammps_simulation

@pytest.fixture
def populated_lammps_simulation(universe, lammps_universe):

    """
    Returns:
    A LAMMPSSimulation which has a PyLammps wrapper where the atomic
    configuration and the topology have been added, and the simulation
    parameters have been set. The PyLammps wrapper is ready to run a LAMMPS
    simulation.
    """

    lammps_simulation = lmp_eng.LAMMPSSimulation(universe,
                                                 traj_step=10,
                                                 time_step=1.,
                                                 lmp=lammps_universe.lmp)
    return lammps_simulation

@pytest.fixture
def ensemble(populated_lammps_simulation):

    """
    Returns:
    An Ensemble which has a PyLammps wrapper where the atomic
    configuration and the topology have been added, and the simulation
    parameters have been set. This is required for thermostat and barostats to
    be added to the PyLammps wrapper through the ensemble.
    """

    populated_lammps_simulation.lin_momentum_steps = None
    return lmp_eng.LAMMPSEnsemble(populated_lammps_simulation.lmp,
                                  time_step=1.)

@pytest.fixture
def simulation(universe):
    """
    A mock simulation to give the engine facade its necessary 'parent simulation'
    """
    return Simulation(universe, traj_step=1, time_step=1., engine='lammps')

@pytest.fixture
def lammps_engine(universe, simulation):

    """
    Returns:
    A LAMMPSEngine which is ready to run a LAMMPS simulation with an NVE
    ensemble.
    """

    lammps_engine = lmp_eng.LAMMPSEngine()
    lammps_engine.parent_simulation = simulation
    lammps_engine.setup_universe(universe)
    lammps_engine.setup_simulation()
    return lammps_engine


def test_simulation_setup():

    universe=Universe((10., 10., 10.))
    sim_obj = Simulation(universe,
                         engine="lammps",
                         time_step=10.18893,
                         temperature=300.0,
                         pressure=101325.0,
                         traj_step=15)
    expected_output = (
        'Simulation created with lammps engine and settings:\n'
        'temperature: 300.0 K \n'
        'pressure: 101325.0 Pa \n\n')
    assert expected_output == sim_obj.setup_msg


def test_universe_dimensions(lammps_universe):

    """
    Tests that creating a simulation box from an MDMC universe results in the
    correct universe dimensions

    Lower dimensions should be 0.0
    Upper dimensions should be equal to the MDMC universe dimensions
    """

    assert 0.0 == lammps_universe.system_state.xlo \
               == lammps_universe.system_state.ylo \
               == lammps_universe.system_state.zlo

    assert UNIVERSE_DIM == lammps_universe.system_state.xhi \
                        == lammps_universe.system_state.yhi \
                        == lammps_universe.system_state.zhi


def test_number_atom_types(lammps_universe):

    """
    Tests that creating a simulation box from an MDMC universe results in the
    correct number of atom types
    """

    assert lammps_universe.system_state.ntypes == 4


def test_number_atoms(lammps_universe, atoms):

    """
    Tests that the correct number of atoms has been added to LAMMPS
    """

    assert lammps_universe.system_state.natoms == len(atoms)


def test_number_interaction_types(lammps_universe):

    """
    Tests that creating a simulation box from an MDMC universe results in the
    correct number of each interaction type:

    - bond
    - angle
    - improper

    PyLammps does not allow polling for ndihedraltypes (unlike nbondtypes,
    nimpropertypes, and nangletypes) so there is no test for the number of
    proper dihedral types.
    """

    getter = lammps_universe.lmp.lmp.numpy
    for name, expected in zip(("bonds", "angles", "impropers"),
                              (2, 1, 1)):
        assert (np.max(getattr(getter, f"gather_{name}")()[:, 0]) == expected)


def test_number_interactions(lammps_universe, bonds, angles, propers,
                             impropers):

    """
    Tests that creating a simulation box from an MDMC universe results in the
    correct allowed number of interactions per atom for each interaction type:

    - bond
    - angle
    - dihedral
    - improper

    DIHEDRAL AND IMPROPER ARE NOT CURRENTLY IMPLEMENTED
    """
    getter = lammps_universe.lmp.lmp.numpy
    for var, name in zip((bonds, angles, propers, impropers),
                         ("bonds", "angles", "dihedrals", "impropers")):
        assert (getattr(getter, f"gather_{name}")().shape[0] ==
                sum(len(x.atoms) for x in var))


def test_atom_type_properties(lammps_universe, universe):

    """
    Tests that element and mass are assigned to each list index corresponding to
    atom type equivalent to that index (-1 offset due to atom_type starting from
    1)
    """

    for atom_type, atoms in universe.atom_types.items():
        assert (lammps_universe.atom_type_properties[atom_type - 1]
                == (atoms[0].element, atoms[0].mass))


def test_atom_type_mass(lammps_universe, universe):

    """
    Tests that the mass of each atom type is set correctly in LAMMPS
    """

    for i in range(len(universe.atoms)):
        assert (lammps_universe.lmp.atoms[i].mass
                == universe.atoms[i].mass)


def test_atom_ID(lammps_universe, universe):

    """
    Tests that atoms created in LAMMPS have the correct ID
    """

    # Atom IDs in universe are offset by some integer related to the number of
    # time the atoms fixture is called. If this offset is subtracted, the IDs
    # should agree exactly with the LAMMPS atom IDs
    offset = universe.atoms[0].ID - 1
    for i in range(len(universe.atoms)):
        assert (lammps_universe.lmp.atoms[i].id
                == universe.atoms[i].ID - offset)


def test_atom_type(lammps_universe, universe):

    """
    Tests that atoms created in LAMMPS have the correct atom types
    """

    for i in range(len(universe.atoms)):
        assert (lammps_universe.lmp.atoms[i].type
                == universe.atoms[i].atom_type)


def test_atom_position(lammps_universe, universe):

    """
    Tests that atoms created in LAMMPS have the correct position
    """

    for i in range(len(universe.atoms)):
        assert (np.array(lammps_universe.lmp.atoms[i].position)
                == universe.atoms[i].position).all()


def test_unimplemented_interactions(lammps_universe, universe):

    """
    Tests that if a universe passed to LAMMPSUniverse._add_topology has any
    interactions which have not been implemented in LAMMPS, NotImplementedError
    is raised
    """

    # Add unimplemented interaction type to universe
    # Dummy class which does not require docstring
    #pylint: disable=missing-docstring, multiple-statements
    class Unimplemented(Dispersion): pass
    unimplemented_interaction = Unimplemented(universe, (1, 1))

    # Create LAMMPS topology from universe, raising NotImplementedError
    with pytest.raises(NotImplementedError):
        lammps_universe._add_topology(universe)


@pytest.mark.parametrize('interactions, expected',
                         [('bonds', 'harmonic'),
                          ('angles', 'harmonic'),
                          ('propers', 'fourier'),
                          ('impropers', 'harmonic')])
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
    assert lmp_eng.parse_bonded_styles(interactions[0]) == expected


@pytest.mark.parametrize('inters, index, expected, solver_attr',
                         [('dispersions', 0, ['buck', 10.], None),
                          ('dispersions', 1, ['lj/cut', 10.], None),
                          ('coulombics', 0, ['coul/cut', 8.], None),
                          ('dispersions', 0, ['buck/long', 10.],
                           'kspace_solver'),
                          ('dispersions', 1, ['lj/long', 10.], 'kspace_solver'),
                          ('coulombics', 0, ['coul/long', 8.], 'kspace_solver'),
                          ('dispersions', 0, ['buck/long', 10.],
                           'dispersive_solver'),
                          ('dispersions', 1, ['lj/long', 10.],
                           'dispersive_solver'),
                          ('coulombics', 0, ['coul/cut', 8.],
                           'dispersive_solver'),
                          ('dispersions', 0, ['buck', 10.],
                           'electrostatic_solver'),
                          ('dispersions', 1, ['lj/cut', 10.],
                           'electrostatic_solver'),
                          ('coulombics', 0, ['coul/long', 8.],
                           'electrostatic_solver')])
def test_parse_nonbonded_styles(inters, index, expected, solver_attr,
                                universe, request):

    """
    Tests that the return from parse_nonbonded_styles is the correct input for
    creating a LAMMPS pair style

    The pair style is modified if a solver is provided:
    - kspace_solver modifies lj, buck, and coul
    - dispersive_solver modifies both lj and buck
    - coulombic_solver modifies coul

    The parameters should be modified whenever a new nonbonded style is
    implemented.
    """

    # As fixtures cannot be included in parameterization, the names of the
    # fixtures are included instead - the return values of the fixtures are then
    # recovered using request.getfixturevalue
    inters = request.getfixturevalue(inters)[index]
    # If a solver_attr is specified, add a PPPM solver to this attribute
    if solver_attr:
        setattr(universe, solver_attr, PPPM(accuracy=1e-4))
    assert lmp_eng.parse_nonbonded_styles(inters)[0] == expected


@pytest.mark.parametrize("inters, indices, solver_attr, expected",
                         [(('coulombics', 'dispersions', 'dispersions'),
                           (0, 0, 1),
                           None,
                           [('buck/coul/cut',
                             '{0} {1}'.format(DISP_CUTOFF, COUL_CUTOFF)),
                            ('lj/cut/coul/cut',
                             '{0} {1}'.format(DISP_CUTOFF, COUL_CUTOFF))]),
                          (('coulombics', 'dispersions', 'dispersions'),
                           (0, 0, 1),
                           'electrostatic_solver',
                           [('buck/coul/long',
                             '{0} {1}'.format(DISP_CUTOFF, COUL_CUTOFF)),
                            ('lj/cut/coul/long',
                             '{0} {1}'.format(DISP_CUTOFF, COUL_CUTOFF))])
                         ])
def test_parse_all_nonbonded_styles_valid_diff_cutoffs(inters, indices,
                                                       solver_attr, expected,
                                                       universe, request):

    """
    Tests the generation of valid LAMMPS pair_styles of Dispersive and
    Coulombic interactions for various solver attributes, where the
    Dispersive and Coulombic cutoff distances are different.

    Doesn't test for interactions created in a universe with a
    kspace_solver attribute as this creates an invalid LAMMPS command.

    Doesn't test for interactions created in a universe with a
    dispersive_solver attribute as this creates an invalid pair style.
    """

    assert COUL_CUTOFF != DISP_CUTOFF
    inters = [request.getfixturevalue(inter)[idx]
              for inter, idx in zip(inters, indices)]
    if solver_attr:
        setattr(universe, solver_attr, PPPM(accuracy=1e-4))
    assert list(lmp_eng.parse_all_nonbonded_styles(inters).keys()) == expected


@pytest.mark.parametrize("inters, indices, solver_attr, cutoff, expected",
                         [(('coulombics', 'dispersions', 'dispersions'),
                           (0, 0, 1),
                           None,
                           CUTOFF,
                           [('buck/coul/cut', '{0}'.format(CUTOFF)),
                            ('lj/cut/coul/cut', '{0}'.format(CUTOFF))]),
                          (('coulombics', 'dispersions', 'dispersions'),
                           (0, 0, 1),
                           'kspace_solver',
                           CUTOFF,
                           [('buck/long/coul/long', 'long long',
                             '{0}'.format(CUTOFF)),
                            ('lj/long/coul/long', 'long long',
                             '{0}'.format(CUTOFF))]),
                          (('coulombics', 'dispersions', 'dispersions'),
                           (0, 0, 1),
                           'electrostatic_solver',
                           CUTOFF,
                           [('buck/coul/long', '{0}'.format(CUTOFF)),
                            ('lj/cut/coul/long', '{0}'.format(CUTOFF))])
                         ])
def test_parse_all_nonbonded_styles_valid_same_cutoff(inters, indices,
                                                      solver_attr, cutoff,
                                                      expected, universe,
                                                      request):

    """
    Tests the generation of valid LAMMPS pair_styles of Dispersive and
    Coulombic interactions for various solvent attributes, where the
    Dispersive and Coulombic cutoff distances are the same.

    Doesn't test for interactions created in a universe with a
    dispersive_solver attribute as this creates an invalid pair style.
    """

    inters = [request.getfixturevalue(interaction)[idx]
              for interaction, idx in zip(inters, indices)]
    # Set the cutoff to the same value for all interactions
    for interaction in inters:
        interaction.cutoff = cutoff
    if solver_attr:
        setattr(universe, solver_attr, PPPM(accuracy=1e-4))
    assert list(lmp_eng.parse_all_nonbonded_styles(inters).keys()) == expected


@pytest.mark.parametrize('index', [0, 1])
def test_parse_all_nonbonded_styles_diff_cutoffs_error(dispersions, index,
                                                       coulombics, universe,
                                                       request):

    """
    Tests that a ValueError is raised when trying to create the following
    pair styles when the Dispersive and Coulombic interactions are created
    with different cut offs:

        - buck/long/coul/long
        - lj/long/coul/long
    """

    assert COUL_CUTOFF != DISP_CUTOFF
    interactions = [request.getfixturevalue('dispersions')[index],
                    request.getfixturevalue('coulombics')[0]]
    # Use kspace solver for long range Dispersive and Coulombic interactions
    setattr(universe, 'kspace_solver', PPPM(accuracy=1e-4))
    with pytest.raises(ValueError):
        lmp_eng.parse_all_nonbonded_styles(interactions)


@pytest.mark.parametrize("interactions, indices, solver_attr",
                         [(('coulombics', 'dispersions'),
                           (0, 0), 'dispersive_solver'),
                          (('coulombics', 'dispersions'),
                           (0, 1), 'dispersive_solver')])
def test_parse_all_nonbonded_styles_invalid_styles(interactions, indices,
                                                   solver_attr, universe,
                                                   request):

    """
    Tests that a ValueError is raised when trying to create the following
    invalid LAMMPS pair_styles:

        - buck/long/coul/cut
        - lj/long/coul/cut
    """

    interactions = [request.getfixturevalue(interaction)[idx]
                    for interaction, idx in zip(interactions, indices)]
    setattr(universe, solver_attr, PPPM(accuracy=1e-4))
    with pytest.raises(ValueError):
        lmp_eng.parse_all_nonbonded_styles(interactions)

def test_parse_nonbonded_styles_no_cutoff_error(request):

    """
    Tests that an AttributeError is raised when trying to create LAMMPS pair_styles from
    nonbonded interactions which have no `cutoff` attribute set.
    """

    interactions = [request.getfixturevalue('dispersions')[0],
                    request.getfixturevalue('coulombics')[0]]
    for interaction in interactions:
        interaction.cutoff = None
    with pytest.raises(AttributeError):
        lmp_eng.parse_all_nonbonded_styles(interactions)

@pytest.mark.parametrize('interaction, arguments, parser',
                         [(Bond, ['atom_pair'], 'parse_bonded_styles'),
                          (Dispersion, ['universe', (1, 1)],
                           'parse_nonbonded_styles')
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
    # atom_type which is equal to 1)
    for index, arg in enumerate(arguments):
        if isinstance(arg, str):
            arguments[index] = request.getfixturevalue(arg)

    # Add interaction without defining InteractionFunction
    undefined_interaction_function = interaction(*arguments)

    with pytest.raises(NotImplementedError):
        # Pass undefined_interaction_function as an argument to parser
        getattr(lmp_eng, parser)(undefined_interaction_function)



@pytest.mark.parametrize('inter_type, fun_type, parameters, settings, expected',
                         [('Bond',
                           'HarmonicPotential',
                           (5., 2.5),
                           {'interaction_type':'bond'},
                           ['harmonic', 0.5975143403441683, 5.]),
                          ('BondAngle',
                           'HarmonicPotential',
                           (90., 1.),
                           {'interaction_type':'angle'},
                           ['harmonic', 0.2390057361376673, 90.]),
                          ('DihedralAngle',
                           'Periodic',
                           (1., 2, 30.),
                           {},
                           ['fourier', 1, 0.2390057361376673, 2, 30.]),
                          ('DihedralAngle',
                           'Periodic',
                           (4.184, 2, 30., 8.368, 8, -45.),
                           {},
                           ['fourier', 2, 1., 2, 30., 2., 8, -45.]),
                          ('DihedralAngle',
                           'HarmonicPotential',
                           (110., 15.),
                           {'improper':True, 'interaction_type':'improper'},
                           ['harmonic', 3.585086042065009, 110.]),
                          ('DihedralAngle',
                           'Periodic',
                           (5.5, 3, 0.),
                           {'improper':True},
                           ['cvff', 1.31453154875717, 1, 3]),
                          ('DihedralAngle',
                           'Periodic',
                           (2.5, 4, 180.),
                           {'improper':True},
                           ['cvff', 0.5975143403441683, -1, 4])])
def test_parse_bonded_coefficients(inter_type, fun_type, parameters, settings,
                                   expected):

    """
    Tests that parsing the bonded coefficients produces the expected input for
    the LAMMPS coeff commands

    Creates an Interaction and InteractionFunction of the types specified. The
    parameters for the InteractionFunction are specified by 'parameters' and
    all required keywords for both the Interaction and InteractionFunction are
    in 'settings'.

    The differences between the values specified in 'parameters' and those in
    'expected' are due to unit conversion which occurs in bond coefficient
    parsing. The differences between the order is because LAMMPS requires some
    Parameters to be ordered differently to MDMC.

    Note that the first numerical coefficient of parsed Periodic interactions
    is the order of the Periodic interaction.

    The following BondedInteractions are tested:
    - Bond with HarmonicPotential
    - BondAngle with HarmonicPotential
    - Proper DihedralAngle with Periodic (first order)
    - Proper DihedralAngle with Periodic (second order)
    - Improper DihedralAngle with HarmonicPotential
    - Improper DihedralAngle with Periodic (d = 0)
    - Improper DihedralAngle with Periodic (d = 180)
    """

    # Create InteractionFunction and Interaction classes from classes that have
    # been imported (and so are in the global namespace)
    # Pass the settings dict to both of these - this is valid as long as the
    # InteractionFunction and Interaction do not have any of the same keywords
    # try/except accounts for InteractionFunctions which do not accept keywords
    try:
        function = globals()[fun_type](*parameters, **settings)
    except TypeError:
        function = globals()[fun_type](*parameters)
    interaction = globals()[inter_type](function=function, **settings)
    assert lmp_eng.parse_bonded_coefficients(interaction) == expected


@pytest.mark.parametrize('system_attr, expected',
                         [('bond_style', 'hybrid'),
                          ('angle_style', 'hybrid'),
                          ('pair_style', 'hybrid/overlay')])
def test_create_interaction_style(lammps_universe, system_attr,
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
    assert getattr(lammps_universe.system_state, system_attr) == expected


def test_atom_charge_set(lammps_universe, universe):

    """
    Tests that atom charges are set correctly
    """

    for i in range(len(universe.atoms)):
        assert (lammps_universe.lmp.atoms[i].charge
                == universe.atoms[i].charge)


def test_atom_charges_update(lammps_universe, universe):

    """
    Tests that atom charges are updated correctly

    Change the charges on the atoms in the universe and test if LAMMPS charges
    update after LAMMPUniverse._update_charges is called
    """

    # Change charges and update LAMMPSEngine
    for atom in universe.atoms:
        atom.charge *= 2.
    lammps_universe._update_charges()

    for i in range(len(universe.atoms)):
        assert (lammps_universe.lmp.atoms[i].charge
                == universe.atoms[i].charge)


@pytest.mark.parametrize('interaction_fixture, lmp_name',
                         [('bonds', 'bond'),
                          ('angles', 'angle'),
                          ('propers', 'dihedral'),
                          ('impropers', 'improper'),
                          ('dispersions', None)])
def test_update_individual_interactions(lammps_universe, interaction_fixture,
                                        lmp_name, request):

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
        for parameter in interaction.parameters:
            interaction.parameters[parameter].value *= 2

    if interaction_fixture == 'dispersions':
        lammps_universe._update_dispersions(lammps_universe.universe)
    else:
        lammps_universe._update_bonded_interactions(lmp_name, interactions)


def test_update_all_interactions(lammps_universe, interactions):

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
        for parameter in interaction.parameters:
            interaction.parameters[parameter].value *= 2

    lammps_universe.update_parameters()


def test_update_charges_error():

    """
    Tests that an error is raised when trying to create a LAMMPS universe
    from a universe that contains atoms with a charge of None.
    """

    universe = Universe(10., verbose=False)
    universe.add_structure(Atom('H'))
    with pytest.raises(AttributeError):
        lmp_eng.LAMMPSUniverse(universe)


@pytest.mark.parametrize('mix', ['GEOMETRIC',
                                 'geometric',
                                 'arithmetic',
                                 'SIXTHPOWER'])
def test_mixing(mix, universe):

    """
    Tests that applying different nonbonded interaction mixing styles does not
    result in a fatal error, where the LAMMPS Python interface causes Python to
    exit without throwing an error, presumably due to a segfault

    A more stringent test would check the that values of pair_modify have been
    set in LAMMPS, however there is no way to check this through the Python
    interface. Therefore the minimum test of checking for a fatal error is used.
    """

    lammps_universe = lmp_eng.LAMMPSUniverse(universe, nonbonded_mix=mix)


@pytest.mark.parametrize('mix', ['geometrix',
                                 'equal'])
def test_mixing_unimplemented(lammps_universe, mix):

    """
    Tests that applying different nonbonded interaction mixing styles does not
    result in a fatal error, where the LAMMPS Python interface causes Python to
    exit without throwing an error, presumably due to a segfault

    A more stringent test would check the that values of pair_modify have been
    set in LAMMPS, however there is no way to check this through the Python
    interface. Therefore the minimum test of checking for a fatal error is used.
    """

    with pytest.raises(ValueError):
        lammps_universe.nonbonded_mix = mix


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
    assert lmp_eng.parse_kspace_solver(solver) == expected


def test_parse_kspace_solver_unimplemented():

    """
    Tests that parsing an unimplemented kspace solver raises a
    NotImplementedError
    """

    solver = KSpaceSolver(accuracy=0.0001)
    with pytest.raises(NotImplementedError):
        unimplemented_solver = lmp_eng.parse_kspace_solver(solver)


@pytest.mark.parametrize('solver_cls, style, omp_style',
                         [(PPPM, 'pppm', 'pppm/omp'),
                          (Ewald, 'ewald', 'ewald/omp')])
def test_set_kspace_solver_styles(populated_lammps_simulation, universe,
                                  dispersions, solver_cls, style, omp_style):

    """
    Tests setting the kspace solver if the Universe has a kspace_solver
    """

    # Create a kspace solver and add it to the universe of a LAMMPSEngine which
    # has not had the topology created. Then create topology to set kspace style
    # in LAMMPS.
    solver = solver_cls(accuracy=0.0001)
    populated_lammps_simulation.universe.kspace_solver = solver
    # LAMMPS requires a single cutoff for LJ and coulombic long range
    # interactions (i.e. kspace calculations), so change the cutoff for the
    # Dispersion interactions
    populated_lammps_simulation._set_kspace_solver()
    assert populated_lammps_simulation.system_state.kspace_style == style or \
           populated_lammps_simulation.system_state.kspace_style == omp_style


@pytest.mark.parametrize('solver_cls', [PPPM, Ewald])
def test_set_different_cutoffs(lammps_universe, universe, dispersions,
                               solver_cls):

    """
    Tests that if cutoffs for dispersion and coulombic interaction are different
    it results in a ValueError
    """

    # Create a kspace solver and add it to an MDMC universe. Pass this universe
    # to a LAMMPSUniverse._add_topology to set this kspace style in LAMMPS.
    solver = solver_cls(accuracy=0.0001)
    universe.kspace_solver = solver
    # Set cutoffs for dispersion interactions to be different to cutoffs for
    # coulombic interactions
    for dispersion in dispersions:
        dispersion.cutoff = COUL_CUTOFF + 2.0
    with pytest.raises(ValueError):
        lammps_universe._add_topology(lammps_universe.universe)


@pytest.mark.parametrize('solver_attr, expected, omp_expected',
                         [('kspace_solver', 'pppm', 'pppm/omp'),
                          ('electrostatic_solver', 'pppm', 'pppm/omp'),
                          ('dispersive_solver', TypeError, TypeError)])
def test_set_kspace_solver_single_solver_error(populated_lammps_simulation,
                                               solver_attr, expected, omp_expected):

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
    setattr(populated_lammps_simulation.universe, solver_attr, solver)
    if expected is TypeError:
        with pytest.raises(expected):
            populated_lammps_simulation._set_kspace_solver()
    else:
        populated_lammps_simulation._set_kspace_solver()
        assert populated_lammps_simulation.system_state.kspace_style == expected or \
               populated_lammps_simulation.system_state.kspace_style == omp_expected


def test_set_kspace_solver_multiple_solvers(populated_lammps_simulation):

    """
    Tests setting the kspace solver if the Universe has both an
    electrostatic_solver and a dispersion_solver and they are equal
    """

    # Create a kspace solver and add it to the universe as both an
    # electrostatic_solver and a dispersive_solver. Then call set_kspace_solver
    # to apply kspace style in LAMMPS.
    solver = PPPM(accuracy=0.0001)
    populated_lammps_simulation.universe.electrostatic_solver = solver
    populated_lammps_simulation.universe.dispersive_solver = solver
    populated_lammps_simulation._set_kspace_solver()
    assert populated_lammps_simulation.system_state.kspace_style == 'pppm' or \
           populated_lammps_simulation.system_state.kspace_style == 'pppm/omp'


def test_set_kspace_solver_multiple_solvers_error(populated_lammps_simulation):

    """
    Tests setting the kspace solver if the Universe has both an
    electrostatic_solver and a dispersion_solver and they are not equal
    """

    # Create different kspace solvers for universe's electrostatic_solver and
    # dispersive_solvers. Then call set_kspace_solver to apply kspace style in
    # LAMMPS.
    universe = populated_lammps_simulation.universe
    universe.electrostatic_solver = PPPM(accuracy=0.0001)
    universe.dispersive_solver = PPPM(accuracy=0.0005)
    with pytest.raises(TypeError):
        populated_lammps_simulation._set_kspace_solver()


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
    assert name == lmp_eng.parse_constraint(constraint_algorithm,
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
        invalid_constraint = lmp_eng.parse_constraint(constraint_algorithm,
                                                      bonds=constrained_bonds,
                                                      bond_ID_dict=bond_ID_dict)


@pytest.mark.parametrize('accuracy', [1.0, 1e-4, 5])
def test_parse_constraint_accuracy(accuracy, constrained_bonds, bond_ID_dict):
    # ID is an acronym
    #pylint: disable=invalid-name

    """
    Tests that accuracy is correct in the input to LAMMPS fix

    Excluding the fix ID and and group-ID, the accuracy is the index 1
    entry passed to a LAMMPS fix. The accuracy must be a float.
    """

    constraint_algorithm = Shake(accuracy=accuracy, max_iterations=1)
    algorithm_accuracy = lmp_eng.parse_constraint(constraint_algorithm,
                                                  bonds=constrained_bonds,
                                                  bond_ID_dict=bond_ID_dict)[1]
    assert float(accuracy) == algorithm_accuracy


@pytest.mark.parametrize('max_iter', [1, 5.4])
def test_parse_constraint_max_iterations(max_iter, constrained_bonds,
                                         bond_ID_dict):
    # ID is an acronym
    #pylint: disable=invalid-name

    """
    Tests that the max number of iterations is correct in the input to LAMMPS
    fix

    Excluding the fix ID and and group-ID, the number of max iterations is the
    index 2 entry passed to a LAMMPS fix. The number of max iterations must be
    an integer.
    """

    constraint_algorithm = Shake(accuracy=1.0, max_iterations=max_iter)
    algorithm_max_iter = lmp_eng.parse_constraint(constraint_algorithm,
                                                  bonds=constrained_bonds,
                                                  bond_ID_dict=bond_ID_dict)[2]
    assert int(max_iter) == algorithm_max_iter


def test_parse_constraint_bonds(constrained_bonds, bond_ID_dict):
    # ID is an acronym
    #pylint: disable=invalid-name

    """
    Tests that the input to LAMMPS has the correct bond IDs

    Excluding the fix ID and and group-ID, the declaration of bond constraints
    (indicated by 'b') is the index 4 entry passed to a LAMMPS fix. Following
    this the IDs of all of the constrained bonds must be listed.
    """

    constraint_algorithm = Shake(accuracy=1.0, max_iterations=1)
    lmp_input = lmp_eng.parse_constraint(constraint_algorithm,
                                         bonds=constrained_bonds,
                                         bond_ID_dict=bond_ID_dict)
    assert lmp_input[4] == 'b'
    assert sorted(lmp_input[5:]) == sorted([bond_ID_dict[bond] for bond
                                            in constrained_bonds])


def test_parse_constraint_angles(constrained_angles, angle_ID_dict):
    # ID is an acronym
    #pylint: disable=invalid-name

    """
    Tests that the input to LAMMPS has the correct angle IDs

    Excluding the fix ID and and group-ID, the declaration of angle constraints
    (indicated by 'a') is the index 4 entry passed to a LAMMPS fix, if no bonds
    are included. Following this the IDs of all of the constrained angles must
    be listed.
    """

    constraint_algorithm = Shake(accuracy=1.0, max_iterations=1)
    lmp_input = lmp_eng.parse_constraint(constraint_algorithm,
                                         angles=constrained_angles,
                                         angle_ID_dict=angle_ID_dict)
    assert lmp_input[4] == 'a'
    assert sorted(lmp_input[5:]) == sorted([angle_ID_dict[angle] for angle
                                            in constrained_angles])



def test_parse_constraint_bonds_angles(constrained_bonds, constrained_angles,
                                       bond_ID_dict, angle_ID_dict):
    # ID is an acronym
    #pylint: disable=invalid-name

    """
    Tests that the input to LAMMPS has the correct bond IDs and angle IDs

    Excluding the fix ID and and group-ID, the declaration of bond constraints
    (indicated by 'b') is the index 4 entry passed to a LAMMPS fix. Following
    this the IDs of all of the constrained bonds must be listed. The index
    after this must be the declaration of angle constraints (indicated by 'a'),
    and then the IDs of all of the constrained angles must be listed.
    """

    constraint_algorithm = Shake(accuracy=1.0, max_iterations=1)
    lmp_input = lmp_eng.parse_constraint(constraint_algorithm,
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
    # ID is an acronym
    #pylint: disable=invalid-name

    """
    Tests that if neither bonds or angles are provided when parsing the
    constraint, a TypeError is raised
    """

    constraint_algorithm = Shake(accuracy=1.0, max_iterations=1)
    with pytest.raises(TypeError):
        lmp_input = lmp_eng.parse_constraint(constraint_algorithm,
                                             bond_ID_dict=bond_ID_dict)


@pytest.mark.parametrize('arguments', [{'bonds':'constrained_bonds'},
                                       {'bonds':'constrained_bonds',
                                        'angle_ID_dict':'angle_ID_dict'},
                                       {'angles':'constrained_angles'},
                                       {'angles':'constrained_angles',
                                        'bond_ID_dict':'bond_ID_dict'}])
def test_parse_constraint_no_IDs(arguments, request):
    # ID is an acronym
    #pylint: disable=invalid-name

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
        lmp_input = lmp_eng.parse_constraint(constraint_algorithm,
                                             **arg_fixtures)


@pytest.mark.parametrize('temperature', [300., 450.])
def test_initialize_velocities(universe, lammps_universe, temperature):

    """
    Test that the LAMMPS velocities have been set correctly when MDMC velocities are zero

    Initialize the velocities by setting the temperature. Set the ensemble to
    NVE and run for 0 steps. Test if the 0 step temperature is as expected.
    """

    lammps_simulation = lmp_eng.LAMMPSSimulation(universe,
                                                 temperature=temperature,
                                                 traj_step=10,
                                                 lmp=lammps_universe.lmp)

    for i, atom in enumerate(universe.atoms):
        # MDMC atoms should be unchanged, but the LAMMPS atoms should have velocities
        assert np.all(np.array(atom.velocity) == 0)
        assert np.all(np.array(lammps_simulation.lmp.atoms[i].velocity) != 0)

    lammps_simulation.lmp.run(0)
    assert_allclose(lammps_simulation.lmp.runs[0][0].Temp[0], temperature)


@pytest.mark.parametrize('temperature', [150., 300.])
def test_initialize_nonzero_velocities(universe, temperature):

    """
    Test that the LAMMPS velocities have been set correctly when MDMC velocities are non-zero

    Initialize the velocities by setting the temperature. Set the ensemble to
    NVE and run for 0 steps. Test if the 0 step temperature is as expected.
    """

    # Set the MDMC velocities
    velocity = []
    for i, atom in enumerate(universe.atoms):
        velocity.append(np.array((-(i + 1), 0, i + 1)))
        atom.velocity = velocity[i]

    # Create new LAMMPS universe/simulation with these velocities
    lammps_universe = lmp_eng.LAMMPSUniverse(universe)
    lammps_simulation = lmp_eng.LAMMPSSimulation(universe,
                                                 temperature=temperature,
                                                 traj_step=10,
                                                 lmp=lammps_universe.lmp)

    # LAMMPS should scale all velocities by the same amount to ensure the temperature is accurate.
    # Get this factor from the first atom, as it had an initial velocity of 1 in the z direction.
    scale_factor = lammps_simulation.lmp.atoms[0].velocity[2]
    for i, atom in enumerate(universe.atoms):
        assert np.all(np.array(atom.velocity) == velocity[i])
        assert np.all(np.array(lammps_simulation.lmp.atoms[i].velocity)
                      == scale_factor * velocity[i])

    lammps_simulation.lmp.run(0)
    assert_allclose(lammps_simulation.lmp.runs[0][0].Temp[0], temperature)


@pytest.mark.parametrize('skin, neighbor_steps', [(1, 2),
                                                  (1., 2.),
                                                  (3., 100)])
def test_set_neighbor_list_parameters(lammps_universe, skin, neighbor_steps):

    """
    Tests that setting neighbor list parameters does not result in a fatal
    error, where the LAMMPS Python interface causes Python to exit without
    throwing an error, presumably due to a segfault

    A more stringent test would check that the neighbor list parameters have
    been set in LAMMPS, however there is no way to check this through the Python
    interface. Therefore the minimum test of checking for a fatal error is used.
    """

    lammps_universe.skin = skin
    lammps_universe.neighbor_steps = neighbor_steps


@pytest.mark.parametrize('momentum_steps, expected_names',
                         [({'lin_momentum_steps':5},
                           ['RemoveLinearMomentum']),
                          ({'ang_momentum_steps':10},
                           ['RemoveAngularMomentum']),
                          ({'lin_momentum_steps':20, 'ang_momentum_steps':20},
                           ['RemoveMomentum']),
                          ({'lin_momentum_steps':15, 'ang_momentum_steps':20},
                           ['RemoveLinearMomentum', 'RemoveAngularMomentum'])])
def test_remove_momentum(populated_lammps_simulation, momentum_steps,
                         expected_names):

    """
    Tests that linear and/or angular momentum remover fixes are correctly
    created
    """

    # Set momentum_step attributes and apply fixes - ensure momentum_step
    # attributes are both initially None.
    populated_lammps_simulation.lin_momentum_steps = None
    populated_lammps_simulation.ang_momentum_steps = None
    for attr, steps in momentum_steps.items():
        setattr(populated_lammps_simulation, attr, steps)

    # The fix styles of all momentum removers should be 'momentum'. There
    # should be one fix with this fix style.
    assert (Counter(populated_lammps_simulation.fix_styles)['momentum']
            == len(expected_names))

    # The name of the fix is defined by whether linear and/or angular
    # momentum is removed
    for name in expected_names:
        assert name in populated_lammps_simulation.fix_names


@pytest.mark.parametrize('thermostat, styles, omp_styles, attributes',
                         [(None, ['nve'], ['OMP', 'nve/omp'], {}),
                          ('nose', ['nvt'], ['OMP', 'nvt/omp'],
                           {'temperature':400., 't_damp':100}),
                          ('berendsen', ['nve', 'temp/berendsen'],
                           ['OMP', 'nve/omp', 'temp/berendsen'],
                           {'temperature':400., 't_damp':100}),
                          ('langevin', ['nve', 'langevin'], ['OMP', 'nve/omp', 'langevin'],
                           {'temperature':400., 't_damp':100}),
                          ('rescale', ['nve', 'temp/rescale'], ['OMP', 'nve/omp', 'temp/rescale'],
                           {'temperature':100., 't_fraction':0.5,
                            't_window':10., 'rescale_step':100}),
                          ('csvr', ['nve', 'temp/csvr'], ['OMP', 'nve/omp', 'temp/csvr'],
                           {'temperature': 400., 't_damp': 100})
                          ])
def test_apply_thermostat(ensemble, thermostat, styles, omp_styles, attributes):

    """
    Tests that applying a thermostat results in the correct fix being applying
    to LAMMPS
    """

    # Set the attributes required for the specified thermostat
    for attr, value in attributes.items():
        setattr(ensemble, attr, value)

    # Add the thermostat
    ensemble.thermostat = thermostat

    # Test that the fix styles returned from the LAMMPS wrapper fixes attribute
    # are correct
    assert ensemble.fix_styles == styles or ensemble.fix_styles == omp_styles


@pytest.mark.parametrize('barostat, styles, omp_styles',
                         [(None, ['nve'], ['OMP', 'nve/omp']),
                          ('berendsen', ['press/berendsen'], ['OMP', 'press/berendsen']),
                          ('nose', ['nph'], ['OMP', 'nph/omp'])])
def test_apply_barostat(ensemble, barostat, styles, omp_styles):

    """
    Tests that applying a barostat results in the correct fix being applied to
    LAMMPS
    """

    # Set the attributes required for all barostats and add the barostat
    ensemble.pressure = 10.
    ensemble.p_damp = 1000
    ensemble.barostat = barostat

    # Test that the fix styles returned from the LAMMPS wrapper fixes attribute
    # are correct
    assert styles == ensemble.fix_styles or omp_styles == ensemble.fix_styles


@pytest.mark.parametrize('thermostat, barostat, styles, omp_styles, attributes',
                         [(None, None, ['nve'], ['OMP', 'nve/omp'], {}),
                          ('nose', 'nose', ['npt'], ['OMP', 'npt/omp'],
                           {'temperature':400., 't_damp':100, 'pressure':10.,
                            'p_damp':1000}),
                          ('berendsen', 'nose', ['temp/berendsen', 'nph'],
                           ['OMP', 'temp/berendsen', 'nph/omp'],
                           {'temperature':400., 't_damp':100, 'pressure':10.,
                            'p_damp':1000}),
                          ('langevin', 'nose', ['langevin', 'nph'], ['OMP', 'langevin', 'nph/omp'],
                           {'temperature':400., 't_damp':100, 'pressure':10.,
                            'p_damp':1000}),
                          ('rescale', 'nose', ['temp/rescale', 'nph'],
                           ['OMP', 'temp/rescale', 'nph/omp'],
                           {'temperature':400., 't_fraction':.5, 't_window':10.,
                            'rescale_step':100, 'pressure':10., 'p_damp':1000}),
                          ('nose', 'berendsen', ['nvt', 'press/berendsen'],
                           ['OMP', 'nvt/omp', 'press/berendsen'],
                           {'temperature':400., 't_damp':100, 'pressure':10.,
                            'p_damp':1000}),
                          ('berendsen', 'berendsen', ['nve', 'temp/berendsen',
                                                      'press/berendsen'],
                           ['OMP', 'nve/omp', 'temp/berendsen', 'press/berendsen'],
                           {'temperature':400., 't_damp':100, 'pressure':10.,
                            'p_damp':1000}),
                          ('langevin', 'berendsen', ['nve', 'langevin',
                                                     'press/berendsen'],
                           ['OMP', 'nve/omp', 'langevin', 'press/berendsen'],
                           {'temperature':400., 't_damp':100, 'pressure':10.,
                            'p_damp':1000}),
                          ('rescale', 'berendsen', ['nve', 'temp/rescale',
                                                    'press/berendsen'],
                           ['OMP', 'nve/omp', 'temp/rescale', 'press/berendsen'],
                           {'temperature':400., 't_fraction':.5, 't_window':10.,
                            'rescale_step':100, 'pressure':10., 'p_damp':1000})]
                        )
def test_apply_thermostat_barostat(ensemble, thermostat, barostat,
                                   styles, omp_styles, attributes):

    """
    Tests that applying both a thermostat and a barostat results in the correct
    fixes being applied to LAMMPS
    """

    # Set the attributes required by each thermostat/barostat pair
    for attr, value in attributes.items():
        setattr(ensemble, attr, value)

    # Add the thermostat and barostat
    ensemble.thermostat = thermostat
    ensemble.barostat = barostat

    # Test that the fix styles returned from the LAMMPS wrapper fixes attribute
    # are correct
    assert styles == ensemble.fix_styles or omp_styles == ensemble.fix_styles


@pytest.mark.parametrize('n_steps', [1, 10])
def test_trajectory_output(lammps_engine, n_steps):

    """
    Tests if a trajectory file of the correct length has been created by LAMMPS
    wrapper
    """

    # lammps_engine_simulation is setup to output trajectory every step. Run for
    # a total of n_steps
    lammps_engine.run(n_steps)

    n_atoms = lammps_engine.system_state.natoms
    n_lines = (n_atoms + 9) * ((n_steps / lammps_engine.traj_step) + 1)
    assert len(lammps_engine.trajectory_file.readlines()) == n_lines


def test_save_config(lammps_engine, universe):

    """
    Tests that the LAMMPS configuration is correctly saved, by checking the
    positions, mass and charge of the LAMMPS wrapper atoms attribute
    """

    lammps_engine.save_config()
    # Positions should be the same as those of the MDMC universe atoms, which
    # are also ordered by ID
    for i in range(len(universe.atoms)):
        assert (np.array(lammps_engine.saved_config[i][:3])
                == universe.atoms[i].position).all()


def test_reset_config(lammps_engine):

    """
    Tests that the reset_config method correctly changes the positions of the
    LAMMPS wrapper atoms back to the saved values

    To do this the config is saved, a short simulation is run, and the config
    is reset
    """

    lammps_engine.save_config()
    lammps_engine.lmp.run(10)

    n_atoms = lammps_engine.system_state.natoms
    # Ensure that the atoms have moved from their starting positions - see atoms
    # fixture for what the starting positions are
    for i in range(n_atoms):
        assert (np.array(lammps_engine.lmp.atoms[i].position)
                != np.array([0.5 * i]*3)).all()

    lammps_engine.reset_config()
    for i in range(n_atoms):
        assert (np.array(lammps_engine.lmp.atoms[i].position)
                == np.array([0.5 * i]*3)).all()


def test_convert_trajectory_output(lammps_engine):

    """
    Tests that converting a trajectory results in an MDMC CompactTrajectory object

    This does not test the correctness of the converted trajectory, purely that
    a trajectory can be converted with the correct type. The correctness of
    the trajectory conversion is covered by a system test.
    """

    lammps_engine.run(3)
    assert isinstance(lammps_engine.convert_trajectory(), CompactTrajectory)


@pytest.mark.parametrize('args',
                         [{'n_steps':0, 'minimize_every':5,
                           'maxiter':1000},
                          {'n_steps':0, 'minimize_every':5,
                           'etol':0., 'ftol':1.e-8,
                           'maxeval':1000, 'maxiter': 1000},
                          {'n_steps':0, 'minimize_every':5,
                           'ftol':1.e-8, 'maxeval':500,
                           'maxiter':5000}])
def test_minimize(args, lammps_engine):

    """
    Tests that the potential energy has been minimized

    This does not test that the minimization reduces the potential energy into a
    local minima, just that the potential energy of the system reduces

    Parameterization tests for both default and non-default minimization
    arguments
    """

    # LAMMPS needs to run for 0 steps to calculate energies - run directly using
    # LAMMPS wrapper run so that any bugs in LAMMPSEngine.run do not affect test
    lammps_engine.lmp.run(0)
    start_energy = lammps_engine.lmp.eval('pe')
    lammps_engine.minimize(**args)
    assert lammps_engine.lmp.eval('pe') < start_energy


@pytest.mark.parametrize('thermostat, barostat, add_args',
                         [(None, None, {}),
                          ('nose', None, {}),
                          ('nose', 'nose', {'pressure':1.0})])
def test_setup_simulation_run(lammps_engine, thermostat, barostat,
                              add_args):

    """
    Tests that the simulation setup can run an NVE, NVT and NPT simulation with
    the default attribute values
    """

    # Simulation setup requires the traj_step attribute to be set, even though
    # it is not being used in this test
    # add_args is a dictionary of additional arguments that are required for the
    # specific ensemble
    lammps_engine.setup_simulation(temperature=300., thermostat=thermostat,
                                   barostat=barostat, **add_args)

    n_steps = 20
    lammps_engine.lmp.run(n_steps)

    # Test that the largest step number in the LAMMPS wrapper runs attribute
    # (which records ThermoData from the previous run) is correct
    assert max(lammps_engine.lmp.runs[0][0].Step) == n_steps


@pytest.mark.parametrize("value", [1., 5, -100, -13.])
def test_convert_unit_no_unit(value):

    """
    Tests that converting a value without a unit just returns the value
    """

    assert value == lmp_eng.convert_unit(value)


@pytest.mark.parametrize("unit_str, expected",
                         [('m', 1e10), ('nm', 10.), ('Ang', 1.),
                          ('ns', 1e6), ('ps', 1e3), ('fs', 1.),
                          ('kg', 1 / CONST['_amu']), ('g', 1 / (CONST['_amu']
                                                                * 1000)),
                          ('amu', 1.), ('g / mol', 1.),
                          ('J', CONST['_Nav'] / 1000.), ('kJ', CONST['_Nav']),
                          ('kcal', CONST['_Nav'] * 4.184),
                          ('kcal / Ang mol', 4.184),
                          ('atm', 101325), ('bar', 1e5),
                          ('rad', 180 / np.pi), ('deg', 1.)])
def test_convert_unit_conversion_factors(unit_str, expected):

    """
    Tests for correct conversion factors for conversion into MDMC units.
    """

    assert np.isclose(lmp_eng.convert_unit(1.0, units.Unit(unit_str),
                                           to_lammps=False),
                      expected)

@pytest.mark.parametrize('value', [1.0, 2.0])
def test_convert_mdmc_base_units_identity(value):

    """
    Tests converting MDMC base units to LAMMPS base units, where the units are
    the same
    """

    for unit in units.SYSTEM.values():
        if unit.components['numerator'][0] == unit \
            and unit in lmp_eng.SYSTEM.values():
            assert lmp_eng.convert_unit(value, unit) == value


@pytest.mark.parametrize('value', [1.0, 2.0])
def test_convert_lammps_base_units_identity(value):

    """
    Tests converting LAMMPS base units to MDMC base units, where the units are
    the same

    The same units are converted as in test_convert_mdmc_base_units_identity,
    except they are being converted from LAMMPS to MDMC
    """

    for unit in lmp_eng.SYSTEM.values():
        if unit.components['numerator'][0] == unit \
            and unit in units.SYSTEM.values():
            assert lmp_eng.convert_unit(value, unit, to_lammps=False) == value


@pytest.mark.parametrize('mdmc_unit, lmp_value',
                         [(units.Unit('Pa'), 1 / 101325.),
                          (units.Unit('kJ / mol'), 1 / 4.184),
                          (units.Unit('kJ / Ang mol'), 1 / 4.184),
                          (units.Unit('amu'), 1.)])
def test_convert_mdmc_base_units(mdmc_unit, lmp_value):

    """
    Tests converting MDMC base units to LAMMPS base units, where the units are
    not the same in the two systems
    """

    assert np.isclose(lmp_eng.convert_unit(1., mdmc_unit), lmp_value)


@pytest.mark.parametrize('lmp_unit, mdmc_value',
                         [(units.Unit('atm'), 101325.),
                          (units.Unit('kcal / mol'), 4.184)])
def test_convert_lammps_base_units(lmp_unit, mdmc_value):

    """
    Tests converting LAMMPS base units to MDMC base units, where the units are
    not the same in the two systems
    """

    assert np.isclose(lmp_eng.convert_unit(1., lmp_unit, to_lammps=False),
                      mdmc_value)


@pytest.mark.parametrize('mdmc_unit, lmp_value',
                         [(units.Unit('kJ') / units.Unit('mol'),
                           4.184 ** -1),
                          (units.Unit('Pa') * units.Unit('fs'), 101325. ** -1),
                          (units.Unit('amu') ** 2, 1.), # mass units equiv
                          (units.Unit('amu') ** -1, 1.), # mass units equiv,
                          (units.SYSTEM['FORCE'],
                           4.184 ** -1)])
def test_convert_mdmc_compound_units(mdmc_unit, lmp_value):

    """
    Tests converting between MDMC compound units (units made up of multiple base
    units)
    """

    assert np.isclose(lmp_eng.convert_unit(1., mdmc_unit), lmp_value)


@pytest.mark.parametrize("unit_str, conversion_factor",
                         [('rad', 1.), ('deg', 180 / np.pi)])
def test_convert_mdmc_angular_potential_strength(unit_str, conversion_factor):

    """
    Tests converting into LAMMPS angular potential strength units for harmonic
    bond angles (which uses radians as the unit of angle rather than degrees)
    for MDMC units of both radians and degrees
    """

    mdmc_unit = units.SYSTEM['ENERGY'] / units.Unit(unit_str) ** 2
    lmp_value = (conversion_factor) ** 2 / 4.184
    assert np.isclose(lmp_eng.convert_unit(1., mdmc_unit), lmp_value)

@pytest.mark.parametrize('lmp_unit, mdmc_value',
                         [(units.Unit('kcal') / units.Unit('mol'),
                           4.184),
                          (units.Unit('atm') * units.Unit('fs'), 101325.),
                          (units.Unit('bar') * units.Unit('fs'), 1e5),
                          (lmp_eng.SYSTEM['MASS'] ** 2, 1.), # mass units equiv
                          (lmp_eng.SYSTEM['MASS'] ** -1, 1.), # mass units equiv
                          (lmp_eng.SYSTEM['ENERGY'], 4.184),
                          (lmp_eng.SYSTEM['FORCE'], 4.184)])
def test_convert_lammps_compound_units(lmp_unit, mdmc_value):

    """
    Tests converting between MDMC compound units (units made up of multiple base
    units)
    """

    assert np.isclose(lmp_eng.convert_unit(1., lmp_unit, to_lammps=False),
                      mdmc_value)


def test_convert_mdmc_compound_equivalence():

    """
    Tests that converting an MDMC compound unit produces the same answer as
    performing the conversions individually
    """

    P = units.SYSTEM['PRESSURE']
    E = units.SYSTEM['ENERGY']

    assert np.isclose(lmp_eng.convert_unit(1., P / E),
                      lmp_eng.convert_unit(1., P) / lmp_eng.convert_unit(1., E))


@pytest.mark.parametrize("unit, mag, power, to_lammps",
                         [('g / mol', 0, 1, False), ('mol / g', 0, 1, False),
                          ('g / mol', 0, 3, False), ('mol / g', 0, 5, False)])
def test_convert_mass_units_special_case(unit, mag, power, to_lammps):

    """
    Tests the various combinations of conversions amu <---> g / mol, the
    inverses, and different powers of units. In all cases, the values should
    be equal.
    """

    value = 5.67
    assert np.isclose(lmp_eng.convert_unit(units.UnitFloat(value,
                                                           units.Unit(unit)
                                                           ** power),
                                           to_lammps=to_lammps),
                      value * 10 ** (mag * power))


def test_partition_single_interaction(interactions, bonds):

    """
    Tests using partition_interactions function to filter a single interaction
    name from a list
    """

    assert bonds == list(lmp_eng.partition_interactions(interactions,
                                                        ['Bond'])[0])


def test_partition_multiple_interactions(interactions, bonds, angles,
                                         coulombics):

    """
    Tests using partition_interactions function to partition multiple
    interactions based on name
    """

    p_bonds, p_angles, p_coulombics = lmp_eng.partition_interactions(
        interactions, ['Bond', 'BondAngle', 'Coulombic'])
    assert list(p_bonds) == bonds
    assert list(p_angles) == angles
    assert list(p_coulombics) == coulombics


def test_partition_interactions_unpartitioned(interactions, dispersions):

    """
    Tests that when unpartitioned=True is passed to partition_interactions, the
    final entry returned is all interactions in input that did not have a name
    in the names argument
    """

    _, _, _, _, p_disps = lmp_eng.partition_interactions(interactions,
                                                         ['Bond',
                                                          'BondAngle',
                                                          'Coulombic',
                                                          'DihedralAngle'],
                                                         unpartitioned=True)
    assert list(p_disps) == dispersions


def test_partion_interactions_return_list(interactions, bonds, angles):

    """
    Tests that when lst=True is passed to partition_interactions, a tuple of
    lists is returned, rather than a tuple of generators
    """

    assert (bonds, angles) == lmp_eng.partition_interactions(interactions,
                                                             ['Bond',
                                                              'BondAngle'],
                                                             lst=True)


def test_warn_on_invalid_run(simulation):
    """
    Tests that a warning is issued when attempting to run a lammps
    simulation shorter than ``traj_step``
    """

    simulation.traj_step = 10
    lammps_engine.lmp_simulation = populated_lammps_simulation
    with pytest.warns(UserWarning, match="run may not produce usable output"):
        simulation.run(n_steps=3)
