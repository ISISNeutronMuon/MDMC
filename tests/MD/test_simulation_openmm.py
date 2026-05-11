"""Tests for setting up and running MDMC using OPENMM"""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from openmm import unit

import MDMC.MD.engine_facades.openmm_engine as openmm_eng
from MDMC.common import units
from MDMC.MD.interaction_functions import (
    Buckingham,
    Coulomb,
    HarmonicPotential,
    LennardJones,
    Periodic,
)
from MDMC.MD.interactions import Bond, BondAngle, Coulombic, DihedralAngle, Dispersion
from MDMC.MD.kspace_solvers import (
    PPPM,
    Ewald,
    KSpaceSolver
)
from MDMC.MD.constraints import Rattle, Shake
from MDMC.MD.simulation import (
    Simulation,
    Universe,
)
from MDMC.MD.structures import Atom
from MDMC.trajectory_analysis.compact_trajectory import CompactTrajectory

pytestmark = [pytest.mark.openmm]

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
    OPENMM, as this is done ordered by atom_type, rather than necessary the
    order which atoms appear in universe.atoms
    """

    symbols = ["C", "H", "N", "O"]
    masses = [12.011, 1.008, 14.007, 16.000]
    elements = symbols * (N_ATOMS // 4)
    elements[len(elements) : N_ATOMS] = symbols[: N_ATOMS - len(elements)]
    # Sorted so that atoms of same type are grouped
    elements = sorted(elements)
    atom_types = {symbol: n + 1 for n, symbol in enumerate(symbols)}
    atom_masses = {symbol: mass for symbol, mass in zip(symbols, masses)}

    return [
        Atom(
            element,
            position=np.array([0.5 * i] * 3),
            atom_type=atom_types[element],
            mass=atom_masses[element],
        )
        for i, element in enumerate(elements)
    ]


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
    bond1_harmonic = HarmonicPotential(1.0, 2.0, interaction_type="bond")
    bond2_harmonic = HarmonicPotential(2.0, 4.0, interaction_type="bond")
    angle_harmonic = HarmonicPotential(1.0, 0.0005, interaction_type="angle")
    proper_periodic = Periodic(1.0, 1, 90.0, 2.0, 2, 180.0, 0.1, 3, -90.0, 0.5, 4, -45.0)
    improper_harmonic = HarmonicPotential(1.0, 0.0002, interaction_type="improper")

    # Create 2 bonds for some atoms, and one angle, coulombic and dispersive
    # interaction
    bond1_atoms = [(atoms[i], atoms[i + 1]) for i in range(0, len(atoms) - 1, 2)]
    bond2_atoms = [(atoms[i], atoms[i + 2]) for i in range(0, len(atoms) - 2, 3)]
    bonds = [
        Bond(*bond1_atoms, function=bond1_harmonic),
        Bond(*bond2_atoms, function=bond2_harmonic),
    ]

    angles = [BondAngle(*zip(atoms[0::3], atoms[1::3], atoms[2::3]), function=angle_harmonic)]

    propers = [
        DihedralAngle(tuple(atom for atom in atoms[:4]), function=proper_periodic, improper=False),
    ]
    impropers = [
        DihedralAngle(tuple(atom for atom in atoms[:4]), function=improper_harmonic, improper=True),
    ]
    coulombics, dispersions = [], []
    for type_ in empty_universe.atom_types:
        coulombics.append(
            Coulombic(
                empty_universe,
                atom_types=type_,
                function=Coulomb(-1.0 + type_ * 0.5),
                cutoff=COUL_CUTOFF,
            ),
        )
        dispersions.append(
            Dispersion(
                empty_universe,
                (type_, type_),
                function=Buckingham(type_ * 0.1, type_ * 1.0, type_ * 2.0),
                cutoff=DISP_CUTOFF,
                vdw_tail_correction=True,
            ),
        )
        dispersions.append(
            Dispersion(
                empty_universe,
                (type_, type_),
                function=LennardJones(type_ * 0.1, type_ * 1.0),
                cutoff=DISP_CUTOFF,
                vdw_tail_correction=True,
            ),
        )

    return (empty_universe, bonds, angles, propers, impropers, coulombics, dispersions)


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
def simulation(universe):
    """
    A mock simulation to give the engine facade its necessary 'parent simulation'
    """
    return Simulation(universe, traj_step=1, time_step=1.0, temperature=120, engine="openmm")


@pytest.fixture
def openmm_engine(universe, simulation):
    """
    Returns:
    A OPENMMEngine which is ready to run a OPENMM simulation with an NVE
    ensemble.
    """

    openmm_engine = openmm_eng.OpenMMEngine()
    openmm_engine._parent_simulation = simulation
    openmm_engine.setup_universe(universe)
    openmm_engine.setup_simulation(temperature=120)
    return openmm_engine


def test_simulation_setup():

    universe = Universe((10.0, 10.0, 10.0))
    universe.add_structure(
        Atom(
            "C",
            position=np.array([0.0, 0.0, 0.0]),
            atom_type=0,
            mass=13124.0,
        ),
    )

    sim_obj = Simulation(
        universe,
        engine="openmm",
        time_step=10.18893,
        temperature=300.0,
        pressure=101325.0,
        traj_step=15,
    )
    expected_output = (
        "Simulation created with openmm engine and settings:\n"
        "temperature: 300.0 K \n"
        "pressure: 101325.0 Pa \n\n"
    )
    assert expected_output == sim_obj.setup_msg


def test_universe_dimensions(openmm_engine):
    """
    Tests that creating a simulation box from an MDMC universe results in the
    correct universe dimensions

    Lower dimensions should be 0.0
    Upper dimensions should be equal to the MDMC universe dimensions
    """

    box = np.array(
        [(vec.x, vec.y, vec.z) for vec in openmm_engine.system.getDefaultPeriodicBoxVectors()],
        dtype=float,
    )
    assert np.allclose(np.eye(3) * UNIVERSE_DIM * unit.angstrom / 10, box)


@pytest.mark.skip("OpenMM doesn't handle atom types yet.")
def test_number_atom_types(openmm_engine):
    """
    Tests that creating a simulation box from an MDMC universe results in the
    correct number of atom types
    """

    assert openmm_engine.system.ntypes == 4


def test_number_atoms(openmm_engine, atoms):
    """
    Tests that the correct number of atoms has been added to OPENMM
    """

    assert openmm_engine.natoms == len(atoms)


def test_atom_type_mass(openmm_engine, universe):
    """
    Tests that the mass of each atom type is set correctly in OPENMM
    """
    assert np.allclose(
        [(float(i.mass) * unit.amu)._value for i in universe.atoms],
        [openmm_engine.system.getParticleMass(i)._value for i in range(openmm_engine.natoms)],
    )


def test_atom_position(openmm_engine, universe):
    """
    Tests that atoms created in OPENMM have the correct position
    """

    pos = (
        openmm_engine.simulation.context.getState(positions=True)
        .getPositions(
            asNumpy=True,
        )
        ._value
    )

    assert np.allclose([np.array(i.position / 10) for i in universe.atoms], pos)


@pytest.mark.skip("OpenMM currently only handles LennardJones")
@pytest.mark.parametrize(
    "interactions, expected",
    [
        ("bonds", "harmonic"),
        ("angles", "harmonic"),
        ("propers", "fourier"),
        ("impropers", "harmonic"),
    ],
)
def test_parse_bonded_styles(interactions, expected, request):
    """
    Tests that the return from parse_bonded_styles is the correct input for
    creating a OPENMM bond_style or angle_style

    The parameters should be modified whenever a new bonded style is
    implemented
    """

    # As fixtures cannot be included in parameterization, the names of the
    # fixtures are included instead - the return values of the fixtures are then
    # recovered using request.getfixturevalue
    interactions = request.getfixturevalue(interactions)
    # Test the first interaction in each list of interactions
    assert openmm_eng.parse_bonded_styles(interactions[0]) == expected


@pytest.mark.skip("OpenMM currently only handles LennardJones")
@pytest.mark.parametrize(
    "inters, index, expected, solver_attr",
    [
        ("dispersions", 0, ["buck", 10.0], None),
        ("dispersions", 1, ["lj/cut", 10.0], None),
        ("coulombics", 0, ["coul/cut", 8.0], None),
        ("dispersions", 0, ["buck/long", 10.0], "kspace_solver"),
        ("dispersions", 1, ["lj/long", 10.0], "kspace_solver"),
        ("coulombics", 0, ["coul/long", 8.0], "kspace_solver"),
        ("dispersions", 0, ["buck/long", 10.0], "dispersive_solver"),
        ("dispersions", 1, ["lj/long", 10.0], "dispersive_solver"),
        ("coulombics", 0, ["coul/cut", 8.0], "dispersive_solver"),
        ("dispersions", 0, ["buck", 10.0], "electrostatic_solver"),
        ("dispersions", 1, ["lj/cut", 10.0], "electrostatic_solver"),
        ("coulombics", 0, ["coul/long", 8.0], "electrostatic_solver"),
    ],
)
def test_parse_nonbonded_styles(inters, index, expected, solver_attr, universe, request):
    """
    Tests that the return from parse_nonbonded_styles is the correct input for
    creating a OPENMM pair style

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
    assert openmm_eng.parse_nonbonded_styles(inters)[0] == expected


def test_atom_charges_update(openmm_engine, universe):
    """
    Tests that atom charges are updated correctly

    Change the charges on the atoms in the universe and test if OPENMM charges
    update after LAMMPUniverse._update_charges is called
    """

    # Change charges and update OPENMMEngine
    for atom in universe.atoms:
        atom.charge *= 2.0
    openmm_engine.update_parameters()


@pytest.mark.skip("OpenMM currently only handles LennardJones")
@pytest.mark.parametrize(
    "interaction_fixture, openmm_name",
    [
        ("bonds", "bond"),
        ("angles", "angle"),
        ("propers", "dihedral"),
        ("impropers", "improper"),
        ("dispersions", None),
    ],
)
def test_update_individual_interactions(openmm_engine, interaction_fixture, openmm_name, request):
    """
    Tests that updating each individual interaction does not result in a fatal
    error, where the OPENMM Python interface causes Python to exit without
    throwing an error, presumably due to a segfault

    A more stringent test would check that the correct coefficients for each
    interation have been set in OPENMM, however there is no way to check this
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

    if interaction_fixture == "dispersions":
        openmm_engine._update_dispersions(openmm_engine.universe)
    else:
        openmm_engine._update_bonded_interactions(openmm_name, interactions)


def test_update_all_interactions(openmm_engine, interactions):
    """
    Tests that updating all interactions does not result in a fatal error, where
    the OPENMM Python interface causes Python to exit without throwing an error,
    presumably due to a segfault

    A more stringent test would check that the correct coefficients for each
    interation have been set in OPENMM, however there is no way to check this
    through the Python interface. Therefore the minimum test of checking for a
    fatal error is used.
    """

    # Scale all parameters for all interactions
    for interaction in interactions:
        for parameter in interaction.parameters:
            interaction.parameters[parameter].value *= 2

    openmm_engine.update_parameters()


def test_update_charges_error():
    """
    Tests that an error is raised when trying to create a OPENMM universe
    from a universe that contains atoms with a charge of None.
    """

    universe = Universe(10.0, verbose=False)
    universe.add_structure(Atom("H"))
    with pytest.raises(AttributeError):
        openmm_eng.OPENMMUniverse(universe)


@pytest.mark.parametrize(
    "solver_cls, accuracy, expected",
    [(PPPM, 0.001, ["PME", 0.001]), (Ewald, 1e-05, ["Ewald", 1e-05])],
)
def test_parse_kspace_solver(solver_cls, accuracy, expected):
    """
    Tests that parsing the kspace solver returns the correct input for OPENMM
    kspace_style command
    """

    solver = solver_cls(accuracy=accuracy)
    solver = openmm_eng.parse_kspace_solver(solver)
    type_, accuracy_ = expected
    assert solver is not None
    assert solver.getNonbondedMethod() == getattr(solver, type_)
    assert solver.getEwaldErrorTolerance() == accuracy_


def test_parse_kspace_solver_unimplemented():
    """
    Tests that parsing an unimplemented kspace solver raises a
    NotImplementedError
    """

    solver = KSpaceSolver(accuracy=0.0001)
    with pytest.raises(NotImplementedError):
        openmm_eng.parse_kspace_solver(solver)


@pytest.mark.parametrize("constraint, name", [(Shake, "shake"), (Rattle, "rattle")])
def test_parse_constraint_algorithm_name(constraint, name, constrained_bonds, bond_ID_dict):
    """
    Tests that passing different ConstraintAlgorithms produces the expected
    algorithm name for the input to OPENMM fix

    Excluding the fix ID and and group-ID, the algorithm name is the index 0
    entry submitted to OPENMM fix
    """

    constraint_algorithm = constraint(accuracy=1.0, max_iterations=1)
    assert (
        name
        == openmm_eng.parse_constraint(
            constraint_algorithm, bonds=constrained_bonds, bond_ID_dict=bond_ID_dict
        )[0]
    )


@pytest.mark.parametrize("accuracy", [1.0, 1e-4, 5])
def test_parse_constraint_accuracy(accuracy, constrained_bonds, bond_ID_dict):
    # ID is an acronym
    # pylint: disable=invalid-name

    """
    Tests that accuracy is correct in the input to OPENMM fix

    Excluding the fix ID and and group-ID, the accuracy is the index 1
    entry passed to a OPENMM fix. The accuracy must be a float.
    """

    constraint_algorithm = Shake(accuracy=accuracy, max_iterations=1)
    algorithm_accuracy = openmm_eng.parse_constraint(
        constraint_algorithm, bonds=constrained_bonds, bond_ID_dict=bond_ID_dict
    )[1]
    assert float(accuracy) == algorithm_accuracy


@pytest.mark.parametrize("max_iter", [1, 5.4])
def test_parse_constraint_max_iterations(max_iter, constrained_bonds, bond_ID_dict):
    # ID is an acronym
    # pylint: disable=invalid-name

    """
    Tests that the max number of iterations is correct in the input to OPENMM
    fix

    Excluding the fix ID and and group-ID, the number of max iterations is the
    index 2 entry passed to a OPENMM fix. The number of max iterations must be
    an integer.
    """

    constraint_algorithm = Shake(accuracy=1.0, max_iterations=max_iter)
    algorithm_max_iter = openmm_eng.parse_constraint(
        constraint_algorithm, bonds=constrained_bonds, bond_ID_dict=bond_ID_dict
    )[2]
    assert int(max_iter) == algorithm_max_iter


def test_parse_constraint_bonds(constrained_bonds, bond_ID_dict):
    # ID is an acronym
    # pylint: disable=invalid-name

    """
    Tests that the input to OPENMM has the correct bond IDs

    Excluding the fix ID and and group-ID, the declaration of bond constraints
    (indicated by 'b') is the index 4 entry passed to a OPENMM fix. Following
    this the IDs of all of the constrained bonds must be listed.
    """

    constraint_algorithm = Shake(accuracy=1.0, max_iterations=1)
    openmm_input = openmm_eng.parse_constraint(
        constraint_algorithm, bonds=constrained_bonds, bond_ID_dict=bond_ID_dict
    )
    assert openmm_input[4] == "b"
    assert sorted(openmm_input[5:]) == sorted([bond_ID_dict[bond] for bond in constrained_bonds])


@pytest.mark.skip("OpenMM does not support angle constraints.")
def test_parse_constraint_angles(constrained_angles, angle_ID_dict):
    """
    Tests that the input to OPENMM has the correct angle IDs

    Excluding the fix ID and and group-ID, the declaration of angle constraints
    (indicated by 'a') is the index 4 entry passed to a OPENMM fix, if no bonds
    are included. Following this the IDs of all of the constrained angles must
    be listed.
    """

    constraint_algorithm = Shake(accuracy=1.0, max_iterations=1)
    openmm_input = openmm_eng.parse_constraint(
        constraint_algorithm, angles=constrained_angles, angle_ID_dict=angle_ID_dict
    )
    assert openmm_input[4] == "a"
    assert sorted(openmm_input[5:]) == sorted(
        [angle_ID_dict[angle] for angle in constrained_angles]
    )


@pytest.mark.skip("OpenMM does not support constraints.")
@pytest.mark.parametrize(
    "arguments",
    [
        {"bonds": "constrained_bonds"},
        {"bonds": "constrained_bonds", "angle_ID_dict": "angle_ID_dict"},
    ],
)
def test_parse_constraint_no_IDs(arguments, request):
    # ID is an acronym
    # pylint: disable=invalid-name

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
    arg_fixtures = {k: request.getfixturevalue(v) for k, v in arguments.items()}
    constraint_algorithm = Shake(accuracy=1.0, max_iterations=1)
    with pytest.raises(KeyError):
        openmm_input = openmm_eng.parse_constraint(constraint_algorithm, **arg_fixtures)


@pytest.mark.parametrize("temperature", [300.0, 450.0])
def test_initialize_velocities(universe, openmm_engine: openmm_eng.OpenMMEngine, temperature):
    """
    Test that the OPENMM velocities have been set correctly when MDMC velocities are zero

    Initialize the velocities by setting the temperature. Set the ensemble to
    NVE and run for 0 steps. Test if the 0 step temperature is as expected.
    """

    openmm_engine.setup_simulation(temperature=temperature, traj_step=10)

    context = openmm_engine.context
    vel = context.getState(velocities=True).getVelocities(asNumpy=True)
    n_atom = openmm_engine.system.getNumParticles()

    for i, atom in enumerate(universe.atoms):
        # MDMC atoms should be unchanged, but the OPENMM atoms should have velocities
        assert np.all(np.array(atom.velocity) == 0)
        assert np.all(np.array(vel[i]) != 0)

    assert_allclose(
        openmm_engine.current_temperature, temperature, atol=temperature / np.sqrt(n_atom)
    )


@pytest.mark.parametrize("temperature", [150.0, 300.0])
def test_initialize_nonzero_velocities(universe, temperature):
    """
    Test that the OPENMM velocities have been set correctly when MDMC velocities are non-zero

    Initialize the velocities by setting the temperature. Set the ensemble to
    NVE and run for 0 steps. Test if the 0 step temperature is as expected.
    """

    # Set the MDMC velocities
    velocity = []
    for i, atom in enumerate(universe.atoms):
        velocity.append(np.array((-(i + 1), 0, i + 1)))
        atom.velocity = velocity[i]

    # Create new OPENMM universe/simulation with these velocities
    openmm_engine = openmm_eng.OpenMMEngine()
    openmm_engine._parent_simulation = ()
    openmm_engine.setup_universe(universe)
    openmm_engine.setup_simulation(temperature=temperature, traj_step=10)

    # OPENMM should scale all velocities by the same amount to ensure the temperature is accurate.
    # Get this factor from the first atom, as it had an initial velocity of 1 in the z direction.
    vel = openmm_engine.context.getState(velocities=True).getVelocities(asNumpy=True)
    scale_factor = vel[0, 2]
    for i, atom in enumerate(universe.atoms):
        assert np.all(np.array(atom.velocity) == velocity[i])
        assert np.all(vel[i] == scale_factor * velocity[i])

    openmm_engine.run(0)
    assert_allclose(openmm_engine.current_temperature, temperature)


@pytest.mark.parametrize("n_steps", [1, 10])
def test_trajectory_output(openmm_engine, n_steps):
    """
    Tests if a trajectory file of the correct length has been created by OPENMM
    wrapper
    """

    # openmm_engine_simulation is setup to output trajectory every step. Run for
    # a total of n_steps
    openmm_engine.run(n_steps)

    n_atoms = openmm_engine.system.natoms
    n_lines = (n_atoms + 9) * ((n_steps / openmm_engine.traj_step) + 1)
    assert len(openmm_engine.trajectory_file.readlines()) == n_lines


def test_save_config(openmm_engine, universe):
    """
    Tests that the OPENMM configuration is correctly saved, by checking the
    positions, mass and charge of the OPENMM wrapper atoms attribute
    """

    openmm_engine.save_config()
    # Positions should be the same as those of the MDMC universe atoms, which
    # are also ordered by ID
    for i in range(len(universe.atoms)):
        assert (np.array(openmm_engine.saved_config[i][:3]) == universe.atoms[i].position).all()


def test_reset_config(openmm_engine):
    """
    Tests that the reset_config method correctly changes the positions of the
    OPENMM wrapper atoms back to the saved values

    To do this the config is saved, a short simulation is run, and the config
    is reset
    """

    openmm_engine.save_config()
    openmm_engine.openmm.run(10)

    n_atoms = openmm_engine.system.natoms
    # Ensure that the atoms have moved from their starting positions - see atoms
    # fixture for what the starting positions are
    for i in range(n_atoms):
        assert (np.array(openmm_engine.openmm.atoms[i].position) != np.array([0.5 * i] * 3)).all()

    openmm_engine.reset_config()
    for i in range(n_atoms):
        assert (np.array(openmm_engine.openmm.atoms[i].position) == np.array([0.5 * i] * 3)).all()


def test_convert_trajectory_output(openmm_engine):
    """
    Tests that converting a trajectory results in an MDMC CompactTrajectory object

    This does not test the correctness of the converted trajectory, purely that
    a trajectory can be converted with the correct type. The correctness of
    the trajectory conversion is covered by a system test.
    """

    openmm_engine.run(3)
    assert isinstance(openmm_engine.convert_trajectory(), CompactTrajectory)


@pytest.mark.parametrize(
    "args",
    [
        {"n_steps": 0, "minimize_every": 5, "maxiter": 1000},
        {
            "n_steps": 0,
            "minimize_every": 5,
            "etol": 0.0,
            "ftol": 1.0e-8,
            "maxeval": 1000,
            "maxiter": 1000,
        },
        {"n_steps": 0, "minimize_every": 5, "ftol": 1.0e-8, "maxeval": 500, "maxiter": 5000},
    ],
)
def test_minimize(args, openmm_engine):
    """
    Tests that the potential energy has been minimized

    This does not test that the minimization reduces the potential energy into a
    local minima, just that the potential energy of the system reduces

    Parameterization tests for both default and non-default minimization
    arguments
    """

    # OPENMM needs to run for 0 steps to calculate energies - run directly using
    # OPENMM wrapper run so that any bugs in OPENMMEngine.run do not affect test
    openmm_engine.openmm.run(0)
    start_energy = openmm_engine.openmm.eval("pe")
    openmm_engine.minimize(**args)
    assert openmm_engine.openmm.eval("pe") < start_energy


@pytest.mark.parametrize(
    "thermostat, barostat, add_args",
    [(None, None, {}), ("nose", None, {}), ("nose", "nose", {"pressure": 1.0})],
)
def test_setup_simulation_run(openmm_engine, thermostat, barostat, add_args):
    """
    Tests that the simulation setup can run an NVE, NVT and NPT simulation with
    the default attribute values
    """

    # Simulation setup requires the traj_step attribute to be set, even though
    # it is not being used in this test
    # add_args is a dictionary of additional arguments that are required for the
    # specific ensemble
    openmm_engine.setup_simulation(
        temperature=300.0, thermostat=thermostat, barostat=barostat, **add_args
    )

    n_steps = 20
    openmm_engine.openmm.run(n_steps)

    # Test that the largest step number in the OPENMM wrapper runs attribute
    # (which records ThermoData from the previous run) is correct
    assert max(openmm_engine.openmm.runs[0][0].Step) == n_steps


def test_warn_on_invalid_run(populated_simulation):
    """
    Tests that a warning is issued when attempting to run a openmm
    simulation shorter than ``traj_step``
    """

    simulation.traj_step = 10
    openmm_engine.simulation = populated_simulation
    with pytest.warns(UserWarning, match="run may not produce usable output"):
        simulation.run(n_steps=3)
