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
from MDMC.MD.simulation import (
    PPPM,
    ConstraintAlgorithm,
    Ewald,
    KSpaceSolver,
    Rattle,
    Shake,
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
    openmm_engine.parent_simulation = simulation
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
        [
            (vec.x, vec.y, vec.z)
            for vec in openmm_engine.system.getDefaultPeriodicBoxVectors()
        ],
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


@pytest.mark.skip("OpenMM currently only handles LennardJones")
def test_number_interaction_types(openmm_engine):
    """
    Tests that creating a simulation box from an MDMC universe results in the
    correct number of each interaction type:

    - bond
    - angle
    - improper

    PyOpenmm does not allow polling for ndihedraltypes (unlike nbondtypes,
    nimpropertypes, and nangletypes) so there is no test for the number of
    proper dihedral types.
    """

    getter = openmm_engine.openmm.openmm.numpy
    for name, expected in zip(("bonds", "angles", "impropers"), (2, 1, 1)):
        assert np.max(getattr(getter, f"gather_{name}")()[:, 0]) == expected


@pytest.mark.skip("OpenMM currently only handles LennardJones")
def test_number_interactions(openmm_engine, bonds, angles, propers, impropers):
    """
    Tests that creating a simulation box from an MDMC universe results in the
    correct allowed number of interactions per atom for each interaction type:

    - bond
    - angle
    - dihedral
    - improper

    DIHEDRAL AND IMPROPER ARE NOT CURRENTLY IMPLEMENTED
    """
    getter = openmm_engine.openmm.openmm.numpy
    for var, name in zip(
        (bonds, angles, propers, impropers), ("bonds", "angles", "dihedrals", "impropers")
    ):
        assert getattr(getter, f"gather_{name}")().shape[0] == sum(len(x.atoms) for x in var)


@pytest.mark.skip("OpenMM doesn't handle atom types")
def test_atom_type_properties(openmm_engine, universe):
    """
    Tests that element and mass are assigned to each list index corresponding to
    atom type equivalent to that index (-1 offset due to atom_type starting from
    1)
    """

    for atom_type, atoms in universe.atom_types.items():
        assert openmm_engine.atom_type_properties[atom_type - 1] == (
            atoms[0].element,
            atoms[0].mass,
        )


def test_atom_type_mass(openmm_engine, universe):
    """
    Tests that the mass of each atom type is set correctly in OPENMM
    """
    assert np.allclose(
        [(float(i.mass) * unit.amu)._value for i in universe.atoms],
        [
            openmm_engine.system.getParticleMass(i)._value
            for i in range(openmm_engine.natoms)
        ],
    )


@pytest.mark.skip("OpenMM doesn't handle atom types")
def test_atom_ID(openmm_engine, universe):
    """
    Tests that atoms created in OPENMM have the correct ID
    """

    # Atom IDs in universe are offset by some integer related to the number of
    # time the atoms fixture is called. If this offset is subtracted, the IDs
    # should agree exactly with the OPENMM atom IDs
    offset = universe.atoms[0].ID - 1
    for i in range(len(universe.atoms)):
        assert openmm_engine.openmm.atoms[i].id == universe.atoms[i].ID - offset


@pytest.mark.skip("OpenMM doesn't handle atom types")
def test_atom_type(openmm_engine, universe):
    """
    Tests that atoms created in OPENMM have the correct atom types
    """

    for i in range(len(universe.atoms)):
        assert openmm_engine.openmm.atoms[i].type == universe.atoms[i].atom_type


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


@pytest.mark.skip("OpenMM doesn't use add topology")
def test_unimplemented_interactions(openmm_engine, universe):
    """
    Tests that if a universe passed to OPENMMUniverse._add_topology has any
    interactions which have not been implemented in OPENMM, NotImplementedError
    is raised
    """

    # Add unimplemented interaction type to universe
    # Dummy class which does not require docstring
    # pylint: disable=missing-docstring, multiple-statements
    class Unimplemented(Dispersion):
        pass

    unimplemented_interaction = Unimplemented(universe, (1, 1))

    # Create OPENMM topology from universe, raising NotImplementedError
    with pytest.raises(NotImplementedError):
        openmm_engine._add_topology(universe)


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


@pytest.mark.skip("OpenMM currently only handles LennardJones")
@pytest.mark.parametrize(
    "inters, indices, solver_attr, expected",
    [
        (
            ("coulombics", "dispersions", "dispersions"),
            (0, 0, 1),
            None,
            [
                ("buck/coul/cut", "{0} {1}".format(DISP_CUTOFF, COUL_CUTOFF)),
                ("lj/cut/coul/cut", "{0} {1}".format(DISP_CUTOFF, COUL_CUTOFF)),
            ],
        ),
        (
            ("coulombics", "dispersions", "dispersions"),
            (0, 0, 1),
            "electrostatic_solver",
            [
                ("buck/coul/long", "{0} {1}".format(DISP_CUTOFF, COUL_CUTOFF)),
                ("lj/cut/coul/long", "{0} {1}".format(DISP_CUTOFF, COUL_CUTOFF)),
            ],
        ),
    ],
)
def test_parse_all_nonbonded_styles_valid_diff_cutoffs(
    inters, indices, solver_attr, expected, universe, request
):
    """
    Tests the generation of valid OPENMM pair_styles of Dispersive and
    Coulombic interactions for various solver attributes, where the
    Dispersive and Coulombic cutoff distances are different.

    Doesn't test for interactions created in a universe with a
    kspace_solver attribute as this creates an invalid OPENMM command.

    Doesn't test for interactions created in a universe with a
    dispersive_solver attribute as this creates an invalid pair style.
    """

    assert COUL_CUTOFF != DISP_CUTOFF
    inters = [request.getfixturevalue(inter)[idx] for inter, idx in zip(inters, indices)]
    if solver_attr:
        setattr(universe, solver_attr, PPPM(accuracy=1e-4))
    assert list(openmm_eng.parse_all_nonbonded_styles(inters).keys()) == expected


@pytest.mark.skip("OpenMM currently only handles LennardJones")
@pytest.mark.parametrize(
    "inters, indices, solver_attr, cutoff, expected",
    [
        (
            ("coulombics", "dispersions", "dispersions"),
            (0, 0, 1),
            None,
            CUTOFF,
            [("buck/coul/cut", "{0}".format(CUTOFF)), ("lj/cut/coul/cut", "{0}".format(CUTOFF))],
        ),
        (
            ("coulombics", "dispersions", "dispersions"),
            (0, 0, 1),
            "kspace_solver",
            CUTOFF,
            [
                ("buck/long/coul/long", "long long", "{0}".format(CUTOFF)),
                ("lj/long/coul/long", "long long", "{0}".format(CUTOFF)),
            ],
        ),
        (
            ("coulombics", "dispersions", "dispersions"),
            (0, 0, 1),
            "electrostatic_solver",
            CUTOFF,
            [("buck/coul/long", "{0}".format(CUTOFF)), ("lj/cut/coul/long", "{0}".format(CUTOFF))],
        ),
    ],
)
def test_parse_all_nonbonded_styles_valid_same_cutoff(
    inters, indices, solver_attr, cutoff, expected, universe, request
):
    """
    Tests the generation of valid OPENMM pair_styles of Dispersive and
    Coulombic interactions for various solvent attributes, where the
    Dispersive and Coulombic cutoff distances are the same.

    Doesn't test for interactions created in a universe with a
    dispersive_solver attribute as this creates an invalid pair style.
    """

    inters = [
        request.getfixturevalue(interaction)[idx] for interaction, idx in zip(inters, indices)
    ]
    # Set the cutoff to the same value for all interactions
    for interaction in inters:
        interaction.cutoff = cutoff
    if solver_attr:
        setattr(universe, solver_attr, PPPM(accuracy=1e-4))
    assert list(openmm_eng.parse_all_nonbonded_styles(inters).keys()) == expected


@pytest.mark.skip("OpenMM currently only handles LennardJones")
@pytest.mark.parametrize("index", [0, 1])
def test_parse_all_nonbonded_styles_diff_cutoffs_error(
    dispersions, index, coulombics, universe, request
):
    """
    Tests that a ValueError is raised when trying to create the following
    pair styles when the Dispersive and Coulombic interactions are created
    with different cut offs:

        - buck/long/coul/long
        - lj/long/coul/long
    """

    assert COUL_CUTOFF != DISP_CUTOFF
    interactions = [
        request.getfixturevalue("dispersions")[index],
        request.getfixturevalue("coulombics")[0],
    ]
    # Use kspace solver for long range Dispersive and Coulombic interactions
    setattr(universe, "kspace_solver", PPPM(accuracy=1e-4))
    with pytest.raises(ValueError):
        openmm_eng.parse_all_nonbonded_styles(interactions)


@pytest.mark.skip("OpenMM currently only handles LennardJones")
@pytest.mark.parametrize(
    "interactions, indices, solver_attr",
    [
        (("coulombics", "dispersions"), (0, 0), "dispersive_solver"),
        (("coulombics", "dispersions"), (0, 1), "dispersive_solver"),
    ],
)
def test_parse_all_nonbonded_styles_invalid_styles(
    interactions, indices, solver_attr, universe, request
):
    """
    Tests that a ValueError is raised when trying to create the following
    invalid OPENMM pair_styles:

        - buck/long/coul/cut
        - lj/long/coul/cut
    """

    interactions = [
        request.getfixturevalue(interaction)[idx] for interaction, idx in zip(interactions, indices)
    ]
    setattr(universe, solver_attr, PPPM(accuracy=1e-4))
    with pytest.raises(ValueError):
        openmm_eng.parse_all_nonbonded_styles(interactions)


@pytest.mark.skip("OpenMM currently only handles LennardJones")
def test_parse_nonbonded_styles_no_cutoff_error(request):
    """
    Tests that an AttributeError is raised when trying to create OPENMM pair_styles from
    nonbonded interactions which have no `cutoff` attribute set.
    """

    interactions = [
        request.getfixturevalue("dispersions")[0],
        request.getfixturevalue("coulombics")[0],
    ]
    for interaction in interactions:
        interaction.cutoff = None
    with pytest.raises(AttributeError):
        openmm_eng.parse_all_nonbonded_styles(interactions)


@pytest.mark.skip("OpenMM currently only handles LennardJones")
@pytest.mark.parametrize(
    "interaction, arguments, parser",
    [
        (Bond, ["atom_pair"], "parse_bonded_styles"),
        (Dispersion, ["universe", (1, 1)], "parse_nonbonded_styles"),
    ],
)
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
        getattr(openmm_eng, parser)(undefined_interaction_function)


@pytest.mark.skip("OpenMM currently only handles LennardJones")
@pytest.mark.parametrize(
    "inter_type, fun_type, parameters, settings, expected",
    [
        (
            "Bond",
            "HarmonicPotential",
            (5.0, 2.5),
            {"interaction_type": "bond"},
            ["harmonic", 0.5975143403441683, 5.0],
        ),
        (
            "BondAngle",
            "HarmonicPotential",
            (90.0, 1.0),
            {"interaction_type": "angle"},
            ["harmonic", 0.2390057361376673, 90.0],
        ),
        (
            "DihedralAngle",
            "Periodic",
            (1.0, 2, 30.0),
            {},
            ["fourier", 1, 0.2390057361376673, 2, 30.0],
        ),
        (
            "DihedralAngle",
            "Periodic",
            (4.184, 2, 30.0, 8.368, 8, -45.0),
            {},
            ["fourier", 2, 1.0, 2, 30.0, 2.0, 8, -45.0],
        ),
        (
            "DihedralAngle",
            "HarmonicPotential",
            (110.0, 15.0),
            {"improper": True, "interaction_type": "improper"},
            ["harmonic", 3.585086042065009, 110.0],
        ),
        (
            "DihedralAngle",
            "Periodic",
            (5.5, 3, 0.0),
            {"improper": True},
            ["cvff", 1.31453154875717, 1, 3],
        ),
        (
            "DihedralAngle",
            "Periodic",
            (2.5, 4, 180.0),
            {"improper": True},
            ["cvff", 0.5975143403441683, -1, 4],
        ),
    ],
)
def test_parse_bonded_coefficients(inter_type, fun_type, parameters, settings, expected):
    """
    Tests that parsing the bonded coefficients produces the expected input for
    the OPENMM coeff commands

    Creates an Interaction and InteractionFunction of the types specified. The
    parameters for the InteractionFunction are specified by 'parameters' and
    all required keywords for both the Interaction and InteractionFunction are
    in 'settings'.

    The differences between the values specified in 'parameters' and those in
    'expected' are due to unit conversion which occurs in bond coefficient
    parsing. The differences between the order is because OPENMM requires some
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
    assert openmm_eng.parse_bonded_coefficients(interaction) == expected


@pytest.mark.skip("OpenMM currently only handles LennardJones")
@pytest.mark.parametrize(
    "system_attr, expected",
    [("bond_style", "hybrid"), ("angle_style", "hybrid"), ("pair_style", "hybrid/overlay")],
)
def test_create_interaction_style(openmm_engine, system_attr, expected):
    """
    Tests that all interactions are created with a hybrid style, for:

    - bond
    - angle
    - dihedral
    - improper
    - nonbonded interactions

    DIHEDRAL AND IMPROPER ARE NOT CURRENTLY IMPLEMENTED
    """
    assert getattr(openmm_engine.system, system_attr) == expected


def test_atom_charge_set(openmm_engine, universe):
    """
    Tests that atom charges are set correctly
    """
    pot = openmm_engine.coul_force

    for i in range(len(universe.atoms)):
        assert pot.getParticleParameters(i)[0]._value == universe.atoms[i].charge


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

    for pot in openmm_engine.system.getForces():
        if pot.getName() == "Coulomb":
            break
    else:
        raise ValueError("No Coulomb forces defined.")

    for i in range(len(universe.atoms)):
        assert pot.getParticleParameters(i)[0]._value == universe.atoms[i].charge


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


@pytest.mark.skip("OpenMM currently only handles LennardJones")
@pytest.mark.parametrize("mix", ["GEOMETRIC", "geometric", "arithmetic", "SIXTHPOWER"])
def test_mixing(mix, universe):
    """
    Tests that applying different nonbonded interaction mixing styles does not
    result in a fatal error, where the OPENMM Python interface causes Python to
    exit without throwing an error, presumably due to a segfault

    A more stringent test would check the that values of pair_modify have been
    set in OPENMM, however there is no way to check this through the Python
    interface. Therefore the minimum test of checking for a fatal error is used.
    """

    openmm_engine = openmm_eng.OPENMMUniverse(universe, nonbonded_mix=mix)


@pytest.mark.skip("OpenMM currently only handles LennardJones")
@pytest.mark.parametrize("mix", ["geometrix", "equal"])
def test_mixing_unimplemented(openmm_engine, mix):
    """
    Tests that applying different nonbonded interaction mixing styles does not
    result in a fatal error, where the OPENMM Python interface causes Python to
    exit without throwing an error, presumably due to a segfault

    A more stringent test would check the that values of pair_modify have been
    set in OPENMM, however there is no way to check this through the Python
    interface. Therefore the minimum test of checking for a fatal error is used.
    """

    with pytest.raises(ValueError):
        openmm_engine.nonbonded_mix = mix


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


# @pytest.mark.parametrize("solver_cls", [PPPM, Ewald])
# def test_set_different_cutoffs(openmm_engine, universe, dispersions, solver_cls):
#     """
#     Tests that if cutoffs for dispersion and coulombic interaction are different
#     it results in a ValueError
#     """

#     # Create a kspace solver and add it to an MDMC universe. Pass this universe
#     # to a OPENMMUniverse._add_topology to set this kspace style in OPENMM.
#     solver = solver_cls(accuracy=0.0001)
#     universe.kspace_solver = solver

#     # Set cutoffs for dispersion interactions to be different to cutoffs for
#     # coulombic interactions
#     for dispersion in dispersions:
#         dispersion.cutoff = COUL_CUTOFF + 2.0
#     with pytest.raises(ValueError):
#         openmm_engine._add_topology(openmm_engine.universe)


# @pytest.mark.parametrize(
#     "solver_attr, expected, omp_expected",
#     [
#         ("kspace_solver", "pppm", "pppm/omp"),
#         ("electrostatic_solver", "pppm", "pppm/omp"),
#         ("dispersive_solver", TypeError, TypeError),
#     ],
# )
# def test_set_kspace_solver_single_solver_error(
#     populated_simulation, solver_attr, expected, omp_expected
# ):
#     """
#     Tests setting the kspace solver with the different solver attributes that
#     exist for a universe (kspace_solver, electrostatic_solver,
#     dispersive_solver)

#     kspace_solver and electrostatic_solver are valid single solvers for OPENMM,
#     however dispersive_solver must raise a TypeError
#     """

#     # Create a solver and add it to the universe as either a kspace_solver,
#     # electrostatic_solver or a dispersive_solver. Then create topology to set
#     # kspace style in OPENMM.
#     solver = PPPM(accuracy=0.0001)
#     setattr(populated_simulation.universe, solver_attr, solver)
#     if expected is TypeError:
#         with pytest.raises(expected):
#             populated_simulation._set_kspace_solver()
#     else:
#         populated_simulation._set_kspace_solver()
#         assert (
#             populated_simulation.system.kspace_style == expected
#             or populated_simulation.system.kspace_style == omp_expected
#         )


# def test_set_kspace_solver_multiple_solvers(populated_simulation):
#     """
#     Tests setting the kspace solver if the Universe has both an
#     electrostatic_solver and a dispersion_solver and they are equal
#     """

#     # Create a kspace solver and add it to the universe as both an
#     # electrostatic_solver and a dispersive_solver. Then call set_kspace_solver
#     # to apply kspace style in OPENMM.
#     solver = PPPM(accuracy=0.0001)
#     populated_simulation.universe.electrostatic_solver = solver
#     populated_simulation.universe.dispersive_solver = solver
#     populated_simulation._set_kspace_solver()
#     assert (
#         populated_simulation.system.kspace_style == "pppm"
#         or populated_simulation.system.kspace_style == "pppm/omp"
#     )


# def test_set_kspace_solver_multiple_solvers_error(populated_simulation):
#     """
#     Tests setting the kspace solver if the Universe has both an
#     electrostatic_solver and a dispersion_solver and they are not equal
#     """

#     # Create different kspace solvers for universe's electrostatic_solver and
#     # dispersive_solvers. Then call set_kspace_solver to apply kspace style in
#     # OPENMM.
#     universe = populated_simulation.universe
#     universe.electrostatic_solver = PPPM(accuracy=0.0001)
#     universe.dispersive_solver = PPPM(accuracy=0.0005)
#     with pytest.raises(TypeError):
#         populated_simulation._set_kspace_solver()


@pytest.mark.skip("OpenMM does not support constraints.")
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


@pytest.mark.skip("OpenMM does not support constraints.")
def test_parse_constraint_algorithm_unimplemented(constrained_bonds, bond_ID_dict):
    """
    Tests that passing an ConstraintAlgorithm that is not implemented raises a
    NotImplementedError
    """

    constraint_algorithm = ConstraintAlgorithm(accuracy=1.0, max_iterations=1)
    with pytest.raises(NotImplementedError):
        invalid_constraint = openmm_eng.parse_constraint(
            constraint_algorithm, bonds=constrained_bonds, bond_ID_dict=bond_ID_dict
        )


@pytest.mark.skip("OpenMM does not support constraints.")
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


@pytest.mark.skip("OpenMM does not support constraints.")
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


@pytest.mark.skip("OpenMM does not support constraints.")
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


@pytest.mark.skip("OpenMM does not support constraints.")
def test_parse_constraint_angles(constrained_angles, angle_ID_dict):
    # ID is an acronym
    # pylint: disable=invalid-name

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
def test_parse_constraint_bonds_angles(
    constrained_bonds, constrained_angles, bond_ID_dict, angle_ID_dict
):
    # ID is an acronym
    # pylint: disable=invalid-name

    """
    Tests that the input to OPENMM has the correct bond IDs and angle IDs

    Excluding the fix ID and and group-ID, the declaration of bond constraints
    (indicated by 'b') is the index 4 entry passed to a OPENMM fix. Following
    this the IDs of all of the constrained bonds must be listed. The index
    after this must be the declaration of angle constraints (indicated by 'a'),
    and then the IDs of all of the constrained angles must be listed.
    """

    constraint_algorithm = Shake(accuracy=1.0, max_iterations=1)
    openmm_input = openmm_eng.parse_constraint(
        constraint_algorithm,
        bonds=constrained_bonds,
        bond_ID_dict=bond_ID_dict,
        angles=constrained_angles,
        angle_ID_dict=angle_ID_dict,
    )
    assert openmm_input[4] == "b"
    n_bonds = len(constrained_bonds)
    assert sorted(openmm_input[5 : 5 + n_bonds]) == sorted(
        [bond_ID_dict[bond] for bond in constrained_bonds]
    )
    assert openmm_input[5 + n_bonds] == "a"
    assert sorted(openmm_input[5 + n_bonds + 1 :]) == sorted(
        [angle_ID_dict[angle] for angle in constrained_angles]
    )


@pytest.mark.skip("OpenMM does not support constraints.")
def test_parse_constraint_no_interactions(bond_ID_dict):
    # ID is an acronym
    # pylint: disable=invalid-name

    """
    Tests that if neither bonds or angles are provided when parsing the
    constraint, a TypeError is raised
    """

    constraint_algorithm = Shake(accuracy=1.0, max_iterations=1)
    with pytest.raises(TypeError):
        openmm_input = openmm_eng.parse_constraint(constraint_algorithm, bond_ID_dict=bond_ID_dict)


@pytest.mark.skip("OpenMM does not support constraints.")
@pytest.mark.parametrize(
    "arguments",
    [
        {"bonds": "constrained_bonds"},
        {"bonds": "constrained_bonds", "angle_ID_dict": "angle_ID_dict"},
        {"angles": "constrained_angles"},
        {"angles": "constrained_angles", "bond_ID_dict": "bond_ID_dict"},
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



# @pytest.mark.parametrize(
#     "momentum_steps, expected_names",
#     [
#         ({"lin_momentum_steps": 5}, ["RemoveLinearMomentum"]),
#         ({"ang_momentum_steps": 10}, ["RemoveAngularMomentum"]),
#         ({"lin_momentum_steps": 20, "ang_momentum_steps": 20}, ["RemoveMomentum"]),
#         (
#             {"lin_momentum_steps": 15, "ang_momentum_steps": 20},
#             ["RemoveLinearMomentum", "RemoveAngularMomentum"],
#         ),
#     ],
# )
# def test_remove_momentum(populated_simulation, momentum_steps, expected_names):
#     """
#     Tests that linear and/or angular momentum remover fixes are correctly
#     created
#     """

#     # Set momentum_step attributes and apply fixes - ensure momentum_step
#     # attributes are both initially None.
#     populated_simulation.lin_momentum_steps = None
#     populated_simulation.ang_momentum_steps = None
#     for attr, steps in momentum_steps.items():
#         setattr(populated_simulation, attr, steps)

#     # The fix styles of all momentum removers should be 'momentum'. There
#     # should be one fix with this fix style.
#     assert Counter(populated_simulation.fix_styles)["momentum"] == len(expected_names)

#     # The name of the fix is defined by whether linear and/or angular
#     # momentum is removed
#     for name in expected_names:
#         assert name in populated_simulation.fix_names


# @pytest.mark.parametrize(
#     "thermostat, styles, omp_styles, attributes",
#     [
#         (None, ["nve"], ["OMP", "nve/omp"], {}),
#         ("nose", ["nvt"], ["OMP", "nvt/omp"], {"temperature": 400.0, "t_damp": 100}),
#         (
#             "berendsen",
#             ["nve", "temp/berendsen"],
#             ["OMP", "nve/omp", "temp/berendsen"],
#             {"temperature": 400.0, "t_damp": 100},
#         ),
#         (
#             "langevin",
#             ["nve", "langevin"],
#             ["OMP", "nve/omp", "langevin"],
#             {"temperature": 400.0, "t_damp": 100},
#         ),
#         (
#             "rescale",
#             ["nve", "temp/rescale"],
#             ["OMP", "nve/omp", "temp/rescale"],
#             {"temperature": 100.0, "t_fraction": 0.5, "t_window": 10.0, "rescale_step": 100},
#         ),
#         (
#             "csvr",
#             ["nve", "temp/csvr"],
#             ["OMP", "nve/omp", "temp/csvr"],
#             {"temperature": 400.0, "t_damp": 100},
#         ),
#     ],
# )
# def test_apply_thermostat(ensemble, thermostat, styles, omp_styles, attributes):
#     """
#     Tests that applying a thermostat results in the correct fix being applying
#     to OPENMM
#     """

#     # Set the attributes required for the specified thermostat
#     for attr, value in attributes.items():
#         setattr(ensemble, attr, value)

#     # Add the thermostat
#     ensemble.thermostat = thermostat

#     # Test that the fix styles returned from the OPENMM wrapper fixes attribute
#     # are correct
#     assert ensemble.fix_styles == styles or ensemble.fix_styles == omp_styles


# @pytest.mark.parametrize(
#     "barostat, styles, omp_styles",
#     [
#         (None, ["nve"], ["OMP", "nve/omp"]),
#         ("berendsen", ["press/berendsen"], ["OMP", "press/berendsen"]),
#         ("nose", ["nph"], ["OMP", "nph/omp"]),
#     ],
# )
# def test_apply_barostat(ensemble, barostat, styles, omp_styles):
#     """
#     Tests that applying a barostat results in the correct fix being applied to
#     OPENMM
#     """

#     # Set the attributes required for all barostats and add the barostat
#     ensemble.pressure = 10.0
#     ensemble.p_damp = 1000
#     ensemble.barostat = barostat

#     # Test that the fix styles returned from the OPENMM wrapper fixes attribute
#     # are correct
#     assert styles == ensemble.fix_styles or omp_styles == ensemble.fix_styles


# @pytest.mark.parametrize(
#     "thermostat, barostat, styles, omp_styles, attributes",
#     [
#         (None, None, ["nve"], ["OMP", "nve/omp"], {}),
#         (
#             "nose",
#             "nose",
#             ["npt"],
#             ["OMP", "npt/omp"],
#             {"temperature": 400.0, "t_damp": 100, "pressure": 10.0, "p_damp": 1000},
#         ),
#         (
#             "berendsen",
#             "nose",
#             ["temp/berendsen", "nph"],
#             ["OMP", "temp/berendsen", "nph/omp"],
#             {"temperature": 400.0, "t_damp": 100, "pressure": 10.0, "p_damp": 1000},
#         ),
#         (
#             "langevin",
#             "nose",
#             ["langevin", "nph"],
#             ["OMP", "langevin", "nph/omp"],
#             {"temperature": 400.0, "t_damp": 100, "pressure": 10.0, "p_damp": 1000},
#         ),
#         (
#             "rescale",
#             "nose",
#             ["temp/rescale", "nph"],
#             ["OMP", "temp/rescale", "nph/omp"],
#             {
#                 "temperature": 400.0,
#                 "t_fraction": 0.5,
#                 "t_window": 10.0,
#                 "rescale_step": 100,
#                 "pressure": 10.0,
#                 "p_damp": 1000,
#             },
#         ),
#         (
#             "nose",
#             "berendsen",
#             ["nvt", "press/berendsen"],
#             ["OMP", "nvt/omp", "press/berendsen"],
#             {"temperature": 400.0, "t_damp": 100, "pressure": 10.0, "p_damp": 1000},
#         ),
#         (
#             "berendsen",
#             "berendsen",
#             ["nve", "temp/berendsen", "press/berendsen"],
#             ["OMP", "nve/omp", "temp/berendsen", "press/berendsen"],
#             {"temperature": 400.0, "t_damp": 100, "pressure": 10.0, "p_damp": 1000},
#         ),
#         (
#             "langevin",
#             "berendsen",
#             ["nve", "langevin", "press/berendsen"],
#             ["OMP", "nve/omp", "langevin", "press/berendsen"],
#             {"temperature": 400.0, "t_damp": 100, "pressure": 10.0, "p_damp": 1000},
#         ),
#         (
#             "rescale",
#             "berendsen",
#             ["nve", "temp/rescale", "press/berendsen"],
#             ["OMP", "nve/omp", "temp/rescale", "press/berendsen"],
#             {
#                 "temperature": 400.0,
#                 "t_fraction": 0.5,
#                 "t_window": 10.0,
#                 "rescale_step": 100,
#                 "pressure": 10.0,
#                 "p_damp": 1000,
#             },
#         ),
#     ],
# )
# def test_apply_thermostat_barostat(ensemble, thermostat, barostat, styles, omp_styles, attributes):
#     """
#     Tests that applying both a thermostat and a barostat results in the correct
#     fixes being applied to OPENMM
#     """

#     # Set the attributes required by each thermostat/barostat pair
#     for attr, value in attributes.items():
#         setattr(ensemble, attr, value)

#     # Add the thermostat and barostat
#     ensemble.thermostat = thermostat
#     ensemble.barostat = barostat

#     # Test that the fix styles returned from the OPENMM wrapper fixes attribute
#     # are correct
#     assert styles == ensemble.fix_styles or omp_styles == ensemble.fix_styles


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


@pytest.mark.parametrize("value", [1.0, 5, -100, -13.0])
def test_convert_unit_no_unit(value):
    """
    Tests that converting a value without a unit just returns the value
    """

    assert value == openmm_eng.convert_unit(value)


@pytest.mark.parametrize(
    "unit_str, expected",
    [
        ("m", 1e10),
        ("nm", 10.0),
        ("Ang", 1.0),
        ("ns", 1e6),
        ("ps", 1e3),
        ("fs", 1.0),
        ("kg", 1 / CONST["_amu"]),
        ("g", 1 / (CONST["_amu"] * 1000)),
        ("amu", 1.0),
        ("g / mol", 1.0),
        ("J", CONST["_Nav"] / 1000.0),
        ("kJ", CONST["_Nav"]),
        ("kcal", CONST["_Nav"] * 4.184),
        ("kcal / Ang mol", 4.184),
        ("atm", 101325),
        ("bar", 1e5),
        ("rad", 180 / np.pi),
        ("deg", 1.0),
    ],
)
def test_convert_unit_conversion_factors(unit_str, expected):
    """
    Tests for correct conversion factors for conversion into MDMC units.
    """

    assert np.isclose(openmm_eng.convert_unit(1.0, units.Unit(unit_str), to_openmm=False), expected)


@pytest.mark.parametrize("value", [1.0, 2.0])
def test_convert_mdmc_base_units_identity(value):
    """
    Tests converting MDMC base units to OPENMM base units, where the units are
    the same
    """

    for unit in units.SYSTEM.values():
        if unit.components["numerator"][0] == unit and unit in openmm_eng.SYSTEM.values():
            assert openmm_eng.convert_unit(value, unit) == value


@pytest.mark.parametrize("value", [1.0, 2.0])
def test_convert_openmm_base_units_identity(value):
    """
    Tests converting OPENMM base units to MDMC base units, where the units are
    the same

    The same units are converted as in test_convert_mdmc_base_units_identity,
    except they are being converted from OPENMM to MDMC
    """

    for unit in openmm_eng.SYSTEM.values():
        if unit.components["numerator"][0] == unit and unit in units.SYSTEM.values():
            assert openmm_eng.convert_unit(value, unit, to_openmm=False) == value


@pytest.mark.parametrize(
    "mdmc_unit, openmm_value",
    [
        (units.Unit("Pa"), 1 / 101325.0),
        (units.Unit("kJ / mol"), 1 / 4.184),
        (units.Unit("kJ / Ang mol"), 1 / 4.184),
        (units.Unit("amu"), 1.0),
    ],
)
def test_convert_mdmc_base_units(mdmc_unit, openmm_value):
    """
    Tests converting MDMC base units to OPENMM base units, where the units are
    not the same in the two systems
    """

    assert np.isclose(openmm_eng.convert_unit(1.0, mdmc_unit), openmm_value)


@pytest.mark.parametrize(
    "openmm_unit, mdmc_value", [(units.Unit("atm"), 101325.0), (units.Unit("kcal / mol"), 4.184)]
)
def test_convert_openmm_base_units(openmm_unit, mdmc_value):
    """
    Tests converting OPENMM base units to MDMC base units, where the units are
    not the same in the two systems
    """

    assert np.isclose(openmm_eng.convert_unit(1.0, openmm_unit, to_openmm=False), mdmc_value)


@pytest.mark.parametrize(
    "mdmc_unit, openmm_value",
    [
        (units.Unit("kJ") / units.Unit("mol"), 4.184**-1),
        (units.Unit("Pa") * units.Unit("fs"), 101325.0**-1),
        (units.Unit("amu") ** 2, 1.0),  # mass units equiv
        (units.Unit("amu") ** -1, 1.0),  # mass units equiv,
        (units.SYSTEM["FORCE"], 4.184**-1),
    ],
)
def test_convert_mdmc_compound_units(mdmc_unit, openmm_value):
    """
    Tests converting between MDMC compound units (units made up of multiple base
    units)
    """

    assert np.isclose(openmm_eng.convert_unit(1.0, mdmc_unit), openmm_value)


@pytest.mark.parametrize("unit_str, conversion_factor", [("rad", 1.0), ("deg", 180 / np.pi)])
def test_convert_mdmc_angular_potential_strength(unit_str, conversion_factor):
    """
    Tests converting into OPENMM angular potential strength units for harmonic
    bond angles (which uses radians as the unit of angle rather than degrees)
    for MDMC units of both radians and degrees
    """

    mdmc_unit = units.SYSTEM["ENERGY"] / units.Unit(unit_str) ** 2
    openmm_value = (conversion_factor) ** 2 / 4.184
    assert np.isclose(openmm_eng.convert_unit(1.0, mdmc_unit), openmm_value)


@pytest.mark.skip("Broken")
# @pytest.mark.parametrize(
#     "openmm_unit, mdmc_value",
#     [
#         (units.Unit("kcal") / units.Unit("mol"), 4.184),
#         (units.Unit("atm") * units.Unit("fs"), 101325.0),
#         (units.Unit("bar") * units.Unit("fs"), 1e5),
#         (openmm_eng.SYSTEM["MASS"] ** 2, 1.0),  # mass units equiv
#         (openmm_eng.SYSTEM["MASS"] ** -1, 1.0),  # mass units equiv
#         (openmm_eng.SYSTEM["ENERGY"], 4.184),
#         (openmm_eng.SYSTEM["FORCE"], 4.184),
#     ],
# )
def test_convert_openmm_compound_units(openmm_unit, mdmc_value):
    """
    Tests converting between MDMC compound units (units made up of multiple base
    units)
    """

    assert np.isclose(openmm_eng.convert_unit(1.0, openmm_unit, to_openmm=False), mdmc_value)


def test_convert_mdmc_compound_equivalence():
    """
    Tests that converting an MDMC compound unit produces the same answer as
    performing the conversions individually
    """

    P = units.SYSTEM["PRESSURE"]
    E = units.SYSTEM["ENERGY"]

    assert np.isclose(
        openmm_eng.convert_unit(1.0, P / E),
        openmm_eng.convert_unit(1.0, P) / openmm_eng.convert_unit(1.0, E),
    )


@pytest.mark.parametrize(
    "unit, mag, power, to_openmm",
    [
        ("g / mol", 0, 1, False),
        ("mol / g", 0, 1, False),
        ("g / mol", 0, 3, False),
        ("mol / g", 0, 5, False),
    ],
)
def test_convert_mass_units_special_case(unit, mag, power, to_openmm):
    """
    Tests the various combinations of conversions amu <---> g / mol, the
    inverses, and different powers of units. In all cases, the values should
    be equal.
    """

    value = 5.67
    assert np.isclose(
        openmm_eng.convert_unit(
            units.UnitFloat(value, units.Unit(unit) ** power), to_openmm=to_openmm
        ),
        value * 10 ** (mag * power),
    )


def test_partition_single_interaction(interactions, bonds):
    """
    Tests using partition_interactions function to filter a single interaction
    name from a list
    """

    assert bonds == list(openmm_eng.partition_interactions(interactions, ["Bond"])[0])


def test_partition_multiple_interactions(interactions, bonds, angles, coulombics):
    """
    Tests using partition_interactions function to partition multiple
    interactions based on name
    """

    p_bonds, p_angles, p_coulombics = openmm_eng.partition_interactions(
        interactions, ["Bond", "BondAngle", "Coulombic"]
    )
    assert list(p_bonds) == bonds
    assert list(p_angles) == angles
    assert list(p_coulombics) == coulombics


def test_partition_interactions_unpartitioned(interactions, dispersions):
    """
    Tests that when unpartitioned=True is passed to partition_interactions, the
    final entry returned is all interactions in input that did not have a name
    in the names argument
    """

    _, _, _, _, p_disps = openmm_eng.partition_interactions(
        interactions, ["Bond", "BondAngle", "Coulombic", "DihedralAngle"], unpartitioned=True
    )
    assert list(p_disps) == dispersions


def test_partion_interactions_return_list(interactions, bonds, angles):
    """
    Tests that when lst=True is passed to partition_interactions, a tuple of
    lists is returned, rather than a tuple of generators
    """

    assert (bonds, angles) == openmm_eng.partition_interactions(
        interactions, ["Bond", "BondAngle"], lst=True
    )


def test_warn_on_invalid_run(populated_simulation):
    """
    Tests that a warning is issued when attempting to run a openmm
    simulation shorter than ``traj_step``
    """

    simulation.traj_step = 10
    openmm_engine.simulation = populated_simulation
    with pytest.warns(UserWarning, match="run may not produce usable output"):
        simulation.run(n_steps=3)
