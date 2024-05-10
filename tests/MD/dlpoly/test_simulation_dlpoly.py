"""
Tests for setting up and running MDMC using DLPOLY infrastructure.
"""

# pylint: disable=redefined-outer-name

import numpy as np
import pytest

from MDMC.common import units
from MDMC.common.units import UnitNDArray
from MDMC.MD.engine_facades import dlpoly_engine
from MDMC.MD.interaction_functions import (Buckingham, Coulomb,
                                           HarmonicPotential, LennardJones,
                                           Periodic)
from MDMC.MD.interactions import (Bond, BondAngle, Coulombic, DihedralAngle,
                                  Dispersion)
from MDMC.MD.simulation import Simulation, Universe
from MDMC.MD.structures import Atom

CUTOFF = 3.14
COUL_CUTOFF = 8.0
DISP_CUTOFF = 10.0
N_ATOMS = 10
UNIVERSE_DIM = UnitNDArray((3, ), "Ang")
UNIVERSE_DIM[:] = 50.0
CONST = units.CODATA[units.CODATA_VERSION]

############
# Fixtures #
############


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
    DLPOLY, as this is done ordered by atom_type, rather than necessary the
    order which atoms appear in universe.atoms
    """

    symbols = ['C', 'H', 'N', 'O']
    masses = [12.011, 1.008, 14.007, 16.000]
    elements = symbols * (N_ATOMS // 4)
    elements[len(elements):N_ATOMS] = symbols[:N_ATOMS-len(elements)]
    # Sorted so that atoms of same type are grouped
    elements = sorted(elements)
    atom_types = {symbol: n for n, symbol in enumerate(symbols, 1)}
    atom_masses = dict(zip(symbols, masses))

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
    improper_harmonic = HarmonicPotential(1.0, 0.0002, interaction_type='improper')

    # Create 2 bonds for some atoms, and one angle, coulombic and dispersive
    # interaction
    bond1_atoms = [(atoms[i], atoms[i+1]) for i in range(0, len(atoms)-1, 2)]
    bond2_atoms = [(atoms[i], atoms[i+2]) for i in range(0, len(atoms)-2, 3)]
    bonds = [Bond(*bond1_atoms, function=bond1_harmonic),
             Bond(*bond2_atoms, function=bond2_harmonic)]
    angles = [BondAngle(*[(atoms[i], atoms[i+1], atoms[i+2])
                          for i in range(0, len(atoms)-2, 3)],
                        function=angle_harmonic)]
    propers = [DihedralAngle(tuple(atoms[:4]), function=proper_periodic, improper=False)]
    impropers = [DihedralAngle(tuple(atoms[:4]), function=improper_harmonic, improper=True)]
    coulombics = [Coulombic(empty_universe, atom_types=type_,
                            function=Coulomb(-1.0+type_*0.5), cutoff=COUL_CUTOFF)
                  for type_ in empty_universe.atom_types]
    dispersions = []
    for type_ in empty_universe.atom_types:
        dispersions.append(Dispersion(empty_universe, (type_, type_),
                                      function=Buckingham(type_ * 0.1,
                                                          type_ * 1.0,
                                                          type_ * 2.0),
                                      cutoff=DISP_CUTOFF,
                                      vdw_tail_correction=True))
        dispersions.append(Dispersion(empty_universe, (type_, type_),
                                      function=LennardJones(type_*0.1,
                                                            type_*1.0),
                                      cutoff=DISP_CUTOFF,
                                      vdw_tail_correction=True))

    return (empty_universe, bonds, angles, propers, impropers,
            coulombics, dispersions)


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
def dlpoly_universe(universe):
    """
    Returns:
    A DLPOLYUniverse where the atomic configuration and the topology have been
    added
    """

    dlpoly_universe = dlpoly_eng.DLPOLYUniverse(universe)
    return dlpoly_universe


@pytest.fixture
def dlpoly_simulation(universe):
    """
    Returns:
    A DLPOLYSimulation where the simulation parameters have been set. The
    dlpoly-py wrapper belonging to this DLPolySimulation does not have an atomic
    configuration or topology, and so it not ready to run DLPOLY.
    """

    # Simulation setup requires the traj_step attribute to be set. All other
    # attributes that are required are set to defaults.
    dlpoly_simulation = dlpoly_eng.DLPOLYSimulation(universe, traj_step=10)
    return dlpoly_simulation


@pytest.fixture
def populated_dlpoly_simulation(universe, dlpoly_universe):
    """
    Returns:
    A DLPOLYSimulation which has a dlpoly-py wrapper where the atomic
    configuration and the topology have been added, and the simulation
    parameters have been set. The dlpoly-py wrapper is ready to run a DLPOLY
    simulation.
    """

    dlpoly_simulation = dlpoly_eng.DLPOLYSimulation(universe,
                                                    traj_step=10,
                                                    time_step=1.,
                                                    lmp=dlpoly_universe.dlpoly)
    return dlpoly_simulation


@pytest.fixture
def ensemble(populated_dlpoly_simulation):

    """
    Returns:
    An Ensemble which has a dlpoly-py wrapper where the atomic
    configuration and the topology have been added, and the simulation
    parameters have been set. This is required for thermostat and barostats to
    be added to the dlpoly-py wrapper through the ensemble.
    """
    populated_dlpoly_simulation.lin_momentum_steps = None
    return dlpoly_eng.DLPOLYEnsemble(populated_dlpoly_simulation.dlpoly,
                                     time_step=1.)


@pytest.fixture
def simulation(universe):
    """
    A mock simulation to give the engine facade its necessary 'parent simulation'
    """
    return Simulation(universe, traj_step=1, time_step=1., engine='dlpoly')


@pytest.fixture
def dlpoly_eng(universe, simulation):

    """
    Returns:
    A DLPOLYEngine which is ready to run a DLPOLY simulation with an NVE
    ensemble.
    """
    engine = dlpoly_engine.DLPOLYEngine()
    engine.parent_simulation = simulation
    engine.setup_universe(universe)
    engine.setup_simulation()
    return engine


######################
# DLPOLYEngine Tests #
######################

@pytest.mark.parametrize("attr, val", (("temperature", 5),
                                       ("pressure", 20),
                                       ("ensemble", 'nvt')),
                         ids=["Error with temperature Getter/Setter",
                              "Error with pressure Getter/Setter",
                              "Error with ensemble Getter/Setter",
                              ])
def test_attr_set_get(dlpoly_eng, attr, val):
    """
    Test DLPoly params are passed through correctly
    """
    setattr(dlpoly_eng, attr, val)
    assert getattr(dlpoly_eng, attr) == val


@pytest.mark.parametrize("attr, val", [("thermostat", 'langevin'),
                                       ("barostat", 'andersen')],
                         ids=["Error with thermostat Getter/Setter",
                              "Error with barostat Getter/Setter"])
def test_barostat_set_get(dlpoly_eng, attr, val):
    """
    Test baro/thermostats are passed through correctly
    """
    dlpoly_eng.pressure = 20
    dlpoly_eng.temperature = 5

    setattr(dlpoly_eng, attr, val)
    assert getattr(dlpoly_eng, attr) == val


traj_hist_one_step = '''CONFIG generated by ASE
         0         3         2                    1                   10
timestep         0         2 0 3           10.188930            0.000000
        1.0000000000        2.0000000000        3.0000000000
        4.0000000000        5.0000000000        6.0000000000
        7.0000000000        8.0000000000        9.0000000000
Ar               1      1.000000      0.000000      0.000000
     1.000000000         2.000000000         3.000000000
Ne               2      2.000000      0.000000      0.000000
     4.000000000         5.000000000         6.000000000
'''


traj_hist_two_steps = '''CONFIG generated by ASE
         0         3         2                    3                   26
timestep         0         2 0 3           10.188930            0.000000
        1.0000000000        2.0000000000        3.0000000000
        4.0000000000        5.0000000000        6.0000000000
        7.0000000000        8.0000000000        9.0000000000
Ar               1      1.000000      0.000000      0.000000
     3.000000000         4.000000000         5.000000000
Ne               2      2.000000      0.000000      0.000000
     6.000000000         7.000000000         8.000000000
timestep         1         2 0 3           10.188930            10.188930
        1.0000000000        2.0000000000        3.0000000000
        4.0000000000        5.0000000000        6.0000000000
        7.0000000000        8.0000000000        9.0000000000
Ar               1      1.000000      0.000000      0.000000
     2.000000000         3.000000000         4.000000000
Ne               2      2.000000      0.000000      0.000000
     5.000000000         6.000000000         7.000000000
timestep         2         2 0 3           10.188930            20.37786
        1.0000000000        2.0000000000        3.0000000000
        4.0000000000        5.0000000000        6.0000000000
        7.0000000000        8.0000000000        9.0000000000
Ar               1      1.000000      0.000000      0.000000
     1.000000000         2.000000000         3.000000000
Ne               2      2.000000      0.000000      0.000000
     4.000000000         5.000000000         6.000000000
'''


@pytest.mark.parametrize("traj_hist_n_steps, n_steps, position",
                         [(traj_hist_one_step, 1, [[[1., 2., 3.], [4., 5., 6.]]]),
                          (traj_hist_two_steps, 3, [[[3., 4., 5.], [6., 7., 8.]],
                                                    [[2., 3., 4.], [5., 6., 7.]],
                                                    [[1., 2., 3.], [4., 5., 6.]]])],
                         ids=["Error with convert trajectory for one step.",
                              "Error with convert trajectory for two steps."])
def test_convert_trajectory(tmp_path, dlpoly_eng, traj_hist_n_steps, n_steps, position):
    """
    Test trajectory converter handles this correctly.
    """

    traj_file = tmp_path / "traj"
    traj_file.write_text(traj_hist_n_steps)

    dlpoly_eng.dlpoly.control['io_file_history'] = traj_file

    dlpoly_eng.universe = None

    traj = dlpoly_eng.convert_trajectory()

    for attr, value, err in [("n_atoms", 2, "Incorrect n_atoms."),
                             ("n_steps", n_steps, "Incorrect n_steps."),
                             ("atom_types", [1, 2], "Incorrect atom_types."),
                             ("atom_masses", [1., 2.], "Incorrect atom_masses."),
                             ("atom_charges", [0., 0.], "Incorrect atom_charges."),
                             ("position", position, "Incorrect position.")]:
        assert np.all(getattr(traj, attr) == value), err

# TODO: setup_universe, setup_simulation, update_parameter, save_config, reset_config
