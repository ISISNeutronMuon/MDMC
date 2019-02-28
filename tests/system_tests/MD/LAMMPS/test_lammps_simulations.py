"""System tests for LAMMPS MD simulations

Compares the thermodynamic and simulation properties calculated from the MDMC
run using LAMMPS with the same properties calculated from an equivalent LAMMPS
setup run externally. This occurs for NVE, NVT and NPT ensembles.  The
calculations of the properties in both cases are performed by LAMMPS, the only
difference is whether the LAMMPS simulation was run through MDMC.

AUTHOR :    Thomas Farmer        START DATE :    22/02/2019, 13:50:29"""

import numpy as np
import pytest

from MDMC.MD.simulation import Universe, Simulation, Shake, PPPM
from MDMC.MD.structural_units import Atom, Bond, BondAngle, Coulombic, \
    Dispersion, Molecule

N_MOLECULES = 216
DIMENSION = 18.60
TEMPERATURE = 300.
# Number of steps between logging of thermo_style variables
THERMO_STEPS = 100
EQUILIBRIUM_STEPS = 10000
MD_STEPS = 20000

# Each EXPECTED dictionary contains all of the required properties as keys. The
# corresponding values are a tuple of (mean value, standard deviation),
# where both the mean value and the standard deviation have been calculated from
# 10 repeats (with different random velocity seeds) of an external LAMMPS
# simulation with the same simulation parameters.
#
# The NVE temperature differs from the set value due to the effects of SHAKE
NVE_EXPECTED = {'Atoms':(N_MOLECULES*3, 0), 'Bonds':(N_MOLECULES*2, 0),
                'Angles':(N_MOLECULES, 0), 'KinEng':(1520.31, 5.4),
                'PotEng':(-1236.56, 4.5), 'Temp':(1183.37, 4.2),
                'Press':(18177.22, 155.9), 'Volume':(DIMENSION**3, 0),
                'E_bond':(0, 0), 'E_angle':(0, 0), 'E_vdwl':(393.75, 3.6),
                'E_coul':(11596.25, 5.6), 'E_long':(-13226.57, 0.32),
                'Nbuild':(920.03, 3.8), 'Ndanger':(0, 0)}

NVT_EXPECTED = {'Atoms':(N_MOLECULES*3, 0), 'Bonds':(N_MOLECULES*2, 0),
                'Angles':(N_MOLECULES, 0), 'KinEng':(385.43, 0.98),
                'PotEng':(-2408.66, 4.2), 'Temp':(300.01, 0.77),
                'Press':(168.6, 84.6), 'Volume':(DIMENSION**3, 0),
                'E_bond':(0, 0), 'E_angle':(0, 0), 'E_vdwl':(457.47, 2.6),
                'E_coul':(10388.92, 5.6), 'E_long':(-13255.06, 0.11),
                'Nbuild':(384.18, 3.4), 'Ndanger':(0, 0)}

NPT_EXPECTED = {'Atoms':(N_MOLECULES*3, 0), 'Bonds':(N_MOLECULES*2, 0),
                'Angles':(N_MOLECULES, 0), 'KinEng':(384.85, 0.9),
                'PotEng':(-2408.42, 6.4), 'Temp':(299.56, 0.70),
                'Press':(3.87, 31.5), 'Volume':(6478.08, 30.3),
                'E_bond':(0, 0), 'E_angle':(0, 0), 'E_vdwl':(455.36, 3.2),
                'E_coul':(10390.32, 18.4), 'E_long':(-13254.11, 13.5),
                'Nbuild':(415.84, 2.7), 'Ndanger':(0, 0)}

NVE_UNCONSTRAINED_EXPECTED = {'Atoms':(N_MOLECULES*3, 0),
                              'Bonds':(N_MOLECULES*2, 0),
                              'Angles':(N_MOLECULES, 0),
                              'KinEng':(1520.31, 5.4),
                              'PotEng':(-1236.56, 4.5),
                              'Temp':(1183.37, 4.2),
                              'Press':(18177.22, 155.9),
                              'Volume':(DIMENSION**3, 0),
                              'E_bond':(0, 0),
                              'E_angle':(0, 0),
                              'E_vdwl':(393.75, 3.6),
                              'E_coul':(11596.25, 5.6),
                              'E_long':(-13226.57, 0.32),
                              'Nbuild':(920.03, 3.8),
                              'Ndanger':(0, 0)}


# Use module scope so that the simulation only runs once for all functions
@pytest.fixture(scope="module")
def universe():

    """
    Returns:
    An MDMC simulation object setup to run an NVE simulation of 216 SPCE water
    molecules at 300K using LAMMPS
    """

    universe = Universe(dimensions=DIMENSION)
    H1 = Atom('H')
    H2 = Atom('H', position=(0., 1.63298, 0.))
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
    O_dispersion = Dispersion(universe, O.atom_type, cutoff=10.,
                              vdw_tail_correction=True)
    universe.add_force_field('SPCE')

    return universe

@pytest.fixture(scope="module")
def NVE(universe):

    """
    Returns:
    An MDMC simulation object setup to run an NVE simulation of 216 SPCE water
    molecules at 300K using LAMMPS
    """

    md_engine = Simulation(universe,
                           engine='lammps',
                           time_step=1.,
                           temperature=TEMPERATURE,
                           traj_step=10)

    # Manually select which properties to output from LAMMPS
    set_thermo_style(md_engine)

    md_engine.run(EQUILIBRIUM_STEPS)
    md_engine.run(MD_STEPS)
    return md_engine


@pytest.fixture(scope="module")
def NVT(universe):

    """
    Returns:
    An MDMC simulation object setup to run an NVT simulation of 216 SPCE water
    molecules at 300K using LAMMPS
    """

    md_engine = Simulation(universe,
                           engine='lammps',
                           time_step=1.,
                           temperature=TEMPERATURE,
                           thermostat='nose',
                           traj_step=10)

    # Manually select which properties to output from LAMMPS
    set_thermo_style(md_engine)

    md_engine.run(EQUILIBRIUM_STEPS)
    md_engine.run(MD_STEPS)
    return md_engine


@pytest.fixture(scope="module")
def NPT(universe):

    """
    Returns:
    An MDMC simulation object setup to run an NPT simulation of 216 SPCE water
    molecules at 300K using LAMMPS
    """

    md_engine = Simulation(universe,
                           engine='lammps',
                           time_step=1.,
                           temperature=TEMPERATURE,
                           pressure=101325.,
                           thermostat='nose',
                           barostat='nose',
                           p_damp=100,
                           traj_step=10)

    # Manually select which properties to output from LAMMPS
    set_thermo_style(md_engine)

    md_engine.run(EQUILIBRIUM_STEPS)
    md_engine.run(MD_STEPS)
    return md_engine


@pytest.fixture(scope="module")
def NVE_unconstrained(universe):

    """
    Returns:
    An MDMC simulation object setup to run an NVE simulation of 216 SPCE water
    molecules at 300K using LAMMPS, without constrained bonds or bond angles
    """

    for interaction in universe.bonded_interactions:
        interaction.constrained = False

    # Reduced time_step is due to removal of constraints
    md_engine = Simulation(universe,
                           engine='lammps',
                           time_step=0.1,
                           temperature=TEMPERATURE,
                           traj_step=10)

    # Manually select which properties to output from LAMMPS
    set_thermo_style(md_engine)

    md_engine.run(EQUILIBRIUM_STEPS)
    md_engine.run(MD_STEPS)
    return md_engine


def parameterize_decorator(func):

    @pytest.mark.parametrize('ensemble, expected',
                             [('NVE', NVE_EXPECTED),
                              ('NVT', NVT_EXPECTED),
                              ('NPT', NPT_EXPECTED),
                              ('NVE_unconstrained', NVE_UNCONSTRAINED_EXPECTED)]
                            )
    def wrapper(ensemble, expected, request):
        func(ensemble, expected, request)

    return wrapper


@parameterize_decorator
def test_number_atoms(ensemble, expected, request):

    """
    Compare the total number of atoms in the simulation with that calculated
    directly from LAMMPS
    """

    assert_property(ensemble, expected, request, 'Atoms')


@parameterize_decorator
def test_number_bonds(ensemble, expected, request):

    """
    Compare the total number of bonds in the simulation with that calculated
    directly from LAMMPS
    """

    assert_property(ensemble, expected, request, 'Bonds')


@parameterize_decorator
def test_number_angles(ensemble, expected, request):

    """
    Compare the total number of angles in the simulation with that calculated
    directly from LAMMPS
    """

    assert_property(ensemble, expected, request, 'Angles')


@parameterize_decorator
def test_kinetic_energy(ensemble, expected, request):

    """
    Compare the kinetic energy with that calculated directly from LAMMPS
    """

    assert_property(ensemble, expected, request, 'KinEng')


@parameterize_decorator
def test_potential_energy(ensemble, expected, request):

    """
    Compare the potential energy with that calculated directly from LAMMPS
    """

    assert_property(ensemble, expected, request, 'PotEng')


@parameterize_decorator
def test_temperature(ensemble, expected, request):

    """
    Compare the temperature with that calculated directly from LAMMPS
    """

    assert_property(ensemble, expected, request, 'Temp')


@parameterize_decorator
def test_pressure(ensemble, expected, request):

    """
    Compare the pressure with that calculated directly from LAMMPS
    """

    assert_property(ensemble, expected, request, 'Press')


@parameterize_decorator
def test_volume(ensemble, expected, request):

    """
    Compare the simulation box volume with that calculated directly from LAMMPS
    """

    assert_property(ensemble, expected, request, 'Volume')


@parameterize_decorator
def test_bond_energy(ensemble, expected, request):

    """
    Compare the total energy of all bonds with that calculated directly from
    LAMMPS
    """

    assert_property(ensemble, expected, request, 'E_bond')


@parameterize_decorator
def test_angle_energy(ensemble, expected, request):

    """
    Compare the total energy of all bond angle with that calculated directly
    from LAMMPS
    """

    assert_property(ensemble, expected, request, 'E_angle')


@parameterize_decorator
def test_vdw_energy(ensemble, expected, request):

    """
    Compare the total energy of the dispersive interactions with that calculated
    directly from LAMMPS
    """

    assert_property(ensemble, expected, request, 'E_vdwl')


@parameterize_decorator
def test_coul_energy(ensemble, expected, request):

    """
    Compare the total energy of the coulombic interactions with that calculated
    directly from LAMMPS
    """

    assert_property(ensemble, expected, request, 'E_coul')


@parameterize_decorator
def test_kspace_correction_energy(ensemble, expected, request):

    """
    Compare the total energy of the kspace correction with that calculated
    directly from LAMMPS
    """

    assert_property(ensemble, expected, request, 'E_long')


@parameterize_decorator
def test_neighbor_builds(ensemble, expected, request):

    """
    Compare the number of times the neighbor list was built
    """

    assert_property(ensemble, expected, request, 'Nbuild')


@parameterize_decorator
def test_dangerous_neighbor_builds(ensemble, expected, request):

    """
    Compare the number of times a neighbor list build was dangerous
    """

    assert_property(ensemble, expected, request, 'Ndanger')


def set_thermo_style(sim):

    """
    Applies a LAMMPS thermo_style to the LAMMPS wrapper in the MDMC Simulation
    object so that the required properties can be determined
    """

    sim.engine.lmp.thermo_style('custom', 'step', 'temp', 'press', 'ke', 'pe',
                                'atoms', 'bonds', 'angles', 'nbuild', 'ndanger',
                                'vol', 'evdwl', 'ecoul', 'elong', 'ebond',
                                'eangle')
    # Set number of steps between logging thermo_style variables
    sim.engine.lmp.thermo(THERMO_STEPS)


def average_property(sim, prop):

    """
    Averages the property over all of the steps in the simulation

    Arguments:
    sim - a Simulation object
    prop - a string specifying a LAMMPS simulation thermo_style property

    Returns:
    A float average of all of the values of prop during the simulation run
    """

    # runs[1] is the thermo_styles properties from the second time the run
    # method of LAMMPS wrapper is called - this is the production run (index 0
    # is the equilibration run)
    return np.mean(getattr(sim.engine.lmp.runs[1].thermo, prop))


def assert_property(ensemble, expected, request, property):

    """
    Performs an assertion on a property using an ensemble returned using request
    """

    # As fixtures cannot be included in parameterization, the names of the
    # fixtures are included instead - the return values of the fixtures are then
    # recovered using request.getfixturevalue
    average = average_property(request.getfixturevalue(ensemble), property)

    # expected[property][1] is the standard_deviation of the property. The
    # absolute tolerance is set to 3 times this value.
    # Small relative tolerance accounts for rounding differences
    assert np.allclose(average, expected[property][0],
                       atol=expected[property][1]*3, rtol=1e-8)
