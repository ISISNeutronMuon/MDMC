"""
System tests for LAMMPS MD simulations using Buckingham potential interactions

Compares the thermodynamic and simulation properties calculated from the MDMC
run using LAMMPS with the same properties calculated from an equivalent LAMMPS
setup run externally. This occurs for NVE, NVT and NPT ensembles.  The
calculations of the properties in both cases are performed by LAMMPS, the only
difference is whether the LAMMPS simulation was run through MDMC.
"""

import numpy as np
import pytest

from MDMC.MD.simulation import Universe, Simulation, Shake, PPPM
from MDMC.MD.structures import Atom, Molecule
from MDMC.MD.interactions import Bond, BondAngle, Dispersion, Coulombic
from MDMC.MD.interaction_functions import Buckingham

pytestmark = [pytest.mark.mpi, pytest.mark.lammps]

""" 
STDEV_FAC is the number of standard deviations within which the calculated
property must lie for it to be considered equivalent to the expected value
i.e. it is the tolerance of the assertion on the property 
"""
STDEV_FAC = 4.
N_MOLECULES = 216
DIMENSION = 18.60
TEMPERATURE = 300.

# Number of steps between logging of thermo_style variables
THERMO_STEPS = 100
EQUILIBRIUM_STEPS = 10000
MD_STEPS = 20000

"""Each EXPECTED dictionary contains all of the required properties as keys. The
corresponding values are a tuple of (mean value, standard deviation),
where both the mean value and the standard deviation have been calculated from
20 repeats (with different random velocity seeds) of an external LAMMPS
simulation with the same simulation parameters.

The NVE temperature differs from the set value due to the effects of SHAKE"""

NVE_EXPECTED = {'Atoms': (N_MOLECULES*3, 0.0),
                'Bonds': (N_MOLECULES*2, 0.0),
                'Angles': (N_MOLECULES, 0.0),
                'KinEng': (1442.2, 4.32),
                'PotEng': (-223.61, 2.94),
                'Temp': (1122.57, 3.36),
                'Press': (54231.3, 106.51),
                'Volume': (DIMENSION**3, 0.0),
                'E_bond': (0.0, 0.0),
                'E_angle': (0.0, 0.0),
                'E_vdwl': (1041.53, 3.1),
                'E_coul': (11963.71, 3.17),
                'E_long': (-13228.85, 0.37),
                'Nbuild': (900.96, 3.9),
                'Ndanger': (0.0, 0.0)}

NVT_EXPECTED = {'Atoms': (N_MOLECULES*3, 0.0),
                'Bonds': (N_MOLECULES*2, 0.0),
                'Angles': (N_MOLECULES, 0.0),
                'KinEng': (382.67, 1.27),
                'PotEng': (-1164.64, 2.44),
                'Temp': (297.86, 0.99),
                'Press': (25978.54, 124.53),
                'Volume': (DIMENSION**3, 0.0),
                'E_bond': (0.0, 0.0),
                'E_angle': (0.0, 0.0),
                'E_vdwl': (535.22, 4.01),
                'E_coul': (11554.9, 3.13),
                'E_long': (-13254.76, 0.11),
                'Nbuild': (431.76, 2.54),
                'Ndanger': (0.0, 0.0)}

NPT_EXPECTED = {'Atoms': (N_MOLECULES*3, 0.0),
                'Bonds': (N_MOLECULES*2, 0.0),
                'Angles': (N_MOLECULES, 0.0),
                'KinEng': (382.97, 0.83),
                'PotEng': (-1089.87, 2.95),
                'Temp': (298.09, 0.65),
                'Press': (-1.29, 43.98),
                'Volume': (11983.8, 94.82),
                'E_bond': (0.0, 0.0),
                'E_angle': (0.0, 0.0),
                'E_vdwl': (55.13, 2.61),
                'E_coul': (11918.41, 134.36),
                'E_long': (-13063.41, 133.72),
                'Nbuild': (491.63, 2.73),
                'Ndanger': (0.0, 0.0)}

NVE_UNCONSTRAINED_EXPECTED = {'Atoms': (N_MOLECULES*3, 0.0),
                              'Bonds': (N_MOLECULES*2, 0.0),
                              'Angles': (N_MOLECULES, 0.0),
                              'KinEng': (1624.92, 3.63),
                              'PotEng': (-86.4, 3.64),
                              'Temp': (842.55, 1.88),
                              'Press': (50218.52, 368.08),
                              'Volume': (DIMENSION ** 3, 0.0),
                              'E_bond': (126.85, 3.08),
                              'E_angle': (272.07, 5.16),
                              'E_vdwl': (1018.19, 11.41),
                              'E_coul': (11727.88, 9.61),
                              'E_long': (-13231.39, 0.68),
                              'Nbuild': (93.29, 1.07),
                              'Ndanger': (0.0, 0.0)}


# Use module scope so that the simulation only runs once for all functions
@pytest.fixture(scope="module")
def universe():
    """
    Returns
    -------
    Universe
        A `Universe` object setup to run an NVE simulation of 216 SPCE water
        molecules at 300K using LAMMPS.
        The interaction potential used is the Buckingham potential.
    """

    universe = Universe(dimensions=DIMENSION)
    H1 = Atom('H')
    H2 = Atom('H', position=(0., 1.63298, 0.))
    O = Atom('O', position=(0., 0.81649, 0.57736))
    Coulombic(atoms=[H1, H2], cutoff=10.)
    Coulombic(atoms=O, cutoff=10.)
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
    universe.add_force_field('SPCE')

    """
    The following Buckingham potential parameters were first derived from rearranging the equations
    and given parameters at: https://water.lsbu.ac.uk/water/water_models.html#af.
    
    These were then manually adjusted "by eye" to graphically "fit" that of the Lennard-Jones
    potential in the 3-12 angstrom range. (Hence the expected values should be similar to that 
    of Lennard-Jones, but not identical)
    
    The values have been rounded to 2 d.p. for readability
    """
    buck = Buckingham(1194446.57, 3.67, 4914.96)
    Dispersion(universe, (O.atom_type, O.atom_type), cutoff=10.,
               vdw_tail_correction=True, function=buck)

    return universe


@pytest.fixture(scope="module")
def NVE(universe):
    """
    Returns
    -------
    Simulation
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
    Returns
    -------
    Simulation
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
    Returns
    -------
    Simulation
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
    Returns
    -------
    Simulation
        An MDMC simulation object setup to run an NVE simulation of 216 SPCE water
        molecules at 300K using LAMMPS, without constrained bonds or bond angles
    """

    # Remove constraints from bonds and angles and set potential strengths for
    # those interactions according to SPC/Fd water model
    for interaction in universe.bonded_interactions:
        interaction.constrained = False
        for parameter in interaction.parameters.filter_name("potential_strength"):
            if interaction.name == 'Bond':
                interaction.parameters[parameter].value = 4410.7728 / 2
            elif interaction.name == 'BondAngle':
                interaction.parameters[parameter].value = 158.7828
    # Remove constraint algorithm from universe
    universe.constraint_algorithm = None

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
    """A decorator for parametrizing all tests with each ensemble."""

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
    """Compare the kinetic energy with that calculated directly from LAMMPS"""

    assert_property(ensemble, expected, request, 'KinEng')


@parameterize_decorator
def test_potential_energy(ensemble, expected, request):
    """Compare the potential energy with that calculated directly from LAMMPS"""

    assert_property(ensemble, expected, request, 'PotEng')


@parameterize_decorator
def test_temperature(ensemble, expected, request):
    """Compare the temperature with that calculated directly from LAMMPS"""

    assert_property(ensemble, expected, request, 'Temp')


@parameterize_decorator
def test_pressure(ensemble, expected, request):
    """Compare the pressure with that calculated directly from LAMMPS"""

    assert_property(ensemble, expected, request, 'Press')


@parameterize_decorator
def test_volume(ensemble, expected, request):
    """Compare the simulation box volume with that calculated directly from LAMMPS"""

    assert_property(ensemble, expected, request, 'Volume')


@parameterize_decorator
def test_bond_energy(ensemble, expected, request):
    """Compare the total energy of all bonds with that calculated directly from LAMMPS"""

    assert_property(ensemble, expected, request, 'E_bond')


@parameterize_decorator
def test_angle_energy(ensemble, expected, request):
    """Compare the total energy of all bond angle with that calculated directly from LAMMPS"""

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
    """Compare the number of times the neighbor list was built"""

    assert_property(ensemble, expected, request, 'Nbuild')


@parameterize_decorator
def test_dangerous_neighbor_builds(ensemble, expected, request):
    """Compare the number of times a neighbor list build was dangerous"""

    assert_property(ensemble, expected, request, 'Ndanger')


def set_thermo_style(sim):
    """
    Applies a LAMMPS thermo_style to the LAMMPS wrapper in the MDMC Simulation
    object so that the required properties can be determined

    Parameters
    ----------
    sim : Simulation
        An MDMC Simulation object
    """

    sim.engine.lmp.thermo_style('custom', 'step', 'temp', 'press', 'ke', 'pe',
                                'atoms', 'bonds', 'angles', 'nbuild', 'ndanger',
                                'vol', 'evdwl', 'ecoul', 'elong', 'ebond',
                                'eangle')
    # Set number of steps between logging thermo_style variables
    sim.engine.lmp.thermo(THERMO_STEPS)


def average_property(sim, prop):
    """
    Averages the property over all the steps in the simulation

    Parameters
    ----------
    sim : Simulation
        An MDMC Simulation object
    prop : str
        A string specifying a LAMMPS simulation thermo_style property

    Returns
    -------
    float
        An average of all values of prop during the simulation run
    """

    """runs[1] is the thermo_styles properties from the second time the run
    method of LAMMPS wrapper is called - this is the production run (index 0
    is the equilibration run)"""
    return np.mean(getattr(sim.engine.lmp.runs[1].thermo, prop))


def assert_property(ensemble, expected, request, prop):
    """
    Performs an assertion on a property using an ensemble returned using request

    Parameters
    ----------
    ensemble : Simulation
        A simulation object fixture (e.g. NVE, NPT)
    expected : dict
        a dictionary where key is a string with the thermodynamic/simulation property name and
        the value is the expected value of that property
    request : pytest.Request
        A pytest request object
    prop : str
        a string with the thermodynamic/simulation property to be tested
    """

    # As fixtures cannot be included in parameterization, the names of the
    # fixtures are included instead - the return values of the fixtures are then
    # recovered using request.getfixturevalue
    average = average_property(request.getfixturevalue(ensemble), prop)

    # expected[property][1] is the standard_deviation of the property. The
    # absolute tolerance is set to STDEV_FAC times this value.
    # Small relative tolerance accounts for rounding differences
    assert np.allclose(average, expected[prop][0],
                       atol=expected[prop][1] * STDEV_FAC, rtol=1e-8)
