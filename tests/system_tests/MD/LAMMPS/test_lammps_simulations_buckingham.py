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
10 repeats (with different random velocity seeds) of an external LAMMPS
simulation with the same simulation parameters.

The NVE temperature differs from the set value due to the effects of SHAKE"""
NVE_EXPECTED = {'Atoms': (648.0, 0.0), 'Bonds': (432.0, 0.0), 'Angles': (216.0, 0.0),
                'KinEng': (1416.746153880597, 3.7354172931461607),
                'PotEng': (-386.9053971741294, 2.6771308495720425),
                'Temp': (1102.7586426865669, 2.9075524765940672),
                'Press': (48792.35475273632, 129.3849406201805), 'Volume': (6434.856, 0.0),
                'E_bond': (0.0, 0.0), 'E_angle': (0.0, 0.0),
                'E_vdwl': (907.2063949800993, 4.625442989406503),
                'E_coul': (11935.364917910449, 3.505790583024876),
                'E_long': (-13229.476718905475, 0.40487535702667377),
                'Nbuild': (891.4741293532339, 3.1977582817133685), 'Ndanger': (0.0, 0.0)}

NVT_EXPECTED = {'Atoms': (648.0, 0.0), 'Bonds': (432.0, 0.0), 'Angles': (216.0, 0.0),
                'KinEng': (382.4499767611941, 0.7109164462250214),
                'PotEng': (-1296.7771849751243, 1.6332759412816056),
                'Temp': (297.6891917064676, 0.5533589910133256),
                'Press': (21835.118670646763, 95.76330891271319), 'Volume': (6434.856, 0.0),
                'E_bond': (0.0, 0.0), 'E_angle': (0.0, 0.0),
                'E_vdwl': (443.0671263134329, 3.5889789266942356),
                'E_coul': (11514.907874626868, 3.7871089328673055),
                'E_long': (-13254.752188557211, 0.1362245168071942),
                'Nbuild': (430.2179104477612, 2.1080263646549375), 'Ndanger': (0.0, 0.0)}

NPT_EXPECTED = {'Atoms': (648.0, 0.0), 'Bonds': (432.0, 0.0), 'Angles': (216.0, 0.0),
                'KinEng': (383.0610417761194, 0.8722296098018337),
                'PotEng': (-1193.2133119900498, 2.59987021422615),
                'Temp': (298.1648289900498, 0.6789209813207984),
                'Press': (-3.4173811777860665, 43.05066881088412),
                'Volume': (10981.601679104477, 54.8575141465732), 'E_bond': (0.0, 0.0),
                'E_angle': (0.0, 0.0), 'E_vdwl': (56.49998219945273, 2.0734293492831317),
                'E_coul': (11803.296957711445, 20.82238856812951),
                'E_long': (-13053.010258706468, 22.37345151248904),
                'Nbuild': (485.8572139303483, 1.8986003528700628), 'Ndanger': (0.0, 0.0)}

NVE_UNCONSTRAINED_EXPECTED = {'Atoms': (648.0, 0.0), 'Bonds': (432.0, 0.0), 'Angles': (216.0, 0.0),
                              'KinEng': (1602.0923289552238, 5.580040237658352),
                              'PotEng': (-252.26176035174132, 5.590040660475537),
                              'Temp': (830.709081766169, 2.8933351405976087),
                              'Press': (44778.287628855716, 225.9730795606594),
                              'Volume': (6434.856, 0.0),
                              'E_bond': (129.51483600646765, 3.2251322657817227),
                              'E_angle': (269.5281860746269, 4.127308177136364),
                              'E_vdwl': (884.8944745820896, 7.921781312371781),
                              'E_coul': (11695.374708955223, 9.753957036365723),
                              'E_long': (-13231.57395472637, 0.383206708789476),
                              'Nbuild': (91.35572139303484, 0.8547583670505459),
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
    buck = Buckingham(1194446.573277789, 3.6992957746478874, 4914.958810163367)
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
