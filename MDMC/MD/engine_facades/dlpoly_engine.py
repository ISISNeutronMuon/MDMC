"""Facade for DL_POLY MD engine

This is a facade to the DL_POLY MD engine and the Python wrapper dlpoly-py that can interface with it.

This facade is a skeleton example adapted from the analogous one for the DL_POLY MD engine in lammmps_engine.py.

Notes
-----

"""

from copy import copy
from itertools import tee
import logging
from ase import Atoms,Atom
from ase.io import write,iread
from MDMC.MD.structural_units import Atom as  MAtom

from dlpoly import DLPoly
from dlpoly.field import Field
from dlpoly.new_control import NewControl as Control
import numpy as np

from MDMC.common import units
from MDMC.common.decorators import unit_decorator, repr_decorator
from MDMC.MD.engine_facades.facade import MDEngine
from MDMC.trajectory_analysis.trajectory import Trajectory, TemporalConfiguration


LOGGER = logging.getLogger(__name__)


class DLPOLYAttribute:

    """
    A class which has a ``dlpoly-py`` object as an attribute and possesses attributes and methods relating to it.

    Parameters
    ----------
    dlpoly : dlpoly-py, optional
        Set the ``dlpoly`` attribute to a ``dlpoly-py`` object. Default is `None`,
        which results in a new ``dlpoly-py`` object being initialised.

    Attributes
    -----------
    dlpoly : dlpoly-py
        The ``dlpoly-py`` object owned by this class
    """

    def __init__(self, dlpoly=None, control=None, config=None, field=None, statis=None, output=None,
            destconfig=None, rdf=None, workdir=None):


        if dlpoly:
            self.dlpoly = dlpoly
        else:
            self.dlpoly = DLPoly()


        LOGGER.debug('%s: {dlpoly: %s}. dlpoly-py'
                     ' instance %s.',
                     self.__class__,
                     self.dlpoly,
                     'added to class' if dlpoly else 'created by class')


@repr_decorator('dlpoly', 'dlpoly_universe', 'dlpoly_simulation')
class DLPOLYEngine(DLPOLYAttribute, MDEngine):

    """
    Facade for DL_POLY

    """

    @property
    def saved_config(self):

        """
        Get the saved configuration of the atomic positions

        Must be implemented (abstract method in MDEngine ABC)

        Returns
        -------
        ``Configuration``
            The atomic positions
        """

        return self._saved_config

    @property
    def time_step(self):

        """
        Get or set the simulation time step in ``fs``

        Optional, but should be useful to implement

        Returns
        -------
        `float`
            Simulation time step in ``fs``
        """

        try:
            return self.dlpoly_simulation.time_step
        except AttributeError:
            return None

    @time_step.setter
    @unit_decorator(unit=units.TIME)
    def time_step(self, value):

        self.dlpoly_simulation.time_step = value

    @property
    def traj_step(self):

        """
        Get or set the number of simulation steps between saving the
        ``Trajectory``

        Optional, but should be useful to implement.

        Returns
        -------
        `int`
            Number of simulation steps that elapse between the ``Trajectory``
            being stored
        """

        try:
            return self.dlpoly_simulation.traj_step
        except AttributeError:
            return None

    @traj_step.setter
    def traj_step(self, value):

        self.dlpoly_simulation.traj_step = value

    @property
    def temperature(self):

        """
        Get or set the temperature of the simulation in ``K``

        Optional, but should be useful to implement

        Returns
        -------
        `float`
            Temperature in ``K``
        """

        return self.dlpoly_simulation.temperature

    @temperature.setter
    @unit_decorator(unit=units.TEMPERATURE)
    def temperature(self, value):

        self.dlpoly_simulation.temperature = value

    @property
    def pressure(self):

        """
        Get or set the pressure of the simulation in ``atm``

        Optional, but should be useful to implement

        Returns
        -------
        `float`
            Pressure in ``atm``
        """

        return self.dlpoly_simulation.pressure

    @pressure.setter
    @unit_decorator(unit=units.PRESSURE)
    def pressure(self, value):

        self.dlpoly_simulation.pressure = value

    @property
    def ensemble(self):

        """
        Get or set the ensemble object which applies a ``thermostat`` and/or
        ``barostat`` to DL_POLY

        Optional, not sure if it fits the paradigm of DL_POLY

        Returns
        -------
        DLPOLYEnsemble
            The simulation thermodynamic ensemble
        """

        return self.dlpoly_simulation.ensemble

    @ensemble.setter
    def ensemble(self, value):

        self.dlpoly_simulation.ensemble = value

    @property
    def thermostat(self):

        """
        Get or set the `str` which specifies the thermostat

        Optional, but should be useful to implement

        Returns
        -------
        `str`
            The ``thermostat`` name
        """

        return self.ensemble.thermostat

    @thermostat.setter
    def thermostat(self, value):

        self.ensemble.thermostat = value

    @property
    def barostat(self):

        """
        Get or set the `str` which specifies the barostat

        Optional, but should be useful to implement

        Returns
        -------
        `str`
            The ``barostat`` name
        """

        return self.ensemble.barostat

    @barostat.setter
    def barostat(self, value):

        self.ensemble.barostat = value

    def setup_universe(self, universe, **settings):

        """
        Creates the simulation box, the atomic configuration, and the topology
        in DL_POLY

        Must be implemented (abstract method in MDEngine ABC)

        Parameters
        ----------
        universe : Universe
            The MDMC ``Universe`` which will be setup in DL_POLY.
        **settings
            The majority of these are generic but some are specific to the
            ``MDEngine`` that is being used.
        """

        self.universe = universe
        self.dlpoly_universe = DLPOLYUniverse(self.universe, self.dlpoly, **settings)
        self._saved_config = None

    def setup_simulation(self, **settings):

        """
        Sets the options required to perform a simulation on a setup
        ``Universe``. Must follow a call to ``setup_universe()``.

        Must be implemented (abstract method in MDEngine ABC)

        Parameters
        ----------
        **settings
            The majority of these are generic but some are specific to the
            ``MDEngine`` that is being used.
        """

        self.dlpoly_simulation = DLPOLYSimulation(self.universe, self.dlpoly,
                                                  **settings)

    def minimize(self, n_steps, **settings):
        """
        Minimizes the simulation energy

        Must be implemented (abstract method in MDEngine ABC)

        Parameters
        ----------
        n_steps : int
            Maximum number of steps for the energy minimization.
        **settings
            The majority of these are generic but some are specific to the
            ``MDEngine`` that is being used.
        """

        #example of how to use the **settings to specify parameters, e.g. tolerances
        etol = settings.get('etol', 1.e-4)
        ftol = settings.get('ftol', 0.)
        LOGGER.info('%s minimize: {n_steps: %s,  ftol: %s}',
                    self.__class__, n_steps, ftol)
        if (ftol == 0.0):
            self.dlpoly.control['minimisation_criterion'] = 'energy'
            self.dlpoly.control['minimisation_tolerance'] = (etol,'e.V/mol')
            self.dlpoly.control['minimisation_frequency'] = (10,'steps')
        else:
            self.dlpoly.control['minimisation_criterion'] = 'force'
            self.dlpoly.control['minimisation_tolerance'] = (ftol,'e.V/Ang')
            self.dlpoly.control['minimisation_frequency'] = (10,'steps')
        self.run(n_steps,equilibration=True)

    def run(self, n_steps, equilibration=False):
        """
        Runs a simulation.  Must follow a call to ``setup_universe()`` and
        ``setup_simulation()``.

        Must be implemented (abstract method in MDEngine ABC)

        Parameters
        ----------
        n_steps : int
            Number of steps for the time integrator.
        equilibration : bool
            If `True`, run is equilibration which does not store the
            ``trajectory``. Otherwise run is prodution.
        """

        if equilibration:
            self.dlpoly.control['time_equilibration'] = (n_steps,'steps')
            self.dlpoly.control['traj_calculate'] ='Off'
        else:
            self.dlpoly.control['time_equilibration'] = (0,'steps')
            self.dlpoly.control['traj_calculate'] ='On'
            self.dlpoly.control['traj_start'] =(0,'steps')
            self.dlpoly.control['traj_interval'] =(10,'steps')
            self.dlpoly.control['traj_key'] ='pos'

        self.dlpoly.control['time_run'] = (n_steps,'steps')
        self.dlpoly.run(executable = '/home/drFaustroll/lavello/build-dlpoly-alin/bin/DLPOLY.Z',numProcs = 1, outputFile='test.log')


    def convert_trajectory(self, start=0, stop=None, step=1, **settings):
        """
        Parses the trajectory from the ``MDEngine`` format into MDMC format

        Must be implemented (abstract method in MDEngine ABC)
        Must be implemented (abstract method in MDEngine ABC)

        Parameters
        ----------
        start : int
            The index of the first trajectory, inclusive.
        stop : int
            The index of the last trajectory, exclusive.
        step : int
            The step size between trajectories.
        **settings
            The majority of these are generic but some are specific to the
            ``MDEngine`` that is being used.

        Returns
        -------
        ``Trajectory``
            The MDMC ``Trajectory`` from the most recent production simulation

        """

        def read_cell(f):
            cell = np.zeros((3,3))
            for i in range(3):
                cell[i,:]= np.array([ float(x) for x in  f.readline().split() ])
            return cell


        def create_atom(f,lvl):

            symbol, d1, d2, *_  = f.readline().split()
            mass = float(d2)
            aid = int(d1)
            pos = [ float(x) for x in f.readline().split() ]
            vel = None
            force = None
            if lvl > 0:
                vel = [ float(x) for x in f.readline().split() ]
            if lvl > 1:
                force = [ float(x) for x in f.readline().split() ]

            atom_type = 1
            atom = MAtom(symbol, position=pos, mass=mass)
            atom.atom_type = atom_type
            if self.universe:
                atom.universe = self.universe
            if vel is not None:
                atom.velocity = vel
            return atom

        atom_ids = settings.get('atom_IDs')
        f = open(self.dlpoly.control['io_file_history'],"r")
        title = f.readline()
        lvl, imcon, n_atoms, frames,  _ = [ int(i) for i in f.readline().split() ]
        if self.universe:
            assert n_atoms == len(self.universe.atoms)
        configs = []
        end = stop
        if stop is None:
            end = frames + 1
        print("frames to process", frames)
        for k in range(frames):
            time = float(f.readline().split()[-1])
            cell = read_cell(f)
            atoms = []
            print("process frame",k)
            for a in range(n_atoms):
                atom  = create_atom(f,lvl)
                if not atom_ids or atom.ID in atom_ids:
                    atoms.append(atom)
            if ((k >= start) and ((k - start)%step == 0) and (k < end)):
                configs.append(TemporalConfiguration(time,*atoms))
                print("add frame",k)
        f.close()

        return Trajectory(*configs)

    def update_parameters(self):

        """
        Updates the ``MDEngine`` force field ``Parameter`` objects from the ``Universe``

        Must be implemented (abstract method in MDEngine ABC)
        """

        self.dlpoly_universe.update_parameters()

    def save_config(self):

        """
        Sets ``self.saved_config`` to the current configuration

        Must be implemented (abstract method in MDEngine ABC)
        """

        for atom in self.universe.atoms:
            pass

        self._saved_config = saved_config

    def reset_config(self):

        """
        Resets the configuration of the simulation to that in ``saved_config``

        Must be implemented (abstract method in MDEngine ABC)
        """

        self.dlpoly_universe.set_config(self.saved_config)


@repr_decorator('universe')
class DLPOLYUniverse(DLPOLYAttribute):

    """
    A class with what would be the equivalent in DL_POLy to the MDMC universe (i.e.
    the configuration and topology)

    Parameters
    ----------
    universe : Universe
        The MDMC ``Universe`` used to create the ``DLPOLYUniverse``
    dlpoly : dlpoly-py, optional
        Set the ``dlpoly`` attribute to a ``dlpoly-py`` object. Default is `None`,
        which results in a new ``dlpoly-py`` object being initialised.
    **settings
            The majority of these are generic but some are specific to the
            ``MDEngine`` that is being used.

    Attributes
    ----------
    universe : Universe
        The MDMC ``Universe`` which has been converted to this ``DLPOLYUniverse``.
    There might be a lot of Attributes needed (see DL_POLYUniverse for example)
    """

    def __init__(self, universe, dlpoly=None, **settings):

        #example methods
        super().__init__(dlpoly=dlpoly)
        self.universe = universe
        self._define_simulation_box(self.universe)
        self._build_config(self.universe)
        self._add_topology(self.universe, **settings)
        self.update_parameters()

    def update_parameters(self):

        """
        Updates the DL_POLY force field parameters from the MDMC universe
            self.workdir

        """

        #example methods
        self._update_charges()
        self._update_bonded_interactions('bond', self.bonds)
        self._update_dispersions(self.universe)

    def _define_simulation_box(self, universe):

        """
        Defines a region and creates a simulation box that fills this region

        Parameters
        ----------
        universe : Universe
            The MDMC ``Universe`` used to create the region and simulation box.
        """

        self.dlpoly.control=Control()
        self.dlpoly.control['title'] = 'my simulation title'
        self.dlpoly.control['time_job'] = (10000,'s')
        self.dlpoly.control['time_close'] = (10,'s')
        self.dlpoly.control['data_dump_frequency'] = (5000,'steps')
        self.dlpoly.control['stats_frequency'] = (100,'steps')
        self.dlpoly.control['print_frequency'] = (100,'steps')
        self.dlpoly.control['stack_size'] = (10,'steps')
        self.dlpoly.control['padding'] = (0.5, 'Ang')
        self.dlpoly.control['vdw_method'] = 'direct'

    def _build_config(self, universe):

        """
        Adds atoms to DL_POLY

        Parameters
        ----------
        universe : Universe
            The MDMC ``Universe`` used to fill the DL_POLY box with atoms.
        """
# assume the dimensions are in angstrom
        a = Atoms(cell = universe.dimensions,pbc = True)
        for atom in universe.atoms:
            a.append(Atom(atom.name,atom.position))
        write('test.config', a, format='dlp4')
        self.dlpoly.load_config('test.config')

    def _add_topology(self, universe, **settings):

        """
        Add the bonded and nonbonded interactions to DL_POLY

        Parameters
        ----------
        universe : Universe
            The MDMC ``Universe`` used to define the topology.
        **settings
            The majority of these are generic but some are specific to the
            ``MDEngine`` that is being used.

        Raises
        ------
        NotImplementedError
            If ``universe`` contains an interaction type that has not been
            implemented in the DL_POLY facade
        """

        LOGGER.info('%s Add topology to DL_POLY',
                    self.__class__)

        #example methods
        bonds, angles, dihedrals, disps, couls, others = partition_interactions(
            set(universe.interactions),
            ['Bond', 'BondAngle', 'DihedralAngle', 'Dispersion', 'Coulombic'],
            unpartitioned=True,
            lst=True)

        if others:
            raise NotImplementedError('This interaction type has not been'
                                      ' implemented in the DL_POLY facade')

        self.bonds = bonds
        self.angles = angles


        if bonds:
            LOGGER.debug('%s Add bonds to DL_POLY', self.__class__)
            self.dlpoly.create_bonds(bonds)

        if angles:
            LOGGER.debug('%s Add angles to DL_POLY', self.__class__)
            # Set used to remove duplicate angle styles, which are not required
            # to be (and in fact cannot) be passed to DL_POLY hybrid angle_style
            self.dlpoly.create_angles(angles)
        self.dlpoly.load_field('Ar.field')
        self.dlpoly.control['cutoff'] = (np.max([i.cutoff for i in self.universe.nonbonded_interactions]),'Ang')

    def _update_charges(self):

        """
        Updates the ``charges`` in DL_POLY

        Raises
        ------
        AttributeError
            If one or more ``Atom`` do not have a ``charge`` (or
            ``charge is None``)
        """

        pass

    def _update_dispersions(self, universe, pair_coeff_cmds=None):

        """
        Updates ``Dispersion`` interactions in DL_POLY

        Parameters
        ----------
        universe : Universe
            The MDMC ``Universe`` containing ``NonBondedInteractions``.
        """

        pass

    def _update_bonded_interactions(self, dlpoly_name, bonded_interactions):

        """
        Updates the bonded interaction coefficients, which are then applied to
        any bonded interactions which have previously been set

        Parameters
        ----------
        dlpoly_name : str
            The name of the bonded interaction type used for setting coeffs in
            DL_POLY.
        bonded_interactions : list of BondedInteractions
            ``BondedInteractions`` which will be updated in DL_POLY.
        """

        pass

    def apply_constraints(self):

        """
        Adds a constraint ``fix`` to DL_POLY for all bonds and bond angles which
        are constrained
        """
        pass


@repr_decorator('universe', 'time_step', 'traj_step', 'ensemble')
class DLPOLYSimulation(DLPOLYAttribute):

    """
    The attributes and methods related running a simulation in DL_POLY using a
    ``DLPOLYUniverse`` object

    Parameters
    ----------
    universe : Universe
        The MDMC ``Universe`` used to create the ``DLPOLYUniverse``.
    dlpoly : dlpoly-py, optional
        Set the ``dlpoly`` attribute to a ``dlpoly-py`` object. Default is `None`,
        which results in a new ``dlpoly-py`` object being initialised.
    **settings
        The majority of these are generic but some are specific to the
        ``MDEngine`` that is being used.

    Attributes
    ----------
    universe : Universe
        An MDMC ``Universe`` object.
    time_step : float
        The time difference between MD simulation steps in fs.
    traj_step : int
        Number of simulation steps that elapse between the ``Trajectory`` being stored.
    ensemble : DLPOLYEnsemble
        Simulation ensemble, which applies a ``thermostat`` and ``barostat``.
    """

    def __init__(self, universe, dlpoly=None, **settings):
        super().__init__(dlpoly=dlpoly)

        self.universe = universe
        self.ensemble = DLPOLYEnsemble(self.dlpoly, **settings)
        self.temperature = settings.get('temperature')
        self.time_step = settings.get('time_step')
        self.traj_step = settings['traj_step']

    @property
    def time_step(self):

        """
        Get or set the simulation time step in ``fs``

        Returns
        -------
        `float`
            Simulation time step in ``fs``
        """

        return self._time_step

    @time_step.setter
    @unit_decorator(unit=units.TIME)
    def time_step(self, value):

        self._time_step = value
        try:
            # Set the timestep in DL_POLY wrapper
            self.dlpoly.control['timestep'] = (convert_unit(self._time_step),'fs')
        except ValueError:
            pass

    @property
    def temperature(self):

        """
        Get or set the temperature of the simulation in ``K``

        Returns
        -------
        `float`
            Temperature in ``K``
        """

        return self._temperature

    @temperature.setter
    @unit_decorator(unit=units.TEMPERATURE)
    def temperature(self, value):

        self._temperature = value
        try:
            # Set the initial temperature in the DL_POLY wrapper
            self.dlpoly.control['temperature'] = (convert_unit(self._temperature), 'K')
        except ValueError:
            pass

    @property
    def pressure(self):

        """
        Get or set the pressure of the simulation in ``atm``

        Returns
        -------
        `float`
            Pressure in ``atm``
        """

        return self.ensemble.pressure

    @pressure.setter
    @unit_decorator(unit=units.PRESSURE)
    def pressure(self, value):

        self.ensemble.pressure = value

    @property
    def thermostat(self):

        """
        Get or set the string which specifies the thermostat

        Returns
        -------
        `str`
            The thermostat name
        """

        return self.ensemble.thermostat

    @thermostat.setter
    def thermostat(self, value):

        self.ensemble.thermostat = value

    @property
    def barostat(self):

        """
        Get or set the string which specifies the barostat

        Returns
        -------
        `str`
            The barostat name
        """

        return self.ensemble.barostat

    @barostat.setter
    def barostat(self, value):

        self.ensemble.barostat = value


@repr_decorator('temperature', 'pressure', 'thermostat', 'barostat')
class DLPOLYEnsemble(DLPOLYAttribute):

    """
    A thermodynamic ensemble determined by applying a thermostat and/or barostat

    Parameters
    ----------
    dlpoly : dlpoly-py
        Set the ``dlpoly`` attribute to a ``dlpoly-py`` object.
    temperature : float, optional
        Thermostat temperature. Default is `None`, which is only valid if a
        ``thermostat`` is also `None`.
    pressure : float, optional
        Barostat pressure. Default is `None`, which is only valid if a
        ``barostat`` is also `None`.
    thermostat : str
        Name of a thermostat to be applied.
    barostat : str
        Name of a barostat to be applied.
    **settings
        The majority of these are generic but some are specific to the
        ``MDEngine`` that is being used, e.g.:

    Attributes
    ----------
    rescale_step : int
        Number of steps between applying temperature rescaling. This only
        applies to rescale thermostats.
    """

    def __init__(self, dlpoly, temperature=None, pressure=None, thermostat=None,
                 barostat=None, **settings):

        # Requires a ``dlpoly-py`` object as thermostats cannot be applied before
        # configuration is defined
        super().__init__(dlpoly)
        self.temperature = temperature
        self.pressure = pressure
        self.dlpoly.control['ensemble'] = 'nve'

        self.time_step = settings.get('time_step')
        self.t_damp = settings.get('t_damp')
        self.p_damp = settings.get('p_damp')
        self.t_window = settings.get('t_window')
        self.t_fraction = settings.get('t_fraction')
        self.rescale_step = settings.get('rescale_step')

        self.thermostat = thermostat
        self.barostat = barostat

    @property
    def temperature(self):

        """
        Get or set the temperature of the simulation in ``K``
        """

        return self._temperature

    @temperature.setter
    @unit_decorator(unit=units.TEMPERATURE)
    def temperature(self, value):

        self._temperature = value

    @property
    def pressure(self):

        """
        Get or set the pressure of the simulation in ``atm``
        """

        return self._pressure

    @pressure.setter
    @unit_decorator(unit=units.PRESSURE)
    def pressure(self, value):

        self._pressure = value

    @property
    def thermostat(self):

        """
        Get or set the `str` which specifies the thermostat

        Raises
        ------
        AttributeError
            If ``self.temperature`` has not been set.
        """

        return self._thermostat

    @thermostat.setter
    def thermostat(self, value):

        if value and not self.temperature:
            raise AttributeError('all ensembles with a thermostat must have a'
                                 ' temperature')
        self._thermostat = value
        # Set the thermostat and barostat in DL_POLY wrapper

    @property
    def barostat(self):

        """
        Get or set the `str` which specifies the barostat

        Raises
            If ``self.pressure`` has not been set.
        """

        return self._barostat

    @barostat.setter
    def barostat(self, value):

        if value and not self.pressure:
            raise AttributeError('all ensembles with a barostat must have a'
                                 ' pressure')

        self._barostat = value
        # Set the thermostat and barostat in DL_POLY wrapper

# Define the unit system used in DL_POLY
SYSTEM = {
    'LENGTH':units.Unit('Ang'),
    'TIME':units.Unit('fs'),
    'MASS':units.Unit('g') / units.Unit('mol'),
    'CHARGE':units.Unit('e'),
    'ANGLE':units.Unit('deg'),
    'TEMPERATURE':units.Unit('K'),
    'ENERGY':units.Unit('kcal') / units.Unit('mol'),
    'FORCE':units.Unit('kcal') / (units.Unit('Ang') * units.Unit('mol')),
    'PRESSURE':units.Unit('atm')
}

#some extra utility methods. these might be obsolete or importable from lammps_engine.py
#(in which case they maybe should be refactored into a utility module)
def convert_unit(value, unit=None, to_dlpoly=True):

    """
    Converts between MDMC units and DL_POLY real units

    Parameters
    ----------
    value : array_like or float_like
        The value of the physical property to be converted, in MDMC units.
        Must derive from either ``ndarray`` or `float`.
    unit : Unit, optional
        The unit of the ``value``. If `None`, the ``value`` must possess a
        ``unit`` attribute i.e. derive from ``UnitFloat`` or ``UnitArray``.
        Default is `None`.
    to_dlpoly : bool, optional
        If `True` the conversion is from MDMC units to DL_POLY units. Default is
        `True`.

    Returns
    -------
    `float` or `numpy.ndarray`
        Value in DL_POLY units if ``to_dlpoly`` is `True`, otherwise value in
        MDMC units. Return type is same as ``value`` type.
    """

    def expand_components(unit, system):

        """
        Expands out the ``components`` of a ``Unit``, so that the ``Unit`` is
        expressed purely in terms of ``base`` ``Unit`` objects. The only
        exception to this is ``Unit`` objects which occur in ``System``: these
        are kept in the `list` of ``components``.

        Parameters
        ----------
        unit : Unit
            The ``Unit`` to be expanded out
        system : dict
            A `dict` of {``Property``: ``Unit``} pairs, which is used to
            substitute ``System`` ``Unit`` objects into the expanded
            ``components``.

        Returns
        -------
        `tuple`
            A `tuple` of (``num``, ``denom``), where ``num`` is a `list` of all
            ``base`` ``Unit`` objects in the numerator, and ``denom`` is a
            `list` of all ``base`` ``Unit`` objects in the denominator
        """

        def is_sublist_of_list(sub, lst):

            """
            Determines if all of the elements in a sublist are in a `list`,
            including ensuring that any duplicates in the sublist have at least
            the same number of duplicates in the `list`

            Parameters
            ----------
            sub : list
                The sublist to be tested
            lst : list
                The `list` the ``sub`` is tested against

            Returns
            -------
            `bool`
                `True` if all of the elements of ``sub`` are in ``lst``, and
                `False` if not
            """

            return all(sub.count(x) <= lst.count(x) for x in set(sub))

        def remove_components(remove_comps, comps):

            """
            Removes all elements of a `list` of ``components`` from another
            `list` of ``components``

            Parameters
            ----------
            remove_comps : list
                The ``components`` to be removed
            comps : list
                The ``components`` from which ``remove_comps`` is removed

            Returns
            -------
            `list`
                A `list` of ``components`` from which ``remove_comps`` have been
                removed
            """

            for remove_comp in remove_comps:
                comps.remove(remove_comp)

            return comps

        num, denom = [], []
        # If unit is in system of units, don't break down to components, as
        # a unit in the system of units always has a conversion, even if it is
        # a compound unit.
        if unit.base or unit in system.values():
            num.append(unit)
        else:
            # tuple unpacking style below appends to num and denom lists
            for comp in unit.components['numerator']:
                num[len(num):], denom[len(denom):] = expand_components(comp,
                                                                       system)
            for comp in unit.components['denominator']:
                denom[len(denom):], num[len(num):] = expand_components(comp,
                                                                       system)

            # Substitute in units rather than separated out components if a unit
            # exists in the system of units.  This is because the conversion can
            # always be determined for units in the system of units. Base units
            # are filtered as they can only replace themselves (as they are
            # their only component).
            for sys_unit in (unit for unit in system.values() if not unit.base):
                # while is required for powers of sys_unit, as otherwise only
                # a single power will be removed
                while True:
                    sys_unit_num = sys_unit.components['numerator']
                    sys_unit_denom = sys_unit.components['denominator']
                    # Determine if all of the components of unit are in the
                    # numerator and denominator lists. If so, remove them and
                    # replace with the unit in the numerator.
                    if (is_sublist_of_list(sys_unit_num, num) and
                            is_sublist_of_list(sys_unit_denom, denom)):
                        num = remove_components(sys_unit_num, num)
                        denom = remove_components(sys_unit_denom, denom)
                        num.append(sys_unit)
                    # Do the same for the inverse (i.e. unit's numberator
                    # components in the denominator list and vice versa). If so,
                    # remove them and replace with the unit in the denominator.
                    elif (is_sublist_of_list(sys_unit_num, denom) and
                          is_sublist_of_list(sys_unit_denom, num)):
                        denom = remove_components(sys_unit_num, denom)
                        num = remove_components(sys_unit_denom, num)
                        denom.append(sys_unit)
                    # Breaks if the components of sys_unit are not found in num
                    # and denom
                    else:
                        break
        return num, denom

    # If no unit argument is passed, the value must possess a unit
    if not unit:
        # If value is unitless, no conversion is required
        try:
            unit = value.unit
        except AttributeError:
            if value is None:
                raise ValueError('Cannot convert NoneType value')
            return value
    # Expand the unit in terms of its base units (for numerator and denominator)
    if to_dlpoly:
        l_sys = copy(SYSTEM)
        # For angular potential strength DL_POLY requires the units in rad,
        # rather than degrees (which is uses otherwise). Therefore if the unit
        # is in MDMC angular potential strength units (energy / angle^2), the
        # ANGLE entry in SYSTEM is replaced by radians.
        if unit == units.SYSTEM['ENERGY'] / units.SYSTEM['ANGLE'] ** 2:
            l_sys['ANGLE'] = units.Unit('rad')

        expanded_unit = expand_components(unit, units.SYSTEM)
        system_inv = {unit:property for property, unit in units.SYSTEM.items()}
        # Apply inversion to all components
        unit_nums, unit_denoms = map(lambda comp_list: [l_sys[system_inv[comp]]
                                                        for comp in comp_list],
                                     expanded_unit)

        conv_nums, conv_denoms = [], []
        for component in unit_nums:
            conv_nums[len(conv_nums):], conv_denoms[len(conv_denoms):] = \
                expand_components(component, l_sys)
        for component in unit_denoms:
            conv_denoms[len(conv_denoms):], conv_nums[len(conv_nums):] = \
                expand_components(component, l_sys)
    else:
        conv_denoms, conv_nums = expand_components(unit, SYSTEM)

    for component in conv_nums:
        value /= getattr(units, component)
    for component in conv_denoms:
        value *= getattr(units, component)

    return value


def partition(items, predicate):

    """
    Partitions an ``iterable`` using a predicate

    Parameters
    ----------
    items : iterable
        An ``interable`` to be partitioned.
    predicate : function
        A predicate that can be applied to items to returned `True` or `False`.

    Returns
    -------
    `tuple`
        A `tuple` of (``gen_true``, ``gen_false``), where ``gen_true`` is a
        generator of all items for which the ``predicate`` returned `True`, and
        ``gen_false is a generator of all items for which the ``predicate``
        returned `False`
    """

    iter_a, iter_b = tee((predicate(item), item) for item in items)
    return ((item for pred, item in iter_a if pred),
            (item for pred, item in iter_b if not pred))


def partition_interactions(interactions, names, unpartitioned=False, lst=False):

    """
    Partitions an ``iterable`` of ``Interaction`` objects using a `list` of
    ``Interaction`` ``names``

    This occurs by using ``partition`` to filter out one ``Interaction`` type
    for each loop, so previously identified ``Interactions`` are no longer
    considered.

    Parameters
    ----------
    interactions : iterable of Interactions
        An ``interable`` of ``Interaction`` objects to be partitioned.
    names : list of str
        Names of ``Interaction`` classes.
    unpartitioned : bool, optional
        If `True`, then a generator containing any ``Interaction`` objects that
        did not have a name in ``names`` is returned as an additional item in
        the `tuple`. Default is `False`.
    lst : bool, optional
        If `True`, then the returned `tuple` contains `list` rather than
        generators. ``Interaction`` objects which have the name specified by
        ``names[n]``. Default is `False`.

    Returns
    -------
    `tuple`
        A `tuple` of length ``len(names)`` where ``index`` ``n`` is a generator
        of all of the ``Interaction`` objects which have the name specified by
        ``names[n]``. If ``unpartitioned`` is `True`, `tuple` is length ``n+1``.
        If ``lst`` is `True`, the generators are replaced by `list`.

    Example
    -------
    Partion ``interactions`` into ``Bonds`` and ``BondAngles``:

        .. highlight:: python
        .. code-block:: python

            bonds, angles = partition_interactions(interactions,
                                                   ['Bond, BondAngle'])
    """

    interaction_lst = [None] * len(names)
    i = 0
    for name in names:
        predicate = lambda x, n=name: x.name == n
        interaction_lst[i], interactions = partition(interactions, predicate)
        i += 1
    if unpartitioned:
        interaction_lst += [interactions]
    if lst:
        interaction_lst = [list(i) for i in interaction_lst]
    return tuple(interaction_lst)
