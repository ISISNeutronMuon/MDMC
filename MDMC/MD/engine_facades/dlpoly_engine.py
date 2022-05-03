"""Facade for DL_POLY MD engine

This is a facade to the DL_POLY MD engine and
the Python wrapper dlpoly-py that can interface with it.

"""

from abc import ABC
from copy import copy
import logging
from ase import Atoms, Atom
from ase.io import write

from dlpoly import DLPoly
from dlpoly.species import Species
from dlpoly.field import Field, Bond, Potential, Molecule
from dlpoly.new_control import NewControl as DLPControl
from dlpoly.config import Config
import numpy as np

from MDMC.common import units
from MDMC.common.decorators import unit_decorator, repr_decorator
from MDMC.MD.engine_facades.facade import MDEngine
from MDMC.MD.structural_units import (Atom as MAtom,
                                      Molecule as MMolecule)
from MDMC.trajectory_analysis.trajectory import (Trajectory,
                                                 TemporalConfiguration)
from MDMC.utilities.partitioning import partition_interactions


LOGGER = logging.getLogger(__name__)

POTENTIAL_REF = {
    'LennardJones': 'lj',
    'HarmonicPotential': 'harm',
    'Buckingham': 'buck',
    'Coulomb': 'coul',
    'Periodic': 'cos'
    }
BOND_CLASS_REF = {
    'Bond': 'bonds',
    'BondAngle': 'angles',
    'DihedralAngle': 'dihedrals'
    }


class DLPOLYAttribute(ABC):
    # pylint: disable=too-few-public-methods
    """
    A class which has a ``dlpoly-py`` object as an
    attribute and possesses attributes and methods relating to it.

    Parameters
    ----------
    dlpoly : dlpoly-py, optional
        Set the ``dlpoly`` attribute to a ``dlpoly-py`` object. Default is `None`, which results
        in a new ``dlpoly-py`` object being initialised.

    Attributes
    -----------
    dlpoly : dlpoly-py
        The ``dlpoly-py`` object owned by this class
    """

    def __init__(self, dlpoly=None, control=None,
                 config=None, field=None, statis=None, output=None,
                 dest_config=None, rdf=None, workdir=None):

        if dlpoly:
            self.dlpoly = dlpoly
        else:
            self.dlpoly = DLPoly(control=control, config=config,
                                 field=field, statis=statis,
                                 output=output, dest_config=dest_config,
                                 rdf=rdf, workdir=workdir)

        LOGGER.debug('%s: {dlpoly: %s}. dlpoly-py'
                     ' instance %s.',
                     self.__class__,
                     self.dlpoly,
                     'added to class' if dlpoly else 'created by class')

    def read_settings(self, settings):
        """
        Read DLP parameters from a settings dict

        Parameters
        ----------
        settings : dict
            MDMC settings dictionary to read parameters from.
        """
        new_opt = DLPControl.from_dict(settings, strict=False)
        self.dlpoly.control = self.dlpoly.control + new_opt


@repr_decorator('dlpoly', 'dlpoly_universe', 'dlpoly_simulation')
class DLPOLYEngine(DLPOLYAttribute, MDEngine):

    """
    Facade for DL_POLY

    """
    def __init__(self, dlpoly=None, control=None,
                 config=None, field=None, statis=None, output=None,
                 dest_config=None, rdf=None, workdir=None):

        DLPOLYAttribute.__init__(self, dlpoly, control,
                                 config, field, statis, output,
                                 dest_config, rdf, workdir)

        self.universe = None
        self.dlpoly_universe = None
        self.dlpoly_simulation = None
        self._saved_config = None

    @property
    def saved_config(self):

        """
        Get the saved configuration of the atomic positions

        Returns
        -------
        ``Configuration``
            The atomic positions
        """

        return self._saved_config

    @property
    def temperature(self):

        """
        Get or set the temperature of the simulation in ``K``

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

        Parameters
        ----------
        universe : Universe
            The MDMC ``Universe`` which will be setup in DL_POLY.
        **settings
            The majority of these are generic but some are specific to the
            ``MDEngine`` that is being used.
        """

        self.universe = universe
        self.dlpoly_universe = DLPOLYUniverse(self.universe,
                                              self.dlpoly,
                                              **settings)
        self._saved_config = None

    def setup_simulation(self, **settings):

        """
        Sets the options required to perform a simulation on a setup
        ``Universe``. Must follow a call to ``setup_universe()``.

        Parameters
        ----------
        **settings
            The majority of these are generic but some are specific to the
            ``MDEngine`` that is being used.
        """

        self.dlpoly_simulation = DLPOLYSimulation(universe=self.universe,
                                                  traj_step=self.traj_step,
                                                  time_step=self.time_step,
                                                  dlpoly=self.dlpoly,
                                                  **settings)

    def minimize(self, n_steps, output_log: str = None, work_dir: str = None, **settings):
        """
        Minimizes the simulation energy

        Parameters
        ----------
        n_steps : int
            Maximum number of steps for the energy minimization.
        output_log: str, optional, default None
            file where the output goes.
        work_dir: str, optional, default None
            folder where the run happens
        **settings
            The majority of these are generic but some are specific to the
            ``MDEngine`` that is being used.
        etol: float, energy tolerance criteria for energy minimisation
        ftol: float, force tolerance criteria for force minimisation, active only if non-zero
        minimisation_frequency: how often to do a geometry optimisation during minimisation process
        """

        # Example of how to use the **settings to specify parameters,
        # e.g. tolerances
        etol = settings.get('etol', 1.e-3)
        ftol = settings.get('ftol', 0.)
        min_freq = settings.get('minimisation_frequency', 10)
        LOGGER.info('%s minimize: {n_steps: %s,  ftol: %s}',
                    self.__class__, n_steps, ftol)
        if ftol == 0.0:
            self.dlpoly.control['minimisation_criterion'] = 'energy'
            self.dlpoly.control['minimisation_tolerance'] = (etol, 'internal_e')
        else:
            self.dlpoly.control['minimisation_criterion'] = 'force'
            self.dlpoly.control['minimisation_tolerance'] = (ftol, 'e.V/Ang')
        self.dlpoly.control['minimisation_frequency'] = (min_freq, 'steps')
        self.run(n_steps, equilibration=True, output_log=output_log, work_dir=work_dir,
                 **settings)
        self.dlpoly.control['minimisation_criterion'] = 'off'

    def run(self, n_steps, equilibration=False, output_log: str = None, work_dir: str = None,
            **settings):
        """
        Runs a simulation.  Must follow a call to ``setup_universe()`` and
        ``setup_simulation()``.

        Parameters
        ----------
        n_steps : int
            Number of steps for the time integrator.
        equilibration : bool (optional, default False)
            If `True`, just runs an equilibration which does not store the
            ``trajectory``.
        output_log: str, optional, default None
            file where the output goes.
        work_dir: str, optional, default None
            folder where the run happens
        **settings
            The majority of these are generic but some are specific to the
            ``MDEngine`` that is being used.
        time_equilibration: int, number of steps to run equilibrarion for,
                                 typically 0 in production stage
        traj_calculate: on/off prints trajectory or not
        traj_start: int, at which the trajectory printing shall start
        traj_key: pos,pos-vel,pos-vel-force, level of details for trajectory
                  prints positions, positions and
                  velocities, positions, velocities and forces
        numprocs: int, number of mpi processes to start to run dl_poly
        """

        if equilibration:
            self.dlpoly.control['time_equilibration'] = (n_steps, 'steps')
            self.dlpoly.control['traj_calculate'] = settings.get('traj_calculate', 'Off')
        else:
            self.dlpoly.control['time_equilibration'] = \
                (settings.get('time_equilibration', 0), 'steps')
            self.dlpoly.control['traj_calculate'] = 'On'
            self.dlpoly.control['traj_start'] = (settings.get('traj_start', 0), 'steps')
            self.dlpoly.control['traj_interval'] = (self.traj_step, 'steps')
            self.dlpoly.control['traj_key'] = settings.get('traj_key', 'pos')

        self.dlpoly.control['time_run'] = (n_steps, 'steps')
        self.dlpoly.workdir = work_dir
        # pylint: disable=c-extension-no-member, too-many-lines
        self.dlpoly.run(numProcs=1, outputFile=output_log)
        print("update coordinates from ",self.dlpoly.control['io_file_revcon'])
        self.dlpoly.dest_config = 'minim.config'
        self.dlpoly.load_config(self.dlpoly.control['io_file_revcon'])


    def convert_trajectory(self, start=0, stop=None, step=1, **settings):
        """
        Parses the trajectory from the ``DL_POLY`` format into MDMC format.

        Parameters
        ----------
        start : int
            The index of the first trajectory, inclusive.
        stop : int
            The index of the last trajectory, exclusive.
        step : int
            The step size between trajectories.

        Returns
        -------
        ``Trajectory``
            The MDMC ``Trajectory`` from the most recent production simulation

        """

        def read_cell(f):
            cell = np.zeros((3, 3))
            for i in range(3):
                cell[i, :] = np.array([float(x) for x in f.readline().split()])
            return cell

        def create_atom(f, level_of_detail):
            # level_of_detail of information in the file, 0, indicates only positions,
            # 1 positions and velocities, 2 positions, velocities and forces
            # the first line gives the symbol, mass and atom_ID of the atom
            symbol, _, mass_str, *_ = f.readline().split()
            mass = float(mass_str)
            # the next line gives the position of the atom
            pos = [float(x) for x in f.readline().split()]
            # the next line, if it exists, gives the velocity of the atom
            if level_of_detail > 0:
                vel = [float(x) for x in f.readline().split()]
            else:
                vel = None
            # next line, if existent, gives the force on the atom. currently not used by MDMC
            if level_of_detail > 1:
                force = [float(x) for x in f.readline().split()]
            else:
                force = None
            _ = force

            atom = MAtom(symbol, position=pos, mass=mass)
            atom.atom_type = self.universe.element_dict[symbol].atom_type
            if self.universe:
                atom.universe = self.universe
            if vel is not None:
                atom.velocity = vel
            return atom

        atom_ids = settings.get('atom_IDs')
        with open(self.dlpoly.control['io_file_history'], "r", encoding="ascii") as f:
            _ = f.readline()  # title
            level_of_detail, _, n_atoms, frames, *_ = \
                [int(i) for i in f.readline().split()]  # imcon
            if self.universe:
                assert n_atoms == len(self.universe.atoms)
            configs = []
            end = stop
            if stop is None:
                end = frames + 1
            for iframe in range(frames):
                time = float(f.readline().split()[-1])
                # Just skip, unused for the moment, will need to be
                # used for npt simulations
                read_cell(f)
                atoms = []
                for _ in range(n_atoms):
                    atom = create_atom(f, level_of_detail)
                    if not atom_ids or atom.ID in atom_ids:
                        atoms.append(atom)
                if (start <= iframe < end) and ((iframe - start) % step == 0):
                    configs.append(
                        TemporalConfiguration(
                            convert_unit(time, unit=units.Unit('ps'), to_dlpoly=False), *atoms
                        ))

        return Trajectory(*configs)

    def update_parameters(self):

        """
        Updates the ``MDEngine`` force field ``Parameter`` objects
        from the ``Universe``
        """

        self.dlpoly_universe.update_parameters()

    def save_config(self):

        """
        Sets ``self.saved_config`` to the current configuration
        """
        self._saved_config = Config(self.dlpoly.control['io_file_revcon'])

    def reset_config(self):

        """
        Resets the configuration of the simulation to that in ``saved_config``
        """

        self.dlpoly_universe.set_config(self.saved_config)


@repr_decorator('universe')
class DLPOLYUniverse(DLPOLYAttribute):

    """
    A class with what would be the equivalent in DL_Poly
    to the MDMC universe (i.e.the configuration and topology)

    Parameters
    ----------
    universe : Universe
        The MDMC ``Universe`` used to create the ``DLPOLYUniverse``
    dlpoly : dlpoly-py, optional
        Set the ``dlpoly`` attribute to a ``dlpoly-py`` object.
        Default is `None`,
        which results in a new ``dlpoly-py`` object being initialised.
    **settings
            The majority of these are generic but some are specific to the
            ``MDEngine`` that is being used.

    Attributes
    ----------
    universe : Universe
        The MDMC ``Universe`` which became this ``DLPOLYUniverse``.
    There might be a lot of Attributes needed (see DL_POLYUniverse for example)
    """

    def __init__(self, universe, dlpoly=None, **settings):

        super().__init__(dlpoly=dlpoly)
        self.universe = universe

        self.bonds = []
        self.disps = []
        self.angles = []
        self.dihedrals = []
        self.couls = []
        self._define_simulation_details()
        self._build_config(self.universe)
        self._add_topology(self.universe, **settings)
        self.update_parameters()

    def update_parameters(self):
        """
        Updates the DL_POLY force field parameters from the MDMC universe
        """

        (self.bonds, self.angles, self.dihedrals,
         self.disps, self.couls, _) = partition_interactions(
            set(self.universe.interactions),
            ['Bond', 'BondAngle', 'DihedralAngle', 'Dispersion', 'Coulombic'],
            unpartitioned=True,
            lst=True)

        self._update_charges()
        self._update_bonded_interactions()
        self._update_dispersions()
        self.dlpoly.field.write(self.dlpoly.control['io_file_field'])

    def _define_simulation_details(self, **settings):
        """
        Defines a region and creates a simulation box that fills this region

        Parameters
        ----------
        **settings
            The majority of these are generic but some are specific to the
            ``MDEngine`` that is being used.
        title: str, title for the simulation
        time_job: float, time of seconds for simulation to run
        time_close: float, close time to allow writing
        data_dump_frequency: int, interval to write restart, recovery data
        stats_frequency: int, interval to print stats
        print_frequency: int, interval to print statistics in output
        stack_size: int, how many steps use for averge stacks
        padding: float, buffer for neighbour lists
        vdw_method: direct/tabulated how do compute dispersion operations
        coul_method: spme, off how to compute long range interactions
        """

        self.dlpoly.control = DLPControl()
        self.dlpoly.control['title'] = settings.get('title', 'my simulation title')
        self.dlpoly.control['time_job'] = (settings.get('time_job', 100000.0), 's')
        self.dlpoly.control['time_close'] = (settings.get('time_close', 10.0), 's')
        self.dlpoly.control['data_dump_frequency'] = \
            (settings.get('data_dump_frequency', 5000), 'steps')
        self.dlpoly.control['stats_frequency'] = \
            (settings.get('stats_frequency', 100), 'steps')
        self.dlpoly.control['print_frequency'] = \
            (settings.get('print_frequency', 100), 'steps')
        self.dlpoly.control['stack_size'] = (settings.get('stack_size', 100), 'steps')
        self.dlpoly.control['padding'] = (settings.get('padding', 0.5), 'Ang')
        self.dlpoly.control['vdw_method'] = settings.get('vdw_method', 'direct')

        if self.universe.electrostatic_solver:
            self.dlpoly.control['coul_method'] = settings.get('coul_method', 'spme')
            self.dlpoly.control['ewald_precision'] = self.universe.electrostatic_solver.accuracy
        else:
            self.dlpoly.control['coul_method'] = settings.get('coul_method', 'off')

    def _build_config(self, universe, **settings):

        """
        Adds atoms to DL_POLY

        Parameters
        ----------
        universe : Universe
            The MDMC ``Universe`` used to fill the DL_POLY box with atoms.
        **settings
            The majority of these are generic but some are specific to the
            ``MDEngine`` that is being used.
        config: str, filename for the config file to be written
        """
        # assume the dimensions are in angstrom
        atoms = Atoms(cell=universe.dimensions, pbc=True)
        for atom in universe.atoms:
            atoms.append(Atom(atom.name, atom.position))
        config_filename = settings.get('config', 'test.config')
        write(config_filename, atoms, format='dlp4')
        self.dlpoly.load_config(config_filename)
        LOGGER.info('%s configuration written in %s',
                    self.__class__, config_filename)

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
        field: str, filename for the field file to be written
        Raises
        ------
        NotImplementedError
            If ``universe`` contains an interaction type that has not been
            implemented in the DL_POLY facade
        """

        LOGGER.info('%s Add topology to DL_POLY',
                    self.__class__)

        self.dlpoly.field = self._create_field(universe)
        self.dlpoly.field.write(settings.get('field', 'FIELD'))

        mx = max(i.cutoff for i in self.universe.nonbonded_interactions)
        print([i.cutoff for i in self.universe.nonbonded_interactions])
        self.dlpoly.control['cutoff'] = (mx, 'Ang')

    def _create_field(self, universe, **settings) -> Field:
        """
        Creates a dlpoly Field object

        Parameters
        ----------
        universe : Universe
            The MDMC ``Universe`` used to define the topology.
        **settings
            The majority of these are generic but some are specific to the
            ``MDEngine`` that is being used.
        header: str, header to be written to the field file
        units: str, kJ,kcal,eV, internal in what units, field file is written, default kJ
        Raises
        ------
        NotImplementedError
            If ``universe`` contains an interaction type that has not been
            implemented in the DL_POLY facade
        """

        (self.bonds, self.angles, self.dihedrals,
         self.disps, self.couls, others) = partition_interactions(
            set(universe.interactions),
            ['Bond', 'BondAngle', 'DihedralAngle', 'Dispersion', 'Coulombic'],
            unpartitioned=True,
            lst=True)

        if others:
            raise NotImplementedError('This interaction type has not been'
                                      ' implemented in the DL_POLY facade')

        out = Field()
        out.header = settings.get('header', 'MDMC Generated Field File')
        out.units = settings.get('units', 'kJ')

        spec = universe.element_lookup
        mols = {}

        for structure in universe.top_level_structure_list:
            if structure.name not in mols:
                newMol = Molecule()
                newMol.name = structure.name

                if isinstance(structure, MAtom):
                    species = structure.name
                    MDMC_spec = universe.element_dict[species]
                    newSpec = Species(species, 0,
                                      charge=MDMC_spec.charge, mass=MDMC_spec.mass)
                    newMol.nAtoms = 1
                    newMol.species = {1: newSpec}
                    mols[newMol.name] = newMol

                elif isinstance(structure, MMolecule):
                    mapping = {}
                    for ind, atm in enumerate(structure.atoms, 1):
                        MDMC_spec = universe.element_dict[atm.element]
                        newSpec = Species(atm.element, ind,
                                          MDMC_spec.charge, MDMC_spec.mass)
                        newMol.species[ind] = newSpec
                        newMol.nAtoms += 1
                        mapping[atm.ID] = ind

                    for bond, atms in structure.bonded_interaction_pairs:
                        currAtm = [mapping[atm.ID] for atm in atms]
                        if bond.constrained:
                            if not type(bond).__name__ == 'Bond':
                                raise NotImplementedError(f'{type(bond).__name__} constraints '
                                                          'not supported in DLPOLY')

                            pot = Bond('constraints',
                                       ['',
                                        *map(str, currAtm),
                                        str(bond.parameters[0].value.real)
                                        ])
                        else:
                            pot = Bond(BOND_CLASS_REF[type(bond).__name__],
                                       [POTENTIAL_REF[bond.function.name],
                                        *map(str, currAtm),
                                        *map(lambda x: str(x.value.real), bond.parameters)
                                        ])
                        newMol.add_potential(currAtm, pot)
                    mols[newMol.name] = newMol

                else:
                    raise TypeError(f'Unknown species type: {type(structure).__name__}')
            out.add_molecule(mols[structure.name])

        for disp in self.disps:
            currAtm = [spec[atm] for parm in disp.atom_types for atm in parm]
            pot = Potential('vdw', [*currAtm,
                                    POTENTIAL_REF[disp.function.name],
                                    *map(lambda x: str(x.value.real), disp.parameters)
                                    ])
            out.add_potential(currAtm, pot)

        return out

    def _update_charges(self):

        """
        Updates the ``charges`` in DL_POLY

        Raises
        ------
        AttributeError
            If one or more ``Atom`` do not have a ``charge`` (or
            ``charge is None``)
        """

        for atom in self.universe.top_level_structure_list:
            currAtom = self.dlpoly.field.molecules[atom.name]
            currAtom.charge = atom.charge

    def _update_dispersions(self, pair_coeff_cmds=None):

        """
        Updates ``Dispersion`` interactions in DL_POLY
        """

        spec = self.universe.element_lookup

        for disp in self.disps:
            currAtm = [spec[atm]
                       for parm in disp.atom_types
                       for atm in parm]
            currPot = next(self.dlpoly.field.get_pot(species=currAtm,
                                                     potType='lj'))
            currPot.params = [*map(lambda x: str(x.value.real), disp.parameters)]

    def _update_bonded_interactions(self):

        """
        Updates the bonded interaction coefficients, which are then applied to
        any bonded interactions which have previously been set
        """

        mols = self.dlpoly.field.molecules

        for structure in (x for x in self.universe.top_level_structure_list
                          if isinstance(x, MMolecule)):
            mapping = {atm.ID: str(ind) for ind, atm in enumerate(structure.atoms, 1)}
            mol = mols[structure.name]

            for bond, atms in structure.bonded_interaction_pairs:
                currAtm = [mapping[atm.ID] for atm in atms]
                if bond.constrained:
                    pot = next(mol.get_pot(species=currAtm))
                    pot.params = str(bond.parameters[0].value.real)

                else:
                    pot = next(mol.get_pot(species=currAtm,
                                           potClass=BOND_CLASS_REF[type(bond).__name__],
                                           potType=POTENTIAL_REF[bond.function.name]))
                    pot.params = [*map(lambda x: str(x.value.real), bond.parameters)]

    def apply_constraints(self):
        """
        Adds a constraint ``fix`` to DL_POLY
        for all bonds and bond angles which are constrained
        """

    def set_config(self, config):
        """
        Set DL_POLY config file
        """
        self.dlpoly.config = config


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
        Set the ``dlpoly`` attribute to a ``dlpoly-py`` object.
        Default is `None`,
        which results in a new ``dlpoly-py`` object being initialised.
    traj_step : int
        How many steps the simulation should take between dumping each
        ``Trajectory`` frame
    time_step : float, optional
        Simulation timestep in ``fs``, default is ``1.``
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
        Number of simulation steps that elapse
        between the ``Trajectory`` being stored.
    ensemble : DLPOLYEnsemble
        Simulation ensemble, which applies a ``thermostat`` and ``barostat``.
    **settings
        The majority of these are generic but some are specific to the
        ``MDEngine`` that is being used.
    temperature: float, temperatjre of the stimulation
    """

    def __init__(self, universe, traj_step: int, time_step: float = 1., dlpoly=None, **settings):
        super().__init__(dlpoly=dlpoly)

        self.universe = universe
        self.ensemble = DLPOLYEnsemble(self.dlpoly, **settings)
        self.temperature = settings.get('temperature')
        self.traj_step = traj_step
        self.time_step = time_step

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
        self.dlpoly.control['timestep'] = (
                convert_unit(self._time_step), str(SYSTEM['TIME']))

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
            self.dlpoly.control['temperature'] = (
                convert_unit(self._temperature), 'K')
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
    A thermodynamic ensemble determined by
    applying a thermostat and/or barostat

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

        # Requires a ``dlpoly-py`` object as thermostats
        # cannot be applied before configuration is defined
        super().__init__(dlpoly)
        self.temperature = temperature
        self.pressure = pressure
        self.dlpoly.control['ensemble'] = settings.get('ensemble', 'nve')

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
    'LENGTH': units.Unit('Ang'),
    'TIME': units.Unit('ps'),
    'MASS': units.Unit('amu'),
    'CHARGE': units.Unit('e'),
    'ANGLE': units.Unit('deg'),
    'TEMPERATURE': units.Unit('K'),
    'ENERGY': units.Unit('kcal') / units.Unit('mol'),
    'FORCE': units.Unit('kcal') / (units.Unit('Ang') * units.Unit('mol')),
    'PRESSURE': units.Unit('katm')
}


# some extra utility methods. these might be obsolete or
# importable from lammps_engine.py
# (in which case they maybe should be refactored into a utility module)
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
        except AttributeError as err:
            if value is None:
                raise ValueError('Cannot convert NoneType value') from err
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
        system_inv = {unit: property for property, unit in units.SYSTEM.items()}
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
