"""Facade for LAMMPS MD engine

This is a facade to ``PyLammps`` (added in 30th-Jul-2016 version), a convenience
wrapper for the LAMMPS Python interface i.e. where Python is extended with
LAMMPS.

Defining all interaction types requires that LAMMPS was built with the MOLECULE
package.

Notes
-----
When variables are either passed to or from ``PyLammps``, the ctypes conversion
can mean that they are unnecessarily cast, particularly from `float` to `int`.
This can cause issues as LAMMPS requires certain variables, e.g. number of
steps, to be `int`.  Therefore it is always a good idea to be cast these
variables when they are read from ``PyLammps`` e.g.
``int(lmp.variables['steps'].value)``.

A minor bug in LAMMPS (Dec 2018 version) means that ``nangletypes`` returned
by ``PyLammps`` is incorrectly set to ``ndihedraltypes``.  This was corrected in
Mar 2020 release.

The PyLammps class (which is what MDMC interfaces to) only outputs on the rank 0
process when using MPI. This means that accessing PyLammps properties should
only be done on rank 0, and broadcast to other ranks if necessary. This is also
true of calls to PyLammps.eval, and may also occur in other cases (anything that
ends up using the OutputCapture class).
"""
from __future__ import annotations

import warnings
from collections import defaultdict, namedtuple
from copy import copy
from itertools import chain, combinations, count, product
import logging
from random import randint
from tempfile import NamedTemporaryFile
from typing import Union
import os

import numpy as np

try:
    from lammps import PyLammps
except ModuleNotFoundError as err:
    raise ModuleNotFoundError('The Python interface for LAMMPS (lammps.py) is'
                              ' not in the PYTHONPATH. See LAMMPS documentation'
                              ' on Python to rectify this.'
                              ).with_traceback(err.__traceback__)

from MDMC.common.units import Unit
from MDMC.common import units
from MDMC.common.decorators import unit_decorator, unit_decorator_getter, \
    repr_decorator
from MDMC.MD.simulation import Universe, KSpaceSolver, ConstraintAlgorithm
from MDMC.MD.engine_facades.facade import MDEngine
from MDMC.MD.structures import Atom
from MDMC.MD.interactions import BondedInteraction, Interaction, \
    NonBondedInteraction, Bond, BondAngle
from MDMC.trajectory_analysis.compact_trajectory import CompactTrajectory
from MDMC.utilities.partitioning import partition, partition_interactions

LOGGER = logging.getLogger(__name__)

# pylint: disable=too-many-lines

class PyLammpsAttribute:

    """
    A class which has a ``PyLammps`` object as an attribute

    It possesses attributes and methods relating to the ``PyLammps`` object

    Parameters
    ----------
    lmp : PyLammps, optional
        Set the ``lmp`` attribute to a ``PyLammps`` object. Default is `None`,
        which results in a new ``PyLammps`` object being initialised.
    atom_style : str, optional
        The LAMMPS ``atom_style``, which determines the properties that can be
        associated which the atoms (e.g. ``charge``, ``bonds``). Default is
        ``full``.

    Attributes
    -----------
    lmp : PyLammps
        The ``PyLammps`` object owned by this class
    """

    def __init__(self, lmp: PyLammps = None, atom_style: str = 'full'):

        if lmp:
            self.lmp = lmp
        else:
            # Pass communicator to PyLammps to ensure consistency with process
            # ranks
            self.lmp = PyLammps()
            self.lmp.units('real')
            self.lmp.atom_style(atom_style)

        LOGGER.debug('%s: {lmp: %s, communicator: %s, atom_style: %s}. PyLammps'
                     ' instance %s.',
                     self.__class__,
                     self.lmp,
                     None,
                     atom_style,
                     'added to class' if lmp else 'created by class')

    @property
    def system_state(self) -> namedtuple:
        """
        Get the ``PyLammps`` wrapper system ``state`` `dict`

        Returns
        -------
        ``System``
                Contains the properties of the simulation box.
        """

        # Conversion from System class (which is a namedtuple) to
        # ordered dict required as System cannot be pickled
        system_state = self.lmp.system._asdict()
        # Cast back to namedtuple to remain consist with LAMMPS system attribute
        return namedtuple('System', system_state.keys())(*system_state.values())

    @property
    def fixes(self) -> "list[dict]":
        """
        Get the ``PyLammps`` wrapper `list` of ``fixes``

        Returns
        -------
        `list` of `dict`
            Each `dict` states the ``group``, ``name`` and ``style`` of a LAMMPS
            ``fix` which is applied
        """

        # PyLammps.fixes only exists on rank 0 process, so bcast
        fixes = self.lmp.fixes
        return fixes

    @property
    def fix_styles(self) -> "list[str]":
        """
        Get the styles of the ``fixes`` applied in LAMMPS

        Returns
        -------
        `list` of `str`
            The styles of the ``fixes``
        """

        return [fix['style'] for fix in self.fixes]

    @property
    def fix_names(self) -> "list[str]":
        """
        Get the names of the ``fixes`` applied in LAMMPS

        Returns
        -------
        `list` of `str`
            The names of the ``fixes``
        """

        return [fix['name'] for fix in self.fixes]

    @property
    def dumps(self) -> list:
        """
        Get the PyLammps wrapper list of dumps

        Dumps are LAMMPS commands which write atom quantities to file for
        specified timesteps
        """

        # PyLammps.dumps only exists on rank 0 process, so bcast.
        dumps = self.lmp.dumps
        return dumps


@repr_decorator('lmp', 'lmp_universe', 'lmp_simulation')
class LAMMPSEngine(PyLammpsAttribute, MDEngine):

    """
    Facade for LAMMPS

    One notable issue with creating the LAMMPS facade is that some
    ``pair_styles`` are combinations of both dispersion and coulombic
    interactions. For example, although LAMMPS has ``lj/cut``, ``coul/cut``, and
    ``coul/long``, there is no ``lj/long`` pair_style; it only exists in
    combination with ``coul/long`` (i.e. ``lj/long/coul/long``). This means that
    the facade has to not just parse each nonbonded interaction, but also has to
    determine if that ``pair_style`` should be combined with another
    ``pair_style`` (e.g. ``lj/long`` and ``coul/long`` have to be combined
    before passing to ``pair_style`` and ``pair_coeff``). An additional
    complication of this is that when setting the coefficients (``Parameters``)
    of these interactions, they again have to be set together. So in theory a
    coulombic interaction needs to know the dispersion interaction it has been
    paired with an also pass those coefficients when calling ``pair_coeff``. In
    practice a simplification can be made, at least in the case of currently
    implemented pair styles: As coulombic ``pair_styles`` do no take any
    coefficients, any coulombic ``pair_styles`` that comprise a combined
    ``pair_style`` can be ignored for the purposes of setting the coulombic
    ``pair_style``; this can be taken care of when ``pair_coeff`` is called from
    the correspoding dispersion interaction, as this possesses the dispersion
    coefficients that also need to be passed when the coefficients of this pair
    style are set.
    """

    def __init__(self):
        super().__init__()
        self.universe = None
        self.lmp_universe = None
        self._saved_config = None
        self.trajectory_file = None
        self.lmp_simulation = None

    @property
    def saved_config(self) -> np.ndarray:
        """
        Get the saved configuration of the atomic positions

        Returns
        -------
        numpy.ndarray
            The configuration from the start of the run. Each row of the
            ``array`` corresponds to the LAMMPS atom ``ID - 1`` (offset is due
            to array zero indexing) and the columns of the ``array`` are the
            ``x``, ``y``, ``z`` components of the ``position``, the ``mass`` and
            the ``charge`` of each ``Atom``.
        """

        return self._saved_config

    @property
    def temperature(self) -> float:
        """
        Get or set the temperature of the simulation in ``K``

        Returns
        -------
        `float`
            Temperature in ``K``
        """

        return self.lmp_simulation.temperature

    @temperature.setter
    @unit_decorator(unit=units.TEMPERATURE)
    def temperature(self, value: float) -> None:

        self.lmp_simulation.temperature = value

    @property
    def pressure(self) -> float:
        """
        Get or set the pressure of the simulation in ``atm``

        Returns
        -------
        `float`
            Pressure in ``atm``
        """

        return self.lmp_simulation.pressure

    @pressure.setter
    @unit_decorator(unit=units.PRESSURE)
    def pressure(self, value: float) -> None:

        self.lmp_simulation.pressure = value

    @property
    def ensemble(self) -> 'LAMMPSEnsemble':
        """
        Get or set the ensemble object which applies a ``thermostat`` and/or
        ``barostat`` to LAMMPS

        Returns
        -------
        LAMMPSEnsemble
            The simulation thermodynamic ensemble
        """

        return self.lmp_simulation.ensemble

    @ensemble.setter
    def ensemble(self, value: 'LAMMPSEnsemble') -> None:

        self.lmp_simulation.ensemble = value

    @property
    def thermostat(self) -> str:
        """
        Get or set the `str` which specifies the thermostat

        Returns
        -------
        `str`
            The ``thermostat`` name
        """

        return self.ensemble.thermostat

    @thermostat.setter
    def thermostat(self, value: str) -> None:

        self.ensemble.thermostat = value

    @property
    def barostat(self) -> str:
        """
        Get or set the `str` which specifies the barostat

        Returns
        -------
        `str`
            The ``barostat`` name
        """

        return self.ensemble.barostat

    @barostat.setter
    def barostat(self, value: str) -> None:

        self.ensemble.barostat = value

    def setup_universe(self, universe: Universe, **settings: dict) -> None:
        """
        Creates the simulation box, the atomic configuration, and the topology
        in LAMMPS

        Parameters
        ----------
        universe : Universe
            The MDMC ``Universe`` which will be setup in LAMMPS.
        **settings
            ``atom_style`` (`str`)
                A LAMMPS ``atom_style`` `str`. The default setting of ``real``
                will generally be appropriate.
        """

        super().__init__(atom_style=settings.get('atom_style', 'full'))
        self.universe = universe
        self.lmp_universe = LAMMPSUniverse(self.universe, self.lmp, **settings)
        self._saved_config = None

    def setup_simulation(self, **settings: dict) -> None:
        """
        Sets simulation parameters in LAMMPS, such as the thermodynamic
        variables, thermostat/barostat parameters and trajectory settings

        Parameters
        ----------
        **settings
            Passed to ``LAMMPSSimulation``
        """
        self.lmp_simulation = LAMMPSSimulation(universe=self.universe,
                                               time_step=self.time_step,
                                               traj_step=self.traj_step,
                                               lmp=self.lmp,
                                               **settings)

    def minimize(self, n_steps: int, output_log: str = None,
                 work_dir: str = None, **settings: dict) -> None:

        # Check fix styles for shake or rattle styles and remove them
        if 'constrain' in self.fix_names:
            LOGGER.debug('%s Remove %s constraint from fixes: %s',
                         self.__class__,
                         self.universe.constraint_algorithm,
                         self.fixes)
            self.lmp.unfix('constrain')

        etol = settings.get('etol', 1.e-4)
        ftol = settings.get('ftol', 0.)
        maxeval = settings.get('maxeval', 10000)
        LOGGER.info('%s minimize: {n_steps: %s, etol: %s, ftol: %s,'
                    ' maxeval: %s}',
                    self.__class__,
                    n_steps,
                    etol,
                    ftol,
                    maxeval)
        self.lmp.minimize(etol,
                          ftol,
                          n_steps,
                          maxeval)

        # Reapply the constraints
        if self.universe.constraint_algorithm:
            self.ensemble.remove_ensemble_fixes()
            self.lmp_universe.apply_constraints()
            self.ensemble.apply_ensemble_fixes()

    def run(self, n_steps: int, equilibration=False, output_log: str = None,
            work_dir: str = None, **settings: dict):
        if not equilibration:
            # Remove previous dumps if they exist
            if 'traj1' in [dump['name'] for dump in self.dumps]:
                self.lmp.undump('traj1')
            # Store the trajectory in a NamedTemporaryFile
            # pylint: disable=consider-using-with
            self.trajectory_file = NamedTemporaryFile()
            #     f_name = self.trajectory_file.name

            # Custom trajectory output just saves the atom ID, type and
            # positions
            LOGGER.debug('%s set trajectory dump output to %s',
                         self.__class__,
                         self.trajectory_file)
            self.lmp.dump('traj1', 'all', 'custom', self.traj_step,
                          self.trajectory_file.name, 'id', 'type', 'x', 'y',
                          'z')
        else:
            reset_to_nve = False
            if self.thermostat is self.barostat is None:
                # If NVE ensemble, add a berendsen thermostat for equilibration
                reset_to_nve = True
                self.thermostat = 'berendsen'

        LOGGER.info('%s run: {n_steps: %s, equilibration: %s}',
                    self.__class__,
                    n_steps,
                    equilibration)
        self.lmp.run(n_steps)

        if equilibration and reset_to_nve:
            self.thermostat = None

    def convert_trajectory(self, start: int = 0, stop: int = None,
                           step: int = 1, **settings: dict) -> CompactTrajectory:
        """
        Converts between a LAMMPS trajectory dump and an MDMC
        ``CompactTrajectory``

        The LAMMPS dump must include at least ``id``, ``atom_type``, and ``xyz``
        ``positions``. The ``xyz`` ``positions`` must be consecutive and in that
        order. The same is true of the ``xyz`` components of the ``velocity``, if
        they are provided.

        Parameters
        ----------
        start : int
            The index of the first trajectory, inclusive.
        stop : int
            The index of the last trajectory, exclusive.
        step : int
            The step size between trajectories.
        **settings
            ``scaled_positions`` (`bool`)
                If the ``trajectory_file`` has scaled ``positions``
            ``atom_IDs`` (`list`)
                LAMMPS ``ID`` of the atoms which should be included. If not passed
                then all atoms are included in the converted trajectory.

        Returns
        -------
        ``CompactTrajectory``
            The MDMC ``CompactTrajectory`` corresponding to the LAMMPS
            ``trajectory_file``.

        Raises
        ------
        AssertionError
            If ``universe`` is passed, and the number of atoms in the
            ``trajectory_file`` is not the same as in the ``universe``.
        TypeError
            If ``trajectory_file`` describes a triclinic universe.
        """

        # Change expected position string if scaled positions are used
        pos_string = 'xs' if settings.get('scaled_positions', False) else 'x'

        traj_dimensions = np.zeros(3)
        frame_n = start
        # Use count to create range so that stop can be undefined
        frame_indexes = count(start, step)
        # next_frame_n next attribute is assigned dynamically
        next_frame_n = next(frame_indexes)  # pylint: disable=no-member

        traj = CompactTrajectory() #the instance of our new trajectory object

        def _make_gen(reader):
            """A support function for splitting a binary file into buffers

            Args:
                reader (file descriptor): an open file or a file-like object

            Yields:
                bytes: a byte string read from the file
            """
            b = reader(1024 * 1024)
            while b:
                yield b
                b = reader(1024*1024)

        # here we check how long the trajectory really is
        with open(self.trajectory_file.name, 'rb') as file_handler:
            file_generator = _make_gen(file_handler.raw.read)
            line_count = sum( buf.count(b'\n') for buf in file_generator )
        # And header_size will tell us how many lines per frame
        # are added on top of the atom positions
        header_size = 0

        with open(self.trajectory_file.name, 'r', encoding='UTF-8') as file_handler:
            line = file_handler.readline()
            while line:

                # LAMMPS TIMESTEP is the number of time steps that have elapsed. To
                # avoid confusion with time_step (the amount of time that elapses in
                # a single simulation step, i.e. dt), these are referred to as
                # frames.
                if 'ITEM: TIMESTEP' in line:
                    line = file_handler.readline()
                    frame = int(line.split()[0])
                    header_size += 2

                if 'ITEM: NUMBER OF ATOMS' in line:
                    line = file_handler.readline()
                    n_atoms = int(line.split()[0])
                    header_size += 2
                    # Check that n_atoms is as expected, if a universe was passed
                    if self.universe:
                        assert n_atoms == len(self.universe.atoms)

                if 'ITEM: BOX BOUNDS' in line:
                    header_size += 1
                    traj_dimensions *= 0.0
                    temp_dim = []
                    # CURRENTLY ASSUMES ORTHOGONAL SIMULATION BOX
                    if 'xy' in line:
                        raise TypeError('triclinic simulation boxes have not'
                                        ' been implemented')
                    # Test dimensions are as expected, if a universe was passed
                    # and we are not using an NPT or NPH ensemble
                    for i in range(3):
                        line = file_handler.readline()
                        header_size += 1
                        dmin, dmax = [float(splt) for splt in line.split()]
                        temp_dim.append((dmin, dmax))
                        traj_dimensions[i] = dmax-dmin
                    if self.universe and not ('npt' in self.fix_names or 'nph' in self.fix_names):
                        for i in range(3):
                            dmin, dmax = temp_dim[i]
                            assert dmin == 0.0
                            # unit is taken from universe dimensions (which is a
                            # UnitArray)
                            assert dmax == convert_unit(self.universe.dimensions[i],
                                                        self.universe.dimensions.unit)

                if 'ITEM: ATOMS' in line:
                    header_size += 1
                    if frame_n == start:
                        # LAMMPS dump files contain order of LAMMPS atom properties,
                        # at each time step. As these should not change with time
                        # step only determine this order for first required time
                        # step. Assumes that position components (x y z) and
                        # velocity components (vx vy vz) are always adjacent and
                        # ordered as shown.
                        splt = line.split()
                        # Requires id, type and position to be defined, velocity is
                        # optional
                        i_id, i_type, i_pos = [splt.index(prop) - 2 for prop
                                               in ['id', 'type', pos_string]]
                        if 'vx' in splt:
                            i_vel = splt.index('vx')
                        else:
                            i_vel = None

                        # now we try to get the correct number of frames in the trajectory
                        real_n_steps = 1 + line_count // (n_atoms + header_size)

                        traj.preAllocate(n_steps = real_n_steps,
                                         n_atoms = n_atoms,
                                         useVelocity = i_vel is not None)

                        traj.setDimensions(traj_dimensions)

                    if frame_n == next_frame_n:
                        # Reads all atom lines before creating any atoms. By
                        # creating a list of tuples of (LAMMPS_ID, atom), this
                        # allows the lines to be reordered based on LAMMPS_ID. This
                        # is required as by default LAMMPS does not sort by ID, so
                        # the same atom will not appear in the same place for each
                        # time step.
                        header_size = 0
                        lines = []
                        for _ in range(n_atoms):
                            split_line = file_handler.readline().split()
                            # convert id to int
                            split_line[i_id] = int(split_line[i_id])
                            lines.append(split_line)
                        # sort list of lists based on id
                        lines = sorted(lines, key=lambda x: x[i_id])

                        sorted_lines = np.array(lines, dtype = traj.dtype)

                        atom_types = sorted_lines[:,i_type].astype(np.int64)
                        if not traj.validateTypes(atom_types):
                            raise TypeError("CompactTrajectory received wrong atom type array")

                        if i_vel is not None:
                            traj.writeOneStep(step_num = frame_n,
                                          time = frame * self.time_step,
                                          positions = sorted_lines[:,i_pos:i_pos+3],
                                          velocities = sorted_lines[:,i_vel:i_vel+3])
                        else:
                            traj.writeOneStep(step_num = frame_n,
                                          time = frame * self.time_step,
                                          positions = sorted_lines[:,i_pos:i_pos+3])

                        traj.setDimensions(traj_dimensions, step_num = frame_n)
                        # next_frame_n next attribute is assigned dynamically
                        # pylint: disable=no-member
                        next_frame_n = next(frame_indexes)
                    frame_n += 1
                    if stop is not None and frame_n >= stop:
                        break

                line = file_handler.readline()
        atom_symbols = {}
        atom_masses = {}
        for at_id in np.unique(traj.atom_types):
            symbol, mass = self.lmp_universe.atom_type_properties[at_id-1]
            atom_symbols[at_id] = symbol
            atom_masses[at_id] = mass
        traj.labelAtoms(atom_symbols, atom_masses)
        # electric charges, for completeness
        # (otherwise we cannot output an Atom from CompactTrajectory)
        charge_list = []
        for struc in self.parent_simulation.universe.structure_list:
            if isinstance(struc, Atom):
                charge_list.append([struc.ID, struc.charge])
        charges = np.array(charge_list)
        sequence = np.argsort(charges[:, 0])
        charges = charges[sequence][:, 1]
        traj.setCharge(charges)
        # we conclude the creation of the trajectory
        traj.postProcess()
        return traj

    def update_parameters(self) -> None:

        self.lmp_universe.update_parameters()

    def save_config(self) -> None:


        # It is not possible to deepcopy the LAMMPS wrapper atoms attribute,
        # or the individual atoms, so instead this saves the x, y, z, mass
        # and charge in a NumPy array with the indexes given by the atom ID
        # (with a -1 offset due to zero index)
        # The atoms attribute also is not iterable
        n_atoms = self.system_state.natoms
        LOGGER.info('%s save_config: {n_atoms: %s}. Config saved.',
                    self.__class__,
                    n_atoms)
        atoms = np.zeros([n_atoms, 5])
        tmp_mass = {}
        for type_ID, atom_type_group in self.lmp_universe.atom_types.items():
            tmp_mass[type_ID] = float(atom_type_group[0].mass)
        for i in range(n_atoms):
            atom = self.lmp.atoms[i]
            atom_type = atom.type
            # _, mass = self.lmp_universe.atom_type_properties[atom_type-1]
            atoms[atom.id-1, :] = (list(atom.position) + [tmp_mass[atom_type], atom.charge])
        saved_config = atoms
        self._saved_config = saved_config

    def reset_config(self) -> None:
        self.lmp_universe.set_config(self.saved_config)


@repr_decorator('universe')
class LAMMPSUniverse(PyLammpsAttribute):
    # Class has to maintain a lot of state (attributes) as PyLammps class does
    # not
    # pylint: disable=too-many-instance-attributes

    """
    A class with what would be the equivalent in LAMMPS to the universe (i.e.
    the configuration and topology)

    Parameters
    ----------
    universe : Universe
        The MDMC ``Universe`` used to create the ``LAMMPSUniverse``
    lmp : PyLammps, optional
        Set the ``lmp`` attribute to a ``PyLammps`` object. Default is `None`,
        which results in a new ``PyLammps`` object being initialised.
    **settings
        ``atom_style`` (`str`)
            A LAMMPS ``atom_style`` string. The default setting of ``real`` will
            generally be appropriate.
        ``nonbonded_mix`` (`str`)
            The name of the formula which determines non-bonded mixing

    Attributes
    ----------
    universe : Universe
        The MDMC ``Universe`` which has been converted to this
        ``LAMMPSUniverse``.
    atom_dict : dict
        A `dict` with {``MDMC_atom``: ``LAMMPS_atom_id``}, where ``MDMC_atom``
        is an MDMC ``Atom`` and ``LAMMPS_atom`` is the corresponding LAMMPS
        ``Atom``.
    atom_types : dict
        A `dict` with {``type_ID``: ``MDMC_atom_group``}, where the ``type_ID``
        is a unique `int` and ``MDMC_atom_group`` is a `list` of ``Atom`` which
        are identical in terms of element and interactions.
    atom_type_properties : list of tuples
        Each `tuple` is (``symbol``, ``mass``) for all ``atom_types`` (ordered)
        by ``atom_type``, where ``symbol`` is a `str` specifying the element of
        the ``atom_type`` and ``mass`` is a `float` specifying the ``mass`` of
        the ``atom_type``.
    bonds : list of Bonds
        All ``Bond`` interactions in the MDMC ``Universe``.
    angles : list of BondAngles
        All ``BondAngle`` interactions in the MDMC ``Universe``.
    couls : list of Coulombics
        All ``Coulombic`` interactions in the MDMC ``Universe``.
    disps : list of Dispersions
        All ``Dispersion`` interactions in the MDMC ``Universe``.
    bond_ID : dict
        A `dict` of {``bond``: ``ID pairs``} relating each ``Bond`` to a LAMMPS
        ``ID``.
    angle_ID : dict
        A `dict` of {``angle``: ``ID pairs``} relating each ``BondAngle`` to a
        LAMMPS ``ID``.
    proper_ID : dict
        A `dict` of {``proper``: ``ID pairs``} relating each ``DihedralAngle``
        (with ``improper == False``) to a LAMMPS ``ID``.
    improper_ID : dict
        A `dict` of {``improper``: ``ID pairs``} relating each ``DihedralAngle``
        (with ``improper == True``) to a LAMMPS ``ID``.
    """

    def __init__(self, universe: Universe, lmp: PyLammps = None, **settings: dict):

        super().__init__(lmp, settings.get('atom_style', 'full'))
        self.universe = universe
        self.atom_dict = {}
        self.atom_types = {}
        self.atom_type_properties = []

        self.bonds = []
        self.angles = []
        self.dihedrals = []
        self.disps = []
        self.couls = []
        self.propers = []
        self.impropers = []
        # ID is an acronym
        # pylint: disable=invalid-name
        self.bond_ID = {}
        self.angle_ID = {}
        self.proper_ID = {}
        self.improper_ID = {}
        self.nonbonded_mix = None

        if "OMP_NUM_THREADS" in os.environ:
            omp_num_threads_str = os.environ["OMP_NUM_THREADS"]
            try:
                omp_num_threads = int(omp_num_threads_str)
            except ValueError:
                pass
            else:
                if omp_num_threads > 1:
                    self.lmp.command("package omp 0")  # use OMP_NUM_THREADS
                    self.lmp.command("suffix omp")  # add /omp to relevant styles

        self._define_simulation_box(self.universe)
        self._build_config(self.universe)
        self._add_topology(self.universe, **settings)
        self.update_parameters()

    @property
    def nonbonded_mix(self) -> str:
        """
        Get or set the formula used to calculate nonbonded interactions between
        different ``atom_types``

        Options are ``geometric``, ``arithmetic`` and ``sixthpower``, which are
        defined in the LAMMPS documentation.

        Returns
        -------
        `str`
            The name of the formula which determines non-bonded mixing

        Raises
        ------
        ValueError
            `str` specifies an unsupported mix name
        """

        return self._nonbonded_mix

    @nonbonded_mix.setter
    def nonbonded_mix(self, value: str) -> None:

        if not value:
            self._nonbonded_mix = value
        else:
            supported = ['geometric', 'arithmetic', 'sixthpower']
            if value.lower() not in supported:
                raise ValueError('The supported mixes are: {0}, {1}, {2}'
                                 ''.format(*supported))
            self._nonbonded_mix = value.lower()

    def update_parameters(self) -> None:
        """
        Updates the LAMMPS force field parameters from the MDMC universe
        """

        self._update_charges()
        self._update_bonded_interactions('bond', self.bonds)
        self._update_bonded_interactions('angle', self.angles)
        self._update_bonded_interactions('dihedral', self.propers)
        self._update_bonded_interactions('improper', self.impropers)
        self._update_dispersions(self.universe)

    def _define_simulation_box(self, universe: Universe) -> None:
        """
        Defines a region and creates a simulation box that fills this region

        Parameters
        ----------
        universe : Universe
            The MDMC ``Universe`` used to create the region and simulation box.
        """

        # ID is an acronym
        # pylint: disable=invalid-name
        region_ID = 'universe'
        self._create_lammps_region(universe, region_ID)
        n_atom_types = len(universe.atom_types)

        # Determine number of bond and angle types
        bonded_interaction_types = [i.name for i in set(universe.interactions)
                                    if issubclass(type(i), BondedInteraction)]
        # Dihedrals are only separated into proper and improper at the attribute
        # level
        dihedrals = partition_interactions(set(universe.interactions),
                                           ['DihedralAngle'])[0]
        dihedral_types = [dihedral.improper for dihedral in list(dihedrals)]
        n_bond_types = bonded_interaction_types.count('Bond')
        n_angle_types = bonded_interaction_types.count('BondAngle')
        n_dihedral_types = dihedral_types.count(False)
        n_improper_types = dihedral_types.count(True)

        # Determine max number of bonds and angles per atom
        atoms = universe.atoms
        max_bonds_per_atom = self._max_n_interaction(atoms, 'Bond')
        max_angles_per_atom = self._max_n_interaction(atoms, 'BondAngle')
        max_dihedrals_per_atom = self._max_n_interaction(atoms, 'proper')
        max_impropers_per_atom = self._max_n_interaction(atoms, 'improper')
        LOGGER.debug('%s create LAMMPS box: {n_atom_types: %s'
                     ' n_bond_types: %s, n_angle_types: %s,'
                     ' n_dihedral_types: %s, n_improper_types: %s,'
                     ' max_bonds_per_atom: %s, max_angles_per_atom: %s,'
                     ' max_dihedrals_per_atom: %s, max_impropers_per_atom: %s}',
                     self.__class__,
                     n_atom_types,
                     n_bond_types,
                     n_angle_types,
                     n_dihedral_types,
                     n_improper_types,
                     max_bonds_per_atom,
                     max_angles_per_atom,
                     max_dihedrals_per_atom,
                     max_impropers_per_atom)
        self.lmp.create_box(n_atom_types,
                            region_ID,
                            'bond/types', n_bond_types,
                            'angle/types', n_angle_types,
                            'dihedral/types', n_dihedral_types,
                            'improper/types', n_improper_types,
                            'extra/bond/per/atom', max_bonds_per_atom,
                            'extra/angle/per/atom', max_angles_per_atom,
                            'extra/dihedral/per/atom', max_dihedrals_per_atom,
                            'extra/improper/per/atom', max_impropers_per_atom
                            )

    # ID is an acronym
    # pylint: disable=invalid-name
    def _create_lammps_region(self, universe: Universe, region_ID: str) -> None:
        """
        Create a geometry of the simulation box in LAMMPS

        Parameters
        ----------
        universe : Universe
            The MDMC ``Universe`` used to create the region.
        region_ID : str
            The LAMMPS region ``ID``
        """

        xlo = ylo = zlo = 0.
        xhi, yhi, zhi = universe.dimensions
        region_ID = 'universe'
        # 'block' gives a cuboidal universe
        self.lmp.region(region_ID, 'block', xlo, xhi, ylo, yhi, zlo, zhi,
                        units='box')

    def _build_config(self, universe: Universe) -> None:
        """
        Adds atoms to LAMMPS

        Parameters
        ----------
        universe : Universe
            The MDMC ``Universe`` used to fill the LAMMPS box with atoms.
        """
        self.atom_types = universe.atom_types
        # Assume all atoms of the same type have the same element and mass
        # Sort atoms based on atom_type (i.e. numerically starting at 1) so
        # that it is possible to index into atom_type_properties with
        # atom_type - 1 (where the -1 is to account for 0 indexing on lists)
        self.atom_type_properties = [(atom[0].element, atom[0].mass)
                                     for type, atom
                                     in sorted(self.atom_types.items())]

        LOGGER.info('%s Add atoms to LAMMPS',
                    self.__class__)
        for type_ID, atom_type_group in self.atom_types.items():
            # The mass below has associated units, which causes a segfault when
            # it is passed to LAMMPS - cast to float to remove units
            self.lmp.mass(type_ID, float(atom_type_group[0].mass))
            for atom in atom_type_group:
                self.lmp.create_atoms(type_ID, 'single', *atom.position)
                # As PyLammps has a bug preventing getting atom id from
                # self.lmp.atoms[index].id, use number of atoms as proxy for
                # new atom id (as it is sequential)
                lmp_atom_id = self.lmp.atoms.natoms
                self.lmp.set('atom', lmp_atom_id,
                                'vx', atom.velocity[0],
                                'vy', atom.velocity[1],
                                'vz', atom.velocity[2])

                self.atom_dict[atom] = lmp_atom_id

    def set_config(self, config: np.ndarray) -> None:
        """
        Changes the positions of all of the atoms in the LAMMPS wrapper

        Parameters
        ----------
        config : numpy.ndarray
            The ``positions``, ``mass`` and ``charge`` of the ``Atom`` objects,
            used to set the LAMMPS configuration. Each row of the array must
            correspond to the LAMMPS ``atom ID - 1`` (offset is due to ``array``
            zero indexing) and the columns of the ``array`` must be the ``x``,
            ``y``, ``z`` components of the ``position``, the ``mass`` and the
            ``charge`` of each ``Atom``.

        Raises
        ------
        IndexError
            If ``config`` does not contain the same number of atoms as LAMMPS
            possesses.
        """

        # Raise an IndexError if the config is not the correct size
        n_atoms = self.system_state.natoms
        if len(config) != n_atoms:
            raise IndexError('the new configuration does not specify the'
                             ' correct number of atoms')

        # The LAMMPS wrapper does not allow the configuration to be updated
        # simply by setting all atoms. Instead the position of the atoms must be
        # reset.
        index_components = list(enumerate(['x', 'y', 'z']))
        # LAMMPS IDs start at 1, so are offset from config indexes
        for id_offset in range(n_atoms):
            for index, component in index_components:
                self.lmp.set('atom', id_offset+1, component,
                             config[id_offset][index])

    @staticmethod
    def _max_n_interaction(atoms: list[Atom], name: str) -> int:
        """
        Parameters
        ----------
        atoms : list of Atoms
        name : str
            An ``Interaction`` type, for example ``Bond``.

        Returns
        -------
        `int`
            The maximum number of interactions with a given name that any
            ``Atom`` in ``atoms`` possesses
        """

        max_inters = 0
        for atom in atoms:
            # Filter interactions by name
            if name in ['proper', 'improper']:
                inters = filter(lambda i: i.name == 'DihedralAngle' and
                                i.improper == bool(name == 'improper'), atom.interactions)
            else:
                inters = filter(lambda i: i.name == name, atom.interactions)
            n_inters = len(list(inters))
            max_inters = n_inters if n_inters > max_inters else max_inters
        return max_inters

    def _add_topology(self, universe: Universe, **settings: dict) -> None:
        """
        Add the bonded and nonbonded interactions to LAMMPS

        Parameters
        ----------
        universe : Universe
            The MDMC ``Universe`` used to define the topology.
        **settings
            ``nonbonded_mix`` (`str`)
                The name of the formula which determines non-bonded mixing

        Raises
        ------
        NotImplementedError
            If ``universe`` contains an interaction type that has not been
            implemented in the LAMMPS facade
        """

        LOGGER.info('%s Add topology to LAMMPS',
                    self.__class__)
        # *_ is for pylint as it does not know about the output of partition_interactions
        bonds, angles, dihedrals, disps, couls, others, *_ = partition_interactions(
            set(universe.interactions),
            ['Bond', 'BondAngle', 'DihedralAngle', 'Dispersion', 'Coulombic'],
            unpartitioned=True,
            lst=True)

        if others:
            raise NotImplementedError('This interaction type has not been'
                                      ' implemented in the LAMMPS facade')

        self.bonds = bonds
        self.angles = angles
        self.dihedrals = dihedrals
        self.disps = disps
        self.couls = couls

        pair_styles, pair_mods, pair_coeff_cmds = self._pair_commands(universe)
        if pair_styles:
            LOGGER.debug('%s Add nonbonded pair styles to LAMMPS: {pair_styles:'
                         ' %s, pair_mods: %s, pair_coeff_cmds: %s}',
                         self.__class__,
                         pair_styles,
                         pair_mods,
                         pair_coeff_cmds)
            # hybrid/overlay allows multiple pair_styles for same atom_type pair
            self.lmp.pair_style('hybrid/overlay', *pair_styles)
            self._update_charges()
            self.nonbonded_mix = settings.get('nonbonded_mix')
            self._update_dispersions(universe, pair_coeff_cmds)
            # Apply LAMMPS modifications to nonbonded interactions
            self._modify_nonbonded_styles(pair_mods)

        if bonds:
            LOGGER.debug('%s Add bonds to LAMMPS',
                         self.__class__)
            # Set used to remove duplicate bond styles, which are not required
            # to be (and in fact cannot) be passed to LAMMPS hybrid bond_style
            self.lmp.bond_style('hybrid',
                                *set(tuple(parse_bonded_styles(b)
                                           for b in list(bonds))))
            self._create_bonded_interactions('bond', bonds)

        if angles:
            LOGGER.debug('%s Add angles to LAMMPS',
                         self.__class__)
            # Set used to remove duplicate angle styles, which are not required
            # to be (and in fact cannot) be passed to LAMMPS hybrid angle_style
            # pylint: disable=not-an-iterable
            # false positive raised here
            self.lmp.angle_style('hybrid',
                                 *set(tuple(parse_bonded_styles(a)
                                            for a in angles)))
            self._create_bonded_interactions('angle', angles)

        if dihedrals:
            LOGGER.debug('%s Add dihedrals to LAMMPS',
                         self.__class__)
            # Split dihedrals into proper and impropers
            impropers, propers = partition(dihedrals, lambda d: d.improper)
            self.impropers, self.propers = list(impropers), list(propers)
            # Set used to remove duplicate dihedral styles, which are not
            # required to be (and in fact cannot) be passed to LAMMPS hybrid
            # dihedral_style or improper_style
            proper_styles = set(tuple(parse_bonded_styles(p) for p
                                      in self.propers))
            improper_styles = set(tuple(parse_bonded_styles(i) for i
                                        in self.impropers))
            if proper_styles:
                self.lmp.dihedral_style('hybrid', *proper_styles)
                self._create_bonded_interactions('dihedral', self.propers)
            if improper_styles:
                self.lmp.improper_style('hybrid', *improper_styles)
                self._create_bonded_interactions('improper', self.impropers)

        if self.universe.constraint_algorithm:
            self.apply_constraints()

    @staticmethod
    def _pair_commands(universe: Universe) -> tuple:
        """
        Parses all the ``NonBondedInteractions`` for every appropriate
        combination of ``atom_type`` pairs in an MDMC ``Universe``, returning
        the correctly formatted input for ``pair_style`` and ``pair_coeff``
        LAMMPS commands.

        THE PARSING OF PAIR MODIFIERS IS LIMITED - if there is more than one
        interaction with the same ``pair_style``, and if these pair styles have
        different modifiers, all of the modifiers will be applied to this
        ``pair_style``. The correct implementation would apply each modifier to
        a copy of the ``pair_style``.

        Parameters
        ----------
        universe : Universe
            The MDMC ``Universe`` containing the ``NonBondedInteraction``
            objects to be parsed.

        Returns
        -------
        `tuple`
            (``pair_styles``, ``pair_modifiers``, ``pair_coeff_cmds``), where
            ``pair_styles`` is a flattened `list` of `str` for the
            ``pair_style`` commands to be set in the LAMMPS interface,
            ``pair_modifiers`` is a `list` of `list` of `str` for the
            ``pair_modify`` commands to be set in the LAMMPS interface, and
            ``pair_coeff_cmds`` is a `list` of `str` for the ``pair_coeff``
            commands for each ``atom_type`` pair to be set individually in the
            LAMMPS interface.
        """

        pair_styles = set()
        pair_modifiers = defaultdict(set)
        pair_coeff_cmds = []

        for pair, inters in universe.nbis_by_atom_type_pairs.items():
            # Generates list of tuples containing styles and cutoffs
            styles_mods = parse_all_nonbonded_styles(inters)
            # One list of pair modifier for
            for style, mod in styles_mods.items():
                # Only append if there is a modifier on the pair style
                if mod:
                    # Only the first style index is required as the rest contain
                    # cutoffs etc
                    pair_modifiers[style[0]].update(set(mod))
            # Generates list of tuples that contain each pair_coeff command
            parsed_coeffs = parse_dispersion_coefficients(inters,
                                                          styles_mods.keys())
            coeff_cmds = [' '.join([str(a_type) for a_type in pair]) + ' '
                          + coeff for coeff in parsed_coeffs]
            pair_styles.update(styles_mods.keys())
            pair_coeff_cmds += coeff_cmds

        # Remove duplicates in pair_styles, create flattened list
        pair_styles = list(chain.from_iterable(pair_styles))
        pair_modifiers = [[style, *mod] for style, mod
                          in pair_modifiers.items()]

        return pair_styles, pair_modifiers, pair_coeff_cmds

    def _update_charges(self) -> None:
        """
        Updates the ``charges`` in LAMMPS

        Raises
        ------
        AttributeError
            If one or more ``Atom`` do not have a ``charge`` (or
            ``charge is None``)
        """

        for atom, lmp_atom_id in self.atom_dict.items():
            try:
                self.lmp.set('atom',
                             lmp_atom_id,
                             'charge',
                             convert_unit(atom.charge))
            except ValueError as error:
                raise AttributeError('LAMMPS requires all atoms in the universe'
                                     ' to have a charge.') from error

    def _update_dispersions(self, universe: Universe, pair_coeff_cmds: 'list[str]' = None) -> None:
        """
        Updates ``Dispersion`` interactions in LAMMPS

        Parameters
        ----------
        universe : Universe
            The MDMC ``Universe`` containing ``NonBondedInteractions`` used to
            generate the ``pair_coeff`` commands if ``pair_coeff_cmds`` is not
            passed.
        pair_coeff_cmds : list of str
            A `list` of ``pair_coeff`` commands for each ``atom_type`` pair to
            be set individually in the LAMMPS interface.
            NOTE: the ``Coulombic`` for the appropriate ``atom_type`` pairs are
            also set in this method.
        """

        if not pair_coeff_cmds:
            _, _, pair_coeff_cmds = self._pair_commands(universe)

        for cmd in pair_coeff_cmds:
            self.lmp.pair_coeff(cmd)

    def _modify_nonbonded_styles(self, pair_mods: list) -> None:
        """
        Applies modifications to nonbonded ``pair_styles``, such as the VdW tail
        correction or setting a mixing style for interactions acting on unlike
        ``atom_type`` pairs

        Parameters
        ----------
        pair_mods : list
            A `list` of `list` which are valid ``pair_modify`` commands
        """

        # MDMC only allows nonbonding mixing on all NonbondedInteractions or
        # none
        if self.nonbonded_mix:
            self.lmp.pair_modify('mix', self.nonbonded_mix)

        for mod in pair_mods:
            self.lmp.pair_modify('pair', *mod)

    def _create_bonded_interactions(self, lmp_name: str,
                                    bonded_interactions: list[BondedInteraction]) -> None:
        """
        Creates coefficients and new bonded interactions in LAMMPS, and fills
        the relevant ``BondedInteraction`` ID (e.g. ``self.bond_ID`` for bonds,
        ``self.angle_ID`` for angles) `dict` with
        ``bonded_interaction``: ``ID pairs``

        This generic method can be used for bonds, angles, dihedrals (proper),
        and impropers. It can only be used for a single type of
        ``bonded_interactions`` (e.g. only bonds)

        Parameters
        ----------
        lmp_name : str
            The name of the bonded interaction type used for setting coeffs in
            LAMMPS. This must be one of: ``bond``, ``angle``, ``dihedral``,
            ``improper``. In the case of ``dihedral``, LAMMPS is just referring
            to proper dihedral interactions.
        bonded_interactions : list of bonded_interactions (or single type)
            ``BondedInteractions`` which will be created in LAMMPS. This list
            must only contain a single type of ``bonded_interactions`` (e.g.
            only bonds), which must correspond to ``lmp_name``.
        """

        special = 'no'
        ID_attr = getattr(self, '{0}_ID'.format(lmp_name)
                          if lmp_name != 'dihedral' else 'proper_ID')
        coeff_function = getattr(self.lmp, f'{lmp_name}_coeff')
        # If bonds already exist, new bond IDs are generated from lowest unused
        # integer
        if ID_attr:
            start = max(ID_attr.values()) + 1
        else:
            start = 1
        for ID, b_i in enumerate(bonded_interactions, start=start):
            # Create the bonded interaction coefficients
            coeff_function(ID, *parse_bonded_coefficients(b_i))

            # Relate each bonded interaction with its ID
            ID_attr[b_i] = ID

            # Create the bonded_interactions
            # Special triggers the internal interaction list in LAMMPS
            # This must at least occur at the end, and is an expensive
            # operation
            if b_i is bonded_interactions[-1]:
                special = 'yes'

            # LAMMPS create_bonds is used for creating all types of bonded
            # interactions, by appending the lmp_name to 'single/'
            c_b_type = f'single/{lmp_name}'
            for atom_tpl in b_i.atoms:
                atom_IDs = [self.atom_dict[atom] for atom in atom_tpl]
                self.lmp.create_bonds(c_b_type,
                                      ID,
                                      *atom_IDs,
                                      'special',
                                      special)

    def _update_bonded_interactions(self, lmp_name: str,
                                    bonded_interactions: list[BondedInteraction]) -> None:
        """
        Updates the bonded interaction coefficients, which are then applied to
        any bonded interactions which have previously been set

        Parameters
        ----------
        lmp_name : str
            The name of the bonded interaction type used for setting coeffs in
            LAMMPS. This must be one of: ``bond``, ``angle``, ``dihedral``,
            ``improper``. In the case of ``dihedral``, LAMMPS is just referring
            to proper dihedral interactions.
        bonded_interactions : list of BondedInteractions
            ``BondedInteractions`` which will be updated in LAMMPS. This list
            must only contain a single type of ``bonded_interactions`` (e.g.
            only bonds), which must correspond to ``lmp_name``.
        """

        # Get LAMMPS function for setting bonded interaction attributes (e.g.
        # bond_coeff)
        coeff_function = getattr(self.lmp, f'{lmp_name}_coeff')
        # Get ID dict attribute from self (e.g. bond_ID)
        b_i_IDs = getattr(self, '{0}_ID'.format(lmp_name)
                          if lmp_name != 'dihedral' else 'proper_ID')
        for b_i in bonded_interactions:
            coeff_function(b_i_IDs[b_i], *parse_bonded_coefficients(b_i))

    def apply_constraints(self) -> None:
        """
        Adds a constraint ``fix`` to LAMMPS for all bonds and bond angles which
        are constrained
        """

        # Sort bonded interactions in the Universe which are constrained into
        # bonds and angles
        b_inters = set(self.universe.bonded_interactions)
        # *_ is for pylint as it does not know about the output of partition_interactions
        bonds, angles, *_ = partition_interactions([inter for inter
                                                    in b_inters
                                                    if getattr(inter, 'constrained', False)],
                                                   ['Bond', 'BondAngle'], lst=True)
        algorithm = parse_constraint(self.universe.constraint_algorithm,
                                     bonds=bonds, bond_ID_dict=self.bond_ID,
                                     angles=angles, angle_ID_dict=self.angle_ID)

        # Create a group from all of the atom types in the constrained bonds and
        # angles - the fix will be applied to this group. Applying constraint
        # fix only to this group improves performance.
        # chain is used to flatten inter.atoms, which is a list of tuples
        atom_types = {atom.atom_type for inter in bonds+angles
                      for atom in chain.from_iterable(inter.atoms)}
        constrain_group = 'constrain_group'
        self.lmp.group(constrain_group, 'type', *atom_types)
        self.lmp.fix('constrain', constrain_group, *algorithm)


@repr_decorator('universe', 'time_step', 'traj_step', 'skin', 'neighbor_steps',
                'lin_momentum_steps', 'ang_momentum_steps', 'ensemble')
class LAMMPSSimulation(PyLammpsAttribute):
    # Class has to maintain a lot of state (attributes) as PyLammps class does
    # not
    # pylint: disable=too-many-instance-attributes

    """
    The attributes and methods related running a simulation in LAMMPS using a
    ``LAMMPSUniverse`` object

    Parameters
    ----------
    universe : Universe
        The MDMC ``Universe`` used to create the ``LAMMPSUniverse``.
    traj_step : int
        How many steps the simulation should take between dumping each
        ``CompactTrajectory`` frame
    time_step : float, optional
        Simulation timestep in ``fs``, default is ``1.``
    lmp : PyLammps, optional
        Set the ``lmp`` attribute to a ``PyLammps`` object. Default is `None`,
        which results in a new ``PyLammps`` object being initialised.
    **settings
        ``temperature`` (`float`)
        ``skin`` (`float`)
        ``neighbor_steps`` (`int`)
        ``remove_linear_momentum`` (`int`)
        ``remove_angular_momentum`` (`int`)
        ``velocity_seed`` (`int`): The seed to be used by LAMMPS to create velocities.

    Attributes
    ----------
    universe : Universe
        An MDMC ``Universe`` object.
    traj_step : int
        Number of simulation steps that elapse between the ``CompactTrajectory``
        being stored.
    ensemble : LAMMPSEnsemble
        Simulation ensemble, which applies a ``thermostat`` and ``barostat``.
    """

    def __init__(self, universe, traj_step: int, time_step: float = 1., lmp=None, **settings: dict):

        super().__init__(lmp, settings.get('atom_style', 'full'))
        self.universe = universe
        self.ensemble = LAMMPSEnsemble(self.lmp, **settings)
        self.velocity_seed = settings.get('velocity_seed', randint(1, 9999))
        self.temperature = settings.get('temperature')
        self.traj_step = traj_step
        self.time_step = time_step

        self.skin = settings.get('skin', 2.0)
        self.neighbor_steps = settings.get('neighbor_steps', 1)

        # Setting _lin_momentum_steps and _ang_momentum_steps allows
        # _set_momentum_removers to be called when setting
        # self.lin_momentum_steps and self.ang_momentum_steps
        self._lin_momentum_steps = None
        self._ang_momentum_steps = None
        self.lin_momentum_steps = settings.get('remove_linear_momentum', 1)
        self.ang_momentum_steps = settings.get('remove_angular_momentum')

        self._set_kspace_solver()

    @property
    def time_step(self) -> float:
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
    def time_step(self, value: float) -> None:

        self._time_step = value
        try:
            # Set the timestep in LAMMPS wrapper
            self.lmp.timestep(convert_unit(self._time_step))
        except ValueError:
            pass

    @property
    def temperature(self) -> float:
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
    def temperature(self, value: float) -> None:

        self._temperature = value
        self.ensemble.temperature = value
        try:
            # Set the initial temperature in the LAMMPS wrapper
            if self.system_state.natoms > 0:
                zero_velocity = [np.array_equal(atom.velocity, (0, 0, 0))
                                 for atom in self.universe.atoms]
                if all(zero_velocity):
                    # If we have not set any velocities (they are all the default value of zero)
                    # then "create" a velocity for each atom according to user-specified or
                    # random seed
                    self.lmp.velocity('all', 'create',
                                      convert_unit(self._temperature), self.velocity_seed)
                else:
                    if any(zero_velocity):
                        msg = ('Some but not all atom velocities set. Atoms with non-zero velocity'
                               ' will be re-scaled to match target temperature, atoms with zero '
                               'velocity will remain stationary.')
                        LOGGER.info('%s %s', self.__class__, msg)
                        print(msg)
                    # If we have set velocities then "scale" the velocities we have to the correct
                    # temperature
                    self.lmp.velocity(
                        'all', 'scale', convert_unit(self._temperature))
        except ValueError:
            pass

    @property
    def pressure(self) -> float:
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
    def pressure(self, value: float) -> None:

        self.ensemble.pressure = value

    @property
    def thermostat(self) -> str:
        """
        Get or set the string which specifies the thermostat

        Returns
        -------
        `str`
            The thermostat name
        """

        return self.ensemble.thermostat

    @thermostat.setter
    def thermostat(self, value: str) -> None:

        self.ensemble.thermostat = value

    @property
    def barostat(self) -> str:
        """
        Get or set the string which specifies the barostat

        Returns
        -------
        `str`
            The barostat name
        """

        return self.ensemble.barostat

    @barostat.setter
    def barostat(self, value: str) -> None:

        self.ensemble.barostat = value

    @property
    def skin(self) -> float:
        """
        Get or set the skin distance in ``Ang``

        Returns
        -------
        `float`
            The skin distance in ``Ang``. This distance plus the force
            ``cutoff`` distance determines which atom pairs are stored in the
            neighbor list.
        """

        return self._skin

    @skin.setter
    @unit_decorator(unit=units.LENGTH)
    def skin(self, value: float) -> None:

        self._skin = value
        # Set the neighor list parameters in the LAMMPS wrapper
        self.lmp.neighbor(convert_unit(self._skin), 'bin')

    @property
    def neighbor_steps(self) -> int:
        """
        Get or set the number of steps between neighbor list updates

        Returns
        -------
        `int`
            Number of steps between neighbor list updates
        """

        return self._neighbor_steps

    @neighbor_steps.setter
    def neighbor_steps(self, value: int) -> None:

        self._neighbor_steps = value
        try:
            # Set the modifiers to the neighbor list in the LAMMPS wrapper
            self.lmp.neigh_modify('every', int(self.neighbor_steps), 'delay', 0,
                                  'check', 'yes')
        except TypeError:
            pass

    @property
    def lin_momentum_steps(self) -> int:
        """
        Get or set the number of steps between resetting the linear momentum

        Returns
        -------
        `int`
            Number of steps between the linear momentum being removed
        """

        return self._lin_momentum_steps

    @lin_momentum_steps.setter
    def lin_momentum_steps(self, value: int) -> None:

        self._lin_momentum_steps = value
        # Set the momentum removers in the LAMMPS wrapper
        self._set_momentum_removers()

    @property
    def ang_momentum_steps(self) -> int:
        """
        Get or set the number of steps between resetting the angular momentum

        Returns
        -------
        `int`
            Number of steps between the angular momentum being removed
        """

        return self._ang_momentum_steps

    @ang_momentum_steps.setter
    def ang_momentum_steps(self, value: int) -> None:

        self._ang_momentum_steps = value
        # Set the momentum removers in the LAMMPS wrapper
        self._set_momentum_removers()

    def _set_momentum_removers(self) -> None:
        """
        Creates the ``fixes`` in LAMMPS which remove the linear and angular
        momentum of the simulation

        Removes all pre-existing momentum remover ``fixes``
        """

        for name in self.fix_names:
            if name in ['RemoveMomentum', 'RemoveLinearMomentum',
                        'RemoveAngularMomentum']:
                self.lmp.unfix(name)

        if self.system_state.natoms > 0:
            if self.lin_momentum_steps == self.ang_momentum_steps is not None:
                self.lmp.fix('RemoveMomentum', 'all', 'momentum',
                             self.lin_momentum_steps, 'linear', 1, 1, 1, 'angular')
            else:
                if self.lin_momentum_steps:
                    self.lmp.fix('RemoveLinearMomentum', 'all', 'momentum',
                                 self.lin_momentum_steps, 'linear', 1, 1, 1)
                if self.ang_momentum_steps:
                    self.lmp.fix('RemoveAngularMomentum', 'all', 'momentum',
                                 self.ang_momentum_steps, 'angular')

    def _set_kspace_solver(self) -> None:
        """
        Creates a k-space solver in LAMMPS using ``kspace_style``, if one is
        required

        Uses either the ``kspace_solver``, the ``electrostatic_solver`` or both
        the ``electrostatic_solver`` and ``dispersive_solver`` attribute of the
        MDMC ``Universe`` to set the ``kspace_style``. Note that LAMMPS does not
        support different electrostatic and dispersive solvers. Setting with
        equivalent electrostatic and dispserive solvers is equivalent to setting
        with ``kspace_solver``. LAMMPS also does not support just applying a
        dispersive solver.

        Raises
        ------
        TypeError
            If ``self.universe`` has both a ``kspace_solver`` and either an
            ``electrostatic_solver`` or a ``dispersive_solver``.
        TypeError
            If ``self.universe`` has a different ``electrostatic_solver`` and
            ``dispersive_solver``.
        TypeError
            If ``self.universe`` only has a ``dispersive_solver``.
        """

        kspace = self.universe.kspace_solver
        electrostatic = self.universe.electrostatic_solver
        dispersive = self.universe.dispersive_solver

        # LAMMPS supports a single kspace solver, which can be taken from kspace
        # electrostatic or dispersive solvers of the universe. If any other
        # solver is defined as well as kspace (which effecitvely defines both
        # other solves), raise an error.
        if kspace and (electrostatic or dispersive):
            raise TypeError('kspace solver cannot be applied in conjunction'
                            ' with either electrostatic or dispersive solvers')
        if kspace:
            self.lmp.kspace_style(*parse_kspace_solver(kspace))
        # If either an electrostatic or a dispersive solver has been defined,
        # use this. If both have been defined, this is valid as long as they are
        # equivalent.  If not, raise an error.
        if electrostatic:
            if dispersive and electrostatic != dispersive:
                raise TypeError('LAMMPS only supports a single solver, so if'
                                ' both dispersive and electrostatic solvers'
                                ' have been defined then they must be'
                                ' equivalent')
            self.lmp.kspace_style(*parse_kspace_solver(electrostatic))
        elif dispersive:
            raise TypeError('LAMMPS does not support only applying a dispersive'
                            ' solver - it must be applied in conjunction with'
                            ' an electrostatic solver')


@repr_decorator('temperature', 'pressure', 'thermostat', 'barostat')
class LAMMPSEnsemble(PyLammpsAttribute):
    # Class has to maintain a lot of state (attributes) as PyLammps class does
    # not
    # pylint: disable=too-many-instance-attributes

    """
    A thermodynamic ensemble determined by applying a thermostat and/or barostat

    Parameters
    ----------
    lmp : PyLammps
        Set the ``lmp`` attribute to a ``PyLammps`` object.
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
        ``time_step`` (`float`)
        ``t_damp`` (`int`)
        ``p_damp`` (`int`)
        ``t_window`` (`float`)
        ``t_fraction`` (`float`)
        ``rescale_step`` (`int`)

    Attributes
    ----------
    rescale_step : int
        Number of steps between applying temperature rescaling. This only
        applies to rescale thermostats.
    """

    def __init__(self, lmp, temperature: float = None, pressure: float = None,
                 thermostat: str = None, barostat: str = None, **settings: dict):

        # Requires a lmp object as thermostats cannot be applied before
        # configuration is defined
        super().__init__(lmp)
        # Setting _thermostat and _barostat allows apply_ensemble_fixes to be
        # called when setting self.temperature and self.pressure
        self._thermostat = None
        self._barostat = None
        self.temperature = temperature
        self.pressure = pressure

        self.time_step = settings.get('time_step', 1.0)
        self.t_damp = settings.get('t_damp', 100)
        self.p_damp = settings.get('p_damp', 1000)
        self.t_window = settings.get('t_window')
        self.t_fraction = settings.get('t_fraction')
        self.rescale_step = settings.get('rescale_step')

        self.thermostat = thermostat
        self.barostat = barostat

    @property
    def time_step(self) -> float:
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

    @property
    def temperature(self) -> float:
        """
        Get or set the temperature of the simulation in ``K``
        """

        return self._temperature

    @temperature.setter
    @unit_decorator(unit=units.TEMPERATURE)
    def temperature(self, value: float) -> None:

        self._temperature = value
        self.apply_ensemble_fixes()

    @property
    def pressure(self) -> float:
        """
        Get or set the pressure of the simulation in ``atm``
        """

        return self._pressure

    @pressure.setter
    @unit_decorator(unit=units.PRESSURE)
    def pressure(self, value: float) -> None:

        self._pressure = value
        self.apply_ensemble_fixes()

    # Unit has to be applied to getter due to operation in setter
    @property
    @unit_decorator_getter(unit=units.TIME)
    def t_damp(self) -> int:
        """
        Get or set the number of time steps over which the ``temperature`` is
        relaxed

        Required for ``Nose-Hoover``, ``Berendsen`` and ``Langevin``
        thermostats. The ``time_step`` must have been set before ``t_damp``.

        Returns
        -------
        `int`
            Number of time steps

        Raises
        ------
        AttributeError
            If ``self.time_step`` has not been set.
        """

        # t_damp is stored in units of time - convert back to number of steps
        # here
        try:
            return self._t_damp / self.time_step
        except TypeError:
            return self._t_damp

    @t_damp.setter
    def t_damp(self, value: int) -> None:

        try:
            # LAMMPS requires t_damp to be given in units of time, but it is
            # more natural to give it in units of steps - convert between them
            # here
            self._t_damp = value * self.time_step
        except TypeError as error:
            if value is None:
                self._t_damp = value
            else:
                raise AttributeError('the time_step attribute must be set'
                                     ' before t_damp') from error

    # Unit has to be applied to getter due to operation in setter
    @property
    @unit_decorator_getter(unit=units.TIME)
    def p_damp(self) -> int:
        """
        Get or set the number of time steps over which the ``pressure`` is
        relaxed

        Required for ``Nose-Hoover`` or ``Berendsen`` barostats. The
        ``time_step`` must have been set before ``p_damp``.

        Returns
        -------
        `int`
            Number of time steps

        Raises
        ------
        AttributeError
            If ``self.time_step`` has not been set.
        """

        # p_damp is stored in units of time - convert back to number of steps
        # here
        try:
            return self._p_damp / self.time_step
        except TypeError:
            return self._p_damp

    @p_damp.setter
    def p_damp(self, value: int) -> None:

        try:
            # LAMMPS requires p_damp to be given in units of time, but it is
            # more natural to give it in units of steps - convert between them
            # here
            self._p_damp = value * self.time_step
        except TypeError as error:
            if value is None:
                self._p_damp = value
            else:
                raise AttributeError('the time_step attribute must be set'
                                     ' before p_damp') from error

    @property
    def t_fraction(self) -> float:
        """
        Get or set the fraction by which the ``temperature`` is rescaled to the
        target temperature

        This is required for the rescale ``thermostat``.

        Returns
        -------
        `float`
            Fraction (i.e. between 0.0 and 1.0 inclusive) by which the
            ``temperature`` is rescaled

        Raises
        ------
        ValueError
            If set to a value outside of 0.0 and 1.0 inclusive.
        """

        return self._t_fraction

    @t_fraction.setter
    def t_fraction(self, value: float) -> None:

        # Must be a fraction
        if value is None or 0. <= value <= 1.0:
            self._t_fraction = value
        else:
            raise ValueError('the t_fraction must be between 0.0 and 1.0')

    @property
    def t_window(self) -> float:
        """
        Get or set the ``temperature`` range in ``K`` in which the
        ``temperature`` is not rescaled

        This only applies to rescale ``thermostats``.

        Returns
        -------
        `float`
            temperature range in ``K``
        """

        return self._t_window

    @t_window.setter
    @unit_decorator(unit=units.TEMPERATURE)
    def t_window(self, value: float) -> None:

        self._t_window = value

    @property
    def thermostat(self) -> str:
        """
        Get or set the `str` which specifies the thermostat

        Raises
        ------
        AttributeError
            If ``self.temperature`` has not been set.
        """

        return self._thermostat

    @thermostat.setter
    def thermostat(self, value: str) -> None:

        if value and not self.temperature:
            raise AttributeError('all ensembles with a thermostat must have a'
                                 ' temperature')
        self._thermostat = value
        # Set the thermostat and barostat in LAMMPS wrapper
        self.apply_ensemble_fixes()

    @property
    def barostat(self) -> str:
        """
        Get or set the `str` which specifies the barostat

        Raises
        ------
        AttributeError
            If ``self.pressure`` has not been set.
        """

        return self._barostat

    @barostat.setter
    def barostat(self, value: str) -> None:

        if value and not self.pressure:
            raise AttributeError('all ensembles with a barostat must have a'
                                 ' pressure')

        self._barostat = value
        # Set the thermostat and barostat in LAMMPS wrapper
        self.apply_ensemble_fixes()

    def remove_ensemble_fixes(self) -> None:
        """
        Removes all LAMMPS ``fixes`` relating to the ensemble i.e. removes all
        thermostats and barostats

        This must be done before thermostat and barostat fixes are added, so
        that there is no conflict with existing thermostat and barostats
        ``fixes``. It is also required for Shake and Rattle ``fixes`` which
        cannot be added after barostat fixes have been applied.
        """

        for name in self.fix_names:
            if name in ['nve', 'nvt', 'npt', 'nph', 't_berendsen',
                        'p_berendsen', 'langevin', 'rescale']:
                LOGGER.debug('%s Remove %s fix.',
                             self.__class__,
                             name)
                self.lmp.unfix(name)

    def apply_ensemble_fixes(self) -> None:
        """
        Passes the required LAMMPS ``fixes`` to apply a specific thermodynamic
        ensemble to the simulation

        Removes all pre-existing thermostat and barostat fixes
        """

        self.remove_ensemble_fixes()

        LOGGER.debug('%s apply_ensemble_fixes before fixes applied:'
                     ' {fixes: %s}',
                     self.__class__,
                     self.fixes)

        if not self.thermostat and not self.barostat:
            self.lmp.fix('nve', 'all', 'nve')
        else:
            if self.thermostat:
                temp = convert_unit(self.temperature)
                if self.thermostat != 'rescale':
                    t_damp = convert_unit(self.t_damp)
                    thermo_parameters = [temp, temp, t_damp]
            if self.barostat:
                press = convert_unit(self.pressure)
                p_damp = convert_unit(self.p_damp)
                press_parameters = ['iso', press, press, p_damp]

            # Apply thermostat
            # we create local funcs for each thermostat, and use a dict to get the correct one.
            # think of this like a proto-factory pattern, or like a switch-case block in C++.

            def nose():
                if self.barostat == 'nose':
                    self.lmp.fix('npt', 'all', 'npt', 'temp',
                                 *thermo_parameters + press_parameters)
                else:
                    self.lmp.fix('nvt', 'all', 'nvt',
                                 'temp', *thermo_parameters)

            def berendsen():
                # berendsen does not do time integration so also requires nve
                self.lmp.fix('nve', 'all', 'nve')
                self.lmp.fix('t_berendsen', 'all', 'temp/berendsen',
                             *thermo_parameters)

            def langevin():
                # langevin does not do time integration so also requires nve
                self.lmp.fix('nve', 'all', 'nve')
                self.lmp.fix('langevin', 'all', 'langevin',
                             *thermo_parameters + [randint(0, 9999)])

            def rescale():
                # temp/rescale does not do time integration so also requires nve
                t_window = convert_unit(self.t_window)
                self.lmp.fix('nve', 'all', 'nve')
                self.lmp.fix('rescale', 'all', 'temp/rescale',
                             self.rescale_step, temp, temp, t_window,
                             self.t_fraction)

            def csvr():
                # csvr does not do time integration so also requires nve
                self.lmp.fix('nve', 'all', 'nve')
                self.lmp.fix('csvr', 'all', 'temp/csvr',
                             *thermo_parameters + [randint(0, 9999)])

            thermostat = {'nose': nose,
                          'berendsen': berendsen,
                          'langevin': langevin,
                          'rescale': rescale,
                          'csvr': csvr}

            try:
                thermostat[self.thermostat]()
            except KeyError:
                warnings.warn("Thermostat type not recognised. No thermostat will be applied."
                              " Your thermostat type may be misspelt or not currently implemented.")

            # Apply barostat
            if self.barostat == 'berendsen':
                self.lmp.fix('p_berendsen', 'all', 'press/berendsen',
                             *press_parameters)
            elif self.barostat == 'nose' and self.thermostat != 'nose':
                if 'nve' in self.fix_names:
                    self.lmp.unfix('nve')
                self.lmp.fix('nph', 'all', 'nph', *press_parameters)

        LOGGER.debug('%s apply_ensemble_fixes after fixes applied:'
                     ' {fixes: %s}',
                     self.__class__,
                     self.fixes)


# Define the unit system used in LAMMPS
# NB: LAMMPS uses deg for angle but radian for derived quantities of angle:
# e.g. harmonic angle potential strength is in kcal / mol radian ^ 2
SYSTEM = {
    'LENGTH': units.Unit('Ang'),
    'TIME': units.Unit('fs'),
    'MASS': units.Unit('g') / units.Unit('mol'),
    'CHARGE': units.Unit('e'),
    'ANGLE': units.Unit('deg'),
    'TEMPERATURE': units.Unit('K'),
    'ENERGY': units.Unit('kcal') / units.Unit('mol'),
    'FORCE': units.Unit('kcal') / (units.Unit('Ang') * units.Unit('mol')),
    'PRESSURE': units.Unit('atm')
}


def convert_unit(value: Union[np.ndarray, float],
                 unit: Unit = None, to_lammps: bool = True):
    """
    Converts between MDMC units and LAMMPS real units

    Parameters
    ----------
    value : array_like or float_like
        The value of the physical property to be converted, in MDMC units.
        Must derive from either ``ndarray`` or `float`.
    unit : Unit, optional
        The unit of the ``value``. If `None`, the ``value`` must possess a
        ``unit`` attribute i.e. derive from ``UnitFloat`` or ``UnitArray``.
        Default is `None`.
    to_lammps : bool, optional
        If `True` the conversion is from MDMC units to LAMMPS units. Default is
        `True`.

    Returns
    -------
    `float` or `numpy.ndarray`
        Value in LAMMPS units if ``to_lammps`` is `True`, otherwise value in
        MDMC units. Return type is same as ``value`` type.
    """

    def expand_components(unit: Unit, system: dict) -> tuple:
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

        def is_sublist_of_list(sub: list, lst: list) -> bool:
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

        def remove_components(remove_comps: list, comps: list) -> list:
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
        except AttributeError as error:
            if value is None:
                raise ValueError('Cannot convert NoneType value') from error
            return value
    # Expand the unit in terms of its base units (for numerator and denominator)
    if to_lammps:
        # SYSTEM represents the LAMMPS system units, units.SYSTEM the MDMC
        # system units
        l_sys = copy(SYSTEM)
        mdmc_sys = copy(units.SYSTEM)

        # For angular potential strength LAMMPS uses units derived from 'rad',
        # namely 'kcal / mol rad^2', but uses 'deg' when measuring angles
        # themselves. As a result 'deg' is recorded as their system unit. In
        # order to prevent us from erroneously expressing potential strength in
        # 'kcal / mol deg^2', we must override the LAMMPS system unit for angle
        # ONLY in the case where we are dealing with angular potential, namely
        # when the MDMC unit is measuring 'energy / angle^2'.
        if (unit in (mdmc_sys['ENERGY'] / mdmc_sys['ANGLE'] ** 2,
                     mdmc_sys['ENERGY'] / units.Unit('rad') ** 2)):
            l_sys['ANGLE'] = units.Unit('rad')

        expanded_unit = expand_components(unit, mdmc_sys)
        # For each MDMC unit, determine the LAMMPS system unit that corresponds
        # to that physical property (e.g. 'kJ / mol' -> 'ENERGY' -> 'kcal / mol')
        l_unit_nums, l_unit_denoms = map(lambda unit_comps: [l_sys[comp.physical_property]
                                                             for comp in unit_comps],
                                         expanded_unit)

        # In general we may have an MDMC measurement that is not wholly
        # expressed in MDMC system units, for example 'kJ / mol rad^2' uses
        # 'rad' rather than 'deg'. To account for this, multiply by the
        # conversion_factor of the original MDMC unit to get the value in terms
        # of MDMC system units only.
        value *= unit.conversion_factor

        # Then, for each component of the LAMMPS unit, divide by the
        # conversion_factor to go from MDMC system units to LAMMPS system units.
        for component in l_unit_nums:
            value /= component.conversion_factor
        for component in l_unit_denoms:
            value *= component.conversion_factor

    else:
        value *= unit.conversion_factor

    return value


def parse_bonded_styles(interaction: BondedInteraction) -> str:
    """
    Converts MDMC ``InteractionFunction`` names for ``BondedInteractions`` to
    LAMMP bond styles

    Parameters
    ----------
    interaction : BondedInteraction
        ``BondedInteraction`` to be parsed into LAMMPS bond style.

    Returns
    -------
    `str`
        LAMMPS bond style corresponding to ``interaction``

    Raises
    ------
    NotImplementedError
        If ``interaction`` has a ``function`` that has not been implemented in
        the LAMMPS facade.
    """

    if interaction.function_name == 'HarmonicPotential':
        if interaction.name == 'DihedralAngle' and not interaction.improper:
            raise TypeError(
                'LAMMPS does not support harmonic proper dihedrals')
        return 'harmonic'
    if interaction.function_name == 'Periodic':
        if interaction.name != 'DihedralAngle':
            raise TypeError('LAMMPS only supports periodic interaction function'
                            ' for Dihedrals')
        if interaction.improper:
            # LAMMPS has an improper_style called cvff, which is a Simplified
            # version of Periodic (it is only first order and d1=0). Return cvff
            # here and deal with incompatible Periodic interactions (e.g. d1!=0)
            # in parse_bonded_coefficients
            return 'cvff'
        return 'fourier'
    raise NotImplementedError('This InteractionFunction has not been'
                              ' implemented in the LAMMPS facade')


def parse_nonbonded_styles(interaction: NonBondedInteraction) -> tuple:
    """
    Converts MDMC ``InteractionFunction`` names for ``NonBondedInteractions`` to
    LAMMPS pair styles

    Parameters
    ----------
    interaction : NonBondedInteraction
        ``NonBondedInteraction`` to be parsed into LAMMPS pair style.

    Returns
    -------
    `tuple`
        (``styles``, ``mods``) where ``styles`` is a `list` of `str` of LAMMPS
        pair style corresponding to the ``interaction``, and ``mods`` is a
        `list` of `str` of LAMMPS pair modifications corresponding to the
        interaction

    Raises
    ------
    NotImplementedError
        If ``interaction`` has a ``function`` that has not been implemented in
        the LAMMPS facade.
    """

    lmp_str = []
    if interaction.function_name == 'Buckingham':
        lmp_str.append('buck')
    elif interaction.function_name == 'LennardJones':
        lmp_str.append('lj')
    elif interaction.function_name == 'Coulomb':
        lmp_str.append('coul')
    else:
        raise NotImplementedError('This InteractionFunction has not been'
                                  ' implemented in the LAMMPS facade')

    if interaction.cutoff:
        cutoff = convert_unit(interaction.cutoff)
        kspace = interaction.universe.kspace_solver
        electrostatic = interaction.universe.electrostatic_solver
        dispersive = interaction.universe.dispersive_solver
        if (kspace or (electrostatic and interaction.name == 'Coulombic')
                or (dispersive and interaction.name == 'Dispersion')):

            lmp_str[-1] += '/long'
        else:
            if interaction.function_name != 'Buckingham':
                lmp_str[-1] += '/cut'
        lmp_str.append(cutoff)
    else:
        raise AttributeError(f'You have not specified a `cutoff` for the'
                                  f'{interaction} InteractionFunction.')

    return lmp_str, parse_nonbonded_modifications(interaction)


def parse_nonbonded_modifications(interaction: Interaction) -> 'list[str]':
    """
    Parses MDMC ``Interaction`` attributes into `list` that can be used with
    LAMMPS ``pair_modify`` command

    Attributes
    ----------
    interaction : Interaction
        The ``Interaction`` for which the attributes are parsed

    Returns
    -------
    `list`
        A `list` of `str` which are a valid ``pair_modify`` command, or an empty
        `list` if no ``Interaction`` attributes require setting ``pair_modify``
    """

    # LAMMPS pair_modify is of the following form:
    # pair_modify('pair', 'lj/cut', 'mix', 'geometric', 'tail', 'yes')
    # pair_modify('coul/long', 'mix', 'arithmetic')
    # where each pair style (lj/cut, coul/long etc) has any mix or tail keywords
    # defined after the pair style. If the same pair_style occurs multiple times
    # but with different modifiers
    # (e.g. 'lj/cut', 'mix', 'geometric' and 'lj/cut', 'mix', 'arithmetic')
    # whichever pair_modify occurs last will be applied to all identical
    # pair_styles.

    mods = []
    if interaction.name == 'Dispersion' and interaction.vdw_tail_correction:
        mods.extend(['tail yes'])

    return mods


def parse_all_nonbonded_styles(interactions: NonBondedInteraction) -> 'dict[tuple, list[str]]':
    """
    Converts all ``NonBondedInteractions`` to LAMMPS pair styles

    This is required because LAMMPS frequently treats ``Coulombic`` and
    ``Disperion`` interactions together; these cases need to be dealt with to
    generate the correct input to LAMMPS ``pair_styles``.  For example, while
    the ``pair_styles`` ``buck``, ``lj/cut``, ``coul/cut`` and ``coul/long`` can
    all be passed separately, the dispersive and coulombic styles are combined
    (dispersive always preceding coulombic) as this is dealt with more
    efficiently in the LAMMPS engine. ``buck/long`` and ``lj/long`` only exist
    as part of other pair styles, such as ``buck/long/coul/long`` and
    ``lj/long/coul/long``, so must be combined.

    Parameters
    ----------
    interactions : list of NonBondedInteractions
        ``NonBondedInteractions`` to be parsed into LAMMPS ``pair_styles``.

    Returns
    -------
    `dict`
        A `dict` with {``pair_styles``: ``pair_modifiers``} where
        ``pair_styles`` is a `tuple` of all of the combined LAMMPS pair styles
        corresponding to ``interactions``. Each `tuple` contains the combined
        ``pair_style`` as one element, and the ``cutoffs`` as the second element
        e.g. ``('lj/cut/coul/cut', 5.0 10.0)``. If additional arguments are
        required when combining styles, they will be added and returned in the
        `tuple` as the third and middle element
        e.g. ``('buck/long/coul/long', 'long long', '10.0'``).
        ``pair_modifiers`` is a `list` of `str` for setting ``pair_modify`` for
        the corresponding ``pair_style``.

    Raises
    ------
    ValueError
        If ``Dispersion`` and ``Coulombic`` interactions have different long
        range ``cutoff``, which is not implemented in LAMMPS.
    ValueError
        If long range ``Dispersion`` not defined in conjunction with a long
        range ``Coulombic``.
    """

    def check_validity(pair_style: str, cutoffs: 'list[float]' = None) -> 'list[str]':
        """
        Tests the validity of a LAMMPS ``pair_style``.

        Parameters
        ----------
        pair_style : str
            The LAMMPS ``pair_style`` to be checked.
        cutoffs : list of float
            The cutoff distances for each style in the ``pair_style``.

        Returns
        -------
        `list` of `str`
            Contains the inputted ``pair_style`` (if valid) as well as any extra
            terms required in specific cases (i.e. ``'long long'`` if the pair
            style ``'buck/long/coul/long'`` is passed).

        Raises
        ------
        ValueError
            If the ``pair_style`` passed is not valid.
        """

        if pair_style in ['buck/long/coul/cut', 'lj/long/coul/cut']:
            raise ValueError('Invalid pair_style: ' + pair_style
                             + '. LAMMPS requires long range Coulombics to be'
                             ' defined in conjunction with long range Dispersion'
                             ' interactions.')
        if pair_style in ['buck/long/coul/long', 'lj/long/coul/long']:
            if cutoffs[0] != cutoffs[1]:
                raise ValueError('LAMMPS requires both cutoffs to be the same'
                                 ' for long range Dispersion and Coulombic pair'
                                 ' styles.')
            return [pair_style, 'long long']

        return [pair_style]

    # Some pair styles cannot have vdw tail corrections modifiers - exclude
    # these and warn about this exclusion
    # This is dealth with here rather than in parse_nonbonded_modifications as
    # it is dependent on combined pair_styles
    excluded = ['lj/long/coul/long']

    # Change parse_nonbonded_styles to return the parsed nonbonded style and the
    # parsed_ modifiers from that interaction
    # Create a dictionary of parsed_interactions (keys) and modifiers (values)
    single_parsed_inters = {tuple(parsed[0]): parsed[1] for parsed in
                            [parse_nonbonded_styles(nb) for nb in interactions]}
    combined_parsed_inters = copy(single_parsed_inters)
    # Define all Dispersion and Coulombic styles currently supported
    # Dispersion styles always precede coulombic styles in LAMMPS pair styles
    disp_styles = ['buck', 'buck/long', 'lj/cut', 'lj/long']
    coul_styles = ['coul/cut', 'coul/long']

    for d_style, c_style in product(disp_styles, coul_styles):
        dc_styles = [d_style, c_style]
        for int1, int2 in combinations(single_parsed_inters.keys(), 2):
            if (int1[0] in dc_styles and int2[0] in dc_styles):
                indiv_cmd = check_validity('/'.join(dc_styles),
                                           cutoffs=[int1[1], int2[1]])
                # Format cutoffs so that disp cutoff precedes coul cutoff if
                # they are different
                if int1[1] == int2[1]:
                    cutoffs = str(int1[1])
                else:
                    d_cut, c_cut = ((int1[1], int2[1]) if int1[0] == d_style
                                    else (int2[1], int1[1]))
                    cutoffs = f'{d_cut} {c_cut}'

                # Add indiv_cmd to parsed_interactions dict instead. Use
                # modifier from parsed_interactions as value. set is used to
                # remove duplicated modifiers which occur for both interactions.
                # Delete int1 and int2 from parsed_interactions
                mod = set(single_parsed_inters[int1]
                          + single_parsed_inters[int2])
                # Warn if incompatible pair_style and pair_modification have
                # been used
                if indiv_cmd in excluded and 'tail yes' in mod:
                    LOGGER.warning('parse_all_nonbonded_styles: %s pair style'
                                   ' cannot have a vdw tail correction applied',
                                   indiv_cmd)
                    mod = set(md for md in mod if md != 'tail yes')
                combined_parsed_inters[tuple(
                    indiv_cmd + [cutoffs])] = list(mod)
                for key in [int1, int2]:
                    try:
                        del combined_parsed_inters[key]
                    except KeyError:
                        pass

    return combined_parsed_inters


def parse_bonded_coefficients(interaction: BondedInteraction) -> 'list[str]':
    """
    Orders MDMC ``Parameter`` objects for input to LAMMPS ``bond_coeff`` and
    ``angle_coeff``

    Parameters
    ----------
    interaction : BondedInteraction
        ``BondedInteraction`` where its style and parameters will be parsed.

    Returns
    -------
    `list` of `str`
        Style and parameters converted to the input format for LAMMPS
        ``bond_coeff`` and ``angle_coeff``

    Raises
    ------
    NotImplementedError
        If ``interaction`` has a ``function`` that has not been implemented in
        the LAMMPS facade.
    """

    parameters = {p.type: convert_unit(p.value)
                  for p in interaction.parameters.as_array}

    style = parse_bonded_styles(interaction)

    if style == 'harmonic':
        ordered_parameters = [parameters['potential_strength'],
                              parameters['equilibrium_state']]
    elif style == 'fourier':
        # There are three parameters (K, n and d) for each order of the equation
        fourier_order = int(len(interaction.parameters) / 3)
        # LAMMPS requires parameters to be ordered K1, n1, d1, K2, n2, d2 etc
        # So the letter sequence is:
        ord_let = ('K', 'n', 'd')
        # Sort by fourier_order first and then by letter sequence
        ordered_p_names = sorted(parameters,
                                 key=lambda p_type: (p_type[1],
                                                     ord_let.index(p_type[0])))
        # Get parameters values and prepend the order of the equation
        ordered_parameters = [fourier_order] + [parameters[p_name] for p_name
                                                in ordered_p_names]
    elif style == 'cvff':
        if len(parameters) > 3:
            raise TypeError('LAMMPS improper dihedrals can only have a Periodic'
                            ' interaction function with first order'
                            ' coefficients (e.g. K1, n1, d1)')
        if parameters['d1'] != 0. and parameters['d1'] != 180.:
            raise TypeError('LAMMPS improper dihedrals can only have a Periodic'
                            ' interaction with d1 = 0 deg or d = 180 deg')
        if not 0 <= parameters['n1'] <= 6:
            raise TypeError('LAMMPS improper dihedrals can only have a Periodic'
                            ' interaction 0 <= n1 <= 6')
        # fourier and cvff have different d parameters. The definition of
        # Periodic is the same as the fourier style, so need to convert to cvff
        # d parameter (which can only take values of 1 or -1)
        cvff_d = 1 if parameters['d1'] == 0. else -1
        ordered_parameters = [parameters['K1'], cvff_d, parameters['n1']]
    else:
        raise NotImplementedError('This InteractionFunction has not been'
                                  ' implemented in the LAMMPS facade')

    return [style] + ordered_parameters


def parse_dispersion_coefficients(interactions: list[NonBondedInteraction],
                                  nonbonded_styles: 'list[tuple]' = None) -> 'list[str]':
    """
    Orders MDMC ``Parameter`` objects for input to LAMMPS ``pair_coeff``

    Parameters
    ----------
    interactions : list of NonBondedInteraction
        ``NonBondedInteractions`` where their style and parameters will be
        parsed.
    nonbonded_styles : list of tuple
        The parsed ``NonBondedInteractions``, where the combined ``pair_styles``
        have been created by the ``parse_all_nonbonded_styles`` function.
        If `None` (default), the ``interactions`` are parsed manually.

    Returns
    -------
    `list` of `str`
        Each element is a partial ``pair_coeff`` command, containing a
        ``pair_style``, the ordered ``Dispersion`` ``Parameter`` objects
        converted to the correct input format for LAMMPS ``pair_coeff``, and the
        ``cutoff``.
        For example, if interactions contains a ``Buckingham`` with
        ``Parameter`` ``A, B, C = 4.184, 1.0, 4.184``, and ``cutoff = 20.0``,
        and a ``Coulombic`` with a ``cutoff`` of ``10.0``, the `str` element of
        the `list` will be::

            'buck/coul/cut 1.0 1.0 1.0 20.0 10.0'

    Raises
    ------
    NotImplementedError
        If ``interaction`` has a ``function`` that has not been implemented in
        the LAMMPS facade.
    ValueError
        If the LAMMPS Buckingham parameter rho is less than or equal to zero.
    """

    if not nonbonded_styles:
        nonbonded_styles = parse_all_nonbonded_styles(interactions)

    coul_styles = ['coul/cut', 'coul/long']
    coeff_cmds = []
    for style in nonbonded_styles:
        pair_style = style[0]
        cutoffs = style[-1]
        if 'buck' in pair_style:
            for inter in interactions:
                if inter.function.name == 'Buckingham':
                    parameters = {p.type: convert_unit(p.value)
                                  for p in inter.parameters.as_array}

            ordered_parameters = [parameters['A'],
                                  parameters['B'] ** -1,
                                  parameters['C']]
            try:
                assert ordered_parameters[1] > 0
            except AssertionError as error:
                raise ValueError('LAMMPS Buckingham parameter rho (= 1 / B)'
                                 ' must be greater than 0') from error
            coeff_cmd = (pair_style + ' '
                         + ' '.join(str(p) for p in ordered_parameters) + ' '
                         + cutoffs)
        elif 'lj' in pair_style:
            for inter in interactions:
                if inter.function.name == 'LennardJones':
                    parameters = {p.type: convert_unit(p.value)
                                  for p in inter.parameters.as_array}

            ordered_parameters = [parameters['epsilon'],
                                  parameters['sigma']]
            coeff_cmd = (pair_style + ' '
                         + ' '.join(str(p) for p in ordered_parameters) + ' '
                         + cutoffs)
        elif pair_style in coul_styles:
            coeff_cmd = pair_style
        else:
            raise NotImplementedError('This InteractionFunction has not been'
                                      ' implemented in the LAMMPS facade')
        coeff_cmds.append(coeff_cmd)

    return coeff_cmds


def parse_kspace_solver(solver: KSpaceSolver) -> list:
    """
    Converts an MDMC ``KSpaceSolver`` for input to LAMMPS ``kspace_style``

    Parameters
    ----------
    solver : KSpaceSolver
        A ``KSpaceSolver`` to be parsed.

    Returns
    -------
    `list`
        Style and parameters for input to LAMMPS ``kspace_style``

    Raises
    ------
    NotImplementedError
        If ``solver`` type has has not been implemented in the LAMMPS facade.
    """

    solver_name = solver.name.lower()
    if solver_name not in ('ewald', 'pppm'):
        raise NotImplementedError(f'This k-space solver ({solver_name}) has not been implemented'
                                  ' in the LAMMPS facade')

    return [solver_name, solver.accuracy]


def parse_constraint(constraint_algorithm: ConstraintAlgorithm,
                     bonds: list[Bond] = None, bond_ID_dict: dict = None,
                     angles: list[BondAngle] = None, angle_ID_dict: dict = None) -> list:
    # ID is an acronym
    # pylint: disable=invalid-name
    """
    Converts an MDMC ``ConstraintAlgorithm`` for input to LAMMPS fix

    At least one of bonds and angles must be passed.

    Parameters
    ----------
    constraint_algorithm : ConstraintAlgorithm
        An object that derives from ``ConstraintAlgorithm`` to be parsed.
    bonds : list of Bonds, optional
        Constrained ``Bond`` interactions.
    bond_ID_dict : dict, optional
        `dict` with {``bond``: ``ID pairs``} relating each ``Bond`` object to a
        LAMMPS ``ID``.
    angles : list of BondAngles, optional
        Constrained ``BondAngle`` interactions.
    angle_ID_dict : dict, optional
        `dict` with {``angle``: ``ID pairs``} relating each ``BondAngle`` object
        to a LAMMPS ``ID``.

    Returns
    -------
    `list`
        Input parameters for LAMMPS ``fix``, not including the first two terms
        (``fix ID``, ``group-ID``).  The output list has a maximum length of 7,
        where the last four entries are optional but a minimum of two is
        required::

            [algorithm name, accuracy, max iterations, 'b', bond IDs,
             'a', angle IDs]

    Raises
    ------
    TypeError
        If there is not at least one ``constrained`` ``Interaction`` passed.
    NotImplementedError
        If ``constraint_algorithm`` not been implemented in the LAMMPS facade.
    """

    # Raise error if there is not at least one constrained interaction passed
    if not (bonds or angles):
        raise TypeError('A LAMMPS constraint fix must have constraints on at'
                        ' least one bond or one bond angle')

    lmp_str = []

    # Add algorithm name
    if constraint_algorithm.name.upper() == 'SHAKE':
        lmp_str.append('shake')
    elif constraint_algorithm.name.upper() == 'RATTLE':
        lmp_str.append('rattle')
    else:
        raise NotImplementedError('This constraint is not implemented in the'
                                  ' LAMMPS facade')

    # Add accuracy and max iterations
    lmp_str.append(float(constraint_algorithm.accuracy))
    lmp_str.append(int(constraint_algorithm.max_iterations))

    # Never display the constraint statistics
    lmp_str.append(0)

    # Add bonds and their LAMMPS IDs and angles and their LAMMPS IDs
    if bonds:
        if not bond_ID_dict:
            bond_ID_dict = {}
        lmp_str.append('b')
        lmp_str += [bond_ID_dict[bond] for bond in bonds if bond.constrained]
    if angles:
        if not angle_ID_dict:
            angle_ID_dict = {}
        lmp_str.append('a')
        lmp_str += [angle_ID_dict[angle] for angle in angles
                    if angle.constrained]

    return lmp_str
