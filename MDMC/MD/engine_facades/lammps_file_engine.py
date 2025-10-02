"""
Parameter-file-based runner for LAMMPS simulations.
"""

import logging
from collections import namedtuple
from itertools import count
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import BinaryIO, Iterator, Optional, Union

try:
    from lammps import Atom, PyLammps
except ModuleNotFoundError as err:
    raise ModuleNotFoundError('The Python interface for LAMMPS (lammps.py) is'
                              ' not in the PYTHONPATH. See LAMMPS documentation'
                              ' on Python to rectify this.',
                              ) from err
import numpy as np

from MDMC.common import units
from MDMC.common.decorators import unit_decorator_getter
from MDMC.MD.engine_facades.file_facade import FileSimulation
from MDMC.trajectory_analysis.compact_trajectory import CompactTrajectory

PathLike = Union[str, Path]
LOGGER = logging.getLogger(__name__)


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
    'PRESSURE': units.Unit('atm'),
}


class LAMMPSFileSimulation(FileSimulation):
    """
    Class to control LAMMPS run through parametrised file.

    Attributes
    ----------
    parser : ParamFileParser
        Parser to control temporary parametrised files.
    lmp : PyLammps
        Main LAMMPS instance.
    traj_step : int
        Steps between dumps.
    time_step : float
        MD integrator time-step.
    type_map : dict[int, str]
        Map of LAMMPS types to chemical symbol.
    trajectory_file : PathLike
        Current trajectory to read.
    """

    def __init__(self,
                 script: PathLike,
                 traj_step: int,
                 time_step: float,
                 type_map: dict[int, str],
                 *extra_files,
                 **settings):
        """
        Class to control LAMMPS run through parametrised file.

        Parameters
        ----------
        script : PathLike
            Path to parametrised control file to load.
        traj_step : int
            Steps between dumps.
        time_step : float
            MD integrator time-step.
        type_map : dict[int, str]
            Map of LAMMPS types to chemical symbol.
        *extra_files : Sequence[PathLike]
            Extra files to include in the running.
        **settings : dict[str, Any]
            Extra config options.

        Notes
        -----

        Using the `LAMMPSFileSimulation` assumes a cut-down version of
        a lammps run-script, i.e. without any ``run`` statements.
        """
        files = {"main_script": script}
        files.update({f"extra_script_{i}": scpt for i, scpt in enumerate(extra_files)})

        super().__init__(files)
        self._setup()
        self.lmp = PyLammps()
        self.time_step = time_step
        self.traj_step = traj_step
        self.type_map = type_map
        self.trajectory_file = None
        self._saved_config = None
        self.update_vals_from_settings(settings)

    @property
    @unit_decorator_getter(unit=units.LENGTH)
    def dimensions(self) -> np.typing.NDArray:
        """
        System box dimensions.

        Returns
        -------
        np.NDArray
            System dimensions in Angstrom.
        """
        return np.asarray([self.lmp.system.xhi - self.lmp.system.xlo,
                           self.lmp.system.yhi - self.lmp.system.ylo,
                           self.lmp.system.zhi - self.lmp.system.zlo])

    @property
    def atoms(self) -> Iterator[Atom]:
        """
        Generator of atoms in system.

        Yields
        ------
        lammps.Atom
            Each atom in the system.
        """
        for i in range(self.lmp.system.natoms):
            yield self.lmp.atoms[i]

    @property
    def thermostat(self) -> Optional[str]:
        """
        Return currently defined thermostat.

        Returns
        -------
        Optional[str]
            Thermostat name if found, else `None`.

        Notes
        -----
        If multiple thermostatic fixes defined, only first found is returned.
        """

        for fix in self.fix_styles:
            if fix in ("nvt",
                       "nvt/sphere",
                       "nvt/asphere",
                       "nvt/sllod",
                       "temp/berendsen",
                       "temp/csvr",
                       "langevin",
                       "temp/rescale"):
                return fix

        return None

    @property
    def barostat(self) -> Optional[str]:
        """
        Return currently defined barostat.

        Returns
        -------
        Optional[str]
            Barostat name if found, else `None`.

        Notes
        -----
        If multiple thermostatic fixes defined, only first found is returned.
        """

        for fix in self.fix_styles:
            if fix in ("npt", "npt/sphere", "npt/asphere",
                       "nph", "press/berendsen"):
                return fix

        return None

    @property
    def fixes(self) -> list[dict]:
        """
        Get the ``PyLammps`` wrapper `list` of ``fixes``.

        Returns
        -------
        list[dict]
            Each `dict` states the ``group``, ``name`` and ``style`` of a LAMMPS
            ``fix` which is applied.
        """
        return self.lmp.fixes

    @property
    def fix_styles(self) -> list[str]:
        """
        Get the styles of the ``fixes`` applied in LAMMPS.

        Returns
        -------
        list[str]
            The styles of the ``fixes``.
        """
        return [fix['style'] for fix in self.fixes]

    @property
    def fix_names(self) -> "list[str]":
        """
        Get the names of the ``fixes`` applied in LAMMPS.

        Returns
        -------
        list[str]
            The names of the ``fixes``.
        """
        return [fix['name'] for fix in self.fixes]

    @property
    def system_state(self) -> namedtuple:
        """
        Get the ``PyLammps`` wrapper system ``state`` `dict`.

        Returns
        -------
        namedtuple
            Contains the properties of the simulation box.
        """

        # Conversion from System class (which is a namedtuple) to
        # ordered dict required as System cannot be pickled
        system_state = self.lmp.system._asdict()
        # Cast back to namedtuple to remain consist with LAMMPS system attribute
        return namedtuple('System', system_state.keys())(*system_state.values())

    @property
    def saved_config(self) -> Path:
        """
        Get the saved configuration of the atomic positions.

        Returns
        -------
        Path
            Path to the configuration to use.
        """
        return self._saved_config

    def save_config(self) -> None:
        """
        Set the current atomic configuration as the saved config.
        """
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
        tmp_mass = {atom.id: atom.mass for atom in self.atoms}

        for atom in self.atoms:
            atom_type = atom.type
            # _, mass = self.lmp_universe.atom_type_properties[atom_type-1]
            atoms[atom.id-1, :] = (list(atom.position) + [tmp_mass[atom_type], atom.charge])

        saved_config = atoms
        self._saved_config = saved_config

    def reset_config(self) -> None:
        """
        Reset the configuration of the simulation to that in ``saved_config``.
        """
        self.set_config(self.saved_config)

    def set_config(self, config: np.ndarray) -> None:
        """
        Change the positions of all of the atoms in the LAMMPS wrapper.

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
        index_components = enumerate(['x', 'y', 'z'])
        # LAMMPS IDs start at 1, so are offset from config indexes
        for id_offset in range(n_atoms):
            for index, component in index_components:
                self.lmp.set('atom', id_offset+1, component,
                             config[id_offset][index])

    def minimize(self, n_steps: int, minimize_every: int = 10,
                 verbose: bool = False, output_log: str = None,
                 work_dir: str = None, **settings: dict) -> None:
        """
        Move the atoms towards a potential energy minimum.

        Does so by performing an MD simulation interrupted
        periodically by a structure relaxation. In the end, the
        configuration with the lowest potential energy reached during
        the run is kept.

        Parameters
        ----------
        n_steps : int
            Number of MD simulation steps.
        verbose : bool, optional
            Whether to print statements upon starting and completing the run.
            Default is `False`.
        minimize_every : int, optional, default 10
            The structure relaxation will be performed every
            'minimize_every' steps of the MD simulation.
        output_log : str, optional
            Not used in this facade.
        work_dir : str, optional
            Not used in this facade.
        **settings
            Extra options.

        Other Parameters
        ----------------
        etol : float
            Energy tolerance criteria for energy minimisation.
        ftol : float
            Force tolerance criteria for force minimisation, active only if non-zero.
        maxiter : int
            Maximum number of steps in a single structure relaxation.
        maxeval : int
            Maximum number of force calculations in a single structure relaxation.
        """
        self.update_vals_from_settings(settings)

        with self.parser(*self.parser.file_name.keys()) as files:
            self.lmp.clear()
            # Load first file (main script), which should contain
            # all necessary references to other files.
            if output_log is not None:
                self.lmp.log(output_log)

            self.lmp.file(files[0])

        # Check fix styles for shake or rattle styles and remove them
        if 'constrain' in self.fix_names:
            LOGGER.debug('%s Remove constraint from fixes.', self.__class__)
            self.lmp.unfix('constrain')

        etol = settings.get('etol', 1.e-4)
        ftol = settings.get('ftol', 0.)
        maxiter = settings.get('maxiter', 10000)
        maxeval = settings.get('maxeval', 10000)
        LOGGER.info('%s minimize: {n_steps: %s, minimize_every: %s, etol: %s, ftol: %s,'
                    ' maxiter: %s, maxeval: %s}',
                    self.__class__,
                    n_steps,
                    minimize_every,
                    etol,
                    ftol,
                    maxiter,
                    maxeval)

        self.lmp.minimize(etol,
                          ftol,
                          maxiter,  # this is the number of relaxation steps
                          maxeval)  # this is the number of force evaluations
        best_energy = self.lmp.eval("pe")
        self.save_config()
        for _ in range(int(n_steps/minimize_every)):
            self.lmp.run(minimize_every)
            self.lmp.minimize(etol,
                              ftol,
                              maxiter,  # this is the number of relaxation steps
                              maxeval)  # this is the number of force evaluations
            energy = self.lmp.eval("pe")
            if energy < best_energy:
                self.save_config()
                best_energy = energy

        self.reset_config()

    def run(
            self,
            n_steps: int,
            equilibration=False,
            verbose: bool = False,
            output_log: str = None,
            work_dir: str = None,
            **settings: dict,
    ):
        """
        Run the MD simulation for the specified number of steps.

        Trajectories for the simulation are only saved when
        ``equilibration`` is `False`.  Additionally running
        equilibration for an NVE system (neither barostat nor
        thermostat set) will temporarily apply a Berendsen thermostat
        (it is removed from the simulation after the run is
        completed).

        Parameters
        ----------
        n_steps : int
            Number of simulation steps to run.
        equilibration : bool, optional
            If the run is for equilibration (`True`) or production (`False`).
            Default is `False`.
        verbose : bool, optional
            Whether to print statements upon starting and completing the run.
            Default is `False`.
        output_log : str, optional
            Log file for the MD engine to write to. Default is `None`.
        work_dir : str, optional
            Working directory for the MD engine to write to. Default is `None`.
        **settings
            The majority of these are generic but some are specific to the
            ``MDEngine`` that is being used.
        """
        self.update_vals_from_settings(settings)

        with self.parser(*self.parser.file_name.keys()) as files:
            self.lmp.clear()
            # Load first file (main script), which should contain
            # all necessary references to other files.
            if output_log is not None:
                self.lmp.log(output_log)

            self.lmp.file(files[0])

        if not equilibration:
            # Store the trajectory in a NamedTemporaryFile
            # pylint: disable=consider-using-with
            self.trajectory_file = NamedTemporaryFile()

            # Custom trajectory output just saves the atom ID, type and
            # positions
            LOGGER.debug('%s set trajectory dump output to %s',
                         self.__class__,
                         self.trajectory_file)
            self.lmp.dump('traj1', 'all', 'custom', self.traj_step,
                          self.trajectory_file.name, 'id', 'type', 'x', 'y',
                          'z')

        LOGGER.info('%s run: {n_steps: %s, equilibration: %s}',
                    self.__class__,
                    n_steps,
                    equilibration)
        self.lmp.run(n_steps)

    def convert_trajectory(
            self,
            start: int = 0,
            stop: int = None,
            step: int = 1,
            **settings: dict,
    ) -> CompactTrajectory:
        """
        Converts between a LAMMPS trajectory dump and an MDMC ``CompactTrajectory``.

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
            Extra options.

        Returns
        -------
        CompactTrajectory
            The MDMC ``CompactTrajectory`` corresponding to the LAMMPS
            ``trajectory_file``.

        Other Parameters
        ----------------
        scaled_positions : bool
            If the ``trajectory_file`` has scaled ``positions``.
        atom_IDs : list
            LAMMPS ``ID`` of the atoms which should be included. If not passed
            then all atoms are included in the converted trajectory.

        Raises
        ------
        AssertionError
            If ``universe`` is passed, and the number of atoms in the
            ``trajectory_file`` is not the same as in the ``universe``.
        TypeError
            If ``trajectory_file`` describes a triclinic universe.
        """
        # pylint: disable=E0606

        # Change expected position string if scaled positions are used
        pos_string = 'xs' if settings.get('scaled_positions', False) else 'x'

        traj_dimensions = np.zeros(3)
        frame_n = start
        # Use count to create range so that stop can be undefined
        frame_indexes = count(start, step)
        # next_frame_n next attribute is assigned dynamically
        next_frame_n = next(frame_indexes)  # pylint: disable=no-member

        traj = CompactTrajectory()  # the instance of our new trajectory object

        def _make_gen(reader: BinaryIO) -> bytes:
            """
            A support function for splitting a binary file into buffers.

            Parameters
            ----------
            reader : BinaryIO
               Open file or a file-like object.

            Yields
            ------
            bytes
                A byte string read from the file.
            """
            while b := reader(1024 * 1024):
                yield b

        # here we check how long the trajectory really is
        with open(self.trajectory_file.name, 'rb') as file_handler:
            file_generator = _make_gen(file_handler.raw.read)
            line_count = sum(buf.count(b'\n') for buf in file_generator)

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

                        i_vel = splt.index('vx') if 'vx' in splt else None

                        # now we try to get the correct number of frames in the trajectory
                        real_n_steps = 1 + line_count // (n_atoms + header_size)

                        traj.preAllocate(n_steps=real_n_steps,
                                         n_atoms=n_atoms,
                                         useVelocity=i_vel is not None)

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

                        sorted_lines = np.array(lines, dtype=traj.dtype)

                        atom_types = sorted_lines[:, i_type].astype(np.int64)
                        if not traj.validateTypes(atom_types):
                            raise TypeError("CompactTrajectory received wrong atom type array")

                        if i_vel is not None:
                            traj.writeOneStep(step_num=frame_n,
                                              time=frame * self.time_step,
                                              positions=sorted_lines[:, i_pos:i_pos+3],
                                              velocities=sorted_lines[:, i_vel:i_vel+3])
                        else:
                            traj.writeOneStep(step_num=frame_n,
                                              time=frame * self.time_step,
                                              positions=sorted_lines[:, i_pos:i_pos+3])

                        traj.setDimensions(traj_dimensions, step_num=frame_n)
                        # next_frame_n next attribute is assigned dynamically
                        # pylint: disable=no-member
                        next_frame_n = next(frame_indexes)
                    frame_n += 1
                    if stop is not None and frame_n >= stop:
                        break

                line = file_handler.readline()

        atom_symbols = {atom.id: self.type_map[atom.type] for atom in self.atoms}
        atom_masses = {atom.id: atom.mass for atom in self.atoms}
        traj.labelAtoms(atom_symbols, atom_masses)
        # electric charges, for completeness
        # (otherwise we cannot output an Atom from CompactTrajectory)
        charge_list = [[atom.id, atom.charge] for atom in self.atoms]
        charges = np.array(charge_list)
        sequence = np.argsort(charges[:, 0])
        charges = charges[sequence][:, 1]
        traj.setCharge(charges)
        # we conclude the creation of the trajectory
        traj.postProcess()
        return traj
