"""
Module demonstrating proof-of-concept purely file-driven calculation.

Notes
-----
Not intended for use, please see
:any:`MDMC.MD.engine_facades.lammps_file_engine.LAMMPSFileSimulation`.
"""

# pylint: disable=consider-using-with

from collections import namedtuple
from itertools import count
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Union

import numpy as np
from lammps import lammps
from verbosemanager import VerboseManager

from MDMC.MD.engine_facades.file_facade import FileSimulation
from MDMC.trajectory_analysis.compact_trajectory import CompactTrajectory

PathLike = Union[str, Path]
NamedFile = namedtuple("NamedFile", "name")


class LAMMPSFullFileSimulation(FileSimulation):
    """
    Class to control LAMMPS run through only parametrised files.

    Attributes
    ----------
    parser : ParamFileParser
        Parser to control temporary parametrised files.
    initial_struct : PathLike
        Original structure to revert to.
    known_params : Keys
        Parameters read from parametrised files.
    lmp : PyLammps
        Main LAMMPS instance.
    trajectory_file : PathLike
        Current trajectory file to dump to.
    prev_trajectory_file : PathLike
        Previous trajectory file to read.
    stored_trajectory_file : PathLike
        Trajectory to revert to.
    """

    def __init__(self,
                 struct_file: PathLike,
                 run_script: PathLike,
                 minim_script: PathLike,
                 equil_script: PathLike,
                 traj_step: int,
                 time_step: float,
                 extra_files: dict[str, PathLike] = None,
                 **settings):
        """
        Class to control LAMMPS run through only parametrised files.

        Parameters
        ----------
        struct_file : PathLike
            Path to the initial structure.
        run_script : PathLike
            Path to parametrised control file to load for production run phase.
        minim_script : PathLike
            Path to parametrised control file to load for minimization phase.
        equil_script : PathLike
            Path to parametrised control file to load for equilibration phase.
        traj_step : int
            Steps between dumps.
        time_step : float
            MD integrator time-step.
        extra_files : dict[str, PathLike]
            Extra files to include in the running.
        **settings : dict[str, Any]
            Extra config options.
        """
        self.extra_files = {} if extra_files is None else extra_files

        files = (
            {
                "main_script": run_script,
                "minim_script": minim_script,
                "equil_script": equil_script,
            } | self.extra_files
        )

        super().__init__(files)
        self.parser.required_parameters = (
            "_n_steps",  # Also used for equilibration steps in "equil_script"
            "_pot_file",
            "_struct_file",
            "_etol",
            "_ftol",
            "_time_step",
            "_traj_step",
            "_traj_file",
            "_minimize_every",
            *self.extra_files.keys(),
        )
        self._setup()

        self.initial_struct = struct_file
        self.known_params = self.parser.param_dict.keys()
        initial_params = {'_struct_file': struct_file,
                          '_traj_step': traj_step,
                          '_time_step': time_step}
        self.parser.update_param_dict(initial_params)
        self.update_vals_from_settings(settings)
        self.lmp = lammps(cmdargs=["-screen", "none"])
        self.prev_trajectory_file = None
        self.trajectory_file = None
        self.stored_trajectory_file = None

    def update_vals_from_settings(self, settings: dict):
        """
        Set file parameters from those provided in settings.

        Parameters
        ----------
        settings : dict
            Settings to put into file dump.
        """
        params = {}
        for key, val in settings.items():
            if key in self.known_params:
                params[key] = val
        self.parser.update_param_dict(params)

    @property
    def time_step(self) -> float:
        """
        Return the current known timestep.

        Returns
        -------
        float
            The timestep stored in the `ParamFileParser`.
        """
        return self.parser.param_dict['_time_step']

    @time_step.setter
    def time_step(self, value: float) -> None:
        """
        Set the timestep in the file parameters.

        Parameters
        ----------
        value : float
            New timestep.
        """
        self.parser.update_param_dict({'_time_step': value})

    @property
    def traj_step(self) -> int:
        """
        Return the current known trajectory step.

        Returns
        -------
        int
            The trajectory step stored in the `ParamFileParser`.
        """
        return self.parser.param_dict['_traj_step']

    @traj_step.setter
    def traj_step(self, value: int) -> None:
        """
        Set the trajectory step in the file parameters.

        Parameters
        ----------
        value : int
            New trajectory step.
        """
        self.parser.update_param_dict({'_traj_step': value})

    @property
    def saved_config(self) -> Path:
        """
        Get the saved configuration of the atomic positions.

        Returns
        -------
        Path
            Path to the configuration to use.
        """
        return self.parser.param_dict["_struct_file"]

    def store_config(self) -> None:
        """
        Set ``self.saved_config`` to the current configuration.
        """
        self.stored_trajectory_file = self.trajectory_file
        self.parser.update_param_dict({'_struct_file': self.trajectory_file.name})

    def save_config(self) -> None:
        """
        Dummy.
        """

    def reset_config(self) -> None:
        """
        Reset the configuration of the simulation to that in ``saved_config``.
        """
        self.parser.update_param_dict({'_struct_file': self.stored_trajectory_file.name})

    def minimize(
            self,
            n_steps: int,
            verbose: bool = True,
            minimize_every: int = 10,
            output_log: str = None,
            work_dir: str = None,
            **settings: dict
    ) -> None:
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
        self.parser.update_param_dict({"_etol": settings.get('etol', 1.e-4),
                                       "_ftol": settings.get('ftol', 0.),
                                       "_n_steps": settings.get('maxiter', 10000),
                                       "_minimize_every": settings.get('maxeval', 10000)})
        self.update_vals_from_settings(settings)

        self.run(n_steps,
                 equilibration=True,
                 output_log=output_log,
                 work_dir=work_dir,
                 control_type="minim_script",
                 **settings)
        self.store_config()

    def run(
            self,
            n_steps: int,
            equilibration: bool = False,
            verbose: bool = False,
            output_log: str = None,
            work_dir: str = None,
            **settings: dict
    ) -> None:
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
            Extra options.
        """
        process = "equilibration" if equilibration else "simulation"

        verbose_manager = VerboseManager.instance()
        # to match legacy use of verbose on this function (where verbose was bool) we use bool
        # and convert to int, corresponding to verbose levels 0 or 1; there is only one verbose
        # step in this function so verbose levels 2 or 3 would not provide extra information
        verbose_manager.start(1, verbose=int(verbose))
        verbose_manager.step(f"Running {process} for {n_steps} steps")

        # Get type of control to load
        params = {"_n_steps": n_steps}
        params["_output_log"] = output_log

        self.prev_trajectory_file = self.trajectory_file
        if filename := settings.get("output_traj"):
            self.trajectory_file = NamedFile(filename)
        else:
            self.trajectory_file = NamedTemporaryFile()

        params["_traj_file"] = self.trajectory_file.name
        if equilibration:
            control_type = settings.get("control_type", "equil_script")
        else:
            control_type = settings.get("control_type", "main_script")

        self.update_vals_from_settings(settings)

        files = [NamedTemporaryFile() for extra_file in self.extra_files.keys()]

        for key, file in zip(self.extra_files.keys(), files):
            params[key] = file.name
            self.parser.dump({key: file.name})

        self.parser.update_param_dict(params)

        with self.parser(control_type) as control:
            self.lmp.command("clear")
            self.lmp.file(control[0])

        for file in files:
            file.close()

        verbose_manager.finish(f"{process.capitalize()}")

    def convert_trajectory(self, start: int = 0, stop: int = None,
                           step: int = 1, **settings: dict) -> CompactTrajectory:
        """
        Convert between a LAMMPS trajectory dump and an MDMC ``CompactTrajectory``.

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
        # Change expected position string if scaled positions are used
        pos_string = 'xs' if settings.get('scaled_positions', False) else 'x'

        traj_dimensions = np.zeros(3)
        frame_n = start
        # Use count to create range so that stop can be undefined
        frame_indexes = count(start, step)
        # next_frame_n next attribute is assigned dynamically
        next_frame_n = next(frame_indexes)  # pylint: disable=no-member

        traj = CompactTrajectory()  # the instance of our new trajectory object

        def _make_gen(reader):
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
            while line := file_handler.readline():
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
                        i_id, i_type, i_pos, i_mass, i_elem, i_chg = [splt.index(prop) - 2
                                                                      for prop in
                                                                      ('id', 'type', pos_string,
                                                                       'mass', 'element', 'q')]
                        if 'vx' in splt:
                            i_vel = splt.index('vx')
                        else:
                            i_vel = None

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
                        atom_symbols = {}
                        atom_masses = {}
                        charge_list = []

                        for _ in range(n_atoms):
                            split_line = file_handler.readline().split()
                            # convert id to int
                            split_line[i_id] = int(split_line[i_id])
                            atom_symbols[split_line[i_id]] = split_line[i_elem]
                            atom_masses[split_line[i_id]] = split_line[i_mass]
                            charge_list.append((split_line[i_id], split_line[i_chg]))
                            split_line = [elem for i, elem in enumerate(split_line)
                                          if i not in (i_elem, i_mass, i_chg)]
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

        traj.labelAtoms(atom_symbols, atom_masses)
        # electric charges, for completeness
        # (otherwise we cannot output an Atom from CompactTrajectory)
        charges = np.array(charge_list)
        sequence = np.argsort(charges[:, 0])
        charges = charges[sequence][:, 1]
        traj.setCharge(charges)
        # we conclude the creation of the trajectory
        traj.postProcess()
        return traj
