from tempfile import NamedTemporaryFile
from pathlib import Path
from verbosemanager import VerboseManager

from MDMC.common.decorators import (unit_decorator_getter, mod_docstring,
                                    repr_decorator, unit_decorator)
from MDMC.common import units
from MDMC.readers.simulation.param_file import ParamFileParser


@repr_decorator('files')
class FileSimulation():
    def __init__(self,
                 files: str | list[str],
                 traj_step: int,
                 time_step: float = 1.,
                 **settings):
        self.parser = ParamFileParser(files)
        self._setup()

        self.traj_step = traj_step
        self.time_step = time_step
        self.settings = settings
        self._temp_files = ()

    def _setup(self) -> None:
        """
        Creates a universe within the ``MDEngine`` with the equivalent
        configuration and topology to ``self.universe`` and defines the
        simulation conditions
        """
        self.parser.parse()

    def minimize(self, n_steps: int,
                 minimize_every: int = 10,
                 verbose: bool = False, output_log: str = None,
                 work_dir: str = None, **settings: dict) -> None:
        """
        Performs an MD run intertwined with periodic structure relaxation.
        This way after a local minimum is found, the system is taken
        out of the minimum to explore a larger volume of the parameter
        space.

        Parameters
        ----------
        n_steps : int
            Total number of the MD run steps
        minimize_every: int, optional
            Number of MD steps between two consecutive minimizations
        verbose: bool, optional
            Whether to print statements when the minimization has been started and completed
            (including the number of minimization steps and time taken). Default is `False`.
        output_log: str, optional
            Log file for the MD engine to write to. Default is `None`.
        work_dir: str, optional
            Working directory for the MD engine to write to. Default is `None`.
        **settings
            ``etol`` (`float`)
                If the energy change between iterations is less than ``etol``,
                minimization is stopped. Default depends on engine used.
            ``ftol`` (`float`)
                If the magnitude of the global force is less than ``ftol``,
                minimization is stopped. Default depends on engine used.
            ``maxiter`` (`int`)
                Maximum number of iterations of a single structure
                relaxation procedure. Default depends on engine used.
            ``maxeval`` (`int`)
                Maximum number of force evaluations to perform. Default depends
                on engine used.
        """

        verbose_manager = VerboseManager.instance()
        # to match legacy use of verbose on this function (where verbose was bool) we use bool
        # and convert to int, corresponding to verbose levels 0 or 1; there is only one verbose
        # step in this function so verbose levels 2 or 3 would not provide extra information
        verbose_manager.start(1, verbose=int(verbose))

        verbose_manager.step(f"Running minimization every {minimize_every} steps "
                             f"in an MD run with {n_steps} steps")

        self.run(n_steps=n_steps, equilibration=True, minimise_every=minimise_every,
                 output_log=output_log, work_dir=work_dir, **self.settings)

        verbose_manager.finish("Minimization")

    def run(self, n_steps: int, equilibration: bool = False, verbose: bool = False,
            output_log: str = None, work_dir: str = None, **settings: dict) -> None:
        """
        Runs the MD simulation for the specified number of steps. Trajectories
        for the simulation are only saved when ``equilibration`` is `False`.
        Additionally running equilibration for an NVE system (neither barostat
        nor thermostat set) will temporarily apply a Berendsen thermostat (it
        is removed from the simulation after the run is completed).

        Parameters
        ----------
        n_steps : int
            Number of simulation steps to run
        equilibration : bool, optional
            If the run is for equilibration (`True`) or production (`False`).
            Default is `False`.
        verbose: bool, optional
            Whether to print statements upon starting and completing the run.
            Default is `False`.
        output_log: str, optional
            Log file for the MD engine to write to. Default is `None`.
        work_dir: str, optional
            Working directory for the MD engine to write to. Default is `None`.
        """

        process = 'equilibration' if equilibration else 'simulation'

        verbose_manager = VerboseManager.instance()
        # to match legacy use of verbose on this function (where verbose was bool) we use bool
        # and convert to int, corresponding to verbose levels 0 or 1; there is only one verbose
        # step in this function so verbose levels 2 or 3 would not provide extra information
        verbose_manager.start(1, verbose=int(verbose))
        verbose_manager.step(f"Running {process} for {n_steps} steps")

        temp_files = self._gen_temp_files()
        self._run_command(temp_files, n_steps=n_steps, equilibration=equilibration, verbose=verbose,
                          output_log=output_log, work_dir=work_dir, **self.settings)
        self._del_temp_files()

        verbose_manager.finish(f"{process.capitalize()}")

    def _gen_temp_files(self) -> list[NamedTemporaryFile]:
        """
        Generate a set of temp files representative of the currently stored files
        """
        names = ((pth.stem, pth.suffix) for pth in self.parser.file_names)
        self._temp_files = tuple(NamedTemporaryFile(prefix=pref, suffix=suff)
                                 for pref, suff in names)
        return tuple(file.name for file in self._temp_files)

    def _del_temp_files(self) -> None:
        """
        Delete contained temp files
        """
        for file in self._temp_files:
            file.close()
        self._temp_files = ()

    @abstractmethod
    def _run_command(self, files, **settings):
        """
        Run the command to start the MD engine using the files
        """
        raise NotImplementedError

    @property
    def files(self):
        self.parser.file_names

    @property
    def parameters(self):
        self.parser.params_dict

    @property
    def universe(self):
        return self

    @property
    def engine(self):
        return self

    @property
    def time_step(self) -> float:
        """
        Get or set the simulation time step in ``fs``

        Returns
        -------
        `float`
            Simulation time step in ``fs``
        """

        return self.parameters["time_step"]

    @time_step.setter
    @unit_decorator(unit=units.TIME)
    def time_step(self, value: float) -> None:
        self.parameters["time_step"] = value

    @property
    def traj_step(self) -> int:
        """
        Get or set the number of simulation steps between saving the
        ``CompactTrajectory``

        Returns
        -------
        `int`
            Number of simulation steps that elapse between the
            ``CompactTrajectory`` being stored
        """

        return self.parameters["traj_step"]

    @traj_step.setter
    def traj_step(self, value: int) -> None:
        self.parameters["traj_step"] = value

    def update_parameters(self) -> None:
        """
        Dummy function as not needed for file dump type
        """
        pass
