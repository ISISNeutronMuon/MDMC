"""
Parameter-file-based runner for DLPoly simulations
"""

from pathlib import Path
from typing import Union

from verbosemanager import VerboseManager
from dlpoly import DLPoly
from dlpoly.config import Config, Atom

from MDMC.common import units
from MDMC.common.decorators import repr_decorator, unit_decorator
from .dlpoly_engine import DLPOLYEngine
from .file_facade import FileSimulation

PathLike = Union[str, Path]


class DLPolyFileSimulation(FileSimulation):
    """
    Class to run a DLPoly calculation using the Param file parser.

    Attributes
    ----------
    dlpoly : DLPoly
        DLPoly-py instance.

    """
    def __init__(self,
                 control: PathLike,
                 config: PathLike,
                 field: PathLike,
                 traj_step: int,
                 time_step: float,
                 **settings):
        """Class to control DLPoly run through parametrised file.

        Parameters
        ----------
        control : PathLike
            Path to parametrised control file to load.
        config : PathLike
            Path to parametrised config file to load.
        field : PathLike
            Path to parametrised field file to load.
        traj_step : int
            Steps between dumps.
        time_step : float
            MD integrator time-step.
        **settings : dict[str, Any]
            Extra config options.

        Extra Parameters
        ----------------
        equil_control : PathLike
            Path to parametrised control to be used during equilibration phase.
        minim_control : PathLike
            Path to parametrised control to be used during minimisation phase.

        """
        super().__init__({"run_control": control,
                          "equil_control": settings.get("equil_control", control),
                          "minim_control": settings.get("minim_control", control),
                          "config": config,
                          "field": field},
                         **settings)
        self._setup()
        self.dlpoly = DLPoly(load_default=False)
        self.time_step = time_step
        self.traj_step = traj_step
        self._saved_config = None

    @property
    def atoms(self) -> list[Atom]:
        """
        Atoms in DLPoly configuration.

        Returns
        -------
        list[Atom]
            Atoms in DLPoly configuration.
        """
        return self.dlpoly.config.atoms

    @property
    def saved_config(self) -> Path:
        """
        Get the saved configuration of the atomic positions

        Returns
        -------
        ``Configuration``
            The atomic positions
        """
        return self._saved_config

    def save_config(self) -> None:
        """
        Sets ``self.saved_config`` to the current configuration
        """
        self._saved_config = Path(self.dlpoly.control['io_file_revcon'])

    def reset_config(self) -> None:
        """
        Resets the configuration of the simulation to that in ``saved_config``
        """
        self.parser.file_name['config'] = self.saved_config

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
        Minimizes the simulation energy.

        Parameters
        ----------
        n_steps : int
            Maximum number of steps for the MD run.
        minimize_every : int, optional, default 10
            Number of MD steps between two consecutive minimizations.
        output_log : str, optional, default None
            file where the output goes.
        work_dir : str, optional, default None
            folder where the run happens
        **settings
            The majority of these are generic but some are specific to the
            ``MDEngine`` that is being used.
        etol : float, energy tolerance criteria for energy minimisation
        ftol : float, force tolerance criteria for force minimisation, active only if non-zero
        maxiter : int, not used in this facade
        maxeval : int, not used in this facade
        """

        # Example of how to use the **settings to specify parameters,
        # e.g. tolerances
        etol = settings.get("etol", 1.e-3)
        ftol = settings.get("ftol", None)

        min_freq = minimize_every

        extra_settings = settings.copy()

        if not ftol:  # Should handle ftol == 0 or undefined ftol
            extra_settings["minimisation_criterion"] = "energy"
            extra_settings["minimisation_tolerance"] = (etol, "internal_e")
        else:
            extra_settings["minimisation_criterion"] = "force"
            extra_settings["minimisation_tolerance"] = (ftol, "e.V/Ang")
        extra_settings["minimisation_frequency"] = (min_freq, "steps")

        self.run(n_steps,
                 equilibration=True,
                 output_log=output_log,
                 work_dir=work_dir,
                 control_type="minim_control",
                 **extra_settings)

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
        Runs the MD simulation for the specified number of steps.

        Trajectories for the simulation are only saved when
        ``equilibration`` is `False`.  Additionally running
        equilibration for an NVE system (neither barostat nor
        thermostat set) will temporarily apply a Berendsen thermostat
        (it is removed from the simulation after the run is
        completed).

        Parameters
        ----------
        n_steps : int
            Number of simulation steps to run
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
        """
        process = "equilibration" if equilibration else "simulation"

        verbose_manager = VerboseManager.instance()
        # to match legacy use of verbose on this function (where verbose was bool) we use bool
        # and convert to int, corresponding to verbose levels 0 or 1; there is only one verbose
        # step in this function so verbose levels 2 or 3 would not provide extra information
        verbose_manager.start(1, verbose=int(verbose))
        verbose_manager.step(f"Running {process} for {n_steps} steps")

        # Get type of control to load
        control_type = settings.get("control_type", "run_control")

        with self.parser(control_type, "field", "config") as files:
            control, field, config = files
            self.dlpoly.load_control(control)
            self.dlpoly.load_field(field)
            self.dlpoly.load_config(config)

            # Take keys from settings
            for key, val in settings.items():
                if key in self.dlpoly.control.keys:
                    self.dlpoly.control[key] = val

            self.dlpoly.control.traj_interval = (self.traj_step, "steps")
            self.dlpoly.control.timestep = (self.time_step, "fs")

            if equilibration:
                self.dlpoly.control["time_equilibration"] = (n_steps, "steps")
                self.dlpoly.control["traj_calculate"] = settings.get("traj_calculate", "Off")
            else:
                self.dlpoly.control["time_equilibration"] = \
                    (settings.get("time_equilibration", 0), "steps")
                self.dlpoly.control["traj_calculate"] = "On"
                self.dlpoly.control["traj_start"] = (settings.get("traj_start", 0), "steps")
                self.dlpoly.control["traj_interval"] = (self.traj_step, "steps")
                self.dlpoly.control["traj_key"] = settings.get("traj_key", "pos")

            self.dlpoly.control["time_run"] = (n_steps, "steps")
            self.dlpoly.workdir = work_dir

            self.dlpoly.run(numProcs=settings.get("numprocs", 1),
                            outputFile=output_log,
                            mpi="mpirun --allow-run-as-root -n")

            print("Update coordinates from ", self.dlpoly.control["io_file_revcon"])
            self.dlpoly.dest_config = "minim.config"
            self.dlpoly.load_config(self.dlpoly.control["io_file_revcon"])

        verbose_manager.finish(f"{process.capitalize()}")

    convert_trajectory = DLPOLYEngine.convert_trajectory
