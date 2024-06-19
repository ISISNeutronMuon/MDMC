"""
Parametrised file parser facade for generalised input files.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from MDMC.common.decorators import repr_decorator
from MDMC.MD.parameters import Parameters
from MDMC.readers.simulations.param_file import ParamFileParser, PathsDict


@repr_decorator('files')
class FileSimulation(ABC):
    """
    Base class for setting up generalised jobs using a parametrised files.

    Attributes
    ----------
    parser : ParamFileParser
        Parser to read parametrised files.
    settings : dict[str, Any]
        Extra settings passed by users.

    Notes
    -----
    If all parameters must be set in dumped files, it is advisable to add
    `time_step` and `traj_step` as required parameters in the `ParamFileParser`.
    """
    def __init__(self,
                 files: PathsDict,
                 traj_step: None = None,
                 time_step: None = None,
                 **settings):
        """
        Base class for setting up generalised jobs using a parametrised files.

        Parameters
        ----------
        files : PathsDict
            Files to load and parse for user parameters.
        traj_step : None
            Default number of iterations between dumping data.
        time_step : None
            Initial timestep.
        **settings : dict[str, Any]
            Extra user arguments.
        """
        self.parser = ParamFileParser(files)
        self.settings = settings
        self._temp_files = ()

    def _setup(self) -> None:
        """
        Parse configuration files and load parameters into self.
        """
        self.parser.parse()

    @abstractmethod
    def minimize(self, n_steps: int, **settings: dict) -> None:
        """
        Minimizes the simulation energy

        Parameters
        ----------
        n_steps : int
            Maximum number of steps for the MD run.
        """
        raise NotImplementedError

    @abstractmethod
    def run(self, n_steps: int, **settings: dict) -> None:
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
        """
        raise NotImplementedError

    @abstractmethod
    def convert_trajectory(self):
        """
        Convert trajectory to MDMC trajectory
        """
        raise NotImplementedError

    @property
    def files(self) -> dict[str, Path]:
        """
        Parametrised files used in specifying job

        Returns
        -------
        `dict[str, Path]`
            Dictionary of internal references to file-paths
        """
        return self.parser.file_name

    @property
    def parameters(self) -> Parameters:
        """
        Parameters object containing parameters to fit

        Returns
        -------
        `Parameters`
            Fit parameters to modify
        """

        return self.parser.as_parameters

    @property
    def param_dict(self) -> dict[str, Any]:
        """
        Parameters object as bare dict

        Returns
        -------
        `dict[str, Any]`
            Fit parameters as ordinary dictionary
        """
        return self.parser.param_dict

    def update_parameters(self) -> None:
        """
        Dummy function as not needed for file dump type
        """

    @property
    def universe(self):
        """
        Dummy property as not needed for file dump type
        """
        return self

    @property
    def engine(self):
        """
        Dummy property as not needed for file dump type
        """
        return self
