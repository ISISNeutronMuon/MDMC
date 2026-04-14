import getpass
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, Flag, auto
from pathlib import Path

from MDMC.exporters.observables.MDA_writer import write_MDA
from MDMC.exporters.trajectories import H5MD_build
from MDMC.refinement.FoM.FoM_abs import ObservablePair
from MDMC.trajectory_analysis.compact_trajectory import CompactTrajectory


class DumpFreq(Enum):
    """
    Enum for deciding how often the output files should be dumped.
    """

    #: Dump only the best by FoM.
    BEST = -1
    #: Dump no h5md trajectories.
    NONE = 0
    #: Dump all h5md trajectories.
    EVERY = 1

    @classmethod
    def _missing_(cls, value):
        if not isinstance(value, str):
            return None
        value = value.upper()
        return cls.__members__.get(value)


class DumpExtent(Flag):
    """
    Choose which files are dumped.
    """

    TRAJ = auto()  # Only the trajectory file
    OBS = auto()  # Only the observables
    BOTH = TRAJ | OBS  # Both the trajectory and observables

    @classmethod
    def _missing_(cls, value):
        if not isinstance(value, str):
            return None
        value = value.upper()
        return cls.__members__.get(value)


class ObsFormat(Flag):
    """
    Choose which files are dumped.
    """

    NONE = auto()
    MDA = auto()  # MDANSE MDA file
    ALL = MDA  # All supported formats

    @classmethod
    def _missing_(cls, value):
        if not isinstance(value, str):
            return None
        value = value.upper()
        return cls.__members__.get(value)


@dataclass(slots=True, kw_only=True)
class H5MDControl:
    """
    Controls H5MD dumping features.

    Parameters
    ----------
    file_prefix : str, optional
        Main stem of filename (without timestamp/extension).
    folder : Path, optional
        Main dump location.
    creator : str, optional
        Name to be associated with H5MD dump. Default is username from system.
    email : str, optional
        Email to be associated with H5MD dump. Default is creator+"@unknown".
    frequency : DumpFreq | str, optional
        Defines how often the trajectory should be dumped to a H5MD file. Default is `DumpFreq.NONE`
    extent : DumpExtent | str, optional
        Which files should be written out in the dump. Default is `DumpExtent.BOTH`
    observable_format : ObsFormat, optional
        Format to dump observables in. Default is `ObsFormat.NONE`.
    timestamp: bool, optional
        Whether a time stamp should be added to the output file names. Default is `False`
    """

    file_prefix: str = "trajectory"
    folder: Path = Path(".")
    creator: str = getpass.getuser()
    email: str = f"{getpass.getuser()}@unknown"
    frequency: DumpFreq = DumpFreq.NONE
    extent: DumpExtent = DumpExtent.BOTH
    observable_format: ObsFormat = ObsFormat.NONE
    timestamp: bool = False

    def __post_init__(self):
        self.frequency = DumpFreq(self.frequency)
        self.extent = DumpExtent(self.extent)

    def dump(self, trajectory: CompactTrajectory, observable_pairs: Iterable[ObservablePair]):
        """Dump combined data to appropriate locations.

        Parameters
        ----------
        trajectory : CompactTrajectory
            Trajectory to dump.
        observable_pairs : Iterable[ObservablePair]
            Observable pairs to dump.
        """
        if DumpExtent.TRAJ in self.extent:
            self.write_traj(trajectory)
        if DumpExtent.OBS in self.extent:
            self.write_observables(observable_pairs)

    def should_dump(self, best: bool):
        """Whether the dumper should dump.

        Parameters
        ----------
        best : bool
            Whether the last FoM is the best recorded.
        """

        return self.frequency is DumpFreq.EVERY or (self.frequency is DumpFreq.BEST and best)

    def write_traj(self, trajectory: CompactTrajectory):
        """
        Dump the trajectory as an H5MD file.

        Parameters
        ----------
        trj : CompactTrajectory
            The compact trajectory from the current step

        Notes
        -----
        When Dumping "EVERY" trajectory timestamp should be true
        or the file name must be different for each trajectory,
        as if not the file will be continually overwritten.
        """
        H5MD_build.write_H5MD(
            trajectory,
            filename=self.file_prefix,
            file_loc=self.folder,
            timestamp=self.get_timestamp(),
            creator_name=self.creator,
            creator_email=self.email,
        )

    def write_observables(self, observable_pairs: Iterable[ObservablePair]):
        """
        Dump the observables.

        Parameters
        ----------
        data_format: ObsFormat | str
            File format for writing the observables.
        """
        if ObsFormat.MDA in self.observable_format:
            for obs_pair in observable_pairs:
                write_MDA(
                    obs_pair.MD_obs,
                    filename=self.file_prefix,
                    file_loc=self.folder,
                    timestamp=self.get_timestamp(),
                )

    def get_timestamp(self) -> str:
        """Return a timestamp or an empty string for use in file names.

        The intention is to use a time stamp if the user requested it,
        or if files are written every step (to avoid overwriting files).

        Parameters
        ----------
        use_timestamp : bool, optional
            If True, forces the time stamp to be non-empty, by default True
        dump_frequency : DumpFreq, optional
            File output setting, by default DumpFreq.EVERY

        Returns
        -------
        str
            Either a time stamp string or an empty string.
        """
        if self.timestamp or self.frequency is DumpFreq.EVERY:
            return datetime.now().strftime("_%Y-%m-%d--%H-%M-%S")
        return ""
