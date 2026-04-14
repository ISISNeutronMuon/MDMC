"""Tests that frequency is dumping the correct files that are requested."""

import getpass
from pathlib import Path

import h5py
import pytest

from MDMC.control import Control
from MDMC.control.h5md import DumpFreq, H5MDControl, DumpExtent, ObsFormat
from MDMC.readers import H5MD_reader
from MDMC.refinement.minimizers.minimizer_abs import Minimizer
from MDMC.trajectory_analysis.compact_trajectory import CompactTrajectory

FILE_NAME = "test_file.h5"

FOM = [40, 10, 88, 77]


class MockControl(Control):
    """Mock class created so a simulation does
    not need to be run to get a FoM
    """

    def __init__(self, in_list):
        self.FoM = in_list
        self.iFoM = iter(in_list)
        self.minimizer = MockMinimizer()

    def _generate_FoM(self) -> tuple[float, CompactTrajectory]:
        """A function that overrides the original _generate_FoM
        so it does no calculations and creates CompactTrajectory
        that means nothing but with the FoM in the position
        and iter's through pre determined FoM's

        Returns
        -------
        tuple[float,CompactTrajectory]
            The FoM and the trajectory made in this function
        """
        fom = next(self.iFoM)
        trj = CompactTrajectory()
        trj.position = [fom, fom]
        trj.setCharge(
            [
                6,
            ]
        )
        trj.element_list = ["H", "H", "H"]
        return fom, trj


class MockMinimizer(Minimizer):
    """A Mock Class created as H5MDControl.frequency needs Minimizer._history"""

    def __init__(self):
        self._history = []

    def change_parameters(self):
        pass

    def extract_result(self):
        pass

    def format_result_string(self):
        pass

    def has_converged(self):
        pass

    def history_columns(self):
        pass

    def reset_parameters(self):
        pass

    def step(self):
        pass


@pytest.mark.parametrize(
    "inp, alt",
    [
        (
            # Defaults
            {},
            {
                "file_prefix": "trajectory",
                "folder": Path("."),
                "creator": getpass.getuser(),
                "email": f"{getpass.getuser()}@unknown",
                "frequency": DumpFreq.NONE,
                "extent": DumpExtent.BOTH,
                "observable_format": ObsFormat.NONE,
                "timestamp": False,
            },
        ),
        # Already enum
        ({}, {"frequency": DumpFreq.NONE}),
        # Check case-insensitive
        ({}, {"frequency": "None"}),
        ({}, {"frequency": "none"}),
        ({}, {"frequency": "NoNE"}),
        ({}, {"frequency": "NONE"}),
        # Check numeric
        ({}, {"frequency": 0}),
    ],
)
def test_h5md_control_init(inp, alt):
    assert H5MDControl(**inp) == H5MDControl(**alt)


def test_save_all_trajectory(tmp_path):
    """
    Test that checks that the H5MD dumper.

    Check control dumps every trajectory based on the correct FoM.
    """
    file_names = []
    h5md = H5MDControl(
        frequency=DumpFreq.EVERY,
        file_prefix=FILE_NAME,
        folder=tmp_path,
        timestamp=False,
        creator="test",
        email="test@test",
    )

    control = MockControl(FOM)
    control.h5md = h5md
    for x in FOM:
        fom, trj = control._generate_FoM()
        control.minimizer._history.append(fom)
        filename = f"{x}_{FILE_NAME}"
        file_names.append(filename)
        control.h5md.file_prefix = filename
        control.h5md.write_traj(trj)
    for filename, fom in zip(file_names, FOM):
        file_path = tmp_path / filename
        with h5py.File(file_path, "r") as file:
            expected_fom = H5MD_reader.read_dataset(file, "position")
            assert expected_fom[0] == fom
