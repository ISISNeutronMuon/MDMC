"""Tests that file_dump_frequency is dumping the correct files that are requested.
"""
from pathlib import Path

import h5py
import pytest

from MDMC.control import Control
from MDMC.control.control import DumpFreq
from MDMC.readers import H5MD_reader
from MDMC.refinement.minimizers.minimizer_abs import Minimizer
from MDMC.trajectory_analysis.compact_trajectory import CompactTrajectory

FILE_NAME = Path('test_file.h5')

FOM = [40, 10, 88,77]

class MockControl(Control):
    """Mock class created so a simulation does
    not need to be run to get a FoM
    """
    def __init__(self, in_list, file_dump_frequency, file_dump_prefix, file_dump_loc, file_dump_timestamp):
        self.FoM = in_list
        self.iFoM = iter(in_list)
        self.file_dump_frequency = file_dump_frequency
        self.minimizer = MockMinimizer()
        self.file_dump_prefix = file_dump_prefix
        self.file_dump_loc = file_dump_loc
        self.file_dump_timestamp = file_dump_timestamp
        self.h5md_creator = "test"
        self.h5md_email = "test@test"

    def _generate_FoM(self) -> tuple[float,CompactTrajectory]:
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
        trj.setCharge([6,])
        trj.element_list = ['H', 'H', 'H']
        return fom, trj

class MockMinimizer(Minimizer):
    """A Mock Class created as file_dump_frequency needs Minimizer._history
    """
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

def test_save_best_trajectory(tmp_path):
    """Test that checks that the H5MD dumper in control is dumping the best FoM
    """
    control = MockControl(FOM, DumpFreq.BEST, FILE_NAME, tmp_path, False)
    for _ in range(len(FOM)):
        fom, trj = control._generate_FoM()
        control.minimizer._history.append(fom)
        control.dump_h5md(trj)
    file_path = tmp_path / FILE_NAME
    with h5py.File(file_path, "r") as file:
        expected_fom = H5MD_reader.read_dataset(file, "position")
        assert expected_fom[0] == min(FOM)

def test_save_all_trajectory(tmp_path):
    """
    Test that checks that the H5MD dumper.

    Check control dumps every trajectory based on the correct FoM.
    """
    file_names = []
    control = MockControl(FOM, DumpFreq.EVERY, FILE_NAME, tmp_path, False)
    for x in FOM:
        fom, trj = control._generate_FoM()
        control.minimizer._history.append(fom)
        filename = f"{x}_{FILE_NAME}"
        file_names.append(filename)
        control.file_dump_prefix = filename
        control.dump_h5md(trj)
    for filename, fom in zip(file_names, FOM):
        file_path = tmp_path / filename
        with h5py.File(file_path, "r") as file:
            expected_fom = H5MD_reader.read_dataset(file, "position")
            assert expected_fom[0] == fom
