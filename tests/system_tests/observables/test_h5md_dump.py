"""Tests that h5md_dumper is dumping the correct files that are requested.
"""
from pathlib import Path
import h5py

from MDMC.control import Control
from MDMC.trajectory_analysis.compact_trajectory import CompactTrajectory
from MDMC.control.control import Dump
from MDMC.readers import H5MD_reader
from MDMC.refinement.minimizers.minimizer_abs import Minimizer


FILE_NAME = 'test_file'

FOM = [40, 10, 88,77]

class MockControl(Control):
    """Mock class created so a simulation does
    not need to be run to get a FoM
    """
    def __init__(self, in_list, h5md_dump, *args, **kwargs):
        self.FoM = in_list
        self.iFoM = iter(in_list)
        self.h5md_dump = h5md_dump
        self.minimizer = MockMinamizer()
        self.h5md_filename = None

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

class MockMinamizer(Minimizer):
    """A Mock Class created as h5md_dumper needs Minamizer._history
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

def test_save_best_traject(tmp_path):
    """Test that checks that the H5MD dumper in control is dumping the best FoM
    """
    control = MockControl(FOM, Dump.BEST)
    for _ in range(len(FOM)):
        fom, trj = control._generate_FoM()
        control.minimizer._history.append(fom)
        control.h5md_dumper(trj, h5md_file_loc = tmp_path,
                            h5md_file_name = FILE_NAME,
                            h5md_timestamp = False)
    file_path = Path(f'{tmp_path}/{FILE_NAME}').with_suffix('.h5')
    with h5py.File(file_path, "r") as file:
        expected_fom = H5MD_reader.read_dataset(file, "position")
        assert expected_fom[0] == min(FOM)

def test_save_all_traject(tmp_path):
    """
    Test that checks that the H5MD dumper.

    Check control dumps every trajectory based on the correct FoM.
    """
    file_names = []
    control = MockControl(FOM, Dump.EVERY)
    for x in FOM:
        fom, trj = control._generate_FoM()
        control.minimizer._history.append(fom)
        filename = f"{x}_{FILE_NAME}"
        file_names.append(filename)
        control.h5md_dumper(trj,
                            h5md_file_loc = tmp_path,
                            h5md_file_name = filename,
                            h5md_timestamp = False)
    for filename, fom in zip(file_names, FOM):
        file_path = Path(f'{tmp_path}/{filename}').with_suffix('.h5')
        with h5py.File(file_path, "r") as file:
            expected_fom = H5MD_reader.read_dataset(file, "position")
            assert expected_fom[0] == fom
