"""
Export structures via ASE.
"""
from typing import Any

import ase.io

from MDMC.exporters.exporter import Exporter
from MDMC.MD.ase.convert import MDMC_to_ASE
from MDMC.MD.structures import Structure


class ASEExporter(Exporter):
    """
    Use ASE to export to any format supported by ASE.
    """
    # pylint: disable=too-few-public-methods
    def write(self, obj: Structure, **settings: Any) -> None:
        """
        Write to any format supported by ASE.

        Parameters
        ----------
        obj : Structure
            The structure to export.
        **settings : str
            The format to write to. If not given, will
            be inferred from the file name.
        """

        file_format = settings.get('format')

        ase_atoms = MDMC_to_ASE(obj)
        ase.io.write(self.file, images=ase_atoms, format=file_format)
