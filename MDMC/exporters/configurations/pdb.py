"""A module to export configuations as pdb files"""
from ase.io import proteindatabank

from MDMC.exporters.exporter import Exporter
from MDMC.MD import Structure
from MDMC.MD.ase.conversions import get_ase_atoms

class ProteinDataBankExporter(Exporter):
    """
    Exporter for Protein Data Bank (pdb) files.

    Parameters
    ----------
    file_name: str
        The output file name.
    """

    #pylint: disable=arguments-renamed, too-few-public-methods
    def write(self, structure: Structure, **settings: dict) -> None:
        ase_atoms = get_ase_atoms(structure.atoms)
        proteindatabank.write_proteindatabank(self.file, ase_atoms)
