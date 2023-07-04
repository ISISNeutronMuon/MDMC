"""A module to export configuations as pdb files"""
from MDMC.exporters.configurations.conf_exporter import ConfigurationExporter
from MDMC.MD import Structure
from MDMC.MD.ase.conversions import get_ase_atoms
from ase.io import proteindatabank

class ProteinDataBankExporter(ConfigurationExporter):

    def __init__(self, file_name: str):
        super().__init__(file_name)

    @property
    @staticmethod
    def extension() -> str:
        return ".pdb"

    def write(self, structure: Structure, **settings: dict) -> None:
        ase_atoms = get_ase_atoms(structure.atoms)
        proteindatabank.write_proteindatabank(self.file, ase_atoms)
