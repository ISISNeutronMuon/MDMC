"""A module to export configuations as pdb files"""
from MDMC.exporters.configurations.conf_exporter import ConfigurationExporter
from MDMC.MD import Structure
from ase import Atoms
from ase.io import proteindatabank
class ProteinDataBankExporter(ConfigurationExporter):

    def __init__(self, file_name: str):
        super().__init__(file_name)

    @property
    @staticmethod
    def extension() -> str:
        return ".pdb"

    def write(self, structure: Structure, **settings: dict) -> None:
        # Convert into ASE format
        atom_list = [atom.element for atom in structure.atoms]
        atom_symbols = "".join(atom_list)
        atom_positions = [atom.position for atom in structure.atoms]
        ase_atoms = Atoms(symbols=atom_symbols, positions=atom_positions)
        # Write to pdb file
        proteindatabank.write_proteindatabank(self.file, ase_atoms)
