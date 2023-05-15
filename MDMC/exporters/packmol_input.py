"""A module of a class to export a packmol input file"""
from MDMC.MD.packmol.packmol_setup import PackmolSetup
from MDMC.exporters.exporter import Exporter

class PackmolInputExporter(Exporter):
    """A class to export `PackmolSetup` objects into packmol input files"""
    def __init__(self, file_name: str):
        super().__init__(file_name)

    INDENT = "  "

    @property
    @staticmethod
    def extension() -> str:
        return ".inp"

    def write(self, setup: PackmolSetup,
              molecule_file_names: dict,
              output_name: str = "output_file.pdb",
              **settings: dict) -> None:
        """
        Write the data contained in a `PackmolSetup` object out to a packmol input file

        Parameters
        ----------
        setup: PackmolSetup
            A `PackmolSetup` object which contains the molecules and constraints for the
        output_name: str
            The filename of the output file to write to
        molecule_file_names: dict
            A dictionary mapping molecules in the system to corresponding file names
        """
        system_settings, mol_settings = setup.get_settings()
        tol = system_settings["tolerance"]
        self.file.writeline("# Created by MDMC")
        self.file.writeline(f"tolerance {tol}")
        self.file.writeline("filetype pdb")
        self.file.writeline(f"output {output_name}")
        for molecule in mol_settings:
            # Get structure file name
            struct_file_name = molecule_file_names[molecule]
            struct_file_name if struct_file_name.endswith(".pdb") else struct_file_name += ".pdb"
            self.file.writeline(f"structure {struct_file_name}")
            for setting in mol_settings[molecule].keys():
                self.file.writeline(self.INDENT+f"{setting}")
            self.file.writeline(f"end structure")
