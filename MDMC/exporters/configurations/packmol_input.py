"""A module of a class to export a packmol input file"""
from MDMC.MD.packmol.packmol_setup import PackmolSetup
from MDMC.exporters.configurations.conf_exporter import ConfigurationExporter


class PackmolInputExporter(ConfigurationExporter):
    def __init__(self, file_name: str):
        super().__init__(file_name)

    INDENT = "  "

    @property
    @staticmethod
    def extension() -> str:
        return ".inp"

    def write(self, setup: PackmolSetup, **settings: dict) -> None:
        """
        Write the data contained in a `PackmolSetup` object out to a packmol input file
        Parameters
        ----------
        setup
            A `PackmolSetup` object which contains the molecules and constraints for the
        """
        system_settings, mol_settings = setup.get_settings()
        tol = system_settings["tolerance"]
        self.file.writeline("# Created by MDMC")
        self.file.writeline(f"tolerance {tol}")
        self.file.writeline(f"filetype pdb")
        self.file.writeline("output output-universe.pdb")
        for molecule in mol_settings:
            # Get structure file name
            struct_file_name = ""
            self.file.writeline(f"structure {struct_file_name}")
            for setting in mol_settings[molecule].keys():
                self.file.writeline(self.INDENT+f"{setting}")
            self.file.writeline(f"end structure")


