"""A module for a class to export a packmol input file"""
from __future__ import annotations
from typing import TYPE_CHECKING

from MDMC.exporters.exporter import Exporter

if TYPE_CHECKING:
    from MDMC.MD.packmol.packmol_setup import PackmolSetup

class PackmolInputExporter(Exporter):
    """A class to export `PackmolSetup` objects into packmol input files"""
    INDENT = "  "

    #pylint: disable=arguments-differ, too-few-public-methods
    def write(self, setup: PackmolSetup,
              molecule_file_names: dict,
              output_name: str = "output_file.pdb",
              **settings: dict) -> None:
        """
        Write the data contained in a `PackmolSetup` object to a packmol input file

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
        self.file.writelines(["# Created by MDMC\n",
                              f"tolerance {tol}\n",
                              "filetype pdb\n",
                              f"output {output_name}\n"])
        self.file.write("\n")

        for molecule_setting in mol_settings:
            # Get structure file name
            struct_file_name = molecule_file_names[molecule_setting["molecule"]]
            if not struct_file_name.endswith(".pdb"):
                struct_file_name += ".pdb"

            self.file.write(f"structure {struct_file_name}\n")
            # Write each setting that is relevant to the structure
            constraint_settings = dict(molecule_setting.items())
            constraint_settings.pop("molecule")
            for setting in constraint_settings.keys():
                self.file.writelines(self.INDENT+f"{setting} {constraint_settings[setting]}\n")
            self.file.write("end structure\n")
            self.file.write("\n")
