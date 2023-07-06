"""A module that integrates packmol into MDMC"""
import re
import shutil
import subprocess
import os

from typing import List

from MDMC.MD.packmol.packmol_setup import PackmolSetup
from MDMC.MD import Universe, Molecule, Atom, Structure
from MDMC.exporters.configurations.pdb import ProteinDataBankExporter
from MDMC.exporters.configurations.packmol_input import PackmolInputExporter
from MDMC.readers.configurations.packmol_pdb import PackmolPDBReader


class PackmolFiller:
    """
    A class representing a packmol run based on a `PackmolSetup` object.
    """

    def __init__(self, setup_data: PackmolSetup):
        self._setup_data: str = setup_data
        self._mol_file_names = {}
        self._filled_universe: Universe = None
        self._packmol_files_path: str = None
        self._input_path: str = None
        self._output_path: str = None

    @property
    def filled_universe(self) -> Universe:
        """
        Get the `Universe` filled via the packmol run.

        Returns
        -------
        Universe
            An MDMC Universe.
        """
        return self._filled_universe

    @property
    def setup_data (self) -> PackmolSetup:
        """
        The Packmol setup data for this run.
        
        Returns
        -------
        PackmolSetup
            The setup data for this run.
        """
        return self._setup_data

    def fill_with_packmol(self) -> Universe:
        """
        Parameters
        ----------
        setup_data
            A `PackmolSetup` object containing the data for the packmol run.

        Returns
        -------
        `Universe`
            A `Universe` object filled with the structures requested
            by the `PackmolSetup` object.
        """
        self._setup_data.validate_setup()
        self._create_paths()
        self._export_structures()
        self._export_system_setup()
        self._call_packmol()
        structures = self._read_packmol_output()
        self._filled_universe = self._fill_universe(structures)
        return self._filled_universe

    def _create_paths(self) -> None:
        """
        Creates the necessary paths and (if needed) directories in which to
        place & run packmol files
        """
        original_cwd = os.getcwd()
        self._packmol_files_path = os.path.join(original_cwd, "packmol_files")
        if not os.path.exists(self._packmol_files_path):
            os.makedirs(self._packmol_files_path)

        self._input_path = os.path.join(self._packmol_files_path, "input_file.inp")
        self._output_path = os.path.join(self._packmol_files_path, "output-universe.pdb")

    def _export_structures(self) -> None:
        """
        Exports the molecules from a `PackmolSetup` object to pdb format
        """
        structures = self._setup_data.get_structures()
        # Enumerate structures to ensure that an empty structure name will have a non-empty file name
        for i, structure in enumerate(structures):
            file_name = f"{str(structure.name)}-{str(i)}"
            file_path = os.path.join(self._packmol_files_path, f"{file_name}.pdb")
            pdb_exporter = ProteinDataBankExporter(file_path)
            with pdb_exporter:
                pdb_exporter.write(structure)
            self._mol_file_names[structure] = file_name

    def _export_system_setup(self) -> None:
        """
        Exports the setup of a `PackmolSetup` system to a packmol input file (.inp file).
        """
        # Create packmol input file
        inp_exporter = PackmolInputExporter(self._input_path)
        with inp_exporter:
            inp_exporter.write(self._setup_data, self._mol_file_names, self._output_path)

    def _call_packmol(self) -> None:
        """
        A function to call packmol on the input path using the packmol files directory.
        """
        # Create packmol call
        packmol_exec_path = self.get_packmol_path()
        command_list = [f"{packmol_exec_path}", "<", f"{self._input_path}"]

        # Run packmol on input file
        self._call_external_program(command_list, work_dir=self._packmol_files_path)

    @staticmethod
    def get_packmol_path() -> str:
        """
        Returns a string containing the path to packmol from the PATH environment variable,
        if it exists. Otherwise, returns the current working directory if packmol is not in PATH.
        """
        if shutil.which("packmol") is not None:
            return shutil.which("packmol")
        return "packmol"

    def get_packmol_output_path(self) -> str:
        """
        Obtains the name of the packmol output file, as defined by the input file.
        Returns an empty string if there is no input file name defined.

        Returns
        -------
        str
            The name of the packmol output file name.
        """
        with open(self._input_path, "r", encoding="UTF-8") as inp_file:
            contents = inp_file.readlines()

        pattern = re.compile("output.*")

        name = ""
        for line in contents:
            if pattern.match(line):
                name = line.split()[1]

        return name

    def get_packmol_files_path(self) -> str:
        """
        Get the path in which packmol files are placed and run.

        Returns
        -------
        str
            The path to the packmol files directory.
        """
        return self._packmol_files_path

    def _read_packmol_output(self) -> List[Structure]:
        """
        A function to read in the packmol output and return the molecules read in

        Returns
        -------
        `List`
            A list of `Structure` objects (i.e. `Molecule` or `Atom`) read from the packmol output.
        """
        reader = PackmolPDBReader(self._output_path)
        with reader:
            reader.parse()
            output_structures = reader.structures
        return output_structures

    def _fill_universe(self, output_structures: List[Structure]) -> Universe:
        """
        A function to fill in the universe with the output data from packmol.

        Parameters
        ----------
        output_structures: `List` of `Structure`
            A list containing the `Structure` objects (i.e. Atoms & Molecules)
            read from packmol to MDMC.

        Returns
        -------
        `Universe`
            A `Universe` object filled with the `Structure` objects in `output_structures`.
        """
        dim = self._setup_data.get_max_sizes()
        universe = Universe(dimensions=dim, verbose=-1)
        _, struct_settings = self._setup_data.get_settings()  # All structures + their metadata

        # Loop through all molecules specified for the system
        for structure_setting in struct_settings:
            reference_structure = structure_setting["structure"]
            number_of_structures = structure_setting["number"]
            current_structures = output_structures[:number_of_structures]
            output_structures = output_structures[number_of_structures:]

            # Copy structure positions
            for current_structure in current_structures:
                # Case where the structure we're copying is a molecule
                if isinstance(reference_structure, Molecule):
                    # get list of atom positions
                    output_molecule_pos_data = [atom.position for atom in current_structure]
                    # copy whole molecule
                    copied_molecule = reference_structure.copy((0., 0., 0.))
                    # adjust individual atom positions
                    for copied_atom, new_pos in zip(copied_molecule.atoms,
                                                    output_molecule_pos_data):
                        copied_atom.position = new_pos

                    # Recalculate centre of mass & add everything to universe
                    copied_molecule.position = copied_molecule.calc_CoM()
                    universe.add_structure(copied_molecule)

                # If the structure we want is an atom, we just copy that one atom into the universe
                elif isinstance(reference_structure, Atom):
                    copied_atm = reference_structure.copy(current_structure.atoms[0].position)
                    universe.add_structure(copied_atm)

            # change output_structures to remove structures we have just added
            # ensures we start with the right structure type in the next iteration
            if len(output_structures) != number_of_structures:
                output_structures = output_structures[number_of_structures:]

        return universe

    #TODO possibly move this to common or utils?
    def _call_external_program(self, command_list: List[str], work_dir: str=None) -> None:
        """
        A function to call an external program in a specific working directory - defaults to
        current working directory as a failsafe

        Parameters
        ----------
        command_list: list of str
            The list of string arguments to be passed to the shell in order
        work_dir: str
            The desired working directory for the program to run in

        """
        command_list = " ".join(command_list)
        try:
            subprocess.run(args=command_list, cwd=work_dir, shell=True, check=True)
        except subprocess.CalledProcessError:
            wd = os.getcwd()
            subprocess.run(args=command_list, cwd=wd, shell=True, check=True)
