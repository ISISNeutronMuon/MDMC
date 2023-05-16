"""A module that integrates packmol into MDMC"""

import re
import shutil
import subprocess
import os

import numpy as np

from MDMC.MD.packmol.packmol_setup import PackmolSetup
from MDMC.MD.simulation import Universe
from MDMC.exporters.configurations.pdb import ProteinDataBankExporter
from MDMC.exporters.packmol_input import PackmolInputExporter

def fill_with_packmol(setup_data: PackmolSetup) -> Universe:
    """
    Parameters
    ----------
    setup_data
        A `PackmolSetup` object containing the data for the

    Returns
    -------
    A `Universe` object filled with the molecules requested by the user as per the `PackmolSetup` object
    """
    packmol_path = "./packmol_files/"
    input_path = os.path.join(packmol_path, "input_file.inp")
    output_path =  os.path.join(packmol_path,"output-universe.pdb")
    # Export molecules into PDB format
    molecules = setup_data.get_molecules()
    mol_file_names = {}
    # Enumerate molecules to ensure that an empty molecule name will have a non-empty file name
    for i, molecule in enumerate(molecules):
        if molecule.name:
            file_name = str(i)
        else:
            file_name = molecule.name
        pdb_exporter = ProteinDataBankExporter(os.path.join(packmol_path, f"{file_name}.pdb"))
        with pdb_exporter:
            pdb_exporter.write(molecule)
        mol_file_names[molecule] = file_name

    # Create packmol input file
    inp_exporter = PackmolInputExporter(input_path)
    with inp_exporter:
        inp_exporter.write(setup_data, mol_file_names, output_path)

    # Call packmol
    # Create packmol call
    command_list = [packmol_path, "<"]
    command_list.append(input_path)

    # Run packmol on input file
    _call_external_program(command_list)

    # # Convert into MDMC universe
    # # Read Output
    # reader = PackmolPDBReader(output_path)
    # with reader:
    #     reader.parse()

    # Create Universe from output
    dim = get_packmol_universe_dimensions(input_path)
    universe = Universe(dim)

    # # Identify molecules from output
    # for molecule in reader.molecules:
    #     universe.add_structure(molecule)
    #
    # # Fill molecules with packmol-provided positions


    return universe

#TODO: Create Algorithm for finding which molecule read in from packmol
# corresponds to which molecule from the user's input
#TODO: Compare molecules by stoichiometry and bonds
#TODO: Create algorithm to copy user's molecules into position of those returned by packmol


def get_packmol_path() -> str:
    """
    Returns a string containing the path to packmol from the PATH environment variable,
    if it exists. Otherwise, returns ``None`` if packmol is not in PATH.
    """
    return shutil.which("packmol")

def get_packmol_output_name(inp_file_path: str) -> str:
    """
    Obtains the name of the packmol output file, as defined by the input file
    Returns an empty string if there is no input file name defined

    Parameters
    ----------
    inp_file_path: str
        The path to the packmol input file (.inp) as a string, an empty string if no file defined.

    Returns
    -------
    str
        The name of the packmol output file name
    """
    with open(inp_file_path, "r", encoding="UTF-8") as inp_file:
        contents = inp_file.readlines()

    pattern = re.compile("output.*")

    name = ""
    for line in contents:
        if pattern.match(line):
            name = line.split()[1]

    return name

def get_packmol_universe_dimensions(inp_file_path: str) -> 'list[float]':
    """
    Obtains and calculates the dimensions needed for a universe from a packmol input file
    Parameters
    ----------
    inp_file_path: str
        The path to the packmol input file (.inp) as a string

    Returns
    -------
    list[float]
        The size of the universe needed for the system
    """
    with open(inp_file_path, "r", encoding="UTF-8") as inp_file:
        contents = inp_file.readlines()

    pattern = re.compile(".*inside box.*")

    min_coords = np.zeros(3)
    max_coords = np.zeros(3)
    for line in contents:
        if pattern.match(line):
            line = line.split()
            minimums = [float(i) for i in line[2:5]]
            maximums = [float(i) for i in line[5:]]
            for i in range (0, 3):
                min_coords[i] = minimums[i] if minimums[i] < min_coords[i] else min_coords[i]
                max_coords[i] = maximums[i] if maximums[i] > max_coords[i] else max_coords[i]

    dimensions = max_coords - min_coords

    return dimensions

def _call_external_program(command_list: 'list[str]', work_dir: str=None):
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
