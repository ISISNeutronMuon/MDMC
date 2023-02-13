import re
import shutil
import subprocess
import os
import sys

import numpy as np

from MDMC.MD.simulation import Universe
import MDMC.readers.configurations.pdb as pdb_reader

def fill_with_packmol(inp_file_name: str, desired_cwd: str) -> Universe:
    """
    Parameters
    ----------
    inp_file_name: str
        The name the .inp file used to input into packmol.
    desired_cwd: str
        A string path to the working directory from which you want to call packmol
        (i.e. where your input files are)

    """

    packmol_path = get_packmol_path()
    command_list = [packmol_path, "<"]

    input_file_path = os.path.join(desired_cwd, inp_file_name)
    command_list.append(input_file_path)

    _call_external_program(command_list, desired_cwd)
    output_file_name = get_packmol_output_name(input_file_path)
    output_file_path = os.path.join(desired_cwd, output_file_name)
    reader = pdb_reader.ProteinDataBankReader(output_file_path)
    reader.parse()
    dim = get_packmol_universe_dimensions(input_file_path)
    universe = Universe(dim)
    for molecule in reader.molecules:
        universe.add_structure(molecule)
    return universe
def get_packmol_path() -> str:
    """
    Returns a string containing the path to packmol from the PATH environment variable,
    if it exists. Otherwise, returns ``None`` if packmol is not in PATH.
    """
    path_var = ":".join(sys.path)
    return shutil.which("packmol", path=path_var)

def get_packmol_output_name(inp_file_path: str) -> str:
    """
    Obtains the name of the packmol output file, as defined by the input file

    Parameters
    ----------
    inp_file_path: str
        The path to the packmol input file (.inp) as a string

    Returns
    -------
    str
        The name of the packmol output file name
    """
    with open(inp_file_path) as inp_file:
        contents = inp_file.readlines()

    pattern = re.compile("output.*")

    for line in contents:
        if pattern.match(line):
            return line.split()[1]

def get_packmol_universe_dimensions(inp_file_path: str) -> list[float]:
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
    with open(inp_file_path) as inp_file:
        contents = inp_file.readlines()

    pattern = re.compile(".*inside box.*")

    min_coords = np.array([np.zeros(3)])
    max_coords = np.array([np.zeros(3)])
    for line in contents:
        if pattern.match(line):
            line = line.split()
            minimums = [float(i) for i in line[2:5]]
            maximums = [float(i) for i in line[5:]]
            min_coords = np.append(min_coords, [minimums], axis=0)
            max_coords = np.append(max_coords, [maximums], axis=0)

    dimensions = []
    for i in range(0, 3):
        dimensions.append(max(max_coords[:, i]) - min(min_coords[:, i]))

    return dimensions

def _call_external_program(command_list: list[str], work_dir: str):
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
        subprocess.run(args=command_list, cwd=work_dir, shell=True)
    except:
        wd = os.getcwd()
        subprocess.run(args=command_list, cwd=wd, shell=True)
