"""A module that integrates packmol into MDMC"""
#TODO: Move into MDMC.MD.Packmol
import re
import shutil
import subprocess
import os

import numpy as np

from MDMC.MD.packmol.packmol_setup import PackmolSetup
from MDMC.MD import Universe, Molecule
from MDMC.exporters.configurations.pdb import ProteinDataBankExporter
from MDMC.exporters.packmol_input import PackmolInputExporter
from MDMC.readers.configurations.packmol_pdb import PackmolPDBReader

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
    original_cwd = os.getcwd()
    packmol_file_path = os.path.join(original_cwd, "packmol_files")
    if not os.path.exists(packmol_file_path):
        os.makedirs(packmol_file_path)

    input_path = os.path.join(packmol_file_path, "input_file.inp")
    output_path =  os.path.join(packmol_file_path,"output-universe.pdb")

    # Export molecules into PDB format
    molecules = setup_data.get_molecules()
    mol_file_names = {}
    # Enumerate molecules to ensure that an empty molecule name will have a non-empty file name
    for i, molecule in enumerate(molecules):
        file_name = f"{str(molecule.name)}-{str(i)}"
        file_path = os.path.join(packmol_file_path, f"{file_name}.pdb")
        pdb_exporter = ProteinDataBankExporter(file_path)
        with pdb_exporter:
            pdb_exporter.write(molecule)
        mol_file_names[molecule] = file_name

    # Create packmol input file
    inp_exporter = PackmolInputExporter(input_path)
    with inp_exporter:
        inp_exporter.write(setup_data, mol_file_names, output_path)

    # Call packmol
    # Create packmol call
    packmol_exec_path = get_packmol_path()
    command_list = [f"{packmol_exec_path}", "<", f"{input_path}"]

    # Run packmol on input file
    _call_external_program(command_list, work_dir=packmol_file_path)

    # Convert into MDMC universe
    # Read Output
    reader = PackmolPDBReader(output_path)
    with reader:
        reader.parse()
        output_molecules = reader.molecules

    # Create Universe from output
    dim = setup_data.get_max_sizes()
    universe = Universe(dim)

    _, mol_settings = setup_data.get_settings() # All molecules in setup + their metadata
    # Loops over all molecules in setup
    for molecule_setting in mol_settings:
        molecule = molecule_setting["molecule"]
        number_of_molecules = molecule_setting["number"]
        count = 0
        while count < number_of_molecules:
            # copy atoms from user defined `molecule`
            # apply new positions to atoms
            atom_copies = []
            for input_atom, output_atom in zip(molecule.atoms, output_molecules[count].atoms):
                atom_copies.append(input_atom.copy(position=output_atom.position))
            molecule_copy = Molecule(atoms=atom_copies)
            universe.add_structure(molecule_copy)
            count += 1
        if len(output_molecules) != number_of_molecules:
            output_molecules = output_molecules[number_of_molecules:]

    print(len(universe.bonded_interactions))
    return universe


def get_packmol_path() -> str:
    """
    Returns a string containing the path to packmol from the PATH environment variable,
    if it exists. Otherwise, returns ``None`` if packmol is not in PATH.
    """
    if shutil.which("packmol") is not None:
        return shutil.which("packmol")
    else:
        return "packmol"

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
