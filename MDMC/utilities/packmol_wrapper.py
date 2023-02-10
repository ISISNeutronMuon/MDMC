import shutil
import subprocess
import os
import sys

from MDMC.MD.simulation import Universe

#TODO: create an overaching interface function to call all others
#TODO: make distinction between building an input file for packmol or some other MD simulation

def fill_with_packmol(inp_file_name: str, desired_cwd: str) -> None:
    """
    Parameters
    ----------
    inp_file_name: str
        The name the .inp file used to input into packmol.
    desired_cwd: str
        A string path to the working directory from which you want to call packmol (i.e. where your
        input files are)

    """
    path_var = ":".join(sys.path)
    packmol_path = shutil.which("packmol", path=path_var)
    command_list = [packmol_path, "<"]
    command_list.append(desired_cwd+inp_file_name)
    _call_external_program(command_list, desired_cwd)


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
