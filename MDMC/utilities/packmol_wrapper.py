import subprocess
from MDMC.MD.simulation import Universe

#TODO: create an overaching interface function to call all others
#TODO: make distinction between building an input file for packmol or some other MD simulation
#TODO: create function to check for necessary files needed for packmol usage
#TODO: create function to check for necessary files needed for fftool usage
#TODO: make packmol wrapper function
def _call_fftool(universe: Universe, molecule_dict: dict, density: float=None, md_eng: str=None) -> None:
    """
    A wrapper function to call fftool to create the input file needed for packmol (or an MD software
    if specified)

    Parameters
    ----------
    universe: Universe
        The universe object in which to add the molecules to
    molecule_dict: dict
        A dictionary containing the path of the atomic coordinates file as the key,
        with the (integer) number of molecules as the value
        Current formats accepted are the same as that of fftool
        (see: https://github.com/paduagroup/fftool)
    density: optional, float
        The target density (in mol/L) for the system. If not supplied, the side length(s)
        of the universe will be used
    md_eng: optional, str
        If provided, the function will use fftool to create a file for a specified MD software
        instead of for packmol (currently - LAMMPS, DL_POLY, GROMACS and OpenMM are supported
        by fftool)

    """
    command_list = ["fftool"]
    dimensions = universe.dimensions

    # Add molecules & number
    for molecule_tup in molecule_dict.items():
        command_list.append(str(molecule_tup[1]))
        command_list.append(str(molecule_tup[0]))

    # Input density/boundary conditions to aim for
    if density is not None:
        command_list.append("-r")
        command_list.append(str(density))
    else:
        command_list.append("-b")
        if type(dimensions) == float:
            command_list.append(str(dimensions))
        else:
            for dimension in dimensions:
                command_list.append(str(dimension))

    # MD engine flag for output (if provided)
    if md_eng is not None:
        if md_eng == "LAMMPS":
            command_list.append("-l")
        elif md_eng == "OpenMM":
            command_list.append("-x")
        elif md_eng == "GROMACS":
            command_list.append("-g")
        elif md_eng == "DL_POLY":
            command_list.append("-d")

    subprocess.run(command_list)

def _call_packmol(path_to_inp: str="pack.inp") -> None:
    """
    Parameters
    ----------
    path_to_inp: str
        A string path to the .inp file used to input into packmol. Defaults to "pack.inp" - the
        fftool default under the assumption it was called in the current directory.

    """
    command_list = ["packmol", "<"]
    command_list.append(path_to_inp)