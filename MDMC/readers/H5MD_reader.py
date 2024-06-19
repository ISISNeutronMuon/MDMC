"""Modual for reading a in H5MD file"""
import h5py
import numpy

def particles_file_path(file: h5py.File) -> h5py.File:
    """
    Builds and returns the first part of the file path that is in the H5MD
    file as the first 2 branches do not contain any data

    Parameters
    ----------
    file : h5py.File
        The H5MD file being read from

    Returns
    -------
    h5py.File
        Root to get to the in to the file to where the particle data is stored
    """
    key = list(file['particles'].keys())[0]
    root = file[f'particles/{key}']
    return root

def read_dataset(file: h5py.File, dataset_name: str) -> numpy.ndarray:

    """
    Reads datasets within the H5MD file

    Parameters
    ----------
    file : h5py.File
        The H5MD file being read from
    dataset_name: str
        The dataset atempting to be read

    Returns
    -------
    numpy.ndarray
        Returns a aray of the whole dataset

    Raises
    ------
    KeyError
        If the dataset is not found when the
        recursive check is finished, raise error
    """

    group = file.visit(lambda name: name if dataset_name in name else None)
    if group is None:
        raise KeyError(f"There is no dataset named '{dataset_name}' found in the H5MD")

    grp = file[group]
    if "value" in grp:
        grp = grp["value"]

    # if read in as bytestring read as string
    if isinstance(grp[1], bytes):
        grp = grp.asstr()

    return grp[:]

def read_units(file: h5py.File, data_name: str) -> str:
    """
    Reads units of data stored in the H5MD file

    Parameters
    ----------
    file : h5py.File
        The H5MD file being read from
    data_name : str
        The name of the data teh unit is to be read from

    Returns
    -------
    str
        String abreviation of the units
    """
    key = list(file['particles'].keys())[0]

    # time is stored deeper that the other units so done like this
    if data_name == 'time':
        return file[f'particles/{key}/position/time'].attrs['unit']

    grp = file[f'particles/{key}/{data_name}']

    if "value" in grp:
        grp = grp["value"]

    return grp.attrs['unit']

def read_times(file: h5py.File, step: int) -> float:
    """
    Reads time of a spesified step from the H5MD file and then slices
    read data so return is a float, not a HD5 object reference

    Parameters
    ----------
    file : h5py.File
        The H5MD file being read from
    step: int
        The time step the H5MD file is calculating the time

    Returns
    -------
    float
        Simulation time at simulation step
    """
    group_step = particles_file_path(file)
    time = group_step['position/time'][group_step['position/time'].shape]
    time_offset = group_step['position/time'].attrs['offset']
    return (time*step)+time_offset

def read_number_steps(file: h5py.File) -> int:
    """
    Calculates the total number of steps stored in the H5MD file

    Parameters
    ----------
    file : h5py.File
        The H5MD file being read from

    Returns
    -------
    int
        Number of steps stored in H5MD file
    """
    group_step = particles_file_path(file)

    particle = group_step['position/value']

    no_steps = len(particle)

    return no_steps

def read_box_dimension(file: h5py.File) -> numpy.ndarray:
    """
    Reads box dimenions in from H5MD file

    Parameters
    ----------
    file : h5py.File
        The H5MD file being read from

    Returns
    -------
    numpy.ndaray
        The dimentions of the simulation box
    """
    group_step = particles_file_path(file)
    return group_step['box'].attrs['dimensions']

def read_box_boundary(file: h5py.File) -> numpy.ndarray:
    """
    Reads box boundary in from H5MD file

    Parameters
    ----------
    file : h5py.File
        The H5MD file being read from

    Returns
    -------
    str
        The boundary of the simulation box
    """
    group_step = particles_file_path(file)
    return group_step['box'].attrs['boundary']

def read_all_data(file: h5py.File) -> dict:
    """
    Reads all data from a the H5MD file and stores it in a dict

    Parameters
    ----------
    file : h5py.File
        The H5MD file being read from

    Returns
    -------
    dict
        A dictornary storing all data in the H5MD file
    """
    all_time = [read_times(file, step) for step in range(read_number_steps(file))]
    all_data = {
        'time': all_time,
        'position': read_dataset(file, 'position'),
        'velocity': read_dataset(file, 'velocity'),
        'mass': read_dataset(file, 'mass'),
        'species': read_dataset(file, 'species'),
        'no_steps': read_number_steps(file),
        'box_dimension': read_box_dimension(file),
        'charge': read_dataset(file, 'charge'),
        'atom_symbol': read_dataset(file, 'atom_symbols'),
        'time_unit': read_units(file, 'time'),
        'position_unit': read_units(file, 'position'),
        'velocity_unit': read_units(file, 'velocity'),
        'mass_unit': read_units(file, 'mass'),
        'charge_unit': read_units(file, 'charge')
    }
    return all_data
