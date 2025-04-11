"""
Module for reading in H5MD files.
"""
import h5py
import numpy

# Note: In getter functions within this file, the slices
#       are to ensure that the returned types from the
#       functions are `numpy.ndarray`s and not `h5py.dataset`s.


def particles_file_path(file: h5py.File) -> h5py.Group:
    """
    Returns the path in the H5MD file where the particle data is stored.

    Parameters
    ----------
    file : h5py.File
        The H5MD file being read from.

    Returns
    -------
    h5py.Group
        Particle data group.
    """
    key = next(iter(file['particles']))
    root = file[f'particles/{key}']
    return root

def read_dataset(file: h5py.File, dataset_name: str) -> numpy.ndarray:

    """
    Read a dataset within an H5MD file.

    Parameters
    ----------
    file : h5py.File
        The H5MD file being read from
    dataset_name: str
        The dataset attempting to be read

    Returns
    -------
    numpy.ndarray
        Returns an array of the whole dataset

    Raises
    ------
    KeyError
        If the dataset is not found in the file.
    """

    group = file.visit(lambda name: name if dataset_name in name else None)
    if group is None:
        raise KeyError(f"There is no dataset named '{dataset_name}' found in the H5MD file.")

    grp = file[group]
    if "value" in grp:
        grp = grp["value"]

    # if read in as bytestring read as string
    if isinstance(grp[1], bytes):
        grp = grp.asstr()

    return grp[:]

def read_units(file: h5py.File, data_name: str) -> str:
    """
    Read the SI units of the data stored in an H5MD file.

    Parameters
    ----------
    file : h5py.File
        The H5MD file being read from
    data_name : str
        The name of the data the unit is to be read from

    Returns
    -------
    str
        String abbreviation of the units
    """
    key = list(file['particles'].keys())[0]

    # time is stored deeper than the other units so done like this
    if data_name == 'time':
        return file[f'particles/{key}/position/time'].attrs['unit']

    grp = file[f'particles/{key}/{data_name}']

    if "value" in grp:
        grp = grp["value"]

    return grp.attrs['unit']

def read_times(file: h5py.File, step: int) -> float:
    """
    Read the time of a specified step from an H5MD file.

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
    Calculate the total number of simulation frames stored in an H5MD file.

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
    position_grps = group_step['position/value']

    return len(position_grps)

def read_box_property(file: h5py.File, property_name: str) -> numpy.ndarray:
    """
    Reads box properties from H5MD file.

    Parameters
    ----------
    file : h5py.File
        The H5MD file being read from

    property_name : str
        The name of the box property being read

    Returns
    -------
    numpy.ndaray
        An array returning the property read

    Notes
    -----
    This `box` is equivalent to what MDMC calls a `Universe` internally.
    """
    group_step = particles_file_path(file)
    return group_step['box'].attrs[property_name]

def read_all_data(filename: str) -> dict:
    """
    Read all data from an H5MD file and return it as a dictionary.

    Parameters
    ----------
    file : str
        The name of file being read from.

    Returns
    -------
    dict
        A dictionary storing all data in the H5MD file.
    """
    with h5py.File(filename, 'r') as file:
        all_time = [read_times(file, step) for step in range(read_number_steps(file))]
        all_data = {
            'time': all_time,
            'position': read_dataset(file, 'position'),
            'velocity': read_dataset(file, 'velocity'),
            'mass': read_dataset(file, 'mass'),
            'species': read_dataset(file, 'species'),
            'no_steps': read_number_steps(file),
            'box_dimension': read_box_property(file, 'dimensions'),
            'box_boundary': read_box_property(file, 'boundary'),
            'charge': read_dataset(file, 'charge'),
            'atom_symbol': read_dataset(file, 'atom_symbols'),
            'time_unit': read_units(file, 'time'),
            'position_unit': read_units(file, 'position'),
            'velocity_unit': read_units(file, 'velocity'),
            'mass_unit': read_units(file, 'mass'),
            'charge_unit': read_units(file, 'charge'),
        }
    return all_data
