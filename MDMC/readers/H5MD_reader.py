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

def read_atom_mass(file: h5py.File) -> numpy.ndarray:
    """
    Reads all atom masses from the H5MD file and then slices
    read data so return is just array not a HD5 object reference

    Parameters
    ----------
    file : h5py.File
        The H5MD file being read from

    Returns
    -------
    numpy.ndarray
        returns an array with the masses of all the attoms masses
    """
    group_time = particles_file_path(file)
    return group_time['mass'][:]

def read_atom_mass_unit(file: h5py.File) -> str:
    """
    Reads the atom mass unit of mesurments from the H5MD file

    Parameters
    ----------
    file : h5py.File
        The H5MD file being read from

    Returns
    -------
    str
        String containing abreviation of the atom mass unit
    """
    group_time = particles_file_path(file)
    return group_time['mass'].attrs['unit']

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

def read_time_unit(file: h5py.File) -> str:
    """
    Reads the simulation time unit from the H5MD file

    Parameters
    ----------
    file : h5py.File
        The H5MD file being read from

    Returns
    -------
    str
        String containing abreviation of the atom mass unit
    """
    group_step = particles_file_path(file)
    return group_step['position/time'].attrs['unit']

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

def read_species(file: h5py.File) -> numpy.ndarray:
    """
    Reads atom species in from H5MD file

    Parameters
    ----------
    file : h5py.File
        The H5MD file being read from

    Returns
    -------
    numpy.ndaray
        The species of the atoms in the simulation
    """
    group_step = particles_file_path(file)
    return group_step['species'][:]

def read_positions(file: h5py.File) -> numpy.ndarray:
    """
    Reads particle positions in from H5MD file

    Parameters
    ----------
    file : h5py.File
        The H5MD file being read from

    Returns
    -------
    numpy.ndaray
        The species of the atoms in the simulation
    """
    group_step = particles_file_path(file)
    return group_step['position/value'][:]

def read_positions_unit(file: h5py.File) -> str:
    """
    Reads the positions unit from the H5MD file

    Parameters
    ----------
    file : h5py.File
        The H5MD file being read from

    Returns
    -------
    str
        String containing abreviation of the positions unit
    """
    group_step = particles_file_path(file)
    return group_step['position/value'].attrs['unit']

def read_velocity(file: h5py.File) -> numpy.ndarray:
    """
    Reads velocity of the particles in from H5MD file

    Parameters
    ----------
    file : h5py.File
        The H5MD file being read from

    Returns
    -------
    numpy.ndaray
        The velocity of the atoms in the simulation
    """
    group_step = particles_file_path(file)
    return group_step['velocity/value'][:]

def read_velocity_unit(file: h5py.File) -> str:
    """
    Reads the velocity time unit from the H5MD file

    Parameters
    ----------
    file : h5py.File
        The H5MD file being read from

    Returns
    -------
    str
        String of containing abreviation of the velocity unit
    """
    group_step = particles_file_path(file)
    return group_step['velocity/value'].attrs['unit']

def read_charge(file: h5py.File) -> numpy.ndarray:
    """
    Reads charge of the particles in from H5MD file

    Parameters
    ----------
    file : h5py.File
        The H5MD file being read from

    Returns
    -------
    numpy.ndaray
        The charge of the atoms in the simulation
    """
    group_step = particles_file_path(file)
    return group_step['charge'][:]

def read_charge_unit(file: h5py.File) -> str:
    """
    Reads the charge unit from the H5MD file

    Parameters
    ----------
    file : h5py.File
        The H5MD file being read from

    Returns
    -------
    str
        String containing abreviation of the charge unit
    """
    group_step = particles_file_path(file)
    return group_step['charge'].attrs['unit']

def read_atom_symbols(file: h5py.File) -> numpy.ndarray:
    """
    Reads atom symbols in from H5MD file

    Parameters
    ----------
    file : h5py.File
        The H5MD file being read from

    Returns
    -------
    numpy.ndaray
        The atom symbols of the atoms in the simulation
    """
    loc = 'parameters'
    group_step = file[loc]
    atom_symbol_string = []
    for byte in group_step['atom_symbols'][:]:
        char = byte.decode('utf-8')
        atom_symbol_string.append(char)
    return atom_symbol_string

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
        'possition': read_positions(file),
        'velocity': read_velocity(file),
        'mass': read_atom_mass(file),
        'specie': read_species(file),
        'no_steps': read_number_steps(file),
        'box_dimension': read_box_dimension(file),
        'charge': read_charge(file),
        'atom_symbol': read_atom_symbols(file),
        'time_unit': read_time_unit(file),
        'possition_unit': read_positions_unit(file),
        'velocity_unit': read_velocity_unit(file),
        'mass_unit': read_atom_mass_unit(file),
        'charge_unit': read_charge_unit(file)
    }
    return all_data
