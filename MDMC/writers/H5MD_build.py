"""
A module for building and saving a H5MD file.

Notes
-----
In getter functions within this file, the slices 
are to ensure that the returned types from the 
functions are `numpy.ndarray`s and not `h5py.dataset`s.
"""
from pathlib import Path

from datetime import datetime
import numpy as np
import h5py
import periodictable

from MDMC.trajectory_analysis.compact_trajectory import CompactTrajectory
from MDMC.common import units

METADATA = {
    'creator_name': 'MDMC',
    'creator_version': [0, 2],
    'h5md_version': [1, 1],
    'module_name': ['units'],
    'module_version': [[1, 0]],
    'loc': 'h5md'
}

ROOT_TRAJECTORY = 'particles/all'

def create_no_data_groups(open_file: h5py.File,
                          groups: list):
    """
    Creates all groups and subgroups that make up the main base structure of the file format

    Parameters
    ----------
    open_file : h5py.File
        A pre-opened file that is being writen to
    groups : list
        A list of groups that are being created
    """
    for group_name in groups:
        open_file.create_group(group_name)

def create_metadata_group(open_file: h5py.File, *,
                          creator_name:str = 'Unknown',
                          creator_email:str = 'Unknown'):
    """
    Creates h5md group that contains all Metadata info

    Parameters
    ----------
    open_file : h5py.File
        A pre-opened file that the data is being writen into
    creator_name : str, optional
        Name of person running the MDMC simulation, by default 'Unknown'
    creator_email : str, optional
        Email of the person running the MDMC simulation, by default 'Unknown'
    """
    group = open_file[METADATA['loc']]
    group.attrs['version'] = METADATA['h5md_version']

    author_group = group.create_group('author')
    author_group.attrs['name'] = creator_name
    author_group.attrs['email'] = creator_email

    creator_group = group.create_group('creator')
    creator_group.attrs['name'] = METADATA['creator_name']
    creator_group.attrs['version'] = METADATA['creator_version']

    modules_group = group.create_group('modules')
    for pos, module in enumerate(METADATA['module_name']):
        module_group = modules_group.create_group(module)
        module_group.attrs['version'] = METADATA['module_version'][pos]

def create_simulation_data(open_file: h5py.File,
                           group_name: str,
                           value: int | float | str | np.ndarray,
                           unit: str = None,
                           time_increment: int = None,
                           step_increment: int = None,
                           time_offset: int = None,
                           step_offset: int = None):
    """Stores data about the simulation

    Parameters
    ----------
    open_file : h5py.File
        A pre-opened file that the data is being writen into
    group_name : str
        The group name for the data being stored
    value : int | float | str | np.ndarray
        The data being stored in the H5MD file
    unit : str, optional
        The units for the data beng stored, if the data has units, by default None
    time_increment : int, optional
        The increment in time between each simulation step, by default None
    step_increment : int, optional
        The increment in between steps in a simulation, by default None
    time_offset : int, optional
        The offset that the time starts at, by default None
    step_offset : int, optional
        The offset what is where the steps start at, by default None
    """
    group = open_file[ROOT_TRAJECTORY]
    if time_increment is None:
        subdata = group.create_dataset(group_name, data= value)
    else:
        subgroup = group.create_group(group_name)
        subdata = subgroup.create_dataset('value', data= value)
        time_link = group.visit(find_time)
        step_link = group.visit(find_step)
        if time_link:
            time_data = subgroup.create_dataset('time', data=group[time_link])
            step_data = subgroup.create_dataset('step', data=group[step_link])
        else:
            time_data = subgroup.create_dataset('time', data=time_increment)
            step_data = subgroup.create_dataset('step', data=step_increment)
            time_data.attrs['offset'] = time_offset
            time_data.attrs['unit'] = str(units.SYSTEM['TIME'])
            step_data.attrs['offset'] = step_offset
    if unit is not None:
        subdata.attrs['unit'] = str(unit)

def find_time(name:str) -> str|None:
    """Finds the first instance of time within the H5MD file

    Parameters
    ----------
    name : str
        Name of the current directory being searched for to find time

    Returns
    -------
    str
        String containing the path of the first instance of 
        time and if time is not found returns None
    """
    if 'time' in name:
        return name

def find_step(name:str) -> str|None:
    """Finds the first instance of step within the H5MD file

    Parameters
    ----------
    name : str
        Name of the current directory being searched for to find step

    Returns
    -------
    str
        String containing the path of the first instance of 
        step and if step is not found returns None
    """
    if 'step' in name:
        return name

def create_box_data(open_file: h5py.File,
                    trajectory: CompactTrajectory):
    """
    Creates the box group and adds all atributes associated with this group

    Parameters
    ----------
    trajectory : CompactTrajectory
        The compact trajectory the file is being built from
    open_file : h5py.File
        A pre-opened file that the data is being writen into
    """
    box_group = open_file[f'{ROOT_TRAJECTORY}/box']
    box_group.attrs['dimensions'] = len(trajectory.dimensions)
    boundry = []
    if trajectory.is_fixedbox:
        for _ in range(len(trajectory.dimensions)):
            boundry.append('periodic')  # MDMC assumes perperiodic
        box_group.attrs['boundary'] = boundry

def create_paramater_data(open_file: h5py.File, data: np.array):
    """
    Creates and stores the extra data that has no place elsewhere in the H5MD

    Parameters
    ----------
    open_file : h5py.File
        A pre-opened file that the data is being writen into
    data : np.array
        The data being stored
    """
    paramaters = open_file['parameters']
    paramaters.create_dataset('atom_symbols', data=data)

def build_full(trajectory: CompactTrajectory,
               filename: str = "trajectory", *,
               timestamp: bool = True,
               file_loc: str = "../file/"):
    """
    Creates full H5MD file including all elements.

    Parameters
    ----------
    trajectory : CompactTrajectory
        The compact trajectory the file is being built from
    filename : str, optional
        The name of the H5MD file, by default "trajectory"
    timestamp : bool, optional
        If true adds time timestamp to file name, by default True
    file_loc : str, optional
        The file where the H5MD file should be stored, by defalt ../file/
    """

    time_increment = trajectory.time[1] - trajectory.time[0]
    time_offset = trajectory.time[0]
    step_increment = 1
    step_offset = 0
    file_name = Path(filename).with_suffix(".h5")

    if timestamp:
        time_stamp = datetime.now().strftime('%d%m%y-%H.%M.%S.%f')
        file_name = file_name.with_stem(f'{time_stamp}_{file_name.stem}')

    with h5py.File(file_name, 'w') as file:
        no_data_groups = ['particles',
                          ROOT_TRAJECTORY,
                          'h5md',
                          f'{ROOT_TRAJECTORY}/box',
                          'parameters']
        create_no_data_groups(file, no_data_groups)

        create_metadata_group(file)

        charge = trajectory.atom_charges
        charge[charge == None] = 0.0
        charge = charge.astype(float)

        species = [getattr(periodictable, element).number
                   for element in trajectory.element_list]

        dependent_data = []
        dependent_data.append(['species',
                               species])
        dependent_data.append(['charge',
                               charge,
                               units.SYSTEM['CHARGE']])
        dependent_data.append(['mass',
                               trajectory.atom_masses,
                               units.SYSTEM['MASS']])
        dependent_data.append(['position',
                               trajectory.position,
                               trajectory.position_unit,
                               time_increment,
                               step_increment,
                               time_offset,
                               step_offset])
        dependent_data.append(['box/edges',
                               trajectory.dimensions,
                               trajectory.position_unit,
                               time_increment,
                               step_increment,
                               time_offset,
                               step_offset])
        if trajectory.has_velocity:
            dependent_data.append(['velocity',
                                   trajectory.velocity,
                                   trajectory.velocity_unit,
                                   time_increment, step_increment,
                                   time_offset,
                                   step_offset])

        for data in dependent_data:
            create_simulation_data(file, *data)

        create_box_data(file, trajectory)
        atom_symbols_data = np.array(trajectory.element_list, dtype=object)
        create_paramater_data(file, atom_symbols_data)
    print("")
