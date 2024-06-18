"""
A module for building and saving a H5MD file.
"""
from pathlib import Path

from datetime import datetime
import numpy as np
import h5py
import periodictable

from MDMC.trajectory_analysis.compact_trajectory import CompactTrajectory
from MDMC.common import units

H5MD_DATA = {
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
    group = open_file[H5MD_DATA['loc']]
    group.attrs['version'] = H5MD_DATA['h5md_version']

    author_group = group.create_group('author')
    author_group.attrs['name'] = creator_name
    author_group.attrs['email'] = creator_email

    creator_group = group.create_group('creator')
    creator_group.attrs['name'] = H5MD_DATA['creator_name']
    creator_group.attrs['version'] = H5MD_DATA['creator_version']

    modules_group = group.create_group('modules')
    for pos, module in enumerate(H5MD_DATA['module_name']):
        module_group = modules_group.create_group(module)
        module_group.attrs['version'] = H5MD_DATA['module_version'][pos]

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
        time_link = group.visit(lambda name: name if 'time' in name else None)
        step_link = group.visit(lambda name: name if 'step' in name else None)
        if time_link is not None:
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

def create_box_data(open_file: h5py.File,
                    trajectory: CompactTrajectory):
    """
    Creates the box group and adds all attributes associated with this group

    Parameters
    ----------
    trajectory : CompactTrajectory
        The compact trajectory the file is being built from
    open_file : h5py.File
        A pre-opened file that the data is being writen into
    """
    box_group = open_file[f'{ROOT_TRAJECTORY}/box']
    box_group.attrs['dimensions'] = len(trajectory.dimensions)
    if trajectory.is_fixedbox:
        boundary = ['periodic' for _ in trajectory.dimensions]  # MDMC assumes periodic
        box_group.attrs['boundary'] = boundary

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
               file_loc: Path = Path(__file__).parents[1] / "H5MD_Files"):
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
    file_loc : Path, optional
        The file where the H5MD file should be stored, by default '../file/'
    """

    time_increment = trajectory.time[1] - trajectory.time[0]
    time_offset = trajectory.time[0]
    step_increment = 1
    step_offset = 0
    file_name = Path(filename).with_suffix('.h5')

    if timestamp:
        time_stamp = datetime.now().strftime('%d%m%y-%H.%M.%S.%f')
        file_name = file_name.with_stem(f'{file_name.stem}_{time_stamp}')

    file_path_name = f'{file_loc}/{file_name}'

    with h5py.File(file_path_name, 'w') as file:
        no_data_groups = ['particles',
                          ROOT_TRAJECTORY,
                          'h5md',
                          f'{ROOT_TRAJECTORY}/box',
                          'parameters']
        create_no_data_groups(file, no_data_groups)

        create_metadata_group(file)

        charge = trajectory.atom_charges
        for count, value in enumerate(charge):
            if value is None:
                charge[count] = 0.0
        charge = charge.astype(float)

        species = [getattr(periodictable, element).number
                   for element in trajectory.element_list]

        dependent_data = [
            ['species',species],
            ['charge', charge, units.SYSTEM['CHARGE']],
            ['mass', trajectory.atom_masses, units.SYSTEM['MASS']],
            [
                'position',
                trajectory.position,
                trajectory.position_unit,
                time_increment,
                step_increment,
                time_offset,
                step_offset
            ],
            [
                'box/edges',
                trajectory.dimensions,
                trajectory.position_unit,
                time_increment,
                step_increment,
                time_offset,
                step_offset
            ]
        ]
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
