"""Stores all filenames and additional data that is required for testing

Data for reader unit tests must have the same key as the reader name. Data for
observable system tests must have the same key as the observable name. Data for
MDMC objects must have the same key as the object name.  Descriptions of data
are also supplied.

AUTHOR :    Thomas Farmer        START DATE :    2018-6-7 14:10:35"""

from os import path

_ABS_DIR_PATH = path.split(path.abspath(__file__))[0]
_EXP_DATA_PATH = '/experimental_data'
_CALC_OBS_PATH = '/calculated_observables'
_OBJECT_PATH = '/MDMC_objects'

# TODO: introspective method to add abs_dir_path (use dir() to get all var names)

# Reader/experimental data
#
# LAMPSQw
# From Bertil Halle QENS water data on in5

READER_DATA = {'LAMPSQw':'/263K05Awat_LAMP'}

# Add paths to data values
for key in READER_DATA:
    READER_DATA[key] = _ABS_DIR_PATH + _EXP_DATA_PATH + READER_DATA[key]


# Calculated observable data
#
# Dynamic incoherent structure factor (DISF i.e. incoherent SQw)
# MD simulation on 2048 water molecules for 50000 timesteps of 1 fs length. DISF
# calculated from time start:end:step of 50:5000:10, q start:end:step of
# 1.6:22.4:1.6, with a qshell width of 0.1, maxmimum 50 q vectors per shell
# and a resolution of 0.05. File format is netcdf.
#
# Dynamic coherent structure factor (DCSF i.e. coherent SQw)
# Same MD simulation and nMOLDYN parameters as DISF.

OBS_DATA = {'SQw_incoh':'/nMOLDYN_ASCII_DISF_water.nc',
            'SQw_coh':'/nMOLDYN_ASCII_DCSF_water.nc'}

# Add paths to data values
for key in OBS_DATA:
    OBS_DATA[key] = _ABS_DIR_PATH + _CALC_OBS_PATH + OBS_DATA[key]


# MDMC object data
#
# trajectory
# Calculated from same MMTK simulation on water as used to calculate DISF

OBJECT_DATA = {'trajectory':'/trajectory.pkl'}

# Add paths to data values
for key in OBJECT_DATA:
    OBJECT_DATA[key] = _ABS_DIR_PATH + _OBJECT_PATH + OBJECT_DATA[key]
