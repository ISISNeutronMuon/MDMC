"""Stores all filenames and additional data that is required for testing

Data for reader unit tests must have the same name as the reader. Data for
observable system tests must have the same name as the observable. Origins for
data are also supplied.

AUTHOR :    Thomas Farmer        START DATE :    2018-6-7 14:10:35"""

from os import path

_ABS_DIR_PATH = path.split(path.abspath(__file__))[0]
_EXP_DATA_PATH = '/experimental_data'
_CALC_OBS_PATH = '/calculated_observables'

# TODO: introspective method to add abs_dir_path (use dir() to get all var names)

# Reader/experimental data
#
# LAMPSQw - From Bertil Halle QENS water data on in5

reader_data = {'LAMPSQw':'/263K05Awat_LAMP'}

# Add paths to data values
for key in reader_data:
    reader_data[key] = _ABS_DIR_PATH + _EXP_DATA_PATH + reader_data[key]


# Calculated observable data
#
# Dynamic incoherent structure factor (DISF i.e. incoherent SQw) - MD simulation
# on 2048 water molecules for 50000 timesteps of 1 fs length.  DISF calculated
# from time start:end:step of 50:5000:10, q start:end:step 1.6:22.4:1.6, with a
# qshell width of 0.1, maxmimum 50 q vectors per shell and a resolution of 0.05

obs_data = {'SQw_incoh':'/nMOLDYN_ASCII_DISF_water.txt'}
