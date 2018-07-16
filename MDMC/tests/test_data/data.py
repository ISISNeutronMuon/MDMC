"""Stores all filenames and additional data that is required for testing

Test data must have the same name as the reader. Origins for data are also
supplied.

AUTHOR :    Thomas Farmer        START DATE :    2018-6-7 14:10:35"""

from os import path

_ABS_DIR_PATH = path.split(path.abspath(__file__))[0]
_EXP_DATA_PATH = '/experimental_data'
_CALC_OBS_PATH = '/calculated_observables'

# TODO: introspective method to add abs_dir_path (use dir() to get all var names)

# LAMPSQw - From Bertil Halle QENS water data on in5

data = {'LAMPSQw':_ABS_DIR_PATH + _EXP_DATA_PATH + '/263K05Awat_LAMP'}
