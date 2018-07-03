"""Stores all filenames and additional data that is required for testing

Origins for data are also supplied

AUTHOR :    Thomas Farmer        START DATE :    2018-6-7 14:10:35"""

from os import path

_ABS_DIR_PATH = path.split(path.abspath(__file__))[0]

# TODO: introspective method to add abs_dir_path (use dir() to get all var names)
# From Bertil Halle QENS water data on in5
LAMP_SQW_FILE = _ABS_DIR_PATH + '/263K05Awat_LAMP'
