"""Stores all filenames and additional data that is required for testing

Data for reader unit tests must have the same key as the reader name. Data for
observable system tests must have the same key as the observable name. Data for
MDMC objects must have the same key as the object name.  Descriptions of data
are also supplied."""

from os import path

_ABS_DIR_PATH = path.split(path.abspath(__file__))[0]
_EXP_DATA_PATH = '/experimental_data'
_CALC_OBS_PATH = '/calculated_observables'
_OBJECT_PATH = '/MDMC_objects'


# Reader/experimental data
#
# LAMPSQw
# From Bertil Halle QENS water data on in5
#
# XML_SQw
# Argon data from van Well et al. (1985). Physical Review A, 31(5), 3391-3414.

READER_DATA = {'LAMPSQw':'/263K05Awat_LAMP',
               'xml_SQw':'/Well_s_q_omega_Ar_data.xml'}

# Add paths to data values
for key in READER_DATA:
    READER_DATA[key] = _ABS_DIR_PATH + _EXP_DATA_PATH + READER_DATA[key]


# Calculated observable data
#
# Dynamic incoherent structure factor (DISF i.e. incoherent SQw)
# MD simulation on 2048 water molecules for 50000 timesteps of 1 fs length. DISF
# calculated from time start:end:step of 51:5001:100, q start:end:step of
# 1.6:22.4:1.6, with a qshell width of 0.1, maxmimum 50 q vectors per shell
# and a resolution of 0.05. File format is netcdf.
#
# Dynamic coherent structure factor (DCSF i.e. coherent SQw)
# Same MD simulation and nMOLDYN parameters as DISF.
#
# Q_vectors - pickled
# A list of lists of arrays. Each array is 3 dimensions and contains a single
# Q vector. Each list contains a collection of Q vectors that have the same Q
# value. These Q vectors were used in calculating the SQw_incoh and SQw_coh data
# from nMOLDYN.
#
# Pair distribution function (PDF)
# Same simulation/trajectory as DISF. PDF calculated from time start:end:step of
# 51:5001:1000, with rvalues start:end:step of 0.:1.05:0.01. File format is
# netcdf.

OBS_DATA = {'SQw_incoh':'/nMOLDYN_DISF_water.nc',
            'SQw_coh':'/nMOLDYN_DCSF_water.nc',
            'Q_vectors':'/qVectors.dat',
            'PDF':'/nMOLDYN_PDF_water.nc'}

# Add paths to data values
for key in OBS_DATA:
    OBS_DATA[key] = _ABS_DIR_PATH + _CALC_OBS_PATH + OBS_DATA[key]


# MDMC object data
#
# trajectory
# Calculated from same MMTK simulation on water as used to calculate DISF. Must
# be unzipped using zlib and then unpickled before use.

OBJECT_DATA = {'trajectory':'/trajectory.zip'}

# Add paths to data values
for key in OBJECT_DATA:
    OBJECT_DATA[key] = _ABS_DIR_PATH + _OBJECT_PATH + OBJECT_DATA[key]
