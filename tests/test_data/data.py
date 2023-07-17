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
_CONFIG_PATH = '/configurations'
_GUI_PATH = '/gui'

# Reader/experimental data
#
# LAMPSQw
# From Bertil Halle. QENS water data collected on in5.
# J. Chem. Phys. 134, 144508 (2011); https://doi.org/10.1063/1.3578472
#
# MantidSQw
# TODO cite data once published, the Mantid data file iris70429_graphite002_red was provided by Jeff Armstrong
# IRIS_26176_water_data was provided by Spencer Howells by (H2O at 280K on the IRIS spectrometer)
#
# xml_SQw
# Argon data from van Well et al. (1985). Physical Review A, 31(5), 3391-3414.
#
# LAMPPDF
# Pair distribution function (PDF) imported from the netcdf PDF that was calculated with nMOLDYN
# (OBS_DATA['netcdf_PDF'], using the same trajectory as all OBS_DATA). The data was
# reformatted into LAMP-style format for PDF data.

READER_DATA = {'LAMPSQw':'/experimental_data/263K05Awat_LAMP',
               'MantidSQw_two_files':'/experimental_data/iris70429_graphite002_red',
               'MantidSQw_one_file':'/experimental_data/IRIS_26176_water_data.dat',
               'MDANSESQw':'/experimental_data/MDANSE_Ar_trajectory.dat',
               'xml_SQw':'/experimental_data/Well_s_q_omega_Ar_data.xml',
               'LAMPPDF':'/calculated_observables/LAMP_from_nMOLDYN_PDF_water.ref'}

CONFIG_DATA = {'cif': '/Paracetamol.cif',
               'pdb_ethanol': '/water.pdb',
               'pdb_palmitic_acid': '/example_pdb_export.pdb'}

RESOLUTION_DATA = {'LAMPSQw':'/262p7K0A5van_LAMP'}

# Add paths to data values
for key in READER_DATA:
    READER_DATA[key] = _ABS_DIR_PATH + READER_DATA[key]

for key in CONFIG_DATA:
    CONFIG_DATA[key] = _ABS_DIR_PATH + _CONFIG_PATH + CONFIG_DATA[key]

for key in RESOLUTION_DATA:
    RESOLUTION_DATA[key] = _ABS_DIR_PATH + _EXP_DATA_PATH + RESOLUTION_DATA[key]

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
# netcdf ('netcdf_PDF'). In addition, the data was reformatted file in the style of LAMP output
# and saved as another file ('lamp_pdf').

OBS_DATA = {'SQw_incoh':'/nMOLDYN_DISF_water.nc',
            'SQw_coh':'/nMOLDYN_DCSF_water.nc',
            'Q_vectors':'/qVectors.dat',
            'netcdf_PDF':'/nMOLDYN_PDF_water.nc',
            'lamp_PDF':'/LAMP_from_nMOLDYN_PDF_water.ref'}

# Add paths to data values
for key in OBS_DATA:
    OBS_DATA[key] = _ABS_DIR_PATH + _CALC_OBS_PATH + OBS_DATA[key]


# MDMC object data
#
# trajectory
# Calculated from same MMTK simulation on water as used to calculate DISF.
# Subsequently converted to CompactTrajectory and pickled again. Must
# be unzipped using zlib and then unpickled before use.

OBJECT_DATA = {'compact_trajectory':'/compact_trajectory.zip'}

# Add paths to data values
for key in OBJECT_DATA:
    OBJECT_DATA[key] = _ABS_DIR_PATH + _OBJECT_PATH + OBJECT_DATA[key]

# viewer data
#
# html files (for X3DOM viewer)

GUI_DATA = {
    'atoms_X3DOM': '/atoms.html',
    'water_molecule_X3DOM': '/water_molecule.html',
    'universe_X3DOM': '/universe.html'
}

for key in GUI_DATA:
    GUI_DATA[key] = _ABS_DIR_PATH + _GUI_PATH + GUI_DATA[key]
