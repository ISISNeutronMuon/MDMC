"""Stores all filenames and additional data that is required for testing

Data for reader unit tests must have the same key as the reader name. Data for
observable system tests must have the same key as the observable name. Data for
MDMC objects must have the same key as the object name.  Descriptions of data
are also supplied."""

from pathlib import Path

_ABS_DIR_PATH = Path(__file__).parent.absolute()
_EXP_DATA_PATH = _ABS_DIR_PATH / 'experimental_data'
_CALC_OBS_PATH = _ABS_DIR_PATH / 'calculated_observables'
_OBJECT_PATH = _ABS_DIR_PATH / 'MDMC_objects'
_CONFIG_PATH = _ABS_DIR_PATH / 'configurations'
_GUI_PATH = _ABS_DIR_PATH / 'gui'

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

READER_DATA = {'LAMPSQw': _EXP_DATA_PATH / '263K05Awat_LAMP',
               'MantidSQw_two_files': _EXP_DATA_PATH / 'iris70429_graphite002_red',
               'MantidSQw_one_file': _EXP_DATA_PATH / 'IRIS_26176_water_data.dat',
               'MDANSESQw': _EXP_DATA_PATH / 'MDANSE_Ar_trajectory.dat',
               'xml_SQw': _EXP_DATA_PATH / 'Well_s_q_omega_Ar_data.xml',
               'xml_SQw_2':_EXP_DATA_PATH / 'Argon_test_data.xml',
               'LAMPPDF': _CALC_OBS_PATH / 'LAMP_from_nMOLDYN_PDF_water.ref'}

CONFIG_DATA = {'cif': _CONFIG_PATH / 'Paracetamol.cif',
               'pdb_ethanol': _CONFIG_PATH / 'water.pdb',
               'pdb_palmitic_acid': _CONFIG_PATH / 'example_pdb_export.pdb'}

RESOLUTION_DATA = {'LAMPSQw': _EXP_DATA_PATH / '262p7K0A5van_LAMP'}

# Parse back to str
READER_DATA = {key: str(val) for key, val in READER_DATA.items()}
CONFIG_DATA = {key: str(val) for key, val in CONFIG_DATA.items()}
RESOLUTION_DATA = {key: str(val) for key, val in RESOLUTION_DATA.items()}

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

OBS_DATA = {'SQw_incoh': _CALC_OBS_PATH / 'nMOLDYN_DISF_water.nc',
            'SQw_coh': _CALC_OBS_PATH / 'nMOLDYN_DCSF_water.nc',
            'Q_vectors': _CALC_OBS_PATH / 'qVectors.dat',
            'netcdf_PDF': _CALC_OBS_PATH / 'nMOLDYN_PDF_water.nc',
            'lamp_PDF': _CALC_OBS_PATH / 'LAMP_from_nMOLDYN_PDF_water.ref'}

OBS_DATA = {key: str(val) for key, val in OBS_DATA.items()}

# MDMC object data
#
# trajectory
# Calculated from same MMTK simulation on water as used to calculate DISF.
# Subsequently converted to CompactTrajectory and pickled again. Must
# be unzipped using zlib and then unpickled before use.

OBJECT_DATA = {'compact_trajectory': _OBJECT_PATH / 'compact_trajectory.zip'}

OBJECT_DATA = {key: str(val) for key, val in OBJECT_DATA.items()}

# viewer data
#
# html files (for X3DOM viewer)

GUI_DATA = {
    'atom_X3DOM': _GUI_PATH / 'atom.html',
    'water_molecule_X3DOM': _GUI_PATH / 'water_molecule.html',
    'universe_X3DOM': _GUI_PATH / 'universe.html'
}

GUI_DATA = {key: str(val) for key, val in GUI_DATA.items()}
