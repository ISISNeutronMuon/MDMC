"""Tests for reading in data
Any reader can be added by including the required parameters in
READERS_TEST_INFO and test data in MDMC.tests.test_data, and by setting the test
data variable in MDMC.tests.test_data.data. Test data variable names must be the
same name as the correspoding reader."""

import inspect

import pytest
import numpy as np

from MDMC.readers.observables.netCDFPDF import netCDFPDF
from MDMC.readers.observables.obs_reader import PDFReader
from MDMC.readers.observables.obs_reader_factory import ObservableReaderFactory
from tests.test_data import data


"""
READER_TEST_INFO contains the following:

- Reader name
- Independent data type(s) as a list
- Dependent data type
"""

READERS_TEST_INFO = [('LAMPSQw', 'LAMPSQw', ['Q', 'E'], 'SQw'),
                     ('MantidSQw', 'MantidSQw_one_file', ['Q', 'E'], 'SQw'),
                     ('MantidSQw', 'MantidSQw_two_files', ['Q', 'E'], 'SQw'),
                     ('xml_SQw', 'xml_SQw', ['Q', 'E'], 'SQw'),
                     ('netCDFPDF', 'netcdf_PDF', ['r'], 'PDF'),
                     ('netCDFSQw', "SQw_incoh", ['Q', 'E'], 'SQw'),
                     ('netCDFSQw', "SQw_coh", ['Q', 'E'], 'SQw'),
                     ('LAMPPDF','LAMPPDF', ['r'], 'PDF')]


@pytest.fixture(params=READERS_TEST_INFO)
def reader_info(request):
    """
    Parameterized reader instantiation

    Returns:
    Dictionary of data contained in reader test info
    """

    return {'reader': request.param[0],
            'data_lookup': request.param[1],
            'indep_datatypes': request.param[2],
            'dep_datatype': request.param[3]}

@pytest.fixture()
def unparsed_reader(reader_info):
    try:
        reader_data = data.READER_DATA[reader_info['data_lookup']]
    except KeyError:
        reader_data = data.OBS_DATA[reader_info['data_lookup']]
    reader = ObservableReaderFactory.create_reader(reader_info['reader'], reader_data)
    return reader

@pytest.fixture()
def parsed_reader(reader_info):
    try:
        reader_data = data.READER_DATA[reader_info['data_lookup']]
    except KeyError:
        reader_data = data.OBS_DATA[reader_info['data_lookup']]

    reader = ObservableReaderFactory.create_reader(reader_info['reader'], reader_data)
    with reader:
        reader.parse()
    return reader

def test_parse_data_types(reader_info, parsed_reader):
    """
    Tests to make sure the correct data types have been parsed
    """
    indep_datatypes = reader_info['indep_datatypes']
    dep_datatype = reader_info['dep_datatype']

    for indep_datatype in indep_datatypes:
        assert indep_datatype in parsed_reader.independent_variables
    assert dep_datatype in parsed_reader.dependent_variables

def test_parse_compatible_dimensions(reader_info, parsed_reader):
    """
    Tests to make sure the dependent variables have dimensions compatible with independent variables
    """
    dep_datatype = reader_info['dep_datatype']

    for indep_var in parsed_reader.independent_variables.values():
        assert len(indep_var) in np.shape(
            parsed_reader.dependent_variables[dep_datatype])

def test_parse_all_errors_non_negative(parsed_reader):
    """
    Tests that all parsed error values are non-negative
    """
    assert np.all(list(parsed_reader.errors.values())[0][0] >= 0)

def test_parse_data_is_floats(parsed_reader):
    """
    Tests that all data from parsed files are floats
    """
    all_vars = (list(parsed_reader.independent_variables.values())
                + list(parsed_reader.dependent_variables.values())[0]
                + list(parsed_reader.errors.values())[0])

    for var in all_vars:
        assert float in inspect.getmro(var.dtype.type)

# PDF-specific tests

def test_parse_partial_pdfs(parsed_reader):
    """
    Tests that the partial PDFs are:
        - Existent
        - The correct data type
        - Each partial is the correct datatype
    """
    if issubclass(type(parsed_reader), PDFReader):
        assert parsed_reader.partial_pdfs is not None
        assert type(parsed_reader.partial_pdfs) == dict
        for partial in parsed_reader.partial_pdfs.values():
            assert type(partial) == np.ndarray

def test_parse_incorrect_partial_pdfs(unparsed_reader):
    if type(unparsed_reader) == netCDFPDF:
        # Modify partial name to incorrect name
        unparsed_reader.__enter__()
        intermediate = unparsed_reader.file.variables.pop("pdf-H-O")
        unparsed_reader.file.variables["pdf-None-H"] = intermediate
        unparsed_reader.parse()
        reader = unparsed_reader
        # Check that all other partials (apart from the one with an incorrect name) are present
        for partial in reader.partial_pdfs.items():
            if partial[0] == "pdf-None-H":
                assert partial[1] is None
            else:
                assert type(partial[1]) == np.ndarray
                assert len(partial[1]) == len(reader.r)