"""Tests for reading in data
Any reader can be added by including the required parameters in
READERS_TEST_INFO and test data in MDMC.tests.test_data, and by setting the test
data variable in MDMC.tests.test_data.data. Test data variable names must be the
same name as the correspoding reader."""

import inspect
import tempfile

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
                     ('MDANSESQw', "MDANSESQw", ['Q', 'E'], 'SQw'),
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

mdanse_data_ascii_output = """# variable name: s(q,f)_total
# 	type: surface
# 	axis: q|omega
# 	units: nm2/ps

# 1st column: q (1/nm)
# 1st row: omega (rad/ps)

0.000000000000000000e+00 -1.709779858788470719e+01 -1.282334894091352950e+01 -8.548899293942353594e+00 -4.274449646971176797e+00 0.000000000000000000e+00 4.274449646971176797e+00 8.548899293942353594e+00 1.282334894091352950e+01 1.709779858788470719e+01
1.219999999999999929e+01 3.950986190596459703e-04 1.392843467136946367e-04 5.612785519136486692e-05 1.716099353001354123e-05 8.161975326149337705e-04 1.716099353001355140e-05 5.612785519136484659e-05 1.392843467136946096e-04 3.950986190596460245e-04
2.419999999999999574e+01 1.267940220582672957e-03 2.017535836327090035e-03 1.967565193000387052e-03 2.662871947470733195e-03 7.601364029083551577e-01 2.662871947470733195e-03 1.967565193000387052e-03 2.017535836327089167e-03 1.267940220582672089e-03
3.619999999999999574e+01 1.746544434807437065e-03 4.828235294063881050e-03 7.125406412215911506e-03 9.161006681950002728e-03 1.552414642625137731e+00 9.161006681950000993e-03 7.125406412215912373e-03 4.828235294063879315e-03 1.746544434807435981e-03
"""

mdanse_data_plotter_output = """# Data         : s(q,f)_total
# First row    : q (1/ang)
# First column : omega (meV)
  0.0000e+00    1.2200e+00    2.4200e+00    3.6200e+00
 -1.1254e+01    3.9510e-04    1.2679e-03    1.7465e-03
 -8.4405e+00    1.3928e-04    2.0175e-03    4.8282e-03
 -5.6270e+00    5.6128e-05    1.9676e-03    7.1254e-03
 -2.8135e+00    1.7161e-05    2.6629e-03    9.1610e-03
  0.0000e+00    8.1620e-04    7.6014e-01    1.5524e+00
  2.8135e+00    1.7161e-05    2.6629e-03    9.1610e-03
  5.6270e+00    5.6128e-05    1.9676e-03    7.1254e-03
  8.4405e+00    1.3928e-04    2.0175e-03    4.8282e-03
  1.1254e+01    3.9510e-04    1.2679e-03    1.7465e-03
"""

@pytest.fixture(scope = 'module')
def mdanse_textfile():
    """Writes the mdanse_data_ascii_output to a temporary file"""
    target = tempfile.NamedTemporaryFile('w')
    target.write(mdanse_data_ascii_output)
    target.flush()
    yield target
    target.close()

@pytest.fixture(scope = 'module')
def mdanse_plotterfile():
    """Writes the mdanse_data_plotter_output to a temporary file"""
    target = tempfile.NamedTemporaryFile('w')
    target.write(mdanse_data_plotter_output)
    target.flush()
    yield target
    target.close()

def read_and_parse(file_obj: tempfile._TemporaryFileWrapper):
    reader = ObservableReaderFactory.create_reader('MDANSESQw', file_obj.name)
    with reader:
        reader.parse()
    return reader

@pytest.fixture()
def parsed_textfile(mdanse_textfile):
    parsed_reader = read_and_parse(mdanse_textfile)
    return parsed_reader

@pytest.fixture()
def parsed_plotterfile(mdanse_plotterfile):
    parsed_reader = read_and_parse(mdanse_plotterfile)
    return parsed_reader

def test_axes(parsed_textfile, parsed_plotterfile):
    q_axis_1 = parsed_textfile.independent_variables['Q']
    q_axis_2 = parsed_plotterfile.independent_variables['Q']
    e_axis_1 = parsed_textfile.independent_variables['E']
    e_axis_2 = parsed_plotterfile.independent_variables['E']
    assert np.allclose(q_axis_1, q_axis_2, rtol=1e-4, atol = 1e-3)
    assert np.allclose(e_axis_1, e_axis_2, rtol=1e-4, atol = 1e-3)
     
def test_data(parsed_textfile, parsed_plotterfile):
    data_1 = parsed_textfile.SQw
    data_2 = parsed_plotterfile.SQw
    assert len(data_1.shape) == len(data_2.shape)
    assert np.all(np.array(data_1.shape) == np.array(data_2.shape))
    assert np.allclose(data_1, data_2, rtol=1e-4, atol=1e-3)
