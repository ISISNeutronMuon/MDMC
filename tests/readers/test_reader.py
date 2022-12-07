"""Tests for reading in data

Any reader can be added by including the required parameters in
READERS_TEST_INFO and test data in MDMC.tests.test_data, and by setting the test
data variable in MDMC.tests.test_data.data. Test data variable names must be the
same name as the correspoding reader."""

import inspect

import pytest
import numpy as np

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
                     ('xml_SQw','xml_SQw', ['Q', 'E'], 'SQw'),
                     ('netCDFPDF', 'netcdf_PDF', ['r'], 'PDF'),
                     ('LAMPPDF','LAMPPDF', ['r'], 'PDF')]


@pytest.fixture(params=READERS_TEST_INFO)
def reader_info(request):

    """
    Parameterized reader instantiation

    Returns:
    Dictionary of data contained in reader test info
    """

    return {'reader':request.param[0],
            'data_lookup':request.param[1],
            'indep_datatypes':request.param[2],
            'dep_datatype':request.param[3]}


def test_parse(reader_info):

    """
    Tests if the reader has correctly parsed the data

    - Correct data types have been parsed
    - Dependent variables have dimensions compatible with independent variables
    - All errors are non-negative
    - All data are floats
    """
    try:
        reader_data = data.READER_DATA[reader_info['data_lookup']]
    except KeyError:
        reader_data = data.OBS_DATA[reader_info['data_lookup']]

    reader = ObservableReaderFactory.create_reader(reader_info['reader'], reader_data)

    indep_datatypes = reader_info['indep_datatypes']
    dep_datatype = reader_info['dep_datatype']

    with reader:
        reader.parse()
    for indep_datatype in indep_datatypes:
        assert indep_datatype in reader.independent_variables
    assert dep_datatype in reader.dependent_variables

    for indep_var in reader.independent_variables.values():
        assert len(indep_var) in np.shape(
            reader.dependent_variables[dep_datatype])

    assert np.all(list(reader.errors.values())[0][0] >= 0)

    all_vars = (list(reader.independent_variables.values())
                + list(reader.dependent_variables.values())[0]
                + list(reader.errors.values())[0])
    for var in all_vars:
        assert float in inspect.getmro(var.dtype.type)
