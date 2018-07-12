"""Tests for reading in data

Any reader can be added by including the required parameters in
READERS_TEST_INFO and test data in MDMC.tests.test_data, and by setting the test
data variable in MDMC.tests.test_data.data. Test data variable names must be the
same name as the correspoding reader.

AUTHOR :    Thomas Farmer        START DATE :    09/07/2018, 17:07:03"""

import pytest
import numpy as np
import inspect

from MDMC.src.readers.reader_factory import ReaderFactory
from MDMC.tests.test_data import data

"""
READER_TEST_INFO contains the following:

- Reader name
- Independent data type(s) as a list
- Dependent data type
"""

READERS_TEST_INFO = [('LAMPSQw', ['Q', 'E'], 'SQw')]


@pytest.fixture(params=READERS_TEST_INFO)
def reader_info(request):

    """
    Parameterized reader instantiation

    Returns:
    Dictionary of data contained in reader test info
    """

    return {'reader':ReaderFactory.create_reader(request.param[0]),
            'indep_datatypes':request.param[1],
            'dep_datatype':request.param[2]}

def test_open(reader_info):

    """
    Tests if the reader has opened the file in read only mode
    """

    # TODO: Add test for read only

    reader = reader_info['reader']

    try:
        reader.open(data.data[reader.__class__.__name__])
    except KeyError:
        raise KeyError("The test data must have the same name as the reader")

def test_parse(reader_info):

    """
    Tests if the reader has correctly parsed the data

    - Correct data types have been parsed
    - Dependent variables have dimensions compatible with independent variables
    - All errors are non-negative
    - All data are floats
    """

    reader = reader_info['reader']
    indep_datatypes = reader_info['indep_datatypes']
    dep_datatype = reader_info['dep_datatype']

    reader.open(data.data[reader.__class__.__name__])
    reader.parse()
    for indep_datatype in indep_datatypes:
        assert indep_datatype in reader.independent_variables
    assert dep_datatype in reader.dependent_variables

    for indep_var in reader.independent_variables.values():
        assert len(indep_var) in np.shape(
            reader.dependent_variables[dep_datatype])

    assert np.all(reader.errors >= 0)

    all_vars = reader.independent_variables.values() + \
               reader.dependent_variables.values() + \
               reader.errors.values()
    for var in all_vars:
        assert float in inspect.getmro(var.dtype.type)
