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

READERS_TEST_INFO = [('LAMPRDF', ['Dist', 'rdf'], 'rdf')]

@pytest.fixture(params=READERS_TEST_INFO)
def reader_info(request):

    """
    Parameterized reader instantiation

    Returns:
    Dictionary of data contained in reader test info
    """

    return {'reader':ObservableReaderFactory.create_reader(request.param[0]),
            'indep_datatype':request.param[1],
            'dep_datatypes':request.param[2]}


def test_parse(reader_info):

    """
    Tests if the reader has correctly parsed the data

    - Correct data types have been parsed
    - Dependent variables have dimensions compatible with independent variables
    - All errors are non-negative
    - All data are floats
    """

    reader = reader_info['reader']
    indep_datatype = reader_info['indep_datatype']
    dep_datatypes = reader_info['dep_datatypes']

    reader.open(data.READER_DATA[reader.__class__.__name__])
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

