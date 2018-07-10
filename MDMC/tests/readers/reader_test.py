"""Tests for reading in data

Any reader can be added by including the required parameters in
READERS_TEST_INFO and test data in MDMC.tests.test_data, and by setting the test
data variable in MDMC.tests.test_data.data. Test data variable names must be the
same name as the correspoding reader.

AUTHOR :    Thomas Farmer        START DATE :    09/07/2018, 17:07:03"""

import pytest

from MDMC.src.readers.reader_factory import ReaderFactory
from MDMC.tests.test_data import data

"""
READER_TEST_INFO contains the following:

- Reader name
- Dependent data type
"""

READERS_TEST_INFO = [('LAMPSQw', 'SQw')]


@pytest.fixture(params=READERS_TEST_INFO)
def reader_info(request):

    """
    Parameterized reader instantiation

    Returns:
    Dictionary of data contained in reader test info
    """

    return {'reader':ReaderFactory.create_reader(request.param[0]),
            'dep_datatype':request.param[1]}


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
    """

    reader = reader_info['reader']

    reader.open(data.data[reader.__class__.__name__])
    reader.parse()
    assert reader_info['dep_datatype'] in reader.data['dependent']
