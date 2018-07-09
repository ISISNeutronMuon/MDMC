"""Tests for reading in data

Any reader can added by including its class name and associated test data in the
RD_PAIRS dictionary. Test data must have the same name as the reader.

AUTHOR :    Thomas Farmer        START DATE :    09/07/2018, 17:07:03"""

import pytest

from MDMC.src.readers.reader_factory import ReaderFactory
from MDMC.tests.test_data import data


READERS = ['LAMPSQw']

# Parameterized reader instantiation so that all READERS are tested
@pytest.fixture(params=READERS)
def reader(request):
    reader = ReaderFactory.create_reader(request.param)
    return reader

def test_open(reader):

    """
    Tests if the reader has opened the file in read only mode
    """

    # TODO: Add test for read only

    try:
        reader.open(data.data[reader.__class__.__name__])
    except KeyError:
        raise KeyError("The test data must have the same name as the reader")

def test_parse(reader):

    """
    Tests if the reader has correctly parsed the data
    """
