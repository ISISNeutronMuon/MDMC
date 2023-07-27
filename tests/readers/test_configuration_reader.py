"""Tests relating to ConfigurationReader and subclasses, and ConfigurationReaderFactory"""

import pytest
from numpy.testing import assert_allclose

from MDMC.MD import Atom
from MDMC.readers.configurations import read
from MDMC.readers.configurations.ase import ASEReader
from MDMC.readers.configurations.conf_reader import ConfigurationReader
from MDMC.readers.configurations.conf_reader_factory import \
    ConfigurationReaderFactory
from tests.test_data import data


def test_configuration_reader_extension_error():
    """
    Tests that subclasses of ConfigurationReader require an extension to be
    defined
    """

    class DummyConfig(ConfigurationReader):

        def parse(self, **settings):

            self._atoms = None

        @property
        def atoms(self):

            return self._atoms

    with pytest.raises(TypeError):
        #pylint: disable=abstract-class-instantiated
        DummyConfig()


def test_create_reader_from_ext():
    """Tests that a reader can be created from a correctly specified file extension"""

    cif_ext = "cif"
    reader = ConfigurationReaderFactory.create_reader_from_ext(cif_ext, data.CONFIG_DATA['cif'])
    assert isinstance(reader, ASEReader)


def test_create_reader_from_ext_unimplemented():
    """
    Tests that if an unknown extension has been passed, a NotImplementedError is
    raised
    """

    unknown_ext = 'unknown'
    with pytest.raises(NotImplementedError):
        ConfigurationReaderFactory.create_reader_from_ext(unknown_ext, 'file_name.ext')
