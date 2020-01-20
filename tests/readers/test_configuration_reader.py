"""Tests relating to ConfigurationReader and subclasses, and
ConfigurationReaderFactory
"""

import pytest

from MDMC.readers.configurations.cif import CIF
from MDMC.readers.configurations.conf_reader import ConfigurationReader
from MDMC.readers.configurations.conf_reader_factory import \
    ConfigurationReaderFactory


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

    """
    Tests that a reader can be created from a correctly specified file extension
    """

    cif_ext = CIF.extension
    reader = ConfigurationReaderFactory.create_reader_from_ext(cif_ext)
    assert isinstance(reader, CIF)


def test_create_reader_from_ext_unimplemented():

    """
    Tests that if an unknown extension has been passed, a NotImplementedError is
    raised
    """

    unknown_ext = 'unknown'
    with pytest.raises(NotImplementedError):
        ConfigurationReaderFactory.create_reader_from_ext(unknown_ext)
