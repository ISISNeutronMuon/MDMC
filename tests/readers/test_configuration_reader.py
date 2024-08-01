"""Tests relating to ConfigurationReader and subclasses, and ConfigurationReaderFactory"""

import pytest

from MDMC.readers.configurations.cif import CIFReader
from MDMC.readers.configurations.pdb import ProteinDataBankReader
from MDMC.readers.configurations.ase import ASEReader
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


@pytest.mark.parametrize('ext, reader_type', [('cif', CIFReader),
                                         ('pdb', ProteinDataBankReader),
                                         ('mol', ASEReader)])
def test_create_reader_from_ext(ext, reader_type):
    """Tests that a reader can be created from a correctly specified file extension"""

    reader = ConfigurationReaderFactory.create_reader_from_ext(ext, f"mock.{ext}")
    assert isinstance(reader, reader_type)


def test_create_reader_from_ext_unimplemented():
    """
    Tests that if an unknown extension has been passed, a NotImplementedError is
    raised
    """

    unknown_ext = 'unknown'
    with pytest.raises(NotImplementedError):
        ConfigurationReaderFactory.create_reader_from_ext(unknown_ext, 'file_name.ext')
