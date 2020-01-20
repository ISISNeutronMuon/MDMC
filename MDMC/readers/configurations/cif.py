"""Module for reading cif files
"""

from MDMC.readers.configurations.conf_reader import ConfigurationReader


class CIF(ConfigurationReader):

    """
    A class for reading CIF configuration files
    """

    extension = 'cif'

    def parse(self, **settings):

        self._atoms = None

    @property
    def atoms(self):

        return self._atoms
