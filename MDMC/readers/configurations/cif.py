"""Module for reading cif files
"""

from MDMC.common.decorators import set_docstring
from MDMC.MD.ase.cif import ase_read_cif
from MDMC.readers.configurations.conf_reader import ConfigurationReader


class CIF(ConfigurationReader):

    """
    A class for reading CIF configuration files
    """

    extension = 'cif'

    def __init__(self):

        super().__init__()
        self._atoms = None

    # Dynamically set docstring
    #pylint: disable=missing-docstring
    @set_docstring(ase_read_cif.__doc__)
    def parse(self, **settings):

        self._atoms = ase_read_cif(self.file, **settings)

    @property
    def atoms(self):

        return self._atoms
