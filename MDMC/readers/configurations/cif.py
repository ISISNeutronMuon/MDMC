"""Module for reading cif files
"""

from MDMC.common.decorators import set_docstring
from MDMC.MD.ase.cif import ase_read_cif
from MDMC.readers.configurations.conf_reader import ConfigurationReader


class CIF(ConfigurationReader):

    """
    A class for reading CIF configuration files

    Examples
    --------
    To use a CIF reader to read a file called 'paracetamol.cif' and create a
    ``Molecule`` from it (assuming ``Molecule`` has been imported from
    ``MDMC.MD``):

    .. highlight:: python
    .. code-block:: python

        file = 'paracetamol.cif'
        cif = CIF()
        cif.open(file)
        # See parse docstring for description of ``names`` parameter
        # Lines are oxygen, nitrogen, carbon and hydrogen atoms
        cif.parse(names=['109', '177', # Oxygens
                         '207', # Nitrogen
                         '208', '108', '90', '178', '90', '90', '90', '185',
                         '85', '85', '85', '91', '91', '91', '91', '183', '110']
                 ) # Hydrogens)
        paracetamol = Molecule(atoms=cif.atoms)
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
