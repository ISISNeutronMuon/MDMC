"""A module for defining the CHARMM36 force field. This was generated from the
corresponding TINKER file."""

from MDMC.MD.force_fields.ff import FileForceField


class CHARMM(FileForceField):

    """
    CHARMM36 force field, with defined atoms and interactions
    """

    file_name = 'charmm.dat'
