"""A module for defining the OPLSAA force field. This was generated from the
corresponding TINKER file."""

from MDMC.MD.force_fields.ff import FileForceField


class OPLSAA(FileForceField):

    """
    OPLSAA force field, with defined atoms and interactions
    """

    file_name = 'oplsaa.dat'
