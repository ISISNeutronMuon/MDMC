"""A module for defining the AMBER99SB-ILDN force field. This was generated from the
corresponding TINKER file."""

from MDMC.MD.force_fields.ff import FileForceField


class AMBER(FileForceField):

    """
    AMBER99SB-ILDN force field, with defined atoms and interactions
    """

    file_name = 'amber.dat'
