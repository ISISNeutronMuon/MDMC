"""
Include test for correct number of unique bonds, bond angles and dihedrals
created from the CIF file. This should test for names, atom_types and neither
being defined.
"""


import pytest

from MDMC.MD.ase import cif


def test_get_bonded_interaction_atoms():

    pass


def test_create_coulombic_interactions():

    pass


def test_create_bonded_interactions():

    pass


def test_group_atoms():

    """
    Parameterize with different keys
    """

    pass


def test_reduce_ase_unit_cell():

    pass


def test_get_reduced_chemical_formula():

    """
    Parameterize with different formulae and number of atoms, and same formula
    but different number of atoms
    """

    pass
