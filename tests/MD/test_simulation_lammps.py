"""Tests for setting up and running MDMC using LAMMPS

AUTHOR :    Thomas Farmer        START DATE :    11/02/2019, 16:21:38"""


def test_universe_dims():

    """
    Tests that creating a simulation box from an MDMC universe results in the
    correct universe dimensions
    """

    pass


def test_universe_shape():

    """
    Tests that creating a simulation box from MDMC universe results in the
    correct universe shape
    """

    pass


def test_number_elements():

    """
    Tests that creating a simulation box from an MDMC universe results in the
    correct number of elements
    """

    pass


def test_number_interaction_types():

    """
    Tests that creating a simulation box from an MDMC universe results in the
    correct number of each interaction type:

    - bond
    - angle
    - dihedral
    - improper

    DIHEDRAL AND IMPROPER ARE NOT CURRENTLY IMPLEMENTED
    """

    pass


def test_number_interactions_per_atom():

    """
    Tests that creating a simulation box from an MDMC universe results in the
    correct allowed number of interactions per atom for each interaction type:

    - bond
    - angle
    - dihedral
    - improper

    DIHEDRAL AND IMPROPER ARE NOT CURRENTLY IMPLEMENTED
    """

    pass

def test_partion_interactions():

    """
    Tests that MDMC universe interactions are partioned based on type
    """

    pass


def test_unsupported_interactions():

    """
    Tests that if a universe passed to LAMMPSEngine._add_topology has any
    interactions which have not been implemented in LAMMPS, NotImplementedError
    is raised
    """

    pass


def test_create_interaction_style():

    """
    Tests that all interactions are created with a hybrid style, for:

    - bond
    - angle
    - dihedral
    - improper

    DIHEDRAL AND IMPROPER ARE NOT CURRENTLY IMPLEMENTED
    """

    pass
