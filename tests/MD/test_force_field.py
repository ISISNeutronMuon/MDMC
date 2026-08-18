"""
Tests for force field parametrization
"""

import pytest

from MDMC.MD.force_fields.OPLSAA import add_opls_force_field
from MDMC.MD.force_fields.force_field_factory import ForceFieldFactory
from MDMC.MD.simulation import Universe
from MDMC.MD.structures import (Atom, Molecule)
from MDMC.MD.interactions import Bond, BondAngle, Dispersion, DihedralAngle


@pytest.fixture
def water_universe():
    """
    Returns
    -------
    Universe
        A Universe with a single water molecule
    """

    universe = Universe(10.0, verbose=False)
    H1 = Atom('H', charge=0., cutoff=10., name="64")
    H2 = H1.copy(position=(0., 1.63298, 0.))
    O = Atom('O', position=(0., 0.81649, 0.57736), charge=0., cutoff=10., name="63")
    water_mol = Molecule(position=(0, 0, 0),
                         velocity=(0, 0, 0),
                         atoms=[H1, H2, O],
                         interactions=[Bond((H1, O), (H2, O), constrained=True),
                                       BondAngle(H1, O, H2, constrained=True)],
                         name='water')
    universe.add_structure(water_mol)
    add_opls_force_field(water_universe, cutoff=10.0, ewald=1e-5)
    return universe


@pytest.mark.parametrize('atoms_info, parameters',
                         [([('F', 1), ('C', 2)],
                           [1.38, 1535.528]),
                          ([('C', 3), ('C', 3)],
                           [1.51, 1464.4]),
                          ([('S', 26), ('C', 90)],
                           [1.76, 1046.0]),
                          ([('C', 441), ('C', 602)],
                           [1.352, 2284.464])
                         ])
def test_ff_parametrize_bond(atoms_info, parameters):
    """
    Tests that FileForceField correctly parametrizes Bond interactions

    The atoms_info parametrization provides 2 tuples, one for each atom. Each
    tuple consists of an element and a name (which is just the OPLS atom type)

    The HarmonicPotential parameters shown in the parametrization are listed in
    the following order: equilibrium_state, potential_strength
    """

    atoms = [Atom(element, name=name) for element, name in atoms_info]
    _validate_interaction_parameters(_parametrize_interaction(Bond,
                                                              'OPLSAA',
                                                              *atoms),
                                     parameters)


@pytest.mark.parametrize('atoms_info, parameters',
                         [([('C', 15), ('C', 16), ('C', 16)],
                           [118., 292.88]),
                          ([('C', 2), ('C', 16), ('C', 2)],
                           [124., 292.88]),
                          ([('C', 90), ('N', 54), ('O', 702)],
                           [121., 292.88]),
                          ([('O', 5), ('S', 434), ('C', 279)],
                           [96.4, 313.8])
                         ])
def test_ff_parametrize_bond_angle(atoms_info, parameters):
    """
    Tests that FileForceField correctly parametrizes BondAngle interactions

    The atoms_info parametrization provides 3 tuples, one for each atom. Each
    tuple consists of an element and a name (which is just the OPLS atom type)

    The HarmonicPotential parameters shown in the parametrization are listed in
    the following order: equilibrium_state, potential_strength
    """

    atoms = [Atom(element, name=name) for element, name in atoms_info]
    _validate_interaction_parameters(_parametrize_interaction(BondAngle,
                                                              'OPLSAA',
                                                              *atoms),
                                     parameters)


@pytest.mark.parametrize('atoms_info, parameters',
                         [([('C', 294), ('C', 277), ('N', 207), ('C', 294)],
                           [9.6232, 25.47638, 0., 0., 180., 0., 1, 2, 3]),
                          ([('H', 287), ('C', 277), ('C', 277), ('H', 287)],
                           [0., 0., 1.2552, 0., 180., 0., 1, 2, 3]),
                          ([('O', 4), ('C', 3), ('C', 18), ('Cl', 45)],
                           [-2.7196, 0., 0., 0., 180., 0., 1, 2, 3]),
                          ([('O', 4), ('C', 3), ('O', 121), ('C', 100)],
                           [0., 21.43882, 0., 0., 180., 0., 1, 2, 3])
                         ])
def test_ff_parametrize_proper_dihedral(atoms_info, parameters):
    """
    Tests that FileForceField correctly parametrizes proper dihedral
    interactions

    The atoms_info parametrization provides 4 tuples, one for each atom. Each
    tuple consists of an element and a name (which is just the OPLS atom type)

    The Periodic parameters shown in the parametrization are listed in the
    following order: K1, K2, K3, d1, d2, d3, n1, n2, n3
    """

    atoms = [Atom(element, name=name) for element, name in atoms_info]
    _validate_interaction_parameters(_parametrize_interaction(DihedralAngle,
                                                              'OPLSAA',
                                                              *atoms),
                                     parameters)


@pytest.mark.parametrize('atoms_info, parameters',
                         [([('C', 3), ('C', 294), ('C', 277), ('O', 4)],
                           [87.864, 180., 2]),
                          ([('C', 3), ('H', 287), ('C', 277), ('O', 214)],
                           [87.864, 180., 2]),
                          ([('N', 54), ('C', 84), ('C', 3), ('Cl', 45)],
                           [20.92, 180., 2]),
                          ([('N', 54), ('H', 287), ('C', 294), ('C', 100)],
                           [20.92, 180., 2]),
                          ([('C', 86), ('C', 55), ('O', 53), ('O', 4)],
                           [125.52, 180., 2.]),
                          ([('C', 86), ('O', 4), ('C', 55), ('O', 53)],
                           [125.52, 180., 2.])
                         ])
def test_ff_parametrize_improper_dihedral(atoms_info, parameters):
    """
    Tests that FileForceField correctly parametrizes improper dihedral
    interactions

    The atoms_info parametrization provides 4 tuples, one for each atom. Each
    tuple consists of an element and a name (which is just the OPLS atom type)

    The Periodic parameters shown in the parametrization are listed in the
    following order: K1, d1, n1

    The 3rd and 4th parametrizations are effectively duplicates, as the dihedral
    only requires atom group 54 in the third place (the rest are wildcards).

    The 5th and 6th parametrizations test permuting the order of the atom
    groups.
    """

    atoms = [Atom(element, name=name) for element, name in atoms_info]
    _validate_interaction_parameters(_parametrize_interaction(DihedralAngle,
                                                              'OPLSAA',
                                                              *atoms,
                                                              improper=True),
                                     parameters)


@pytest.mark.parametrize('atoms_info1, atoms_info2, expected',
                         [([('F', 1), ('C', 2), ('C', 2), ('C', 2)],
                           [('F', 106), ('C', 13), ('C', 23), ('C', 31)],
                           [-8.368, 2.9288, 12.552, 0., 180., 0., 1, 2, 3]),
                          ([('F', 1), ('C', 3), ('C', 18), ('H', 89)],
                           [('F', 106), ('C', 55), ('C', 40), ('H', 85)],
                           [0., 0., 1.50624, 0., 180., 0., 1, 2, 3])
                         ])
def test_bonded_valid_atom_groups(atoms_info1, atoms_info2, expected):
    """
    Tests that a bonded interaction which has multiple atom tuples, (with
    the same atom groups) correctly parametrizes the interaction

    This is tested for proper dihedral interactions

    The Periodic parameters shown in the parametrization are listed in the
    following order: K1, K2, K3, d1, d2, d3, n1, n2, n3
    """

    atom_tuples = [tuple([Atom(element, name=name) for element, name
                          in atoms_info])
                   for atoms_info in [atoms_info1, atoms_info2]]
    _validate_interaction_parameters(_parametrize_interaction(DihedralAngle,
                                                              'OPLSAA',
                                                              *atom_tuples),
                                     expected)


@pytest.mark.parametrize('atoms_info1, atoms_info2',
                         [([('F', 1), ('C', 2), ('C', 2), ('C', 2)],
                           [('F', 1), ('C', 2), ('C', 2), ('C', 6)]),
                          ([('H', 287), ('C', 277), ('C', 3), ('O', 214)],
                           [('C', 294), ('C', 277), ('C', 3), ('O', 4)])
                         ])
def test_bonded_invalid_atom_groups(atoms_info1, atoms_info2):
    """
    Tests that a bonded interaction which has multiple atom tuples, (but with
    different atom groups) raises a ValueError

    This is tested for proper dihedral interactions
    """

    atom_tuples = [tuple([Atom(element, name=name) for element, name
                          in atoms_info])
                   for atoms_info in [atoms_info1, atoms_info2]]
    with pytest.raises(ValueError):
        _parametrize_interaction(DihedralAngle, 'OPLSAA', *atom_tuples)


def _validate_interaction_parameters(interaction, expected_parameters):
    """
    Asserts that all interaction_parameters are equal to the expected values

    Parameters
    ----------
    interaction : Interaction
        The interaction for which the parameters are validated
    expected_parameters : list
        A list of the expected parameter values, in the same order as that
        output by interaction.parameters

    Raises
    ------
    AssertionError
        If the interaction parameters are not equal to the expected_parameters
    """

    for actual, expected in zip(interaction.parameters.as_array, expected_parameters):
        assert actual.value == expected


def _parametrize_interaction(interaction_class, force_field_name, *atoms,
                             **settings):
    """
    Parametrizes an interaction using the specified force field

    Parameters
    ----------
    interaction_class : Interaction
        The class of an Interaction
    atoms : list
        A list of atoms
    force_field_name : str
        The force field with which the interaction will be parametrized
    **settings
        Any **settings to be passed when initializing the Interaction
    """

    interaction = interaction_class(*atoms, **settings)
    force_field = ForceFieldFactory.create(force_field_name)

    interaction_type = (interaction_class.__name__.lower() if interaction_class
                        not in [Bond, BondAngle, DihedralAngle]
                        else 'bonded')
    getattr(force_field, '_parametrize_' + interaction_type)(interaction)
    return interaction


@pytest.mark.parametrize('force_field_name, element, expected_number',
                         [('OPLSAA', 'Cl', 11),
                          ('OPLSAA', 'F', 13),
                          ('OPLSAA', 'Ca', 2),
                          ('OPLSAA', 'P', 6)])
def test_filter_element(force_field_name, element, expected_number):
    """
    Tests that filtering the atoms of a FileForceField by an element produces
    the expected number of rows in the returned DataFrame, and that all of these
    rows have the correct element type
    """

    force_field = ForceFieldFactory.create(force_field_name)
    atoms = force_field.filter_element(element)
    assert len(atoms) == expected_number
    assert all(atoms['element'] == element)


def test_specific_force_fields_names():
    """
    Tests that ForceFieldFactory.get_force_field_names includes certain force
    field names

    This is an imperfect test of this method, as it doesn't test exactly what is
    returned; however this avoids the need to update this test with each new
    force field added, and should be a sufficiently robust test.

    To increase robustness, the names of new force fields could be added to this
    test.
    """

    force_field_names = ForceFieldFactory.available_names()
    for name in ['SPC', 'SPCE', 'OPLSAA']:
        assert name in force_field_names

