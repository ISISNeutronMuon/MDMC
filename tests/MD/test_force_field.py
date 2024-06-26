"""
Tests for force field parametrization
"""

import pytest

from MDMC.MD.force_fields.force_field_factory import ForceFieldFactory
from MDMC.MD.simulation import Universe
from MDMC.MD.structures import (Atom, Molecule)
from MDMC.MD.interactions import Bond, BondAngle, Dispersion, Coulombic, DihedralAngle


@pytest.fixture
def water_universe():
    """
    Returns
    -------
    Universe
        A Universe with a single water molecule
    """

    universe = Universe(10.0, verbose=False)
    H1 = Atom('H', charge=0., cutoff=10.)
    H2 = H1.copy(position=(0., 1.63298, 0.))
    O = Atom('O', position=(0., 0.81649, 0.57736), charge=0., cutoff=10.)
    water_mol = Molecule(position=(0, 0, 0),
                         velocity=(0, 0, 0),
                         atoms=[H1, H2, O],
                         interactions=[Bond((H1, O), (H2, O), constrained=True),
                                       BondAngle(H1, O, H2, constrained=True)],
                         name='water')
    universe.add_structure(water_mol)
    O_dispersion = Dispersion(universe, (O.atom_type, O.atom_type), cutoff=10.,
                              vdw_tail_correction=True)
    H_dispersion = Dispersion(universe, (H1.atom_type, H1.atom_type),
                              cutoff=10., vdw_tail_correction=True)
    return universe


@pytest.mark.parametrize('model, O_charge, H_charge',
                         [('TIP3P', -0.8340, 0.4170),
                          ('TIP4P', 0.0000, 0.5200),
                          ('TIP3F', -0.8220, 0.4110),
                          ('TIP4F', 0.0000, 0.5110),
                          ('TIP5P', 0.0000, 0.2410),
                          ('SPC', -0.8200, 0.4100)])
def test_opls_water_model_charges(water_universe, model, O_charge, H_charge):
    """
    Tests that water models using OPLS force field have correct charge
    parametrization for the H and O atoms. It does not test the charge
    assignment of virtual atoms, as these have not been implemented.
    """

    for atom in water_universe.atoms:
        name = model + ' Water '
        atom.name = name + 'H' if atom.element.symbol == 'H' else name + 'O'
        # Check that initial charges are 0.
        assert atom.charge == 0.
    water_universe.add_force_field('OPLSAA')

    for atom in water_universe.atoms:
        if atom.element.symbol == 'H':
            assert atom.charge == H_charge
        else:
            assert atom.charge == O_charge


@pytest.mark.parametrize('model', ['TIP3P', 'TIP4P', 'TIP3F', 'TIP4F', 'TIP5P',
                                   'SPC'])
def test_opls_water_model_masses(water_universe, model):
    """
    Tests that water models using OPLS force field have correct mass
    parametrization for the H and O atoms. It does not test the mass
    assignment of virtual atoms, as these have not been implemented.

    All water models have the same H and O mass.
    """

    for atom in water_universe.atoms:
        name = model + ' Water '
        atom.name = name + 'H' if atom.element.symbol == 'H' else name + 'O'
        # Check that initial masses are not the same as model masses
        assert atom.mass not in [1.008, 15.999]
    water_universe.add_force_field('OPLSAA')

    for atom in water_universe.atoms:
        if atom.element.symbol == 'H':
            assert atom.mass == 1.008
        else:
            assert atom.mass == 15.999

@pytest.mark.parametrize('model, sigma, epsilon',
                         [('TIP3P', 3.15061, 0.63639),
                          ('TIP4P', 3.15365, 0.64852),
                          ('TIP3F', 3.17600, 0.62760),
                          ('TIP4F', 3.27000, 0.41840),
                          ('TIP5P', 3.12000, 0.66944),
                          ('SPC', 3.16557, 0.65019)])
def test_opls_water_model_lj_parameters(water_universe, model, sigma, epsilon):
    """
    Tests that water models using OPLS force field have correct LJ
    parametrization for the H and O atoms. It does not test the LJ
    assignment of virtual atoms, as these have not been implemented.

    All models should have 0. for both H parameters, and so are not
    parametrized.
    """

    for atom in water_universe.atoms:
        name = model + ' Water '
        atom.name = name + 'H' if atom.element.symbol == 'H' else name + 'O'
    water_universe.add_force_field('OPLSAA')

    for interaction in water_universe.nonbonded_interactions:
        if isinstance(interaction, Dispersion):
            if 'O' in interaction.element_list():
                assert interaction.function.sigma.value == sigma
                assert interaction.function.epsilon.value == epsilon
            else:
                assert interaction.function.sigma.value == 0.
                assert interaction.function.epsilon.value == 0.


@pytest.mark.parametrize('model, eq_state, pot_strength',
                         [('TIP3P', 0.9572, 2510.4),
                          ('TIP4P', 0.9572, 2510.4),
                          ('TIP3F', 0.9572, 2215.84640),
                          ('TIP4F', 0.9572, 2510.4),
                          ('TIP5P', 0.9572, 2510.4),
                          ('SPC', 1.0000, 2510.4)])
def test_opls_water_model_bond_parameters(water_universe, model, eq_state,
                                          pot_strength):
    """
    Tests that water models using OPLS force field have correct HO bond
    parametrization.
    """

    for atom in water_universe.atoms:
        name = model + ' Water '
        atom.name = name + 'H' if atom.element.symbol == 'H' else name + 'O'
    water_universe.add_force_field('OPLSAA')

    for interaction in water_universe.nonbonded_interactions:
        if isinstance(interaction, Bond):
            assert interaction.function.sigma.equilibrium_state == eq_state
            assert (interaction.function.epsilon.potential_strength
                    == pot_strength)


@pytest.mark.parametrize('model, eq_state, pot_strength',
                         [('TIP3P', 104.52, 313.8),
                          ('TIP4P', 104.52, 313.8),
                          ('TIP3F', 104.52, 142.46520),
                          ('TIP4F', 109.50, 313.8),
                          ('TIP5P', 104.52, 313.8),
                          ('SPC', 109.47, 313.8)])
def test_opls_water_model_bond_angle_parameters(water_universe, model,
                                                eq_state, pot_strength):
    """
    Tests that water models using OPLS force field have correct HOH bond angles
    parametrization.
    """

    for atom in water_universe.atoms:
        name = model + ' Water '
        atom.name = name + 'H' if atom.element.symbol == 'H' else name + 'O'
    water_universe.add_force_field('OPLSAA')

    for interaction in water_universe.nonbonded_interactions:
        if isinstance(interaction, Bond):
            assert interaction.function.sigma.equilibrium_state == eq_state
            assert (interaction.function.epsilon.potential_strength
                    == pot_strength)


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


@pytest.mark.parametrize('atoms_info, expected',
                         [([('C', 8), ('C', 9)], 0.),
                          ([('S', 24), ('S', 26)], -0.47),
                          ([('C', 22), ('C', 23), ('C', 39), ('C', 40)], 0.265)
                         ])
def test_coulombic_valid_charges(atoms_info, expected):
    """
    Tests that a coulombic interaction which has atoms of different types,
    (with the same charges for the atoms) correctly parametrizes the interaction
    """

    atoms = [Atom(element, name=name) for element, name in atoms_info]
    _validate_interaction_parameters(_parametrize_interaction(Coulombic,
                                                              'OPLSAA',
                                                              atoms=atoms),
                                     [expected])


@pytest.mark.parametrize('atoms_info',
                         [([('C', 8), ('C', 22)]),
                          ([('O', 5), ('C', 6)]),
                          ([('C', 10), ('C', 131), ('C', 22), ('C', 31)])
                         ])
def test_coulombic_invalid_charges(atoms_info):
    """
    Tests that a coulombic interaction which has atoms of different types,
    (with different charges for the atoms) raises a ValueError
    """

    atoms = [Atom(element, name=name) for element, name in atoms_info]
    with pytest.raises(ValueError):
        _parametrize_interaction(Coulombic, 'OPLSAA', atoms=atoms)


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
    force_field = ForceFieldFactory.create_force_field(force_field_name)

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

    force_field = ForceFieldFactory.create_force_field(force_field_name)
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

    force_field_names = ForceFieldFactory.get_force_field_names()
    for name in ['SPC', 'SPCE', 'OPLSAA']:
        assert name in force_field_names


def test_name_element_error():
    """
    Test that atoms with mismatched names and elements raise an error
    """

    uni = Universe(10., verbose=False)
    # name=1 corresponds to a F atom in OPLSAA
    H1 = Atom('H', name=1)
    H2 = Atom('H', name=1)
    uni.add_structure(H1)
    uni.add_structure(H2)
    Bond((H1, H2))
    with pytest.raises(KeyError):
        uni.add_force_field('OPLSAA')


def test_undefined_bond_error():
    """
    Test that atoms without a defined bond raise an error
    """

    uni = Universe(10., verbose=False)
    # There is no OPLSAA bond between two "7" atoms
    H1 = Atom('H', name=7)
    H2 = Atom('H', name=7)
    uni.add_structure(H1)
    uni.add_structure(H2)
    Bond((H1, H2))
    with pytest.raises(ValueError):
        uni.add_force_field('OPLSAA')


def test_coulombic_error():
    """
    Test that a coulombic interaction applied to an ``atom_type`` that is
    missing from the universe raises an error
    """

    uni = Universe(10., verbose=False)
    H1 = Atom('H', name=7)
    H2 = Atom('H', name=7)
    uni.add_structure(H1)
    uni.add_structure(H2)
    # We only have atom_type of 1
    Coulombic(uni, atom_types=[2])
    with pytest.raises(ValueError):
        uni.add_force_field('OPLSAA')


def test_dispersion_error():
    """
    Test that a dispersion interaction applied to an ``atom_type`` that is
    missing from the universe raises an error
    """

    uni = Universe(10., verbose=False)
    H1 = Atom('H', name=7)
    H2 = Atom('H', name=7)
    uni.add_structure(H1)
    uni.add_structure(H2)
    # We only have atom_type of 1
    Dispersion(uni, atom_types=[2])
    with pytest.raises(ValueError):
        uni.add_force_field('OPLSAA')
