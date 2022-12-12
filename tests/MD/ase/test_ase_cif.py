"""
Include test for correct number of unique bonds, bond angles and dihedrals
created from the CIF file. This should test for names, atom_types and neither
being defined.
"""

import numpy as np
import pytest

from MDMC.MD.ase import cif


def atom_names(prefix, number):

    return ['{0}{1}'.format(prefix, i) for i in range(1, number + 1)]


class MockAtom:
    """
    A mock of the MDMC Atom class
    """

    def __init__(self, atom_type=None, name=None, ID=None, position=None):

        self.atom_type = atom_type
        self.name = name
        self.ID = ID
        self.position = position


class MockAtoms:
    """
    A mock of the ase.atoms.Atoms class
    """

    def __init__(self):
        self.n_mols = 8
        H_atoms = atom_names('H', 9)
        O_atoms = atom_names('O', 2)
        C_atoms = atom_names('C', 8)
        N_atoms = atom_names('N', 1)
        atom_sites = np.arange(0., 20, 1.)
        self.info = {'_atom_site_label':H_atoms + O_atoms + C_atoms + N_atoms,
                     '_atom_site_fract_x':atom_sites,
                     '_atom_site_fract_y':atom_sites,
                     '_atom_site_fract_z':atom_sites,
                     '_geom_bond_atom_site_label_1':H_atoms[:-1],
                     '_geom_bond_atom_site_label_2':O_atoms * 4,
                     '_geom_angle_atom_site_label_1':C_atoms[:3],
                     '_geom_angle_atom_site_label_2':N_atoms * 3,
                     '_geom_angle_atom_site_label_3':H_atoms[:3],
                     '_geom_torsion_atom_site_label_1':H_atoms[:1],
                     '_geom_torsion_atom_site_label_2':O_atoms[:1],
                     '_geom_torsion_atom_site_label_3':C_atoms[:1],
                     '_geom_torsion_atom_site_label_4':N_atoms[:1]}
        self.cell = np.array([1.] * 3)

    def __len__(self):
        return self.n_mols * len(self.info['_atom_site_label'])

    def __iter__(self):
        """
        Must be an iterator but return is irrelevant due to monkeypatch
        """
        return self

    def __next__(self):
        """
        Must be an iterator but return is irrelevant due to monkeypatch
        """
        return None

    def get_chemical_symbols(self):
        return ['H'] * 72 + ['O'] * 16 + ['C'] * 64 + ['N'] * 8


@pytest.fixture(scope='module')
def ase_atoms():

    return MockAtoms()


@pytest.mark.parametrize('geom_def, expected',
                         [('_geom_bond_atom_site_label_',
                           np.array([['AtomH1', 'AtomO1'],
                                     ['AtomH2', 'AtomO2'],
                                     ['AtomH3', 'AtomO1'],
                                     ['AtomH4', 'AtomO2'],
                                     ['AtomH5', 'AtomO1'],
                                     ['AtomH6', 'AtomO2'],
                                     ['AtomH7', 'AtomO1'],
                                     ['AtomH8', 'AtomO2']])),
                          ('_geom_angle_atom_site_label_',
                           np.array([['AtomC1', 'AtomN1', 'AtomH1'],
                                     ['AtomC2', 'AtomN1', 'AtomH2'],
                                     ['AtomC3', 'AtomN1', 'AtomH3']])),
                          ('_geom_torsion_atom_site_label_',
                           np.array([['AtomH1', 'AtomO1', 'AtomC1', 'AtomN1']]))
                         ])
def test_get_bonded_interaction_atoms(ase_atoms, geom_def, expected):
    """
    Tests that get_bonded_interactions_atoms correctly gets the atoms associated
    with each interaction defined in the CIF file (and parsed into the
    ase.atoms.Atoms objects)
    """

    # Arbitrary 'atoms' created by prepending 'Atom' to the label - simply to
    # check that correct mapping occurs in get_bonded_interaction (i.e. values
    # do not actually need to be Atom objects)
    atoms_labels = {label:'Atom' + label for label
                    in ase_atoms.info['_atom_site_label']}
    atoms = cif.get_bonded_interactions_atoms(ase_atoms.info, geom_def,
                                              atoms_labels)
    assert np.all(np.shape(atoms) == np.shape(expected))
    assert np.all(atoms == expected)


@pytest.mark.parametrize('interactions_atoms_shape',
                         [(3, 10),
                          (5, 5),
                          (10, 1)])
def test_create_bonded_interactions_error(interactions_atoms_shape):
    """
    Tests that create_bonded_interactions raises a TypeError if
    interactions_atoms contains an incorrect number of atoms for each
    interaction
    """

    interactions_atoms = np.zeros(interactions_atoms_shape, dtype=object)
    for i in range(interactions_atoms_shape[0]):
        for j in range(interactions_atoms_shape[1]):
            interactions_atoms[i, j] = MockAtom()
    with pytest.raises(TypeError):
        cif._create_bonded_interactions(interactions_atoms)


@pytest.mark.parametrize('key, expected',
                         [('atom_type',
                           [(1,) * 5, (2,) * 6, (3,) * 4, (4,) * 2, (6,) * 3]),
                          ('name',
                           [('C',) * 8, ('H',) * 9, ('N',), ('O',) * 2]),
                          ('ID',
                           [(ID,) for ID in range(1, 21, 1)])])
def test_group_atoms(key, expected):
    """
    Parameterize with different keys
    """

    atom_types = [1] * 3 + [2] * 5 + [4] * 2 + [3] * 4 + [6] * 3 + [1] * 2 + [2]
    names = ['H'] * 5 + ['O'] * 2 + ['H'] * 4 + ['C'] * 7 + ['N'] + ['C']
    IDs = range(20, 0, -1)

    atoms = [MockAtom(atom_type, name, ID) for atom_type, name, ID
             in zip(atom_types, names, IDs)]

    grouped_atoms = cif._group_atoms(atoms, key=lambda atom: getattr(atom, key))
    for group, expected_group_keys in zip(grouped_atoms, expected):
        assert expected_group_keys == tuple(getattr(atom, key) for atom
                                            in group)


def test_reduce_ase_unit_cell(ase_atoms):
    """
    Tests that _reduce_ase_unit_cell returns an ASEAtoms object that is
    equivalent to the input except with a fraction of the number of atoms
    """

    reduced_atoms = cif._reduce_ase_unit_cell(ase_atoms)
    assert len(reduced_atoms) == 20
    assert reduced_atoms.info == ase_atoms.info
    assert np.all(reduced_atoms.positions
                  == np.array([ase_atoms.info['_atom_site_fract_x'],
                               ase_atoms.info['_atom_site_fract_y'],
                               ase_atoms.info['_atom_site_fract_z']]).T)


@pytest.mark.parametrize('settings',
                         [{'atom_types':list(range(1, 10))
                                        + list(range(9, 0, -1))},
                          {'names':['H1', 'H2', 'H1', 'O1', 'O1', 'C1']},
                          {'atom_types':[1, 2, 1, 3, 3, 4],
                           'names':['H1', 'H2', 'H1', 'O1', 'O1', 'C1']}])
def test_ase_read_cif_atom_init(monkeypatch, ase_atoms, settings):
    """
    Tests that atoms are created correctly depending on whether atom names and
    atom_types are specified

    Does not test add_charges and add_bonds behaviour as this is covered in
    tests of other functions
    """

    def mock_read_cif(file, **settings):

        return [[1]]

    def mock_reduce_ase_unit_cell(atms):

        return ase_atoms

    def mock_convert_from_ase_atom(atom, name, atom_type, **settings):

        return MockAtom(atom_type=atom_type, name=name)

    def mock_make_atom_positions_valid(atoms):

        return atoms

    monkeypatch.setattr(cif, 'read_cif', mock_read_cif)
    monkeypatch.setattr(cif, '_reduce_ase_unit_cell', mock_reduce_ase_unit_cell)
    monkeypatch.setattr(cif, 'convert_from_ase_atom',
                        mock_convert_from_ase_atom)
    monkeypatch.setattr(cif, '_make_atom_positions_valid',
                        mock_make_atom_positions_valid)

    settings['add_bonds'] = False
    settings['add_charges'] = False
    atoms = cif.ase_read_cif('', **settings)
    if 'atom_types' in settings:
        assert settings['atom_types'] == [atom.atom_type for atom in atoms]
    if 'names' in settings:
        assert settings['names'] == [atom.name for atom in atoms]


@pytest.mark.parametrize('positions, expected',
                         [(([0., 0., 0.], [1., 1., 1.], [2., 2., 2.]),
                           ([0., 0., 0.], [1., 1., 1.], [2., 2., 2.])),
                          (([-0.5, 2., 8.], [-0.4, -6., 10.], [4., 9., -2.]),
                           ([0., 8., 10.], [0.1, 0., 12.], [4.5, 15., 0.])),
                          (([-8., -9., -7.], [1., 1., 1.], [2., 2., 2.]),
                           ([0., 0., 0.], [9., 10., 8.], [10., 11., 9.])),
                          (([5., 4., 9.], [10., 1., 4.], [6., 5., 3.]),
                           ([0., 3., 6.], [5., 0., 1.], [1., 4., 0.])),
                          (([3., 4., 5.], [5., 4., 3.], [3., 5., 3.]),
                           ([0., 0., 2.], [2., 0., 0.], [0., 1., 0.])),
                          (([5., -2., 9.], ),
                           ([0., 0., 0.], ))])
def test_make_atom_positions_valid(positions, expected):
    """
    Tests that `_make_atom_positions_valid` results in the all positions of
    the atoms being >= 0., with at least one of all x, y and z axes equalling 0,
    and that relative distances are preserved.
    """

    atoms = [MockAtom(position=np.array(position)) for position in positions]
    cif._make_atom_positions_valid(atoms)
    for atom, expected_atom_position in zip(atoms, expected):
        assert np.allclose(atom.position, np.array(expected_atom_position))
