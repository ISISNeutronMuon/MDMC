"""Tests conversions between equivalent MDMC and ASE objects
"""

from io import StringIO

import ase
import numpy as np
import pytest

from MDMC.MD.ase import conversions
from MDMC.MD.structures import Atom


FORMULA = 'C8H4O2'
CELL = (1., 2., 3.)
BONDS = [(1, 2), (3, 8), (4, 5)]
IDS = list(range(0, 14, 1))

X3DOMHEADER = ('<html>\n\n'
               ' <head>\n\n'
               '  <title>MDMC atomic visualization</title>\n\n'
               '  <link rel="stylesheet" type="text/css"\n\n'
               '   href="https://www.x3dom.org/x3dom/release/x3dom.css">\n\n'
               '  </link>\n\n'
               '  <script type="text/javascript"\n\n'
               '   src="https://www.x3dom.org/x3dom/release/x3dom.js">\n\n'
               '  </script>\n\n'
               ' </head>\n\n'
               ' <body>\n\n'
               '  <X3D width="800px" height="800px">\n\n'
               '   <Scene>\n\n'
               '   <Viewpoint centerOfRotation="0.50 0.50 0.50" '
               'position="0.50 0.50 3.29"></Viewpoint>\n\n'
               '   </Scene>\n\n'
               '  </X3D>\n\n'
               ' </body>\n\n'
               '</html>\n\n')

X3DHEADER = ('<?xml version="1.0" encoding="UTF-8"?>\n\n'
             '<!DOCTYPE X3D PUBLIC "ISO//Web3D//DTD X3D 3.2//EN" '
             '"http://www.web3d.org/specifications/x3d-3.2.dtd">\n\n'
             '<X3D profile="Interchange" version="3.2" '
             'xmlns:xsd="http://www.w3.org/2001/XMLSchema-instance" '
             'xsd:noNamespaceSchemaLocation='
             '"http://www.web3d.org/specifications/x3d-3.2.xsd">\n\n'
             ' <Scene>\n\n'
             ' <Viewpoint centerOfRotation="0.50 0.50 0.50" '
             'position="0.50 0.50 3.29"></Viewpoint>\n\n'
             '</X3D>\n\n')

class MockAtom:

    def __init__(self, ID):
        self.ID = ID


class MockBond:

    def __init__(self, atoms):
        self.atoms = atoms


@pytest.fixture
def bond_atom_IDs():
    """
    Returns
    -------
    list of tuples
        To be used as IDs for atoms in a bond
    """

    return [(1, 2), (3, 4), (5, 6), (1, 6), (3, 6)]


@pytest.fixture
def ase_atoms():

    ase_atoms = conversions.ASEAtoms(symbols=FORMULA, cell=CELL, bonds=BONDS,
                                     IDs=IDS)
    return ase_atoms


def test_ASEAtoms(ase_atoms):
    """
    Tests that an ASEAtoms class is equal to an ase.atoms.Atoms class
    initialized with the same parameters, but also has bonds and IDs attributes.
    """

    assert ase_atoms == ase.atoms.Atoms(symbols=FORMULA, cell=CELL)
    assert ase_atoms.bonds == BONDS
    assert ase_atoms.IDs == IDS


def test_ASEAtoms_error():
    """
    Tests that a ValueError is raised if there are not the same number of IDs as
    atoms
    """

    with pytest.raises(ValueError):
        conversions.ASEAtoms(symbols='H5', IDs=range(0, 10, 1))


@pytest.mark.parametrize('index, symbol',
                         [(0, 'C'),
                          (9, 'H'),
                          (12, 'O')])
def test_ASEAtoms_getitem_int(ase_atoms, index, symbol):
    """
    Tests that indexing into ASEAtoms with an int returns the correct
    ase.atom.Atom object
    """

    ase_atom = ase_atoms[index]
    assert isinstance(ase_atom, ase.atom.Atom)
    assert ase_atom.symbol == symbol


@pytest.mark.parametrize('indexes, symbols, bonds',
                         [([0, 1, 2], 'C3', [(1, 2)]),
                          ([1, 2, 3, 4, 8], 'C4H', [(1, 2), (3, 8)]),
                          ([1, 2, 3, 4, 5, 8], 'C5H', [(1, 2), (3, 8), (4, 5)]),
                          (list(range(14)), FORMULA, BONDS),
                          ([1, 3, 5], 'C3', [])])
def test_ASEAtoms_getitem_intlist(ase_atoms, indexes, symbols, bonds):
    """
    Tests that indexing into ASEAtoms with a list of int returns the correct
    ASEAtoms object
    """

    indexed_ase_atoms = ase_atoms[indexes]
    assert isinstance(indexed_ase_atoms, conversions.ASEAtoms)
    assert all(indexed_ase_atoms.symbols == symbols)
    assert indexed_ase_atoms.bonds == bonds
    assert indexed_ase_atoms.IDs == indexes


@pytest.mark.parametrize('p_slice, symbols, bonds, IDs',
                         [((0, 3, 1), 'C3', [(1, 2)], range(3)),
                          ((0, 14, 1), FORMULA, BONDS, IDS),
                          ((3, 8, 1), 'C5', [(4, 5)], range(3, 8, 1)),
                          ((3, 9, 1), 'C5H', [(3, 8), (4, 5)], range(3, 9, 1)),
                          ((0, 14, 2), 'C4H2O', [], range(0, 14, 2)),
                          ((None, 14, None), FORMULA, BONDS, IDS)])
def test_ASEAtoms_getitem_slice(ase_atoms, p_slice, symbols, bonds, IDs):
    """
    Tests that indexing into ASEAtoms with a slice returns the correct ASEAtoms
    object
    """

    sliced_ase_atoms = ase_atoms[slice(*p_slice)]
    assert isinstance(sliced_ase_atoms, conversions.ASEAtoms)
    assert all(sliced_ase_atoms.symbols == symbols)
    assert sliced_ase_atoms.bonds == bonds
    assert sliced_ase_atoms.IDs == list(IDs)


@pytest.mark.parametrize('position, index, mass, symbol, charge',
                         [((0., 0., 0.), 1, 12., 'Ca', 1.5),
                          ((-5, -10., -15.), 10, 1., 'H', 0.),
                          ((2., 4., 8.), None, 56., 'H', -0.5)])
def test_convert_to_ase_atom(position, index, mass, symbol, charge):
    """
    Tests that an equivalent ase.atom.Atom object is created from an MDMC Atom
    """

    atom = Atom(symbol, position=position, mass=mass, charge=charge, cutoff=10.)
    ase_atom = conversions.convert_to_ase_atom(atom, index)
    # If an index is not passed, the index should be set to the Atom ID
    index = index if index else atom.ID
    assert ase_atom.symbol == symbol
    assert ase_atom.charge == charge
    assert ase_atom.mass == mass
    assert ase_atom.index == index
    assert all(ase_atom.position == position)


@pytest.mark.parametrize('element, atom_type, name, set_charge',
                         [('H', 1, 'Alcohol OH (UA)', True),
                          ('O', 5, 't-Butanol COH (UA)', False),
                          ('P', None, None, False),
                          ('K', 4, None, True),
                          ('Ca', None, '119', False)])
def test_convert_from_ase_atom(element, atom_type, name, set_charge):
    """
    Tests that an equivalent MDMC Atom object is created from an ase.atom.Atom

    Includes testing that the atom_type, name and charge can be optionally set
    """

    charge = 1.
    ase_atom = ase.atom.Atom(symbol=element, charge=charge)
    atom = conversions.convert_from_ase_atom(ase_atom,
                                             atom_type=atom_type,
                                             name=name,
                                             set_charge=set_charge,
                                             cutoff=10.)
    assert atom.element == element
    assert atom.atom_type == atom_type
    # If a name is not passed, the name should be the symbol
    name = name if name else ase_atom.symbol
    assert atom.name == name
    # If set_charge is False, the charge should be None
    charge = charge if set_charge else None


@pytest.mark.parametrize('IDs, expected',
                         [(range(1, 20, 1),
                           range(0, 19, 1)),
                          (range(1, 50, 2),
                           range(0, 25, 1)),
                          ([7, 4, 19, 55],
                           range(0, 4, 1))])
def test_get_ase_atoms(IDs, expected):
    """
    Tests that an ASEAtoms object can created from a list of atoms

    Parametrized to test with different IDs, as these require conversion due
    to ASE indexing

    This does not test application of bonds, as these are tested in other
    functions
    """

    atoms = []
    for ID in IDs:
        atom = Atom('H')
        atom.ID = ID
        atoms.append(atom)

    index_conversion = dict(zip(IDs, expected))
    ase_atoms = conversions.get_ase_atoms(atoms)
    for i, atom in enumerate(ase_atoms):
        assert atom.index == index_conversion[ase_atoms.IDs[i]]


def test_convert_bond(bond_atom_IDs):
    """
    Tests that bonds are correctly converted to integers indexing atoms, where
    there is no index conversion
    """

    bond_atoms = map(lambda x: (MockAtom(x[0]), MockAtom(x[1])), bond_atom_IDs)
    mock_bond = MockBond(bond_atoms)

    ase_bond_atoms = conversions.convert_bond(mock_bond)
    assert ase_bond_atoms == bond_atom_IDs


def test_convert_bond_index_conversion(bond_atom_IDs):

    """
    Tests that bonds are correctly converted to integers indexing atoms, where
    there is index conversion
    """

    index_conv = {1:100, 2:200, 3:300, 4:400, 5:500, 6:600}
    bond_atoms = map(lambda x: (MockAtom(x[0]), MockAtom(x[1])), bond_atom_IDs)
    mock_bond = MockBond(bond_atoms)

    ase_bond_atoms = conversions.convert_bond(mock_bond, index_conv=index_conv)
    assert np.all(np.shape(ase_bond_atoms) == np.shape(bond_atom_IDs))
    for ase_pair, ID_pair in zip(ase_bond_atoms, bond_atoms):
        for ase_index, ID in zip(ase_pair, ID_pair):
            assert ase_index == index_conv[ID]


@pytest.mark.parametrize('datatype, header',
                         [('X3DOM', X3DOMHEADER),
                          ('X3D', X3DHEADER)])
def test_x3d_write_header_footer(datatype, header):
    """
    Tests that the correct X3DOM header and footer are written

    An empty ASEAtoms object is passed to X3D
    """

    x3d = conversions.X3D(conversions.get_ase_atoms([], cell=[1., 1., 1.]))
    output = StringIO()
    x3d.write(output, datatype=datatype)
    assert output.getvalue() == header


def test_x3d_atom_lines(monkeypatch):
    """
    Tests that x3d.atom_lines returns the correct x3d/html lines (with the
    correct indentation)
    """

    mock_jmol_colors = [(1.000, 1.000, 1.000),
                        (0.851, 1.000, 1.000)]

    mock_covalent_radii = [8., 4.]

    monkeypatch.setattr(ase.data.colors, 'jmol_colors', mock_jmol_colors)
    monkeypatch.setattr(ase.data, 'covalent_radii', mock_covalent_radii)

    atom = Atom('H', position=(1., 2., 3.))
    ase_atom = conversions.convert_to_ase_atom(atom)
    ase_atom.number = 1
    lines = conversions.X3D([]).atom_lines(ase_atom)

    expected = [(0, '<Transform translation="1.00 2.00 3.00">')]
    expected += [(1, '<Shape>')]
    expected += [(2, '<Appearance>')]
    expected += [(3, '<Material diffuseColor="0.851 1.000 1.000"'
                     ' specularColor="0.5 0.5 0.5">')]
    expected += [(3, '</Material>')]
    expected += [(2, '</Appearance>')]
    expected += [(2, '<Sphere radius="1.00">')]
    expected += [(2, '</Sphere>')]
    expected += [(1, '</Shape>')]
    expected += [(0, '</Transform>')]

    assert len(expected) == len(lines)
    for i, line in enumerate(lines):
        assert line == expected[i]


def test_x3d_bond_lines():
    """
    Tests that x3d.bond_lines returns the correct x3d/html lines (with the
    correct indentation)
    """

    atoms = [Atom('H', position=(2., 4., 6.)),
             Atom('C', position=(6., -2., 9.))]
    ase_atoms = conversions.get_ase_atoms(atoms)
    bond = (0, 1)
    lines = conversions.X3D(ase_atoms).bond_lines(bond)

    expected = [(0, '<Transform center="0 -3.9051 0"'
                    ' translation="2.0000 7.9051 6.0000"'
                    ' rotation="0.3841 0.0000 -0.5121 2.4469">')]
    expected += [(1, '<Shape>')]
    expected += [(2, '<Appearance>')]
    expected += [(3, '<Material diffuseColor="0 0 0"'
                  ' specularColor="0.5 0.5 0.5">')]
    expected += [(3, '</Material>')]
    expected += [(2, '</Appearance>')]
    expected += [(2, '<Cylinder height="7.8102" radius="0.02">')]
    expected += [(2, '</Cylinder>')]
    expected += [(1, '</Shape>')]
    expected += [(0, '</Transform>')]

    assert len(expected) == len(lines)
    for i, line in enumerate(lines):
        assert line == expected[i]


@pytest.mark.parametrize('cell', [[10., 10., 10.],
                                  [20., 10., 5.]])
def test_X3D_get_center_of_rotation_cell(cell):
    """
    Tests that the correct center of rotation is calculated when a cell is set
    """

    cell = np.array(cell)
    ase_atoms = conversions.get_ase_atoms([Atom('H')], cell=cell)
    x3d = conversions.X3D(ase_atoms)
    assert np.all(x3d.get_center_of_rotation() == cell / 2.)


@pytest.mark.parametrize('positions, center',
                         [(([0., 0., 0.], [5., 5., 5.]),
                           [2.5, 2.5, 2.5]),
                          (([1., 2., 3.], [5., 5., 5.], [3., 4., 12.]),
                           [3., 3.5, 7.5])])
def test_X3D_get_center_of_rotation_no_cell(positions, center):
    """
    Tests that the correct center of rotation is calculated when no cell is set
    """

    atoms = [Atom('H', position=position) for position in positions]
    ase_atoms = conversions.get_ase_atoms(atoms)
    x3d = conversions.X3D(ase_atoms)
    assert np.all(x3d.get_center_of_rotation() == center)


@pytest.mark.parametrize('cell, z',
                         [([10., 10., 10.],
                           32.8588),
                          ([5., 5., 5.],
                           32.8588 / 2.),
                          ([10., 5., 0.],
                           18.0714),
                          ([5., 10., 0.],
                           18.0714)])
def test_X3D_get_viewpoint_z_cell(cell, z):
    """
    Tests that the correct z position for setting the viewpoint is calculated
    from the cell
    """

    cell = np.array(cell)
    ase_atoms = conversions.get_ase_atoms([Atom('H')], cell=cell)
    x3d = conversions.X3D(ase_atoms)
    assert np.allclose(x3d.get_viewpoint_z(), z)
