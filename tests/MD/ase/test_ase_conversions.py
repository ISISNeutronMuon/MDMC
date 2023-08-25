"""Tests conversions between equivalent MDMC and ASE objects
"""

import ase
from numpy.testing import assert_allclose
import pytest
from pytest_cases import parametrize, fixture_ref

from MDMC.MD.ase import convert
from MDMC.MD.structures import Atom, Molecule
from MDMC.MD.interactions import Coulombic, Bond, BondAngle, DihedralAngle


FORMULA = 'C8H4O2'
CELL = (1., 2., 3.)
BONDS = [(1, 2), (3, 8), (4, 5)]
IDS = list(range(0, 14, 1))

@pytest.fixture
def water():
    H1 = Atom('H')
    H2 = Atom('H', position=(0., 1.63298, 0.))
    O = Atom('O', position=(0., 0.81649, 0.57736))
    H_coulombic = Coulombic(atoms=[H1, H2], cutoff=10.)
    O_coulombic = Coulombic(atoms=O, cutoff=10.)
    water_mol = Molecule(position=(0, 0, 0),
                         velocity=(0, 0, 0),
                         atoms=[H1, H2, O],
                         interactions=[Bond((H1, O), (H2, O), constrained=True),
                                       BondAngle(H1, O, H2, constrained=True)],
                         name='water')
    return water_mol

@pytest.fixture
def methanol():
    HC1 = Atom('H', position=[-0.7006,  0.3636,  0.8900], name='98', charge=0., atom_type=1)
    C = Atom('C', position=[-0.3366, -0.1504,  0.0000], name='99', charge=0., atom_type=2)
    O = Atom('O', position=[ 1.0849, -0.1713,  0.0000], name='96', charge=0., atom_type=3)
    HO = Atom('H', position=[ 1.3606,  0.7699,  0.0000], name='97', charge=0., atom_type=4)
    CH_bond = Bond(C, HC1)
    CO_bond = Bond(C, O)
    OH_bond = Bond(O, HO)

    HCO_angle = BondAngle((HC1, C, O))
    HOC_angle = BondAngle((HO, O, C))

    HCOH_dihedral = DihedralAngle((HC1, C, O, HO))

    HC2 = HC1.copy(position=[-0.7006,  0.3636, -0.8900])

    H1CH2_angle = BondAngle((HC1, C, HC2))

    # Duplicate the HC1 atom
    # This atom will have all bond (CH_bond) and bond angles (HCO_angle and H1CH2_angle) defined
    HC3 = HC1.copy(position=[-0.7076, -1.1754,  0.0000])
    H1CH3_angle = BondAngle((HC1, C, HC3))

    # Create the methanol Molecule
    methanol = Molecule(atoms=[HC1, HC2, HC3, C, O, HO])
    return methanol


@pytest.mark.parametrize('position, mass, symbol, charge',
                         [((0., 0., 0.), 12., 'Ca', 1.5),
                          ((-5, -10., -15.), 1., 'H', 0.),
                          ((2., 4., 8.), 56., 'H', -0.5)])
def test_convert_to_ase_atom(position, mass, symbol, charge):
    """
    Tests that an equivalent ase.Atom object is created from an MDMC Atom
    """

    atom = Atom(symbol, position=position, mass=mass, charge=charge, cutoff=10.)
    ase_atom = convert.MDMC_to_ASE(atom)[0]
    assert ase_atom.symbol == symbol
    assert ase_atom.charge == charge
    assert ase_atom.mass == mass
    assert all(ase_atom.position == position)


@pytest.mark.parametrize('element', ['H', 'O', 'P', 'K', 'Ca'])
def test_convert_from_ase_atom(element):
    """
    Tests that an equivalent MDMC Atom object is created from an ase.Atom

    Includes testing that the atom_type, name and charge can be optionally set
    """

    charge = 1.
<<<<<<< HEAD
    ase_atom = ase.atom.Atom(symbol=element, charge=charge)
    atom = conversions.convert_from_ase_atom(ase_atom,
                                             atom_type=atom_type,
                                             name=name,
                                             set_charge=set_charge,
                                             cutoff=10.)
    assert atom.element.symbol == element
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
=======
    ase_atom = ase.Atoms([ase.Atom(symbol=element, charge=charge)])
    atom = convert.ASE_to_MDMC(ase_atom)[0]
    assert atom.name == ase_atom[0].symbol
    assert atom.charge == ase_atom[0].charge
>>>>>>> master

@parametrize('molecule', [water, methanol])
def test_convert_involution(molecule):
    """
    Test that converting a molecule to an ASE Atoms object and back
    gives back the original molecule.
    """
    converted_molecule = Molecule(atoms=convert.ASE_to_MDMC(convert.MDMC_to_ASE(molecule)))
    assert converted_molecule.formula == molecule.formula
    assert_allclose(converted_molecule.position, molecule.position, atol=1e-7)
    assert len(converted_molecule.bonded_interactions) == len(molecule.bonded_interactions)
