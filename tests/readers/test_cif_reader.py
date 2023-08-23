"""Tests for the .cif file reader"""

from numpy.testing import assert_allclose

from MDMC.MD import Atom
from MDMC.readers.configurations import read
from tests.test_data import data

def test_cif_reader():
    """Tests that the CIF reader returns a configuration as expected."""
    paracetamol_atoms = [Atom('O', position=(5.495160176, 2.815855004, 6.336941424)),
                         Atom('O', position=(5.25876428, 1.251478028, -0.035537139)),
                         Atom('N', position=(5.714629492, -0.20373172, 1.64638243)),
                         Atom('C', position=(5.705049768, 0.648644112, 2.787176095)),
                         Atom('C', position=(5.59405462, 2.09737684, 5.169366147)),
                         Atom('C', position=(5.131562832, 1.923321648, 2.807605658)),
                         Atom('C', position=(5.4993378, 0.1000994, 0.355543067)),
                         Atom('C', position=(6.259377256, 0.127302884, 3.95543808)),
                         Atom('C', position=(6.21097444, 0.849196204, 5.138464287)),
                         Atom('C', position=(5.079126448, 2.640739936, 3.997498945)),
                         Atom('C', position=(5.55551964, -1.079071533, -0.580783291)),
                         Atom('H', position=(5.9134988, -0.82788092, -1.43865326)),
                         Atom('H', position=(6.086366, -1.82063144, -0.24721488)),
                         Atom('H', position=(4.6530088, -1.43201024, -0.7210434)),
                         Atom('H', position=(6.7058068, -0.7772424, 3.94513746)),
                         Atom('H', position=(4.6386032, 3.52467652, 4.02925919)),
                         Atom('H', position=(4.7250368, 2.28933216, 2.01377121)),
                         Atom('H', position=(6.6337788, 0.49814172, 5.95547513)),
                         Atom('H', position=(5.8414708, -1.04927724, 1.84209421)),
                         Atom('H', position=(5.5965756, 2.2846216, 7.01987253))]

    atoms = read(data.CONFIG_DATA['cif'])
    assert isinstance(atoms, list)
    assert len(atoms) == len(paracetamol_atoms)
    for i in range(len(atoms)):
        assert atoms[i].name == paracetamol_atoms[i].name
        assert_allclose(atoms[i].position, paracetamol_atoms[i].position)
