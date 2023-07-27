"""Tests for the .cif file reader"""

from numpy.testing import assert_allclose

from MDMC.MD import Atom
from MDMC.readers.configurations import read
from tests.test_data import data

def test_cif_reader():
    """Tests that the CIF reader returns a configuration as expected."""
    paracetamol_atoms = [Atom('O', position=(0.85655698, 4.63648644, 7.77559468)),
                         Atom('O', position=(0.62016108, 3.07210947, 1.40311612)),
                         Atom('N', position=(1.07602629, 1.61689972, 3.08503569)),
                         Atom('C', position=(1.06644657, 2.46927555, 4.22582936)),
                         Atom('C', position=(0.95545142, 3.91800828, 6.60801941)),
                         Atom('C', position=(0.49295963, 3.74395309, 4.24625892)),
                         Atom('C', position=(0.8607346, 1.92073084, 1.79419633)),
                         Atom('C', position=(1.62077406, 1.94793432, 5.39409134)),
                         Atom('C', position=(1.57237124, 2.66982764, 6.57711755)),
                         Atom('C', position=(0.44052325, 4.46137138, 5.4361522)),
                         Atom('C', position=(0.91691644, 0.74155991, 0.85786997)),
                         Atom('H', position=(1.2748956, 0.99275052, 0.)),
                         Atom('H', position=(1.4477628, 0., 1.19143838)),
                         Atom('H', position=(0.0144056, 0.3886212, 0.71760986)),
                         Atom('H', position=(2.0672036, 1.04338904, 5.38379072)),
                         Atom('H', position=(0., 5.34530796, 5.46791245)),
                         Atom('H', position=(0.0864336, 4.1099636, 3.45242447)),
                         Atom('H', position=(1.9951756, 2.31877316, 7.39412839)),
                         Atom('H', position=(1.2028676, 0.7713542, 3.28074747)),
                         Atom('H', position=(0.9579724, 4.10525304, 8.45852579))]

    atoms = read(data.CONFIG_DATA['cif'])
    assert isinstance(atoms, list)
    assert len(atoms) == len(paracetamol_atoms)
    for i in range(len(atoms)):
        assert atoms[i].name == paracetamol_atoms[i].name
        assert_allclose(atoms[i].position, paracetamol_atoms[i].position)
