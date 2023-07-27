"""Tests for the GUI viewer"""

import pytest
from IPython.core.display import HTML

from tests.test_data.data import GUI_DATA
from MDMC import MD
from MDMC.gui import view

# uses non-fixtures and function dict 
# rather than fixtures and pytest_cases.parametrize because
# for some reason, when we use fixtures and parametrize over the fixtures,
# the tests create new atoms which make ase.conversions fall over.
def atoms():
    """Two unbonded atom objects."""
    H1 = MD.Atom('H', charge=0.4238)
    O = MD.Atom('O', position=(0., 0.81649, 0.57736), charge=-0.8476)

    return [H1, O]

def water_molecule():
    """A water molecule."""
    H1 = atoms()[0]
    O = atoms()[1]
    HO_bond = MD.Bond((H1, O))
    H2 = H1.copy(position=(0., 1.63298, 0.))

    return MD.Molecule(atoms=[H1, O, H2])

def universe():
    """A universe filled with water molecules."""
    box = MD.Universe(10., verbose=False)
    box.fill(water_molecule(), num_density=0.0336)

    return box

fixture_dict = {
    'atoms': atoms(),
    'water_molecule': water_molecule(),
    'universe': universe()
}


@pytest.mark.parametrize('structures', ['atoms', 'water_molecule', 'universe'])
def test_view_X3DOM(structures):
    """Tests that the HTML viewer creates the expected objects."""
    html = view(fixture_dict[structures], viewer='X3D')
    expected = HTML(filename=GUI_DATA[structures + "_X3DOM"])

    assert html.data == expected.data
