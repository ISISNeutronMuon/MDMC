"""Tests for the GUI viewer"""

import pytest
from IPython.core.display import HTML

from MDMC.gui import view
from MDMC.MD import interactions, simulation, structures
from tests.test_data.data import GUI_DATA


# uses non-fixtures and function dict
# rather than fixtures and pytest_cases.parametrize because
# for some reason, when we use fixtures and parametrize over the fixtures,
# the tests create new atoms which make ase.conversions fall over.
def atom():
    """One atom object."""
    Ar = structures.Atom('Ar', position=(0., 0.81649, 0.57736), charge=-0.8476)

    return Ar

def water_molecule():
    """A water molecule."""
    H1 = structures.Atom('H', charge=0.4238)
    O = structures.Atom('O', position=(0., 0.81649, 0.57736), charge=-0.8476)
    HO_bond = interactions.Bond((H1, O))
    H2 = H1.copy(position=(0., 1.63298, 0.))

    return structures.Molecule(atoms=[H1, O, H2])

def universe():
    """A universe filled with water molecules."""
    box = simulation.Universe(10., verbose=False)
    box.fill(water_molecule(), num_density=0.0336)

    return box

fixture_dict = {
    'atom': atom(),
    'water_molecule': water_molecule(),
    'universe': universe()
}


@pytest.mark.parametrize('structures', ['atom', 'water_molecule', 'universe'])
def test_view_X3DOM(structures):
    """Tests that the HTML viewer creates the expected objects."""
    html = view(fixture_dict[structures], viewer='X3D')
    expected = HTML(filename=GUI_DATA[structures + "_X3DOM"])

    assert html.data == expected.data
