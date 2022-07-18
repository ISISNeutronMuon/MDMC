"""Tests for the GUI viewer"""

import pytest
from pytest_cases import parametrize, fixture_ref
from IPython.core.display import HTML

from tests.test_data.data import GUI_DATA
from MDMC import MD
from MDMC.gui import view


@pytest.fixture
def atoms():
    """Two unbonded atom objects."""
    H1 = MD.Atom('H', charge=0.4238)
    O = MD.Atom('O', position=(0., 0.81649, 0.57736), charge=-0.8476)

    return [H1, O]

@pytest.fixture
def water_molecule(atoms):
    """A water molecule."""
    H1 = atoms[0]
    O = atoms[1]
    HO_bond = MD.Bond((H1, O))
    H2 = H1.copy(position=(0., 1.63298, 0.))

    return MD.Molecule(atoms=[H1, O, H2])

@pytest.fixture
def universe(water_molecule):
    """A universe filled with water molecules."""
    box = MD.Universe(10.)
    box.fill(water_molecule, num_density=0.0336)

    return box


@parametrize('structures, expected_html', 
            [(fixture_ref(atoms), 'atoms_X3DOM'),
             (fixture_ref(water_molecule), 'water_molecule_X3DOM'),
             (fixture_ref(universe), 'universe_X3DOM')])
def test_view_X3DOM(structures, expected_html):
    """Tests that the HTML viewer creates the expected objects."""
    html = view(structures, viewer='X3DOM')
    expected = HTML(filename=GUI_DATA[expected_html])

    assert html.data == expected.data
