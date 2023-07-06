import pytest
import numpy as np

from MDMC.MD import Atom, Bond, BondAngle, Molecule
from MDMC.MD.packmol import packmol_setup as packmol_setup_module

# lammps mark used to ensure test runs in docker container
pytestmark = [pytest.mark.lammps]

@pytest.fixture()
def h2o_molecule():
    """A simple water molecule"""
    H1 = Atom('H')
    H2 = Atom('H', position=[0., 1.63298, 0.])
    O = Atom('O', position=[0., 0.81649, 0.57736])
    h2o_bonds = Bond((H1, O), (H2, O))
    HOH_angle = BondAngle(H1, O, H2)
    return Molecule(atoms=[H1, H2, O], interactions=[HOH_angle, h2o_bonds], name="water")

@pytest.mark.parametrize('lengths, container_type, expected',
                         [((20.,), "cube", 8000.),
                          ((10.,), "sphere", 4188.79),
                          ((10., 20., 5.), "box", 1000.)])
def test_calculate_volume(lengths, container_type, expected):
    """Tests that the volume of a container is accurately calculated"""
    volume = packmol_setup_module.calculate_volume(lengths, container_type)
    assert np.isclose(volume, expected, atol=1e-2)

@pytest.mark.parametrize('origin, rotation, centre',
                        [((0.,0.,0.,), (0.,0.,0.), True),
                         ((10.,10.,10.), (10.,10.,10.), False)])
def test_static_molecule_created(h2o_molecule, origin, rotation, centre):
    """Tests that the creation of a static molecule is successful"""
    setup = packmol_setup_module.PackmolSetup()
    setup.add_fixed_structure(structure=h2o_molecule,
                             position=origin,
                             rotation=rotation,
                             centre=centre)
    _, mol_settings = setup.get_settings()
    fixed_mol_setting = mol_settings[0]
    for attribute in ["structure", "center", "number", "fixed"]:
        assert attribute in list(fixed_mol_setting.keys())

    assert fixed_mol_setting["structure"] == h2o_molecule
    assert fixed_mol_setting["center"] == centre
    assert fixed_mol_setting["number"] == 1
    assert fixed_mol_setting["fixed"] == " ".join([str(num) for num in origin+rotation])
@pytest.mark.parametrize('lengths, origin, density, n_molecules',
                         [((30.,20.,15.), (0.,0.,0.), 0.01, 0),
                          ((20., 10., 20.), (10., 10., 10.), 0., 1000),
                          ((20.,20.,20.), (30., 30., 30.), 0.1, 0)])
def test_box_correctly_created(h2o_molecule, lengths, origin, density, n_molecules):
    """
    Tests that a cuboid box is created and has the correct number of molecules
    and coordinates
    """
    setup = packmol_setup_module.PackmolSetup()
    setup.add_box(h2o_molecule, lengths, origin, density, n_molecules)
    _, mol_settings = setup.get_settings()
    settings = mol_settings[0]
    assert settings["structure"] == h2o_molecule
    assert settings["number"] == n_molecules or \
    settings["number"] == int(density * packmol_setup_module.calculate_volume(lengths, "box"))
    assert "inside box" in settings.keys()
    lengths_arr = np.array(lengths)
    origin_arr = np.array(origin)
    expected_lengths = tuple(np.add(lengths_arr, origin_arr))
    assert settings["inside box"] == " ".join([str(num) for num in origin + expected_lengths])

@pytest.mark.parametrize('lengths, origin, density, n_molecules',
                         [(20., (0.,0.,0.), 0.01, 0),
                          (10.5, (10., 10., 10.), 0., 1000),
                          (30., (30., 30., 30.), 0.1, 0)])
def test_cube_correctly_created(h2o_molecule, lengths, origin, density, n_molecules):
    """Tests that a cube is created and has the correct attributes assigned"""
    setup = packmol_setup_module.PackmolSetup()
    setup.add_cube(h2o_molecule, lengths, origin, density, n_molecules)
    _, mol_settings = setup.get_settings()
    settings = mol_settings[0]
    assert settings["structure"] == h2o_molecule
    assert settings["number"] == n_molecules or \
    settings["number"] == int(
        density * packmol_setup_module.calculate_volume((lengths,), "cube"))
    assert "inside cube" in settings.keys()
    assert settings["inside cube"] == " ".join([str(num) for num in origin + (lengths,)])

@pytest.mark.parametrize('lengths, origin, density, n_molecules, expected_molecules',
                         [(20., (0.,0.,0.), 0.01, 0, 335),
                          (10.5, (10., 10., 10.), 0., 1000, 1000),
                          (30., (30., 30., 30.), 0.1, 0, 11310)])
def test_sphere_correctly_created(h2o_molecule, lengths, origin, density, n_molecules, expected_molecules):
    """Tests that a sphere is created and has the correct attributes assigned"""
    setup = packmol_setup_module.PackmolSetup()
    setup.add_sphere(h2o_molecule, lengths, origin, density, n_molecules)
    _, mol_settings = setup.get_settings()
    settings = mol_settings[0]
    assert settings["structure"] == h2o_molecule
    assert settings["number"] == expected_molecules
    assert "inside sphere" in settings.keys()
    assert np.allclose(float(settings["inside sphere"].split()[-1]), lengths, atol=1.e-2)

def test_atoms_can_be_used_for_filling():
    """Tests that we can fill a universe with Atoms as well as Molecules"""
    setup = packmol_setup_module.PackmolSetup()
    hydrogen_atom = Atom('H')
    setup.add_box(hydrogen_atom, (20.,20.,20.,), density=0.1)
    setup.validate_setup()

def test_tolerance():
    """
    Tests that getting tolerance will return the default value (2.0),
    then tests that the tolerance can be changed to a given value
    """
    setup = packmol_setup_module.PackmolSetup()
    assert np.isclose(setup.tolerance, 2.0)
    setup.tolerance = 1.0
    assert np.isclose(setup.tolerance, 1.0)

# Create test for resolving density
@pytest.mark.parametrize("dimensions, density, container_type, expected_dimensions, expected_n_molecules",
                         [((10., 20., 30.,), 0.2132, "box", (9.999, 19.999, 29.998), 1279),
                          ((10.0,), 0.1, "cube", (10.0,), 100),
                          ((20.,), 0.1243, "sphere", (19.999,), 4165)
                          ])
def test_resolve_density_correctly_changes_volume(dimensions, density, container_type,
                                                  expected_dimensions, expected_n_molecules):
    """
    Test that resolving density will accurately change the volume of
    a container to accommodate a density
    """
    setup = packmol_setup_module.PackmolSetup()
    actual_dimensions, actual_n_molecules = setup.resolve_density(dimensions, density, container_type)
    assert np.allclose(actual_dimensions, expected_dimensions, atol=1.e-3)
    assert actual_n_molecules == expected_n_molecules

def test_resolve_density_handles_0_values():
    """Tests that the resolve_density """
    setup = packmol_setup_module.PackmolSetup()
    with pytest.raises(ValueError):
        setup.resolve_density((0.,0.,0.,), 0.5, "box")
    with pytest.raises(ValueError):
        setup.resolve_density((10.,30.,15.,), 0., "box")

# test if resolve density lets user know when volume of container is changed
def test_user_informed_when_volume_changed():
    setup = packmol_setup_module.PackmolSetup()
    dimensions, _ = setup.resolve_density((20.,), 0.1271, "sphere")
    # TODO Look at logger or stdout to check for the message

def test_get_system_settings():
    """Tests to make sure the system settings are correctly returned (currently only tolerance)"""
    setup = packmol_setup_module.PackmolSetup()
    system_settings, _ = setup.get_settings()
    assert "tolerance" in system_settings.keys()
def test_get_molecule_settings(h2o_molecule):
    """Tests to make sure the molecule settings are correctly returned"""
    setup = packmol_setup_module.PackmolSetup()
    setup.add_box(h2o_molecule,(10.,10.,10.), n_structures=1000)
    setup.add_cube(h2o_molecule, 10., n_structures=2000)
    setup.add_sphere(h2o_molecule, 10., n_structures=1000)
    setup.add_fixed_structure(h2o_molecule)
    _, molecule_settings = setup.get_settings()
    for setting in molecule_settings:
        assert "structure" in setting.keys()
        assert "number" in setting.keys()
        if "fixed" in setting.keys():
            assert "center" in setting.keys()
