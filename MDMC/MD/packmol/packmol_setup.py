"""A module containing a class for storing packmol systems and their metadata"""
import math
import numpy as np

from MDMC.MD import Molecule


def get_volume_affecting_dimensions(dimensions: 'tuple[float]', container_type: "str" = None):
    match container_type:
        case "cube":
            return dimensions[3]
        case "box":
            x = dimensions[3] - dimensions[0]
            y = dimensions[4] - dimensions[1]
            z = dimensions[5] - dimensions[2]
            return x, y, z
        case "sphere":
            return dimensions[3]
        case _:
            raise ValueError("The type of container is unsupported or none."
                             "Currently only \"cube\", \"box\", and \"sphere\" are supported.")


def calculate_volume(dimensions: 'tuple[float]', container_type: "str" = None) -> float:
    """
    A method to calculate the volume of a container given the dimensions.

    Parameters
    ----------
    dimensions
        A tuple of float that defines the dimensions of the container
    container_type: str
        A string specifying the type of container.
        Currently only "cube", "box" and "sphere" are supported

    Returns
    -------
    The volume of the dimensions
    """
    dimensions = get_volume_affecting_dimensions(dimensions, container_type)
    match container_type:
        case "cube":
            return dimensions[3] ** 3
        case "box":
            x = dimensions[3] - dimensions[0]
            y = dimensions[4] - dimensions[1]
            z = dimensions[5] - dimensions[2]
            return x * y * z
        case "sphere":
            return (3 / 4) + math.pi * (dimensions[3]) ^ 2
        case _:
            raise ValueError("The type of container is unsupported or none."
                             "Currently only \"cube\", \"box\", and \"sphere\" are supported.")

class PackmolSetup:
    """
    A class that stores molecules and their metadata for use in a packmol generation a universe
    For an explanation of all the settings and constraints see:
    https://m3g.github.io/packmol/userguide.shtml#basic
    """
    _molecules: 'list[Molecule]' = []
    _molecule_settings: 'list[dict]' = []
    _system_settings: dict = {}

    def __init__(self):
        # packmol default tolerance
        self._system_settings["tolerance"] = 2.0

    def add_fixed_molecule(self, molecule: Molecule,
                           position: 'tuple[float]' = (0.,0.,0.),
                           rotation: 'tuple[float]' = (0.,0.,0.),
                           centre: bool = True) -> None:
        """
        Add a single molecule in a fixed position to the setup

        Parameters
        ----------
        molecule: Molecule
            The `Molecule` object to be added to the setup
        position: optional, tuple
            A 3-tuple containing the xyz coordinates of the molecule.
            Defaults to the origin (0.,0.,0.)
        rotation: optional, tuple
            A 3-tuple containing the rotational angles of the molecules (in radians).
            Defaults to (0.,0.,0.) (0 rotation in any direction)
        centre: optional, bool
            True if the molecule is to be centred around the position or not. Defaults to True.
        """
        if molecule not in self._molecules:
            self._molecules.append(molecule)

        self._molecule_settings.append({
            "molecule": molecule,
            "number": 1,
            "center": centre,
            "fixed": position + rotation
        })

    def add_container(self,
                      molecule: Molecule,
                      dimensions: tuple = None,
                      density: float = 0.,
                      n_molecules: int = 0,
                      container_type: str = None):

        if molecule not in self._molecules:
            self._molecules.append(molecule)

        if not n_molecules:
            # Figure out number of molecules
            dimensions, n_molecules = self.resolve_density(density, dimensions, container_type)

        self._molecule_settings.append({
            "molecule": molecule,
            "number": n_molecules,
            f"inside {container_type}": dimensions
        })
    def add_cube(self, molecule: Molecule,
                 origin: 'tuple[float]',
                 size: float,
                 density: float = 0.,
                 n_molecules: int = 0) -> None:
        """
        Add a cube of randomly-packed molecules.
        At least two of "size", "density" or "number" must be filled in order to
        properly create the cube. If "density" is provided, the size of the cube
        may change to allow for a whole number of molecules.

        Parameters
        ----------
        molecule: Molecule
            The `Molecule` object to randomly fill the cube with.
        origin: tuple
            A 3-tuple of xyz coordinates indicating the origin of the cube.
        size: float
            The size (x y and z) of the cube in angstroms.
        density: optional, float
            The density of the molecule within the cube.
        n_molecules: optional, int
            An integer number of molecules to fill the cube with.
        """
        self.add_container(molecule=molecule,
                           dimensions=origin+(size),
                           density=density,
                           n_molecules=n_molecules,
                           container_type="cube")
    def add_box(self, molecule: Molecule,
                     dimensions: 'tuple[float]',
                     density: float = 0.,
                     n_molecules: int = 0) -> None:
        """
        Add a cuboid box of randomly-packed molecules.
        At least two of "size", "density" or "number" must be filled in order to
        properly create the box. If "density" is provided, the size of the box
        may change to allow for a whole number of molecules.

        Parameters
        ----------
        molecule: Molecule
            The `Molecule` object to randomly fill the box with.
        dimensions: optional, tuple
            A 6-tuple of origin and end coordinates of the box,
            in the form: x_min, y_min, z_min, x_max, y_max, z_max
            Defaults to None
        density: optional, float
            The density of the molecule within the box.
            Defaults to 0.
        n_molecules: optional, int
            An integer number of molecules to fill the box with.
            Defaults to 0.
        """
        self.add_container(molecule=molecule,
                           dimensions=dimensions,
                           density=density,
                           n_molecules=n_molecules,
                           container_type="box")

    def add_sphere(self, molecule: Molecule,
                     dimensions: 'tuple[float]' = None,
                     density: float = 0.,
                     n_molecules: int = 0) -> None:
        """
        Add a cuboid box of randomly-packed molecules.
        At least two of "size", "density" or "number" must be filled in order to
        properly create the box. If "density" is provided, the size of the box
        may change to allow for a whole number of molecules.

        Parameters
        ----------
        molecule: Molecule
            The `Molecule` object to randomly fill the box with.
        dimensions: optional, tuple
            A 4-tuple of centre and size (diameter) coordinates of the sphere,
            in the form: x_centre, y_centre, z_centre, diameter
            Defaults to None
        density: optional, float
            The density of the molecule within the box.
            Defaults to 0.
        n_molecules: optional, int
            An integer number of molecules to fill the box with.
            Defaults to 0.
        """
        self.add_container(molecule=molecule,
                           dimensions=dimensions,
                           density=density,
                           n_molecules=n_molecules,
                           container_type="sphere")
    def remove_molecule(self, molecule: Molecule) -> None:
        """
        Remove a molecule and associated setups from the system

        Parameters
        ----------
        molecule: Molecule
            The `Molecule` object to remove from the setup
        """
        if molecule not in self._molecules:
            raise ValueError("This molecule does not exist in the setup.")
        else:
            for setting in self._molecule_settings:
                if setting["molecule"] == molecule:
                    del setting
            self._molecules.remove(molecule)

    def validate_setup(self):
        """Ensures that the setup is valid - shows errors and warnings for issues with the setup"""
        # The system tolerance must be set
        tol = self._system_settings["tolerance"]
        assert (tol is not None and tol > 0.), "The system tolerance must be set"

        # At least one type of molecule must be present
        assert len(self._molecules) >= 1, "There must be at least one type of molecule present"

        for settings_dict in self._molecule_settings:
            settings = settings_dict.keys()
            molecule = settings_dict["dict"]
            assert "molecule" in settings, "There is a setting without a molecule associated to it."
            # Each molecule needs to have at least one "number" setting
            assert "number" in settings, \
                f"The number of {molecule} molecules needs to be specified."
            # Each molecule must have at least one constraint
            assert np.any([self._is_constraint(key) for key in settings]), \
                f"Molecule {molecule} needs to have a spatial constraint attached to it."
            # Each molecule must have values for their respective constraints
            assert np.any([settings[key] is not None for key in settings]), \
                f"Molecule {molecule} has unfilled values for it's respective settings."

    def get_settings(self) -> tuple[dict, dict]:
        pass

    @staticmethod
    def _is_constraint(setting_name: str) -> bool:
        """
        Checks if a setting name is a constraint

        Parameters
        ----------
        setting_name: str
            The name of the setting

        Returns
        -------
        True if the setting is a constraint, False otherwise
        """
        return setting_name in ["inside cube", "outside cube",
                                "inside box", "outside box",
                                "inside sphere", "outside sphere", "fixed"]

    @staticmethod
    def resolve_density(density: float = 0.,
                        dimensions: 'tuple[float]' = (0., 0., 0.),
                        container_type: str = None) -> tuple:
        """
        Finds and returns the number of molecules and
        revised dimensions of a volume given dimensions
        and a target density.

        Parameters
        ----------
        density: optional, float
            A target density to achieve within the system
        dimensions: optional, tuple
            A tuple of the dimensions of the shape
        container_type: str
            A string describing the type of container to use.
            Currently only "cube", "box" and "sphere" are supported

        Returns
        -------
        1) A tuple of the (possibly) revised dimensions of the volume
        2) The number of molecules needed to meet the density
        """
        if density == 0. and dimensions == (0., 0., 0.):
            raise ValueError("Density and sizes are all set to 0.")

        volume = calculate_volume(dimensions, container_type)
        expected_mol = round(density * volume)
        expected_volume = expected_mol / density

        area_factor = expected_volume/volume
        scale_factor = area_factor**(1/3)

        scale_dimensions = get_volume_affecting_dimensions(dimensions, container_type)
        scale_dimensions = [dim*scale_factor for dim in scale_dimensions]

        return scale_dimensions, expected_mol

