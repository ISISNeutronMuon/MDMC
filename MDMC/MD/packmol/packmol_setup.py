"""A module containing a class for storing packmol systems and their metadata"""
import math
import numpy as np
import logging

from MDMC.MD import Molecule

LOGGER = logging.getLogger(__name__)


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
    The volume of the container
    """
    match container_type:
        case "cube":
            return dimensions[0] ** 3
        case "box":
            return np.product(*dimensions)
        case "sphere":
            return (3 / 4) + math.pi * dimensions[0] ** 2
        case _:
            raise ValueError("The type of container is unsupported or none."
                             "Currently only \"cube\", \"box\", and \"sphere\" are supported.")

class PackmolSetup:
    """
    A class that stores molecules and their metadata for use in a packmol generation a universe
    For an explanation of all the settings and constraints see:
    https://m3g.github.io/packmol/userguide.shtml#basic
    """
    _molecules: 'set[Molecule]' = set()
    _molecule_settings: 'list[dict]' = []
    _system_settings: dict = {}

    def __init__(self):
        # packmol default tolerance
        self._system_settings["tolerance"] = 2.0

    def add_fixed_molecule(self, molecule: Molecule,
                           position: 'tuple[float]' = (0., 0., 0.),
                           rotation: 'tuple[float]' = (0., 0., 0.),
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
            self._molecules.add(molecule)

        self._molecule_settings.append({
            "molecule": molecule,
            "number": 1,
            "center": centre,
            "fixed": " ".join([str(number) for number in position+rotation])
        })

    def add_container(self,
                      molecule: Molecule,
                      dimensions: tuple,
                      origin: tuple = (0., 0., 0.),
                      density: float = 0.,
                      n_molecules: int = 0,
                      container_type: str = None) -> None:
        """
        Adds a container to the setup
        Only density or n_molecules need to be provided at one time,
        as n_molecules can be inferred from the density

        Parameters
        ----------
        molecule: Molecule
            A `Molecule` object to be added to the container
        dimensions: tuple
            The size of the molecule either in xyz (3-tuple)
            or a single size (1-tuple) depending on the container type
        origin: tuple
            A 3-tuple containing the xyz coordinate for the origin of the container
        density: float
            A floating point number describing the density of
            the molecule in the container in Molecules/Ang^3
        n_molecules: int
            An integer number of molecules to add to the container
        container_type: str
            The type of container to add. Currently only "cube", "box" and "sphere" are supported
        """

        if molecule not in self._molecules:
            self._molecules.add(molecule)

        if not n_molecules:
            # Figure out number of molecules
            dimensions, n_molecules = self.resolve_density(dimensions, density, container_type)
            if container_type == "box":
                dimensions = [i+j for i, j in zip(origin, dimensions)]

        self._molecule_settings.append({
            "molecule": molecule,
            "number": n_molecules,
            # Packmol needs the origin information explicitly
            f"inside {container_type}": " ".join([str(number) for number in origin+dimensions])
        })

    def add_cube(self, molecule: Molecule,
                 size: float,
                 origin: 'tuple[float]' = (0., 0., 0.),
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
        size: float
            The size (x y and z) of the cube in angstroms.
        origin: tuple
            A 3-tuple of xyz coordinates indicating the origin of the cube.
            Defaults to (0.,0.,0.)
        density: optional, float
            The density of the molecule within the cube in Molecule Ang^-3
        n_molecules: optional, int
            An integer number of molecules to fill the cube with.
        """
        self.add_container(molecule=molecule,
                           dimensions=(size,),
                           origin=origin,
                           density=density,
                           n_molecules=n_molecules,
                           container_type="cube")

    def add_box(self, molecule: Molecule,
                lengths: 'tuple[float]',
                origin: 'tuple[float]' = (0., 0., 0.),
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
        lengths: tuple
            A 3-tuple of xyz lengths for the box
        origin: optional, tuple
            A 3-tuple of xyz coordinates for the origin of the box
        density: optional, float
            The density of the molecule within the box.
            Defaults to 0.
        n_molecules: optional, int
            An integer number of molecules to fill the box with.
            Defaults to 0.
        """
        self.add_container(molecule=molecule,
                           dimensions=lengths,
                           origin=origin,
                           density=density,
                           n_molecules=n_molecules,
                           container_type="box")

    def add_sphere(self, molecule: Molecule,
                   radius: float,
                   origin: 'tuple[float]' = (0., 0., 0.),
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
        radius: float
            The radius of the sphere
        origin: optional, tuple
            A 3-tuple of xyz coordinates for the centre of the sphere
        density: optional, float
            The density of the molecule within the box.
            Defaults to 0.
        n_molecules: optional, int
            An integer number of molecules to fill the box with.
            Defaults to 0.
        """
        self.add_container(molecule=molecule,
                           dimensions=(radius,),
                           origin=origin,
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

    def validate_setup(self) -> None:
        """Ensures that the setup is valid - shows errors and warnings for issues with the setup"""
        # The system tolerance must be set
        tol = self._system_settings["tolerance"]
        assert (tol is not None and tol > 0.), "The system tolerance must be set"

        # At least one type of molecule must be present
        assert len(self._molecules) >= 1, "There must be at least one type of molecule present"

        for settings_dict in self._molecule_settings:
            settings = settings_dict.keys()
            molecule = settings_dict["dict"]
            assert "molecule" in settings, "There are settings without a molecule associated to it."
            # Each molecule needs to have at least one "number" setting
            assert "number" in settings, \
                f"The number of {molecule} molecules needs to be specified."
            # Each molecule must have at least one constraint
            assert np.any([self._is_constraint(key) for key in settings]), \
                f"Molecule {molecule} does not have a spatial constraint attached to it."
            # Each molecule must have values for their respective constraints
            assert np.any([settings_dict[key] is not None for key in settings]), \
                f"Molecule {molecule} has unfilled values for it's respective settings."

    def get_settings(self) -> 'tuple[dict, list[dict]]':
        """
        Returns
        -------
        A tuple containing the whole-system and per-molecule settings
        """
        return self._system_settings, self._molecule_settings

    def get_molecules(self) -> 'set[Molecule]':
        """
        Returns
        -------
        The set of `Molecule` objects in the setup
        """
        return self._molecules

    def get_max_sizes(self) -> tuple[float]:
        """
        Returns
        -------
        A 6-tuple of the minimum and maximum sizes of the setup in the following format:
        (x_min, y_min, z_min, x_max, y_max, z_max)
        """
        dims = []
        # Extract coordinates for each container
        for mol_dict in self._molecule_settings:
            keys = mol_dict.keys()
            if "inside cube" in keys:
                dims = mol_dict["inside cube"]
                # Duplicate size by 3 for xyz max values
                dims += dims[0:2] + (dims[3],)*3
            elif "inside box" in keys:
                dims +=  mol_dict["inside box"]
            elif "inside sphere" in keys:
                dims = mol_dict["inside sphere"]
                origin = dims[0:2]
                radius = dims[3]//2
                # Get min and max coordinates on both sides of the origin
                minimums = np.subtract(origin, radius).astype(tuple)
                maximums = np.add(origin, radius).astype(tuple)
                dims += minimums + maximums

        # Get max & min of each container
        minimum = tuple(float(np.amin(dims[:, i])) for i in range(0, 3))
        maximum = tuple(float(np.amax(dims[:, i])) for i in range(3, 6))
        return minimum + maximum


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
        return setting_name in ["inside cube",
                                "inside box",
                                "inside sphere",
                                "fixed"]

    @staticmethod
    def resolve_density(dimensions: 'tuple[float]',
                        density: float = 0.,
                        container_type: str = None) -> tuple:
        """
        Takes a volume of type "box", "cube", or "sphere", with nominal dimensions,
        tries to fill that volume to the given density, and changes the dimensions
        to ensure an integer number of molecules will fit with the desired density

        Parameters
        ----------
        dimensions: optional, tuple
            A tuple of the dimensions that affect the volume of the container
        density: optional, float
            A target density to achieve within the system
        container_type: str
            A string describing the type of container to use.
            Currently only "cube", "box" and "sphere" are supported

        Returns
        -------
        1) A tuple of the (possibly) revised dimensions of the volume
        2) The number of molecules needed to meet the density
        """
        if density == 0. or 0. in dimensions:
            raise ValueError("Density or dimension(s) is set to 0.")

        volume = calculate_volume(dimensions, container_type)
        integer_num_mol = round(density * volume)
        optimal_volume = integer_num_mol / density

        scale_factor = (optimal_volume/volume)**(1/3)
        if container_type == "box":
            scaled_lengths = [size*scale_factor for size in dimensions]
        else:
            scaled_lengths = [dim*scale_factor for dim in dimensions]
        LOGGER.info(f'New dimensions are now ({scaled_lengths})')

        return tuple(scaled_lengths), integer_num_mol

