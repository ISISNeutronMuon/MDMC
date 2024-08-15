"""A module containing a class for storing packmol systems and their metadata"""
import logging
import math
import warnings
from typing import List, Tuple

import numpy as np

from MDMC.MD import Structure

LOGGER = logging.getLogger(__name__)


def calculate_volume(dimensions: Tuple[float], container_type: str = None) -> float:
    """
    A method to calculate the volume of a container given the dimensions.

    Parameters
    ----------
    dimensions
        A tuple of floats that define the dimensions of the container.
    container_type: str
        A string specifying the type of container.
        Currently only "cube", "box" and "sphere" are supported.

    Returns
    -------
    `float`
        The volume of the container.
    """
    match container_type:
        case "cube":
            return dimensions[0] ** 3
        case "box":
            return np.prod(np.array(dimensions))
        case "sphere":
            return (4 / 3) * math.pi * (dimensions[0] ** 3)
        case _:
            raise ValueError("The type of container is unsupported or none."
                             "Currently only \"cube\", \"box\", and \"sphere\" are supported.")

class PackmolSetup:
    """
    A class that stores structures and their metadata for use in a packmol generation.
    For an explanation of all the settings and constraints see:
    https://m3g.github.io/packmol/userguide.shtml#basic
    """

    def __init__(self):
        self._structures: List[Structure] = []
        self._structure_settings: List[dict] = []
        self._system_settings: dict = {}
        self._system_settings["tolerance"] = 2.0  # packmol default tolerance

    def add_fixed_structure(self, structure: Structure,
                           position: Tuple[float] = (0., 0., 0.),
                           rotation: Tuple[float] = (0., 0., 0.),
                           centre: bool = True) -> None:
        """
        Add a single structure (atom or molecule) in a fixed position to the setup.

        Parameters
        ----------
        structure: Structure
            The `Structure` object (atom or molecule) to be added to the setup.
        position: optional, tuple
            A 3-tuple containing the xyz coordinates of the structure.
            Defaults to the origin (0.,0.,0.)
        rotation: optional, tuple
            A 3-tuple containing the rotational angles of the structure (in radians).
            Defaults to (0.,0.,0.) (0 rotation in any direction). Meaningless if
            the structure is not a Molecule.
        centre: optional, bool
            True if the structure is to be centred around the position or not. Defaults to True.
            Meaningless if the structure is not a Molecule.
        """
        if structure not in self._structures:
            self._structures.append(structure)

        self._structure_settings.append({
            "structure": structure,
            "number": 1,
            "center": centre,
            "fixed": " ".join([str(number) for number in position+rotation]),
        })

    def add_container(self,
                      structure: Structure,
                      dimensions: tuple,
                      origin: tuple = (0., 0., 0.),
                      density: float = 0.,
                      n_structures: int = 0,
                      container_type: str = None) -> None:
        """
        Adds a container to the setup
        Only density or n_structures need to be provided at one time,
        as n_structures can be inferred from the density

        Parameters
        ----------
        structure: Structure
            A `Structure` (atom or molecule) object to be added to the container.
        dimensions: tuple
            The size of the container either in xyz (3-tuple)
            or a single size (1-tuple) depending on the container type.
        origin: tuple
            A 3-tuple containing the xyz coordinate for the origin of the container.
        density: float
            A floating point number describing the density of
            the structures in the container in Ang^-3.
        n_structures: int
            An integer number of structures to add to the container.
        container_type: str
            The type of container to add. Currently only "cube", "box" and "sphere" are supported.
        """

        if structure not in self._structures:
            self._structures.append(structure)

        if not n_structures:
            # Figure out number of structures
            dimensions, n_structures = self.resolve_density(dimensions, density, container_type)

        if container_type == "box":
            dimensions = tuple(i+j for i, j in zip(origin, dimensions))

        self._structure_settings.append({
            "structure": structure,
            "number": n_structures,
            # Packmol needs the origin information explicitly
            f"inside {container_type}": " ".join([str(number) for number in origin+dimensions]),
        })

    def add_cube(self, structure: Structure,
                 size: float,
                 origin: Tuple[float] = (0., 0., 0.),
                 density: float = 0.,
                 n_structures: int = 0) -> None:
        """
        Add a cube of randomly-packed structures.
        At least two of "size", "density" or "number" must be filled in order to
        properly create the cube. If "density" is provided, the size of the cube
        may change to allow for a whole number of structures.

        Parameters
        ----------
        structure: Structure
            The `Structure` (atom or molecule) object to randomly fill the cube with.
        size: float
            The size (x y and z) of the cube in angstroms.
        origin: tuple
            A 3-tuple of xyz coordinates indicating the origin of the cube.
            Defaults to (0.,0.,0.).
        density: optional, float
            The density of the structures within the cube in Ang^-3.
        n_structures: optional, int
            An integer number of structures to fill the cube with.
        """
        self.add_container(structure=structure,
                           dimensions=(size,),
                           origin=origin,
                           density=density,
                           n_structures=n_structures,
                           container_type="cube")

    def add_box(self, structure: Structure,
                lengths: Tuple[float],
                origin: Tuple[float] = (0., 0., 0.),
                density: float = 0.,
                n_structures: int = 0) -> None:
        """
        Add a cuboid box of randomly-packed structures.
        At least two of "lengths", "density" or "number" must be filled in order to
        properly create the box. If "density" is provided, the size of the box
        may change to allow for a whole number of structures.

        Parameters
        ----------
        structure: Structure
            The `Structure` (atom or molecule) object to randomly fill the box with.
        lengths: tuple
            A 3-tuple of xyz lengths for the box in angstroms.
        origin: optional, tuple
            A 3-tuple of xyz coordinates for the origin of the box. Defaults to (0.,0.,0.).
        density: optional, float
            The density of the structures within the box.
            Defaults to 0.
        n_structures: optional, int
            An integer number of structures to fill the box with.
            Defaults to 0.
        """
        self.add_container(structure=structure,
                           dimensions=lengths,
                           origin=origin,
                           density=density,
                           n_structures=n_structures,
                           container_type="box")

    def add_sphere(self, structure: Structure,
                   radius: float,
                   origin: Tuple[float] = (0., 0., 0.),
                   density: float = 0.,
                   n_structures: int = 0) -> None:
        """
        Add a sphere of randomly-packed structures.
        At least two of "size", "density" or "number" must be filled in order to
        properly create the box. If "density" is provided, the size of the box
        may change to allow for a whole number of structures.

        Parameters
        ----------
        structure: Structure
            The `Structure` (atom or molecule) object to randomly fill the sphere with.
        radius: float
            The radius of the sphere in angstroms.
        origin: optional, tuple
            A 3-tuple of xyz coordinates for the centre of the sphere.
        density: optional, float
            The density of the structures within the box.
            Defaults to 0.
        n_structures: optional, int
            An integer number of structures to fill the box with.
            Defaults to 0.
        """
        self.add_container(structure=structure,
                           dimensions=(radius,),
                           origin=origin,
                           density=density,
                           n_structures=n_structures,
                           container_type="sphere")

    def remove_structure(self, structure: Structure) -> None:
        """
        Remove a structure and associated setups from the system.

        Parameters
        ----------
        structure
            The `Structure` (atom or molecule) object to remove from the setup.
        """
        if structure not in self._structures:
            warnings.warn(f"The removal of the structure {structure} was requested,"
                          " but no such structure is in the system.")
        self._structure_settings = [setting
                                    for setting in self._structure_settings
                                    if setting["structure"] != structure]
        self._structures.remove(structure)

    def validate_setup(self) -> None:
        """Ensures that the setup is valid - shows errors and warnings for issues with the setup"""
        error_messages = []
        tol = self._system_settings["tolerance"]
        if (tol is None or tol <= 0.):
            error_messages.append("The system tolerance must be set.")

        if len(self._structures) == 0:
            error_messages.append("There must be at least one type of structure present.")

        for settings_dict in self._structure_settings:
            settings = settings_dict.keys()
            structure = settings_dict["structure"]
            if "structure" not in settings:
                error_messages.append("There are settings without an associated structure.")
            # Each structure needs to have at least one "number" setting
            if "number" not in settings:
                error_messages.append(f"The number of {structure}"
                                      " structures needs to be specified.")
            # Each structure must have at least one constraint
            if not np.any([self._is_constraint(key) for key in settings]):
                error_messages.append(f"Structure {structure} must"
                                      " have a spatial constraint attached.")
            # Each structure must have values for their respective constraints
            if np.any([settings_dict[key] is None for key in settings]):
                error_messages.append(f"Structure {structure} has unfilled values"
                                      " for its settings.")

        if error_messages:
            raise RuntimeError("The packmol setup is invalid for the following reasons:\n"
                               + "\n".join(error_messages))

    def get_settings(self) -> Tuple[dict, list[dict]]:
        """
        Returns
        -------
        `tuple`
            A tuple containing the whole-system and per-structure settings
        """
        return self._system_settings, self._structure_settings

    def get_structures(self) -> Tuple[Structure]:
        """
        Returns
        -------
        `list`
            The set of `Structure` objects in the setup
        """
        return self._structures

    def get_max_sizes(self) -> Tuple[float]:
        """
        Returns
        -------
        `tuple`
            A 6-tuple of the minimum and maximum sizes of the setup in the following format:
            (x_min, y_min, z_min, x_max, y_max, z_max)
        """
        dims = np.zeros(shape=(1,6))
        # Extract coordinates for each container
        for index, mol_dict in enumerate(self._structure_settings):
            keys = mol_dict.keys()
            if "inside cube" in keys:
                container_dims = [float(dim) for dim in mol_dict["inside cube"].split()]
                end_coord = np.add(container_dims[0:3], container_dims[3])
                final_dims = np.concatenate([container_dims[0:3], end_coord])
                dims = np.insert(dims, index, final_dims, axis=0)
            elif "inside box" in keys:
                container_dims = [float(dim) for dim in mol_dict["inside box"].split()]
                dims = np.insert(dims, index, container_dims, axis=0)
            elif "inside sphere" in keys:
                container_dims = [float(dim) for dim in mol_dict["inside sphere"].split()]
                origin = container_dims[0:3]
                radius = container_dims[3]
                # Get min and max coordinates on both sides of the origin
                origin_min = np.subtract(origin, radius)
                origin_max = np.add(origin, radius)
                dims = np.insert(dims, index, np.concatenate([origin_min, origin_max]), axis=0)

        # Get max & min of each container
        minimum = [float(np.amin(dims[:, i])) for i in range(0, 3)]
        maximum = [float(np.amax(dims[:, i])) for i in range(3, 6)]
        return np.subtract(maximum, minimum)


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
        `bool`
            True if the setting is a constraint, False otherwise
        """
        return setting_name in ["inside cube",
                                "inside box",
                                "inside sphere",
                                "fixed"]

    @staticmethod
    def resolve_density(dimensions: Tuple[float],
                        density: float = 0.,
                        container_type: str = None) -> tuple:
        """
        Takes a volume of type "box", "cube", or "sphere", with nominal dimensions,
        tries to fill that volume to the given density, and changes the dimensions
        to ensure an integer number of structures will fit with the desired density.

        Parameters
        ----------
        dimensions: optional, tuple
            A tuple of the dimensions that affect the volume of the container.
        density: optional, float
            A target density to achieve within the system.
        container_type: str
            A string describing the type of container to use.
            Currently only "cube", "box" and "sphere" are supported.

        Returns
        -------
        `tuple`
            A tuple containing (in order):
            1) A tuple of the (possibly) revised dimensions of the volume
            2) The number of structures needed to meet the density
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

        if scale_factor != 1.0:
            info_string = (f"The dimensions of the container are changed to {scaled_lengths} "
                           f"to achieve the requested density of {density} "
                            "with a whole number of structures")
            # TODO: Find a way to silence this to not clog the output during testing
            print(info_string)
            LOGGER.info(msg=info_string)

        return tuple(scaled_lengths), integer_num_mol

    @property
    def tolerance(self) -> float:
        """
        Get or set the tolerance of the packmol run (i.e. how far apart each atom must be)

        Returns
        -------
        `float`
            The tolerance of the distances between atoms
        """
        return self._system_settings["tolerance"]

    @tolerance.setter
    def tolerance(self, new_tolerance: float) -> None:
        self._system_settings["tolerance"] = new_tolerance
