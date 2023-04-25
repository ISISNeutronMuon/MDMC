"""A module containing a class for storing packmol systems and their metadata"""
from MDMC.MD import Molecule

class PackmolSetup:
    """
    A class that stores molecules and their metadata for use in a packmol generation a universe
    For an explanation of all the settings and constraints see:
    https://m3g.github.io/packmol/userguide.shtml#basic
    The currently-supported metadata for the whole system is as follows:
        tolerance
    The currently-supported metadata for each molecule is as follows:
        number
        fixed
        inside/outside cube xmin  ymin  zmin  d
        inside/outside box xmin  ymin  zmin  xmax  ymax  zmax
        inside/outside sphere a  b  c  d
        inside/outside ellipsoid a1  b1  c1  a2  b2  c2  d
        above/below plane x  y  z  d
        inside/outside cylinder a1  b1  c1  a2  b2  c2  d  l
        constrain_rotation x/y/z  angle  variation
        radius
        atoms (?)
    """
    _molecules: 'list[Molecule]' = []
    _molecule_settings: dict = {}
    _system_settings: dict = {}

    def __init__(self, system_settings: dict = None, molecule_settings: dict = None):
        """
        Parameters
        ----------
        system_settings: dict
            A dictionary of settings for the system as a whole
        molecule_settings
            A dictionary of dictionaries.
            The outer dictionary maps `Molecule` objects to an inner dictionary
            The inner dictionary defines the settings of said molecule
        """
        # packmol default tolerance
        self._system_settings["tolerance"] = 2.0
        pass

    def add_molecule(self, molecule: Molecule, settings: dict = None):
        """
        Add a molecule

        Parameters
        ----------
        molecule: Molecule
            The `Molecule` object to be added to the setup
        settings: optional, dict
            A dictionary holding the values for the settings of the molecule
        """
        pass


    def apply_molecule_settings(self, molecule: Molecule, settings: dict) -> None:
        """
        Apply a different set of settings to a specific molecule

        Parameters
        ----------
        molecule: Molecule
            The `Molecule` object to apply the settings to
        settings: dict
            A dictionary of settings to apply to the molecule
        """
        pass

    def remove_molecule(self, molecule: Molecule) -> None:
        """
        Remove molecule(s) from the system

        Parameters
        ----------
        molecule: Molecule
            The `Molecule` object to remove from the setup
        """
        pass

    def validate_setup(self):
        """
        Ensures that the setup is valid - shows errors and warnings for issues with the setup
        """
        # Each molecule needs to have at least one "number" setting
        # Each molecule must have at least one constraint
        # Each molecule must have values for their respective constraints
        # The system tolerance must be set
        pass

    def _is_constraint(self, setting_name: str) -> bool:
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
        return setting_name in ["fixed", ""]