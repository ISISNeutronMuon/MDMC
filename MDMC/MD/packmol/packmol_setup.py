"""A module containing a class for storing packmol systems and their metadata"""
from typing import Tuple

import numpy as np

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
        if system_settings != None:
            self._system_settings = system_settings
        if molecule_settings != None:
            self._molecule_settings = molecule_settings

    def add_molecule(self, molecule: Molecule, settings: dict = None):
        """
        Add a molecule to the setup

        Parameters
        ----------
        molecule: Molecule
            The `Molecule` object to be added to the setup
        settings: optional, dict
            A dictionary holding the values for the settings of the molecule
        """
        if molecule not in self._molecules:
            self._molecules.append(molecule)
        else:
            raise ValueError("The molecule already exists in the setup. "
                             "Please change the settings of the molecule instead")
        self._molecule_settings[molecule] = settings


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
        if molecule not in self._molecules:
            raise ValueError("This molecule does not exist in the setup. "
                             "Please add the molecule to the setup")
        else:
            self._molecule_settings[molecule] = settings

    def remove_molecule(self, molecule: Molecule) -> None:
        """
        Remove molecule(s) from the system

        Parameters
        ----------
        molecule: Molecule
            The `Molecule` object to remove from the setup
        """
        if molecule not in self._molecules:
            raise ValueError("This molecule does not exist in the setup.")
        else:
            del self._molecule_settings[molecule]
            self._molecules.remove(molecule)

    def validate_setup(self):
        """Ensures that the setup is valid - shows errors and warnings for issues with the setup"""
        # The system tolerance must be set
        tol = self._system_settings["tolerance"]
        assert (tol is not None and tol > 0.), "The system tolerance must be set"

        # At least one type of molecule must be present
        assert len(self._molecules) >= 1, "There must be at least one type of molecule present"

        for mol, settings in self._molecule_settings.items():
            # Each molecule needs to have at least one "number" setting
            assert "number" in settings.keys(), \
                f"The number of {mol} molecules needs to be specified."
            # Each molecule must have at least one constraint
            assert np.any([self._is_constraint(key) for key in settings.keys]), \
                f"Molecule {mol} needs to have a spatial constraint attached to it."
            # Each molecule must have values for their respective constraints
            assert np.any([settings[key] is not None for key in settings.keys]), \
                f"Molecule {mol} has unfilled values for it's respective settings."

    def get_settings(self) -> tuple[dict, dict]:
        """Return the dictionaries of the system and per-molecule settings"""
        return self._system_settings, self._molecule_settings

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
        return setting_name in ["inside cube", "outside cube",
                                "inside box", "outside box", "fixed"]
