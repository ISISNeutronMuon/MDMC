"""
Module for the SolventConfig abstract class
"""

from abc import ABC, abstractmethod
from copy import deepcopy

import numpy as np

from MDMC.common.decorators import repr_decorator, unit_decorator,\
    unit_decorator_getter
from MDMC.common import units
from MDMC.MD import structures, interactions


@repr_decorator('description', 'box_dimensions', 'atom_types', 'molecule_name',
                'n_molecules', 'bonded_interactions', 'nonbonded_interactions')
class SolventConfig(ABC):

    """
    Abstract class defining solvent configs

    Classes that inherit from SolventConfig should be named SOLVENTConfig, where
    SOLVENT is the name of the solvent e.g. SPCConfig where SPC is the solvent

    Properties will we be modified are deepcopied, so that these aren't changed
    in the mutable solvent_config_dict reference. All other properties are
    properties are just references, to reduce memory usage.
    """

    def __init__(self):

        self._box_dimensions = deepcopy(
            self._solvent_config_dict['box_dimensions'])
        self._atom_types = deepcopy(self._solvent_config_dict['atom_types'])
        self._nonbonded_interactions = \
            deepcopy(self._solvent_config_dict['nonbonded_interactions'])
        self._molecules = deepcopy(self._solvent_config_dict['molecules'])

    def __str__(self):

        return self.description

    @property
    @abstractmethod
    def _solvent_config_dict(self):
        """
        Returns
        -------
        dict
            A dict with the description, box_dimensions, atom_types,
            bonded_interactions, nonbonded_interactions, constrained,
            molecule_name, and molecules
        """

        raise NotImplementedError

    @property
    def description(self):
        """
        Get a description of the solvent configuration

        Returns
        -------
        str
            The description of the SolventConfig
        """

        return self._solvent_config_dict['description']

    @property
    def box_dimensions(self):
        """
        Get or set the box dimensions in Ang

        Returns
        -------
        array
            The dimensions of the SolventConfig box in Ang
        """

        return self._box_dimensions

    @box_dimensions.setter
    @unit_decorator(unit=units.LENGTH)
    def box_dimensions(self, value):

        self._box_dimensions = value

    @property
    @unit_decorator_getter(unit=units.LENGTH ** 3)
    def volume(self):
        """
        Get the volume of the box in Ang^3

        Returns
        -------
        float
            The volume of the SolventConfig in Ang^3
        """

        return np.prod(self.box_dimensions)

    @property
    def atom_types(self):
        """
        Get or set the element: atom_type pairs in the SolventConfig

        Returns
        -------
        dict
            {element: atom_type} pairs for all atoms in the SolventConfig,
            where element is a str and atom_type is an int
        """

        return self._atom_types

    @atom_types.setter
    def atom_types(self, value):

        self._atom_types = value

    @property
    def bonded_interactions(self):
        """
        Get a dictionary of the bonded interactions that must be instantiated
        when solvating using the SolventConfig

        Returns
        -------
        list
            A list of lists, where each list contains a bonded_interaction (str)
            and one or more tuples containing two or more atom_names. Each
            atom_name_pair is a str which is a key in self.molecules
        """

        return self._solvent_config_dict['bonded_interactions']

    @property
    def nonbonded_interactions(self):
        """
        Get or set a dictionary of the nonbonded interactions that must be
        instantiated when solvating using the SolventConfig

        Returns
        -------
        list
            A list of lists, where each list contains a nonbonded_interaction
            (str) and either an atom_type or one or more tuples containing two
            atom_types. The atom_types must be ints that are values in
            self.atom_types
        """

        return self._nonbonded_interactions

    @nonbonded_interactions.setter
    def nonbonded_interactions(self, value):

        self._nonbonded_interactions = value

    @property
    def constrained(self):
        """
        Get whether or not the bonded interactions must be constrained when
        sovlating with this SolventConfig

        Returns
        -------
        bool
            True if the bonded interactions are constrained
        """

        return self._solvent_config_dict['constrained']

    @property
    def molecule_name(self):
        """
        Get the name of the solvent molecule

        Returns
        -------
        str
            The name of the solvent molecule
        """

        return self._solvent_config_dict['molecule_name']

    @property
    def molecules(self):
        """
        Get or set the specifications of the molecules that must be added to the
        Universe when solvating with this SolventConfig

        Returns
        -------
        dict
            {index: molecule} pairs, where index is an int and molecule is a
            dict of {atom_name: position} pairs, where atom_name is the name of
            the atom (which is a str specifying the element and a number) and
            position is a 3 element array specifying the position of the atom
        """

        return self._molecules

    @molecules.setter
    def molecules(self, value):

        self._molecules = value

    @property
    def n_molecules(self):
        """
        Get the number of molecules in the SolventConfig

        Returns
        -------
        int
            The number of molecules in the SolventConfig
        """

        return len(self.molecules)

    @property
    @unit_decorator_getter(unit=units.MASS)
    def mass(self):
        """
        Get the mass of a solvent molecule in amu

        Returns
        -------
        float
            The mass of the solvent molecule in amu
        """

        return self.molec_from_dict(list(self.molecules.values())[0]).mass

    @property
    @unit_decorator_getter(unit=units.MASS / (units.LENGTH ** 3))
    def density(self):
        """
        Get the density of the SolventConfig in

        Returns
        -------
        float
            The mass density of the SolventConfig in amu / Ang^3
        """

        return self.mass * self.n_molecules / self.volume

    def reset_molecules(self):
        """
        Resets the molecules dict to the original inbuilt dict
        """

        self.molecules = deepcopy(self._solvent_config_dict['molecules'])

    def offset_atom_types(self, offset):
        """
        Increments all atom_types (both in self.atom_types and
        self.nonbonded_interactions) by the offset

        Parameters
        ----------
        offset : int
            The amount by which the increment the atom_types
        """

        self.atom_types = {name: (value + offset) for name, value
                           in self.atom_types.items()}
        for nb_i in self._nonbonded_interactions:
            if nb_i[0] == 'Coulombic':
                nb_i[1] += offset
            else:
                nb_i[1] = [atom_type + offset for atom_type in nb_i[1]]

    def molec_from_dict(self, mol_dict, bonded_interactions=None,
                        universe=None):
        """
        Creates a Molecule object from a dictionaries containing atoms and
        atom_types

        Parameters
        ----------
        mol_dict : dict
            A dict of {atom_name : position} pairs, where atom_name is a str and
            position is a 3 element array
        bonded_interactions : dict, optional
            A dict of {interaction_name : atom_name} pairs, where both are str
        universe : Universe, optional
            The Universe to which the Molecule will be added, with the default
            being None

        Returns
        -------
        Molecule
            A Molecule object comprised of the atoms specified in mol_dict with
            the BondedInteractions in bonded_interactions applied
        """

        atoms = {}
        for name, position in mol_dict.items():
            elem = name.replace('1', '').replace('2', '')
            atoms[name] = structures.Atom(elem,
                                                position=position,
                                                atom_type=self.atom_types[elem],
                                                universe=universe)
        for b_i in bonded_interactions or []:
            # Get the required atom objects based on the atom names specified
            # for each bonded interaction in bonded_interactions
            atom_name_tuples = b_i[1:]
            atom_tuples = map(lambda atom_name_tuple: tuple(atoms[name] for name
                                                            in atom_name_tuple),
                              atom_name_tuples)
            b_i[0].atoms += atom_tuples

        return structures.Molecule(atoms=list(atoms.values()))

    def molecules_from_coords(self, coords, universe=None):
        """
        Creates Molecules from atomic coordinates and atom_types

        Parameters
        ----------
        coords : dict
            A dict of {ID : atom_coordinates} where ID is an int specifying the
            ID of the Molecule (unrelated to Structure.ID, purely for this
            dict) and atom_coordinates is a dict of {atom_name : position}
            pairs, where atom_name is a str and position is a 3 element array
        universe : Universe, optional
                The Universe to which the Molecules are added. The default is
                None.

        Returns
        -------
        molecules : list of Molecules
            Molecule objects generated from the atoms passed in coords, where
            each molecule position is its centre-of-mass.
        """

        if self.bonded_interactions:
            con = self.constrained
            # Initialises an object from the str specifying a BondedInteraction
            # at the start of each list in bonded_interactions. The rest of each
            # list (i.e. the atom names) are left unchanged.
            bonded_interactions = list(map(lambda b_i:
                                           [getattr(interactions,
                                                    el)(constrained=con)
                                            if not n else el
                                            for n, el in enumerate(b_i)],
                                           self.bonded_interactions))
        else:
            bonded_interactions = []

        molecules = []
        for mol_dict in coords.values():
            mol = self.molec_from_dict(mol_dict,
                                       bonded_interactions=bonded_interactions,
                                       universe=universe)
            molecules.append(mol)

        if self.nonbonded_interactions:
            for nb_i in self.nonbonded_interactions:
                # Different __init__ for Coulombic than other
                # NonBondedInteractions
                if nb_i[0] == 'Coulombic':
                    dummy = interactions.Coulombic(
                        universe=universe, atom_types=nb_i[1])
                else:
                    dummy = getattr(interactions, nb_i[0])(universe, *nb_i[1:])
        return molecules
