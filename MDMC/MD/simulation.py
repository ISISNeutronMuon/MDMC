# MDMC is a package for the optimisation of classical potentials with experimental data
# Copyright (C) 2026 MDMC Developers
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Module for setting up and running the simulation

Classes for the simulation box, minimizer and integrator."""

from __future__ import annotations

import logging
import warnings
from collections import defaultdict
from itertools import count, filterfalse, product
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from ase.io import read as ase_read
from MDANSE.Framework.Parsers.pdb import PDBFile
from statsmodels.tsa.stattools import kpss
from verbosemanager import VerboseManager

from MDMC.common import units
from MDMC.common.decorators import (
    mod_docstring,
    repr_decorator,
    unit_decorator,
    unit_decorator_getter,
)
from MDMC.MD.container import AtomContainer
from MDMC.MD.engine_facades.facade_factory import MDEngineFacadeFactory
from MDMC.MD.force_fields.force_field_factory import ForceFieldFactory
from MDMC.MD.interactions import Bond, BondAngle, Coulombic, Dispersion
from MDMC.MD.parameters import Parameters
from MDMC.MD.solvents.solvents import get_solvent_config, get_solvent_names
from MDMC.MD.structures import Atom, Molecule
from MDMC.trajectory_analysis.trajectory import Configuration

if TYPE_CHECKING:
    from pathlib import Path

    from MDMC.MD.interactions import Interaction
    from MDMC.MD.structures import Atom, Molecule, Structure
    from MDMC.trajectory_analysis.compact_trajectory import CompactTrajectory


LOGGER = logging.getLogger(__name__)
_FF_DOCSTRING = {"DYNAMIC_FORCE_FIELD_LIST": ", ".join(ForceFieldFactory.available_names())}

# pylint: disable=too-few-public-methods
# as many classes here are small MD engine compatibility


@repr_decorator(
    "dimensions",
    "kspace_solver",
    "electrostatic_solver",
    "dispersive_solver",
    "constraint_algorithm",
    "parameters",
)
@mod_docstring(_FF_DOCSTRING)
class Universe(AtomContainer):
    """
    Class where the configuration and topology are defined

    Parameters
    ----------
    dimensions : numpy.ndarray, list, float
        Dimensions of the ``Universe``, in units of ``Ang``. A `float` can be
        used for a cubic universe.
    force_fields : ForceField, optional
        A force field to apply to the Universe. The force fields available are:
        DYNAMIC_FORCE_FIELD_LIST. Default is None.
    structures : list, optional
        ``Structure`` objects contained in the ``Universe``. Default is None.
    **settings
        ``kspace_solver`` (`KSpaceSolver`)
            The k-space solver to be used for both electrostatic and dispersive
            interactions. If this is passed then no ``electrostatic_solver`` or
            ``dispersive_solver`` may be passed.
        ``electrostatic_solver`` (`KSpaceSolver`)
            The k-space solver to be used for electrostatic interactions.
        ``dispersive_solver`` (`KSpaceSolver`)
            The k-space solver to be used for dispersive interactions.
        ``constraint_algorithm`` (`ConstraintAlgorithm`)
            The constraint algorithm which will be applied to constrained
            ``BondedInteractions``.
        ``verbose`` (`str`)
            If the output of the class instantiation should be reported, default to True.

    Attributes
    ----------
    dimensions : numpy.ndarray, list, float
        Dimensions of the ``Universe`` in units of ``Ang``.
    configuration : Configuration
        Stores the content, i.e. configuration of atoms etc within the universe
    force_fields : ForceField or None
        Force field applied to apply to the Universe
    kspace_solver : KSpaceSolver
        The k-space solver to be used for both electrostatic and dispersive
        interactions.
    electrostatic_solver : KSpaceSolver
        The k-space solver to be used for electrostatic interactions.
    dispersive_solver : KSpaceSolver
        The k-space solver to be used for dispersive interactions.
    constraint_algorithm : ConstraintAlgorithm
        The constraint algorithm which will be applied to constrained
        ``BondedInteractions``.
    interactions
    bonded_interactions
    nonbonded_interactions
    bonded_interaction_pairs
    n_bonded
    n_nonbonded
    n_interactions
    parameters
    volume
    element_list
    element_dict
    element_lookup
    atoms
    n_atoms
    molecule_list
    n_molecules
    structure_list
    top_level_structure_list
    equivalent_top_level_structures_dict
    force_fields
    atom_types
    atom_type_interactions
    density
    solvent_density
    nbis_by_atom_type_pairs
    """

    def __init__(self, dimensions, force_field=None, structures=None, **settings):
        self.dimensions = dimensions
        self._parameters = None
        self._atom_types = defaultdict(list)
        self._atom_type_interactions = {}
        if structures:
            self.configuration = Configuration(structures)
        else:
            self.configuration = Configuration(universe=self)
        self._solvent_density = 0.0
        self._bonded_interaction_pairs = set()
        self._nonbonded_interactions = set()
        if force_field:
            self._force_fields = ForceFieldFactory.create(force_field)
        else:
            self._force_fields = None

        self.kspace_solver = settings.get("kspace_solver")
        self.electrostatic_solver = settings.get("electrostatic_solver")
        self.dispersive_solver = settings.get("dispersive_solver")

        self.verbose = settings.get("verbose", True)
        # kspace_solver is mutually exclusive with the other two solver
        # attributes
        if self.kspace_solver and (self.electrostatic_solver or self.dispersive_solver):
            msg = "No other solver may be passed if kspace_solver is passed."
            LOGGER.error(
                "%s has kspace_solver and %s. %s",
                self.__class__,
                "electrostatic_solver" if self.electrostatic_solver else "dispersive_solver",
                msg,
            )
            raise ValueError(msg)

        self.constraint_algorithm = settings.get("constraint_algorithm")

        LOGGER.info(r"%s created: {dimensions:%s}", self.__class__, self.dimensions)

        if self.verbose:
            setup_frame = "Universe created with:"
            if self.dimensions is not None:
                setup_frame += f"\nDimensions {np.round(self.dimensions, 2)}"
            if force_field is not None:
                setup_frame += f"\nForce field    {force_field}"
            if self.n_atoms > 0:
                setup_frame += f"\nNumber of atoms    {self.n_atoms}"
            print(setup_frame)

    def __str__(self) -> str:
        return (
            f"Universe with {self.n_atoms} atoms, {self.n_bonded} bonded interactions, "
            f"{self.n_nonbonded} nonbonded interactions, and dimensions of {self.dimensions}"
        )

    def __eq__(self, other) -> bool:
        if id(other) == id(self):
            return True
        if isinstance(other, self.__class__):
            for k, v in self.__dict__.items():
                try:
                    iter(v)
                    if any(v != getattr(other, k)):
                        return False
                except TypeError:
                    if v != getattr(other, k):
                        return False
            return True
        return False

    # Unit decorator on getter due to operations in setter

    @classmethod
    def from_file(self, file_path: str | Path, file_format: str | None = None) -> Universe:
        init_struct = ase_read(file_path, format=file_format)
        new_instance = Universe(np.diag(init_struct.get_cell()))
        for atom in init_struct:
            new_instance.add_structure(
                Atom(atom.symbol, position=atom.position, charge=atom.charge)
            )
        return new_instance

    @classmethod
    def from_pdb_file(
        self,
        file_path: str | Path,
        atom_type_mapping: dict[str, Any] | None = None,
        bonds_per_molecule: dict[str, tuple[str, str]] | None = None,
        angles_per_molecule: dict[str, tuple[str, str, str]] | None = None,
    ) -> Universe:
        init_struct = ase_read(file_path)
        atom_aliases = {atom.symbol: atom.symbol for atom in init_struct}
        mdanse_parser = PDBFile(file_path)
        mdanse_parser.parse()
        topology = mdanse_parser.build_chemical_system(atom_aliases)
        all_indices = topology.all_indices
        atom_types = topology.atom_list
        molecules = topology._clusters
        all_names = topology.name_list
        index_to_label = {
            index: label for label, value in topology._labels.items() for index in value
        }

        if atom_type_mapping is None:
            atom_type_mapping = {name_string: {} for name_string in all_names}

        bond_pairs = {}
        for mol_label, pair_list in bonds_per_molecule.items():
            for pair in pair_list:
                bond_pairs[(mol_label, pair)] = []
        angle_triplets = {}
        for mol_label, triplet_list in angles_per_molecule.items():
            for triplet in triplet_list:
                angle_triplets[(mol_label, triplet)] = []
        unique_atoms = {}

        new_instance = Universe(np.diag(init_struct.get_cell()))
        for molecule_type, mol_list in molecules.items():
            for atom_list in mol_list:
                mol_type = set(index_to_label[index] for index in atom_list)
                if len(mol_type) > 1:
                    raise ValueError(
                        f"Atoms {atom_list} in molecule {molecule_type} "
                        f"have multiple PDB labels {mol_type}"
                    )
                mol_type = mol_type.pop()
                bond_list = bonds_per_molecule[mol_type]
                angle_list = angles_per_molecule[mol_type]
                atom_dict = {}
                for ind in atom_list:
                    if (mol_type, all_names[ind]) not in unique_atoms:
                        atom_instance = Atom(
                            atom_types[ind],
                            position=init_struct[ind].position,
                            charge=init_struct[ind].charge,
                            **atom_type_mapping[all_names[ind]],
                        )
                        unique_atoms[(mol_type, all_names[ind])] = atom_instance
                    else:
                        atom_instance = unique_atoms[(mol_type, all_names[ind])].copy(
                            position=init_struct[ind].position,
                        )
                    atom_dict[all_names[ind]] = atom_instance
                mol_settings = {
                    "atoms": list(atom_dict.values()),
                    "name": molecule_type,
                }
                new_instance.add_structure(Molecule(**mol_settings))
                for at_pair in bond_list:
                    bond_pairs[(mol_type, at_pair)].append(
                        tuple(atom_dict[at_key] for at_key in at_pair)
                    )
                for at_triplets in angle_list:
                    angle_triplets[(mol_type, at_triplets)].append(
                        tuple(atom_dict[at_key] for at_key in at_triplets)
                    )
                all_indices -= set(atom_list)
        for tuple_list in bond_pairs.values():
            Bond(*tuple_list)
        for triplet_list in angle_triplets.values():
            BondAngle(*triplet_list)
        for atom_index in all_indices:
            atom = init_struct[atom_index]
            if atom.symbol not in unique_atoms:
                atom_instance = Atom(atom.symbol, position=atom.position, charge=atom.charge)
                unique_atoms[atom.symbol] = atom_instance
            else:
                atom_instance = unique_atoms[atom.symbol].copy(position=atom.position)
            new_instance.add_structure(atom_instance)
        return new_instance

    @property
    @unit_decorator_getter(unit=units.LENGTH)
    def dimensions(self) -> np.ndarray:
        """
        Get or set the dimensions of the ``Universe``

        Raises
        ------
        TypeError
            If a `float`, `list`, or ``array`` are not passed
        ValueError
            If a `list` or ``array`` is not 1d with 3 elements
        Returns
        -------
        numpy.ndarray
            The dimensions of the ``Universe``
        """

        return self._dimensions

    @dimensions.setter
    def dimensions(self, dimensions: npt.ArrayLike) -> None:
        if isinstance(dimensions, float):
            if dimensions <= 0:
                msg = "Only positive values for the Universe dimensions are currently supported."
                LOGGER.error("%s: {dimensions: %s} %s", self.__class__, dimensions, msg)
                raise ValueError(msg)
            self._dimensions = np.array([dimensions] * 3)
        elif isinstance(dimensions, (list, tuple, np.ndarray)):
            if len(dimensions) == 3:
                if any(dim <= 0 for dim in np.array(dimensions)):
                    msg = (
                        "Only positive values for the Universe dimensions are currently supported."
                    )
                    LOGGER.error("%s: {dimensions: %s} %s", self.__class__, dimensions, msg)
                    raise ValueError(msg)
                self._dimensions = np.array(dimensions)
            else:
                msg = "3 dimensions must be specified"
                LOGGER.error("%s: {dimensions: %s} %s", self.__class__, dimensions, msg)
                raise ValueError(msg)
        else:
            msg = "dimensions must be a float or 3 element list of floats."
            LOGGER.error("%s: {dimensions: %s} %s", self.__class__, dimensions, msg)
            raise TypeError(msg)

    @property
    def interactions(self) -> list:
        """
        Get the interactions in the ``Universe``

        Returns
        -------
        list
            The interactions in the ``Universe``
        """

        return self.bonded_interactions + self.nonbonded_interactions

    @property
    def bonded_interactions(self) -> list:
        """
        Get the bonded interactions in the ``Universe``

        Returns
        -------
        list
            The bonded interactions in the ``Universe``
        """

        return [pair[0] for pair in self.bonded_interaction_pairs]

    @property
    def nonbonded_interactions(self) -> list:
        """
        Get the nonbonded interactions in the ``Universe``

        Returns
        -------
        list
            The nonbonded interactions in the ``Universe``
        """

        return list(self._nonbonded_interactions)

    @property
    def bonded_interaction_pairs(self) -> list:
        """
        Get the bonded interactions and the atoms they apply to

        Returns
        -------
        list
            The (``interaction``, ``atoms``) pairs in the ``Universe``, where
            ``atoms`` is a `tuple` of all ``Atom`` objects for that specific
            ``interaction``

        Example
        -------
        For an O Atom with two bonds, one to H1 and one to H2::

            >>> print(O.bonded_interaction_pairs)
            [(Bond, (H1, O)), (Bond, (H2, O))]
        """

        # bonded_interaction_pairs is a set to avoid double counting of
        # interactions
        return list(self._bonded_interaction_pairs)

    @property
    def n_bonded(self) -> int:
        """
        Get the number of bonded interactions in the ``Universe``

        Returns
        -------
        int
            The number of bonded interactions in the ``Universe``
        """

        return len(self.bonded_interactions)

    @property
    def n_nonbonded(self) -> int:
        """
        Get the number of nonbonded interactions in the ``Universe``

        Returns
        -------
        int
            The number of nonbonded interactions in the ``Universe``
        """

        return len(self.nonbonded_interactions)

    @property
    def n_interactions(self) -> int:
        """
        Get the number of interactions in the ``Universe``

        Returns
        -------
        int
            The number of interactions in the ``Universe``
        """

        return self.n_bonded + self.n_nonbonded

    @property
    def parameters(self) -> Parameters:
        """
        Get the parameters of the interactions that exist within the
        ``Universe``

        Returns
        -------
        Parameters
            The ``Parameters`` objects defined within ``Universe``
        """
        if self._parameters is None:
            self._parameters = Parameters(
                [
                    parameter
                    for interaction in self.interactions
                    for parameter in interaction.parameters.as_array
                ],
            )
        return self._parameters

    @parameters.setter
    def parameters(self, input_list):
        self._parameters = Parameters(input_list)

    @property
    @unit_decorator_getter(unit=units.LENGTH**3)
    def volume(self) -> float:
        """
        Get the volume of the ``Universe``

        Returns
        -------
        float
            Volume in ``Ang^3``
        """

        return np.prod(self.dimensions)

    @property
    def element_list(self) -> list[str]:
        """
        The elements of the atoms in the ``Universe``

        Returns
        -------
        list
            The elements in the ``Universe``
        """

        return [atom.element.symbol for atom in self.atoms]

    @property
    def element_dict(self) -> dict[str, Atom]:
        """
        Get the elements in the ``Universe`` and example ``Atom`` objects for
        each element

        This is required for MD engines which assign the same potential
        parameters for all identical element types.

        Returns
        -------
        dict
            ``element:atom pairs``, where ``atom`` is a single ``Atom`` of the
            specified ``element``.

        """

        return {atom.element.symbol: atom for atom in self.atoms}

    @property
    def element_lookup(self) -> dict[str, Atom]:
        """
        Get the elements by ID in the ``Universe``

        This is required for MD engines which assign the same potential
        parameters for all identical element names.

        Returns
        -------
        dict
            ``element:atom pairs``, where ``atom`` is a single ``Atom`` of the
            specified ``element``.

        """

        return {atom.atom_type: atom.element.symbol for atom in self.atoms}

    @property
    def atoms(self) -> list[Atom]:
        """
        Get a list of the atoms in the ``Universe``

        Returns
        -------
        list
            The atoms in the ``Universe``
        """

        return self.configuration.atoms

    @property
    def n_atoms(self) -> int:
        """
        Get the number of atoms in the ``Universe``

        Returns
        -------
        int
            The number of atoms in the ``Universe``
        """

        return len(self.atoms)

    @property
    def molecule_list(self) -> list[Molecule]:
        """
        Get a list of the ``Molecule`` objects in the ``Universe``

        Returns
        -------
        list
            The ``Molecule`` objects in the ``Universe``
        """

        return self.configuration.molecule_list

    @property
    def n_molecules(self) -> int:
        """
        Get the number of molecules in the ``Universe``

        Returns
        -------
        int
            The number of molecules in the ``Universe``
        """

        return len(self.molecule_list)

    @property
    def structure_list(self) -> None:
        """
        Get a `list` of all ``Structure`` objects that exist in the
        ``Universe``.  This includes all ``Structure`` that are a subunit
        of another structure belonging to the ``Universe``.


        Returns
        -------
        list
            The ``Structure`` objects in the ``Universe``
        """

        def add_all_parents(unit):
            current = unit
            parent = unit.parent
            parents = [parent]
            while parent is not parent.top_level_structure:
                parent = current.parent
                parents.append(parent)
                current = parent
            return parents

        structures = {parent for atom in self.atoms for parent in add_all_parents(atom)}
        structures.update(self.atoms)

        return list(structures)

    @property
    def top_level_structure_list(self) -> list[Structure]:
        """
        Get a `list` of the top level ``Structure`` objects that exist in the
        ``Universe``. This does not include any ``Structure`` that are a subunit
        of another structure belonging to the ``Universe``.


        Returns
        -------
        list
            The top level ``Structure`` objects in the ``Universe``
        """

        # Remove duplicate entries from multiple atoms belonging to the same molecule
        structures = {atom.top_level_structure for atom in self.atoms}

        # and sort by ID for consistency
        # Sorted converts to list
        top_level_structures = sorted(structures, key=lambda x: x.ID)
        return top_level_structures

    @property
    def equivalent_top_level_structures_dict(self) -> dict[Structure, int]:
        """
        Get a `dict` of equivalent top level ``Structure`` objects that
        exist in the ``Universe`` as keys, and the number of ``Structure``
        that are equivalent to the key as a value. This does not include any
        ``Structure`` that are a subunit of another structure belonging
        to the ``Universe``.


        Returns
        -------
        dict
            The top level ``Structure`` objects in the ``Universe`` as
            keys, and the number of each as a value
        """

        equivalent_dict = {}
        for structure in self.top_level_structure_list:
            match = False
            for key in equivalent_dict:
                if structure.is_equivalent(key):
                    equivalent_dict[key] += 1
                    match = True
                    break
            if not match:
                equivalent_dict[structure] = 1

        return equivalent_dict

    @property
    @mod_docstring(_FF_DOCSTRING)
    def force_fields(self) -> None:
        """
        Get the ``ForceField`` acting on the ``Universe``

        The available force fields are:
        DYNAMIC_FORCE_FIELD_LIST

        Returns
        -------
        list
            ``ForceField`` that applies to the ``Universe``
        """

        return self._force_fields

    @property
    def atom_types(self) -> dict[int, list[Atom]]:
        """
        Get the atom types of atoms in the ``Universe``

        Returns
        -------
        list
            The atom types in the ``Universe``
        """

        return self._atom_types

    @property
    def atom_type_interactions(self) -> dict[int, list[Interaction]]:
        """
        Get the atom types and the interactions for each atom type

        Returns
        -------
        dict
            ``atom_type:interactions`` pairs where ``atom_type`` is a `int`
            specifying the atom type and ``interactions`` is a `list` of
            ``Interaction`` objects acting on that ``atom_type``
        """

        return self._atom_type_interactions

    @property
    @unit_decorator_getter(unit=units.MASS / units.LENGTH**3)
    def density(self) -> float:
        """
        Get the mass density of the ``Universe``

        Returns
        -------
        float
            The mass density of all of the atoms of the ``Universe``
        """

        return sum(atom.mass for atom in self.atoms) / self.volume

    @property
    @unit_decorator_getter(unit=units.MASS / units.LENGTH**3)
    def solvent_density(self) -> float:
        """
        Get the mass density of a solvent added using ``solvate()``

        Returns
        -------
        float
            The mass density of a solvent added using sovlate
        """

        return self._solvent_density

    def _update_atom_types(self, atom: Atom) -> None:
        """
        Adds the atom to ``atom_types`` `dict`

        ``atom_type`` is decided on the following criteria:

          - If an ``Atom`` has the same name and element as an ``Atom`` already
            in the ``Universe``, it will be assigned the same ``atom_type``.
          - If an ``Atom`` does not have a name (e.g. ``Atom.name == None``),
            but has the same element and the same ``Interaction`` objects as an
            ``Atom`` already in the ``Universe``, it will be assigned the same
            ``atom_type``.
          - If an ``Atom`` does not fulfil either of the other criteria, it will
            be assigned the lowest integer which is not already an ``atom_type``
            in the ``Universe``

        Parameters
        ----------
        atom : Atom
            An ``Atom`` to add to the ``atom_types`` `dict`
        """

        if atom.name:
            inter_key = (atom.element.symbol, atom.name)
        else:
            # Sorting is just to ensure consistent order. As interactions will have
            # different types, sort by id
            inter_key = (atom.element.symbol,) + tuple(sorted(atom.interactions, k=id))

        if atom.atom_type:
            atom_type = atom.atom_type
            if atom_type not in self.atom_types:
                self._update_atom_type_interactions(inter_key, atom_type)
        else:
            try:
                atom_type = self.atom_type_interactions[inter_key]
            except KeyError:
                # Get lowest missing interger in self.atom_type_interactions
                atom_type = next(
                    filterfalse(set(self.atom_type_interactions.values()).__contains__, count(1)),
                )
                self._update_atom_type_interactions(inter_key, atom_type)
            atom.atom_type = atom_type
        self._atom_types[atom_type].append(atom)

    def _update_atom_type_interactions(self, key: tuple, atom_type: int) -> None:
        """
        Adds a new ``key:atom_type`` to ``atom_type_interactions``, if the
        ``key`` does not already exist

        Parameters
        ----------
        key : tuple
            ``(element, *interactions)``, where ``element`` is a `str`
            specifying the atomic element, and ``*interactions`` is one or more
            ``Interaction`` objects
        atom_type : int
            The atom type

        Raises
        ------
        TypeError
            If an assignment is made to an ``atom_type`` which already has
            associated interactions
        """

        if key not in self.atom_type_interactions:
            self.atom_type_interactions[key] = atom_type
        else:
            msg = (
                "assignments cannot be made to atom_type_interactions keys"
                "which already possess values."
            )
            LOGGER.error(
                "%s: {key: %s, atom_type_interactions: %s} %s",
                self.__class__,
                key,
                self.atom_type_interactions,
                msg,
            )
            raise TypeError(msg)

    @mod_docstring(_FF_DOCSTRING)
    def add_structure(
        self,
        structure: Structure | int,
        force_field: str | None = None,
        center: bool = False,
    ) -> None:
        """
        Adds a single ``Structure`` to the ``Universe``, with optional
        ``ForceField`` applying only to that ``Structure``

        Parameters
        ----------
        structure : Structure or int
            The ``Structure`` (or its ``ID`` as an `int`) to be added to the ``Universe``
        force_field : str, optional
            The force field to be applied to the structure. The available
            ``ForceField`` are:
            DYNAMIC_FORCE_FIELD_LIST
        center : bool, optional
            Whether to center `structure` within the Universe as it is
            added
        """

        if center:
            structure.position = self.dimensions / 2.0
        structure.universe = self
        self.configuration.add_structure(structure)
        for atom in structure.atoms:
            self.add_bonded_interaction_pairs(*atom.bonded_interaction_pairs)
            self.add_nonbonded_interaction(*atom.nonbonded_interactions)
            self._update_atom_types(atom)

        if force_field:
            self.add_force_field(force_field, *structure.interactions)

    @mod_docstring(_FF_DOCSTRING)
    def fill(
        self,
        structures: Structure,
        force_field: str = None,
        num_density: float = None,
        num_struc_units: int = None,
    ) -> None:
        """
        A liquid-like filling of the ``Universe`` independent of existing atoms

        Adds copies of ``structures`` to existing configuration until
        ``Universe`` is full.  As exclusion region is defined by the size of a
        bounding sphere, this method is most suitable for atoms or molecules
        with approximately equal dimensions.

        .. note:: CURRENT APPROACH RESULTS IN NUMBER DENSITY DIFFERENT TO WHAT
                  IS SPECIFIED DEPENDING ON HOW CLOSE CUBE ROOT OF N_MOLECULES
                  IS TO AN `int`.

        .. note:: CURRENT IMPLEMENTATION SHOULD NOT BE USED WITH NON-CUBIC
                  UNIVERSES AS THE DENSITY MAY OR MAY NOT BE ISOTROPIC
                  DEPENDING ON THE DIMENSIONS AND NUMBER OF UNITS.

        Parameters
        ----------
        structures : Structure or int
            The ``Structure`` with which to fill the ``Universe``
        force_field : str
            Applies a ``ForceField`` to the ``Universe``. The available
            ``ForceField`` are:
            DYNAMIC_FORCE_FIELD_LIST
        num_density: float
            Non-negative `float` specifying the number density of the
            ``Structure``, in units of ``Structure / Ang ^ -3``
        num_struc_units: int
            Non-negative `int` specifying the number of passed
            ``Structure`` objects that the universe should be filled
            with, regardless of ``Universe.dimensions``.

        Raises
        ------
        ValueError
            If both ``num_density`` and ``num_struc_units`` are passed
        ValueError
            If neither ``num_density`` or ``num_struc_units`` are passed
        """

        if num_density is None and num_struc_units is not None:
            num_density = num_struc_units / np.prod(self.dimensions)
        elif num_density is not None and num_struc_units is not None:
            msg = "Cannot pass both num_density and num_struc_units to fill the universe with."
            LOGGER.error(
                "%s: {num_density: %s, num_struc_units: %s} %s",
                self.__class__,
                num_density,
                num_struc_units,
                msg,
            )
            raise ValueError(msg)
        elif num_density is None and num_struc_units is None:
            msg = "The fill method takes either num_density or num_struc_units as a parameter."
            LOGGER.error("%s %s", self.__class__, msg)
            raise ValueError(msg)

        n_units_xyz = self.dimensions * (num_density ** (1.0 / 3.0))
        n_units_xyz = n_units_xyz.astype(int)

        # Determine the upper and lower bounds for structural unit with its
        # position (CoM) and its bounding box
        bounds = structures.bounding_box
        mn = np.array((0.0, 0.0, 0.0)) - (bounds.min - structures.position)
        mx = self.dimensions - (bounds.min - structures.position)
        positions = [
            np.linspace(mi, ma, n_units, endpoint=False)
            for mi, ma, n_units in zip(mn, mx, n_units_xyz)
        ]
        positions = sorted(product(*positions))

        # Add the first structural unit and force field (if specified) before
        # copying the structural unit to fill the universe
        self.add_structure(structures, force_field)
        structures.position = positions.pop(0)
        for position in positions:
            new_unit = structures.copy(position)
            self.add_structure(new_unit)

    def set_atom_charge(self, atom_name: str = "", charge=0.0):
        if not atom_name:
            for atom in self.atoms:
                atom.charge = charge
            return
        for atom in self.atoms:
            if atom.name == atom_name:
                atom.charge = charge

    def update_charges(self, charge_parameters: Parameters):
        all_indices = set(range(1, 1 + len(self.atoms)))
        for molecule in self.molecule_list:
            temp_parameters = charge_parameters.filter_molecule(molecule.name).values()
            new_values = [par.value for par in temp_parameters]
            fixed_status = [par.fixed for par in temp_parameters]
            element_names = [par.elements[0] for par in temp_parameters]
            molecule.update_charges(new_values, fixed_status, element_names)
            all_indices -= molecule.indices
        for atom in self.atoms:
            if atom.ID not in all_indices:
                continue
            temp_parameters = charge_parameters.filter_element(atom.element).values()
            if len(temp_parameters) > 1:
                for par in temp_parameters:
                    if par.molecules:
                        continue
                    else:
                        real_par = par
                        break
            elif len(temp_parameters) == 0:
                continue
            else:
                real_par = temp_parameters[0]
            new_value = real_par.value
            fixed_status = real_par.fixed
            if not fixed_status:
                atom.charge = new_value

    @mod_docstring(_FF_DOCSTRING)
    def add_force_field(
        self,
        force_field: str,
        *interactions: Interaction,
        **settings: Any,
    ) -> None:
        """
        Adds a force field to the specified ``interactions``.  If no
        ``interactions`` are passed, the force field is applied to all
        interactions in the ``Universe``.

        Parameters
        ----------
        force_field : str
            The ``ForceField`` to parameterize ``interactions`` (if provided),
            or all the ``interactions`` in the ``Universe``. The available
            ``ForceField`` are:
            DYNAMIC_FORCE_FIELD_LIST
        *interactions
            ``Interaction`` objects to parameterize with the ``ForceField``
        **settings
            ``add_dispersions`` (`bool` or `list` of ``Atoms``)
                If `True`, a ``Dispersion`` interaction will be added to all
                atoms in the ``Universe``. If a list of ``Atom`` objects is
                provided, the ``Dispersion`` will be added to these instead. Any
                added ``Dispersion`` interactions (and any previously defined)
                will then be parametrized by the ``ForceField``. The
                ``Dispersion`` interactions added will only be like-like. By
                default, no ``Dispersion`` interactions are added.
        """

        self._force_fields = ForceFieldFactory.create(force_field)
        add_dispersions = settings.get("add_dispersions", False)

        if add_dispersions:
            if isinstance(add_dispersions, list):
                atoms = add_dispersions
            elif isinstance(add_dispersions, bool):
                atoms = self.atoms
            else:
                msg = (
                    "add_dispersions parameter of add_force_field method"
                    "must be a list of Atoms or a bool."
                )
                LOGGER.error("%s: {add_dispersions: %s}", self.__class__, add_dispersions)
                raise TypeError(msg)
            # Get unique atom types and add dispersions for each of these
            atom_types = {atom.atom_type for atom in atoms}
            dispersions = [Dispersion(self, (atom_type,) * 2) for atom_type in atom_types]

        if not interactions:
            self.force_fields.parameterize_interactions(set(self.interactions))
        else:
            if add_dispersions:
                interactions = interactions + tuple(dispersions)
            self.force_fields.parameterize_interactions(set(interactions))

        # FileForceFields also contain mass definitions for atoms, so set these
        try:
            for atom in self.atoms:
                self.force_fields.set_atom_mass(atom)
        except AttributeError:
            pass

    def add_bonded_interaction_pairs(
        self,
        *bonded_interaction_pairs: tuple[Interaction, Atom],
    ) -> None:
        """
        Adds one or more interaction pairs to the ``Universe``

        Parameters
        ----------
        *bonded_interaction_pairs
            one or more (``interaction``, ``atoms``) pairs, where ``atoms`` is a
            `tuple` of all atoms for that specific bonded ``interaction``
        """

        self._bonded_interaction_pairs.update(bonded_interaction_pairs)

    def add_nonbonded_interaction(self, *nonbonded_interactions: tuple[Interaction, Atom]) -> None:
        # ignore line too long linting as it is necessary for URL formatting
        # pylint: disable=line-too-long
        """
        Adds one or more nonbonded interactions to the ``Universe``

        Parameters
        ----------
        *nonbonded_interactions
            Nonbonded interactions to be added to the ``Universe``.
            Can take any number of non-bonded interactions:
                ``Dispersion()``:
                    either Lennard-Jones or Buckingham dispersion
                ``Coulombic()``:
                    normal or modified Coulomb interaction
            with appropriate parameters for the interaction.
            See http://mdmcproject.org/how-to/use-MDMC/notebooks/defining-molecule-interactions.ipynb
            for more details on non-bonded interactions.
        """

        # Since self._nonbonded_interactions is a set the update method only adds new interactions
        self._nonbonded_interactions.update(nonbonded_interactions)

    @property
    def nbis_by_atom_type_pairs(self) -> dict[tuple[int], list[Interaction]]:
        """
        Generates a `dict` of all nonbonded interactions possessed by all
        combinations of ``atom_type`` pairs in the ``Universe``.

        Returns
        -------
        dict
            A `dict` of ``{pair: interactions}`` pairs, where:
                - ``pair`` is a `tuple` for a pair of ``atom_types`` in the
                  ``Universe``
                - ``interactions`` is a list of nonbonded interactions that
                  exist for this ``pair`` of ``atom_types``

            Any ``Dispersions`` in ``interactions`` are ones that exist
            explicity for this ``pair``, whereas any ``Coulombics`` in
            ``interactions`` are ones that exist for either of the
            ``atom_types`` in ``pair``.
        """

        pairs_interactions = {}
        # Find pairwise combinations of N atom_types in the universe
        # Where each pair (i, j) has 0 < i, j <= N   and    i <= j
        atom_type_pairs = ((i, j) for i, j in product(self.atom_types, repeat=2) if i <= j)
        # Create dict of interactions for each atom type pair
        for pair in atom_type_pairs:
            pairs_interactions[pair] = [
                inter
                for inter in self.interactions
                if (isinstance(inter, Dispersion) and pair in inter.atom_types)
                or (isinstance(inter, Coulombic) and any(elem in inter.atom_types for elem in pair))
            ]

        return pairs_interactions

    def _check_out_of_bounds(self, position: list | np.ndarray) -> bool:
        """
        Checks whether a ``position`` lies outside the bounds of the
        ``Universe``

        Parameters
        ----------
        position : list, array
            The position to be checked against bounds of the ``Universe``.

        Returns
        -------
        bool
            `True` if the ``position`` passed falls outside the ``Universe``,
            `False` otherwise.
        """

        return any(position > self.dimensions) or any(position < [0, 0, 0])

    @property
    def unique_atom_types(self):
        """Set of unique atom types in the system"""
        return set(atom.atom_type for atom in self)

    @property
    def unique_molecule_types(self):
        """Set of unique atom types in the system"""
        return set(molecule.name for molecule in self.molecule_list)

    @mod_docstring({"DYNAMIC_SOLVENT_LIST": ", ".join(get_solvent_names())})
    def solvate(
        self,
        density: float,
        tolerance: float = 1.0,
        solvent: str = "SPCE",
        max_iterations: int = 100,
        **settings: Any,
    ) -> None:
        """
        Fills the ``Universe`` with solvent molecules according to pre-defined
        coordinates.

        Parameters
        ----------
        density : float
            The desired density of the ``Solvent`` that solvates the
            ``Universe``, in units of ``amu Ang ^ -3``
        tolerance : float, optional
            The +/- percentage tolerance of the density to be achieved.
            The default is 1 %. Tolerances of less than 1 % are at risk
            of not converging.
        solvent : str, optional
            A `str` specifying an inbuilt ``Solvent`` from the following:
            DYNAMIC_SOLVENT_LIST.
            The default is 'SPCE'.
        max_iterations: int, optional
            The maximum number of times to try to solvate the universe to
            within the required density before stopping. Defaults to 100.
        **settings
            ``constraint_algorithm`` (`ConstraintAlgorithm`)
                A ``ConstraintAlgorithm`` which is applied to the ``Universe``.
                If an inbuilt ``Solvent`` is selected (e.g. 'SPCE') and
                ``constraint_algorithm`` is not passed, the
                ``ConstraintAlgorithm`` will default to ``Shake(1e-4, 100)``.

        Raises
        ------
        ValueError
            If the ``Universe`` has already been solvated with a different
            density.
        """

        solvent_config = get_solvent_config(solvent)

        # Calculate useful properties from the original box
        solvent_mass = solvent_config.mass
        orig_box_dimensions = solvent_config.box_dimensions
        # density is adjusted to account for density of solvent already in box
        density = density - self.solvent_density
        # If this is already within the specified tolerance then return, as
        # calling solvate is redundant. Otherwise, raise an error, as solvate is
        # not designed to be applied multiple times to change the
        # solvent_density of a Universe.
        if abs(density * 100) <= abs(tolerance):
            return
        if self.solvent_density != 0.0:
            msg = (
                "The universe has already been solvated. The density of a"
                " previously added solvent cannot be changed."
            )
            LOGGER.error(
                "%s: {solvent_density: %s, density: %s}. %s",
                self.__class__,
                self.solvent_density,
                density,
                msg,
            )
            raise ValueError(msg)
        # Get the prelim scaling of the orig box required to achieve density
        dim_scaling = np.array([(solvent_config.density / density) ** (1.0 / 3.0)] * 3)

        scale_factor = 0.0
        counter = 0
        # Offset the atom_types of the solvent_config by the maximum atom_type
        # in the Universe.
        max_atom_type = max(self.atom_types.keys(), default=0)

        solvent_config.offset_atom_types(max_atom_type)
        difference = float("inf")

        while abs(difference * 100) >= abs(tolerance):
            counter += 1
            dim_scaling *= 1 + scale_factor
            box_dimensions = orig_box_dimensions * dim_scaling
            num_tiles = np.array(self.dimensions / box_dimensions)
            # Binary list for axes along which whole num of tiles are used.
            wrap = np.array([1 if direc.is_integer() else 0 for direc in num_tiles])
            num_tiles = np.ceil(num_tiles).astype(int)

            mols = []
            for trans_vect in product(
                range(0, num_tiles[0]),
                range(0, num_tiles[1]),
                range(0, num_tiles[2]),
            ):
                solvent_config.reset_molecules()
                for mol_key, mol in tuple(solvent_config.molecules.items()):
                    atom_positions = mol.values()
                    CoM = solvent_config.molec_from_dict(mol).position

                    remove = False
                    for pos in atom_positions:
                        pos += dim_scaling * (CoM + trans_vect * orig_box_dimensions) - CoM
                        # Create binary list indicating the axes along
                        # which the atom is out of bounds.
                        axes = np.array([1 if i > j else 0 for i, j in zip(pos, self.dimensions)])
                        # Translates position if wrapping required.
                        pos -= wrap * axes * num_tiles * box_dimensions
                        remove = self._check_out_of_bounds(pos)
                        # Check for overlap with solute molecules.
                        if not remove:
                            for solute in self.molecule_list:
                                cond1 = all(pos > solute.bounding_box.min)
                                cond2 = all(pos < solute.bounding_box.max)
                                if cond1 and cond2:
                                    remove = True
                    if remove:
                        del solvent_config.molecules[mol_key]
                mols += solvent_config.molecules_from_coords(
                    solvent_config.molecules,
                    universe=self,
                )

            # Check the density
            actual = (len(mols) * solvent_mass) / self.volume
            difference = (actual - density) / density
            scale_factor = difference / counter
            if counter > max_iterations:
                msg = (
                    "100 Iterations have been tried but the universe is not solvated"
                    " to within tolerance. Try increasing the tolerance or Universe size"
                )
                LOGGER.error(msg)
                raise ValueError(msg)

        # Once the correct density is achieved, add molecules to universe
        # and get all bonded interactions
        # Also determine the total density of the solvent
        bonded_interactions = []
        for molecule in mols:
            self.add_structure(molecule)
            bonded_interactions += molecule.interactions

        # Get nonbonded interactions from atom types
        # Add interaction if any of its atom types are in atom_types
        nonbonded_interactions = []
        for interaction in self.nonbonded_interactions:
            inter_atom_types = np.array(interaction.atom_types).flatten()
            if len(set(inter_atom_types).intersection(solvent_config.atom_types.values())) >= 1:
                nonbonded_interactions.append(interaction)

        # Apply the force field of the solvent to the Universe
        try:
            self.add_force_field(solvent, *set(bonded_interactions + nonbonded_interactions))
            # If BondedInteractions are constrained, apply a constrain algorithm
            if solvent_config.constrained:
                self.constraint_algorithm = settings.get("constraint_algorithm", Shake(1e-4, 100))

            if self.verbose:
                print(f"Force field created by solvent {solvent}")

        except ImportError:
            pass

        self._solvent_density += len(mols) * solvent_mass / self.volume


class KSpaceSolver:
    """
    Class describing the k-space solver that is applied to electrostatic and/or
    dispersion interactions

    Different ``MDEngine`` require different parameters to be specified for a
    k-space solver to be used. These parameters are specified in settings.

    Parameters
    ----------
    **settings
        ``accuracy`` (`float`)
            The relative RMS error in per-atom forces

    Attributes
    ----------
    accuracy : float
        The relative RMS error in per-atom forces
    """

    def __init__(self, **settings: Any):
        self.accuracy = settings.get("accuracy")

    @property
    def name(self):
        """
        Get the name of the class

        Returns
        -------
        str
            The name of the class
        """

        return self.__class__.__name__


class Ewald(KSpaceSolver):
    """
    Holds the parameters that are required for the Ewald solver to be applied to
    both/either the electrostatic and/or dispersion interactions

    Parameters
    ----------
    **settings
        ``accuracy`` (`float`)
            The relative RMS error in per-atom forces
    """


class PPPM(KSpaceSolver):
    """
    Holds the parameters that are required for the PPPM solver to be applied to
    both/either the electrostatic and/or dispersion interactions

    Parameters
    ----------
    **settings
        ``accuracy`` (`float`)
            The relative RMS error in per-atom forces
    """

    def __eq__(self, other) -> bool:
        """
        Two KSpaceSolvers are equal if their __dict__ are equal
        """

        if not isinstance(other, self.__class__):
            return False
        return all(v == getattr(other, k) for k, v in self.__dict__.items())

    def __ne__(self, other) -> bool:
        return not self.__eq__(other)


class ConstraintAlgorithm:
    """
    Class describing the algorithm and parameters which are applied to
    constrain ``BondedInteraction`` objects

    Parameters
    ----------
    accuracy : float
        The accuracy (tolerance) of the applied constraints
    max_iterations : int
        The maximum number of iterations that can be used when calculating the
        additional force that is required to constrain the atoms to satisfy the
        constraints on the bonded interactions

    Attributes
    ----------
    accuracy : float
        The accuracy (tolerance) of the applied constraints
    """

    def __init__(self, accuracy: float, max_iterations: int):
        self.accuracy = accuracy
        self.max_iterations = max_iterations

    @property
    def name(self) -> str:
        """
        Get the name of the class

        Returns
        -------
        str
            The name of the class
        """

        return self.__class__.__name__

    @property
    def max_iterations(self) -> int:
        """
        Get or set the maximum number of iterations that can be used when
        calculating the additional force that is required to constrain the atoms
        to satisfy the constraints on the bonded interactions

        Returns
        -------
        int
            The maximum number of iterations
        """

        return self._max_iterations

    @max_iterations.setter
    def max_iterations(self, value: int) -> None:
        self._max_iterations = int(value)


class Shake(ConstraintAlgorithm):
    """
    Holds the parameters which are required for the SHAKE algorithm to be
    applied to the constrained interactions

    Parameters
    ----------
    accuracy : float
        The accuracy (tolerance) of the applied constraints
    max_iterations : int
        The maximum number of iterations that can be used when calculating the
        additional force that is required to constrain the atoms to satisfy the
        constraints on the bonded interactions
    """


class Rattle(ConstraintAlgorithm):
    """
    Holds the parameters which are required for the RATTLE algorithm to be
    applied to the constrained interactions

    Parameters
    ----------
    accuracy : float
        The accuracy (tolerance) of the applied constraints
    max_iterations : int
        The maximum number of iterations that can be used when calculating the
        additional force that is required to constrain the atoms to satisfy the
        constraints on the bonded interactions
    """


@repr_decorator("universe", "engine", "settings")
class Simulation:
    """
    Sets up the molecular dynamics engine and parameters for how it should run

    An ensemble is defined by whether a thermostat or barostat are present

    Parameters
    ----------
    universe : Universe
        The ``Universe`` on which the simulation is performed.
    traj_step : int
        How many steps the simulation should take between dumping each
        ``CompactTrajectory`` frame. Along with ``time_step`` determines
        the time separation of calculated variables such as energy.
    time_step : float, optional
        Simulation timestep in ``fs``. Default is 1.
    engine : str, optional
        The ``MDEngine`` used for the simulation. Default is ``'lammps'``.
    **settings
        ``temperature`` (`float`)
            Simulation temperature in ``K``. Note that velocities of atoms in
            the MD engine are set based on the ``temperature``. If all atoms in
            the ``universe`` have 0 velocity, then velocities for the MD engine
            will be randomly chosen from a uniform distribution and then scaled
            to give the correct temperature. If one or more velocities are set,
            then these will be scaled to give the correct temperature. In
            either case, only the velocities on the ``engine``, not the
            ``universe``, are affected.
        ``integrator`` (`str`)
            Simulation time integrator.
        ``lj_options`` (`dict`)
            ``option:value`` pairs for Lennard-Jones interactions.
        ``es_options`` (`dict`)
            ``option:value`` pairs for electrostatic interactions.
        ``thermostat`` (`str`)
            The name of the thermostat e.g. Nose-Hoover.
        ``barostat`` (`str`)
            The name of the barostat e.g. Nose-Hoover.
        ``pressure`` (`float`)
            Simulation pressure in ``Pa``. This is required if a barostat is
            passed.
        ``verbose`` (`str`)
            If the output of the class instantiation should be reported, default to True.

    Attributes
    ----------
    universe : Universe
        The ``Universe`` on which the simulation is performed.
    traj_step : int
        How many steps the simulation should take between dumping each
        ``CompactTrajectory`` frame. Along with ``time_step`` determines
        the time separation of calculated variables such as energy.
    engine : MDEngine, optional
        A subclass of ``MDEngine`` which provides the interface to the MD
        library. Default is ``'lammps'``.
    settings : dict
        The settings passed to the ``Simulation``.  See the Parameters section
        for details.
    time_step
    trajectory
    """

    # TODO: Potentially separate out universe and simulation setup
    def __init__(
        self,
        universe: Universe,
        traj_step: int,
        time_step: float = 1.0,
        engine: str = "lammps",
        **settings,
    ):
        self.universe = universe
        self.traj_step = traj_step
        self.time_step = time_step
        self.settings = settings
        self.engine = MDEngineFacadeFactory.create_facade(engine)
        self.engine.parent_simulation = self
        self.verbose = settings.get("verbose", True)
        self._setup()

        self.setup_msg = f"Simulation created with {engine} engine"
        if self.settings:
            settings_strings = "".join(
                [
                    f"{key}: {value} {units.SYSTEM.get(key.upper(), '')} \n"
                    for key, value in self.settings.items()
                ],
            )
            self.setup_msg += f" and settings:\n{str(settings_strings)}\n"
        if self.verbose:
            print(self.setup_msg)

    @property
    def time_step(self) -> float:
        """
        Get or set the simulation time step in ``fs``

        Returns
        -------
        `float`
            Simulation time step in ``fs``
        """

        return self._time_step

    @time_step.setter
    @unit_decorator(unit=units.TIME)
    def time_step(self, value: float) -> None:
        self._time_step = value

    @property
    def traj_step(self) -> int:
        """
        Get or set the number of simulation steps between saving the
        ``CompactTrajectory``

        Returns
        -------
        `int`
            Number of simulation steps that elapse between the
            ``CompactTrajectory`` being stored
        """

        return self._traj_step

    @traj_step.setter
    def traj_step(self, value: int) -> None:
        self._traj_step = value

    def _setup(self) -> None:
        """
        Creates a universe within the ``MDEngine`` with the equivalent
        configuration and topology to ``self.universe`` and defines the
        simulation conditions
        """

        self.engine.setup_universe(self.universe, **self.settings)
        self.engine.setup_simulation(**self.settings)

    def minimize(
        self,
        n_steps: int,
        minimize_every: int = 10,
        verbose: bool = False,
        output_log: str = None,
        work_dir: str = None,
        **settings: Any,
    ) -> None:
        """
        Performs an MD run intertwined with periodic structure relaxation.
        This way after a local minimum is found, the system is taken
        out of the minimum to explore a larger volume of the parameter
        space.

        Parameters
        ----------
        n_steps : int
            Total number of the MD run steps
        minimize_every: int, optional
            Number of MD steps between two consecutive minimizations
        verbose: bool, optional
            Whether to print statements when the minimization has been started and completed
            (including the number of minimization steps and time taken). Default is `False`.
        output_log: str, optional
            Log file for the MD engine to write to. Default is `None`.
        work_dir: str, optional
            Working directory for the MD engine to write to. Default is `None`.
        **settings
            ``etol`` (`float`)
                If the energy change between iterations is less than ``etol``,
                minimization is stopped. Default depends on engine used.
            ``ftol`` (`float`)
                If the magnitude of the global force is less than ``ftol``,
                minimization is stopped. Default depends on engine used.
            ``maxiter`` (`int`)
                Maximum number of iterations of a single structure
                relaxation procedure. Default depends on engine used.
            ``maxeval`` (`int`)
                Maximum number of force evaluations to perform. Default depends
                on engine used.
        """

        verbose_manager = VerboseManager.instance()
        # to match legacy use of verbose on this function (where verbose was bool) we use bool
        # and convert to int, corresponding to verbose levels 0 or 1; there is only one verbose
        # step in this function so verbose levels 2 or 3 would not provide extra information
        verbose_manager.start(1, verbose=int(verbose))

        verbose_manager.step(
            f"Running minimization every {minimize_every} steps in an MD run with {n_steps} steps",
        )
        self.engine.minimize(
            n_steps,
            minimize_every,
            output_log=output_log,
            work_dir=work_dir,
            **settings,
        )

        verbose_manager.finish("Minimization")

    def run(
        self,
        n_steps: int,
        equilibration: bool = False,
        verbose: bool = False,
        output_log: str = None,
        work_dir: str = None,
        **settings: Any,
    ) -> None:
        """
        Runs the MD simulation for the specified number of steps. Trajectories
        for the simulation are only saved when ``equilibration`` is `False`.
        Additionally running equilibration for an NVE system (neither barostat
        nor thermostat set) will temporarily apply a Berendsen thermostat (it
        is removed from the simulation after the run is completed).

        Parameters
        ----------
        n_steps : int
            Number of simulation steps to run
        equilibration : bool, optional
            If the run is for equilibration (`True`) or production (`False`).
            Default is `False`.
        verbose: bool, optional
            Whether to print statements upon starting and completing the run.
            Default is `False`.
        output_log: str, optional
            Log file for the MD engine to write to. Default is `None`.
        work_dir: str, optional
            Working directory for the MD engine to write to. Default is `None`.
        """

        process = "equilibration" if equilibration else "simulation"

        verbose_manager = VerboseManager.instance()
        # to match legacy use of verbose on this function (where verbose was bool) we use bool
        # and convert to int, corresponding to verbose levels 0 or 1; there is only one verbose
        # step in this function so verbose levels 2 or 3 would not provide extra information
        verbose_manager.start(1, verbose=int(verbose))
        verbose_manager.step(f"Running {process} for {n_steps} steps")

        if n_steps < self.traj_step:
            warnings.warn(
                f"Run length ({n_steps}) less than dump frequency ({self.traj_step}), "
                "run may not produce usable output.",
            )

        self.engine.run(
            n_steps=n_steps,
            equilibration=equilibration,
            output_log=output_log,
            work_dir=work_dir,
            **self.settings,
        )

        verbose_manager.finish(f"{process.capitalize()}")

    # pylint: disable=dangerous-default-value
    # this is flagged up by variables having a list as default value
    # but we don't mutate the list anywhere in the function, so
    # this is safe and has better readability
    def auto_equilibrate(
        self,
        variables: list[str] = ("temp", "pe"),
        eq_step: int = 10,
        window_size: int = 100,
        tolerance: float = 0.01,
        max_step=10000,
    ) -> tuple[int, dict]:
        """
        Equilibrate until the specified list of variables have stabilised.
        Uses the KPSS stationarity test to determine
        whether the variables are stationary in a given window.

        Parameters
        ----------
        variables: list[str], default ['temp', 'pe']
            The MD engine variables we use to monitor stability.
        eq_step: int, default 10
            The number of equilibration steps between each stability check.
        window_size: int, default 100
            The size of the rolling window of stability checks. The simulation
            is considered stable if it has been stationary for eq_step*window_size
            equilibration steps.
        tolerance: float, 0.01
            The p-value used for the KPSS test.

        Returns
        -------
        int
            The number of equilibration steps performed.
        vals_dict
            The value of each variable per step, for graphing if desired.
        """

        vals_dict = {var: [] for var in variables}

        def auto_step() -> None:
            """
            Performs one step of an equilibration and records variable values.
            """
            self.run(eq_step, equilibration=True)
            for var in vals_dict:
                vals_dict[var].append(self.engine.eval(var))

        def window_is_stationary(var: str) -> bool:
            """
            Checks stability for a variable by running the KPSS test on a window.
            """
            variable_values = vals_dict[var]
            window = variable_values[-window_size:]
            results = kpss(window, regression="c")
            # results[1] is the p-value from the test
            # we base our tolerance on the p-value, where the alternative hypothesis
            # for KPSS is "NOT stationary" - statsmodels also never gives a p above
            # 0.1 as it doesn't hold critical values above that point. so for 0.05
            # tolerance, we are asking that p be greater than 0.95
            return results[1] > 0.1 - tolerance

        # we perform an initial run of window_size*eq_step steps to create a window.
        for _ in range(window_size):
            auto_step()

        # we consider the simulation stable if the rolling window
        # is stationary for all variables (by KPSS)
        for _ in range(max_step - window_size):
            if all(window_is_stationary(var) for var in vals_dict):
                break
            auto_step()
        else:
            raise ValueError("Auto-equilibration stability not reached after {max_step} steps.")

        total_steps = len(list(vals_dict.values())[0]) * eq_step
        print(f"Auto-equilibration has detected stability after {total_steps} equilibration steps.")
        return total_steps, vals_dict

    @property
    def trajectory(self) -> CompactTrajectory | None:
        """
        The ``CompactTrajectory`` produced by the most recent production run of the
        ``Simulation``.

        Returns
        -------
        CompactTrajectory
            Most recent production run ``CompactTrajectory``, or `None` if no
            production run has been performed
        """

        try:
            return self.engine.convert_trajectory()
        except AttributeError:
            return None
