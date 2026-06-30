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

"""Module in which all interactions between structural units are defined.

``Interaction`` is the abstract base class from which all interactions have to be derived.."""

from __future__ import annotations

import logging
import weakref
from abc import ABC, abstractmethod
from contextlib import suppress
from itertools import permutations
from types import MethodType
from typing import TYPE_CHECKING, Any, NoReturn

import numpy as np

from MDMC.common import units
from MDMC.common.decorators import repr_decorator, unit_decorator
from MDMC.MD.interaction_functions import Coulomb
from MDMC.utilities.structures import is_atom

if TYPE_CHECKING:
    from MDMC.MD.interaction_functions import InteractionFunction
    from MDMC.MD.parameters import Parameters
    from MDMC.MD.simulation import Universe
    from MDMC.MD.structures import Atom


LOGGER = logging.getLogger(__name__)


@repr_decorator("function")
class Interaction(ABC):
    """
    Base class for interactions, both bonded, non-bonded and constraints

    Each different type of interaction should have an ``Interaction`` object.
    This object contains a `list` of the ``Atom`` (or ```Atom`` pairs, triplets
    or quadruplets,  depending on the type of interaction) for which this
    ``Interaction`` applies. For example, an oxygen ``Coulombic`` interaction
    would contain a `list` of `tuple` where each `tuple` contains a different O
    ``Atom``, and a hydrogen-oxygen ``Bond`` interaction would contain a `list`
    of `tuple` where each `tuple` contains a different H and O pair.
    ``Interaction`` objects can be sliced to return a sublist of the `tuple`.

    When an ``Atom`` is passed to an ``Interaction``, the ``Interaction`` is
    also added to the ``Atom``.

    Parameters
    ----------
    **settings
        ``function`` (`InteractionFunction`)
            A class of interaction function (e.g. ``HarmonicPotential``)
    """

    def __init__(self, **settings: Any):
        """
        Arguments:
        atom_tuples - One or more tuples consisting of one or more Atom objects.
        Each tuple contains all of the atoms involved in a single interaction.
        For example:

        H1 = Atom('H')
        H2 = Atom('H')
        O = Atom('O')

        For non-bonded interactions both of the following are equivalent
        (applying a Dispersion interaction to both H atoms):

        H_dispersive = Dispersion((H1, ), (H2, ))
        H_dispersive = Dispersion(H1, H2)

        For bonded interactions both of the following are equivalent:

        HO_bond = Bond((H1, O))
        HO_bond = Bond(H1, O)

        However when multiple bonds are initialized within a single Bond object
        (e.g. creating a Bond between each H and O for a single water molecule),
        tuples must be used to separate the bonds:

        HO_bond = Bond((H1, O), (H2,O))

        whereas the following is not valid:

        HO_bond = Bond(H1, O, H2, O)
        TypeError: object of type 'Atom' has no len()
        """

        self.function = settings.get("function")
        self.name = self.__class__.__name__

    def __deepcopy__(self, memo=None) -> NoReturn:
        """
        Interactions cannot be copied
        """

        raise AttributeError("Interactions cannot be deepcopied")

    def __copy__(self) -> NoReturn:
        """
        Interactions cannot be copied
        """

        raise AttributeError("Interactions cannot be copied")

    @property
    @abstractmethod
    def atoms(self) -> Atom | list[Atom]:
        """
        Get the atoms on which the ``Interaction`` is applied
        """

        raise NotImplementedError

    @property
    def parameters(self) -> Parameters:
        """
        Get the ``Parameter`` objects belonging to the ``InteractionFunction``
        belonging to the ``Interaction``

        Returns
        -------
        Parameters
            A ``Parameters`` object containing each ``Parameter``
        """
        return self.function.parameters

    @property
    def function(self) -> InteractionFunction:
        """
        Get or set the ``InteractionFunction`` of the ``Interaction``

        Returns
        -------
        InteractionFunction
            The interaction function of the ``Interaction``
        """

        return self._function

    @function.setter
    def function(self, value: InteractionFunction):
        self._function = value

    @property
    def function_name(self) -> str | None:
        """
        Get the name of the ``InteractionFunction`` belonging to the
        ``Interaction``

        Returns
        -------
        str
            The name of the ``InteractionFunction``, or `None` if no
            ``InteractionFunction`` has been set
        """

        try:
            return self.function.name
        except AttributeError:
            return None

    @property
    @abstractmethod
    def universe(self) -> Universe:
        """
        Get the ``Universe`` to which the ``Interaction`` belongs

        Returns
        -------
        Universe
            The ``Universe`` to which the ``Interaction`` belongs or `None`
        """

        raise NotImplementedError

    @abstractmethod
    def element_list(self) -> list:
        """
        Get a `list` of the elements for which the ``Interaction`` applies

        Returns
        -------
        list
            The elements for which the ``Interaction`` applies
        """

        raise NotImplementedError

    def sorted_element_list(self) -> list:
        """
        Sort the list of elements for which the ``Interaction`` applies

        Returns
        -------
        list
            The elements for which the ``Interaction`` applies, sorted
            alphabetically
        """

        return sorted(self.element_list())

    def element_tuple(self) -> tuple:
        """
        A `tuple` of elements for which the ``Interaction`` applies

        Returns
        -------
        tuple
            The elements for which the ``Interaction`` applies
        """

        return tuple(self.element_list())

    def _add_interaction_atoms(self, atoms: list[Atom]):
        """
        Add the ``Interaction`` to atoms for which the ``Interaction`` has been
        applied

        Parameters
        ----------
        atoms : list
            ``Atom`` objects which have been added to the ``Interaction``
        """

        for atom in atoms:
            atom.add_interaction(self, from_interaction=True)


@repr_decorator("function", "atom_types", "cutoff")
class NonBondedInteraction(Interaction):
    """
    Base class for non-bonded interactions

    Parameters
    ----------
    universe : Universe
        The ``Universe`` in which the ``NonBondedInteraction`` exists
    *atom_types
        `int` for each ``atom_type`` for which the ``NonBondedInteraction``
        applies
    **settings
        ``cutoff`` (`float`)
            The distance in ``Ang`` at which the interaction potential is
            truncated
    """

    def __init__(self, universe, *atom_types, **settings):
        self.universe = universe
        if self.universe:
            self.universe.add_nonbonded_interaction(self)
        self.cutoff = settings.get("cutoff")
        super().__init__(**settings)
        LOGGER.info(
            "%s created: {function:%s, atom_types:%s}",
            self.__class__,
            self.function,
            self.atom_types,
        )

    @abstractmethod
    def __eq__(self, other) -> bool:
        raise NotImplementedError

    def __ne__(self, other) -> bool:
        return not self == other

    def __hash__(self):
        # Simplified version of immutable hash which Python3 produces
        # (marginally less efficient but shouldn't matter)
        return id(self) // 8

    def __str__(self) -> str:
        """
        Returns
        -------
        str
            The ``type``, ``atom_types`` and ``cutoff`` of the
            ``NonBondedInteraction``
        """

        return (
            f"{self.__class__.__name__} interaction  atom_types: {self.atom_types}  "
            f" cutoff: {self.cutoff}"
        )

    @property
    @abstractmethod
    def atom_types(self) -> list[int]:
        """
        Get the atom types for which the ``NonBondedInteraction`` applies

        Returns
        -------
        list
            A list of `int` for the ``atom_types``
        """

        raise NotImplementedError

    @property
    def universe(self) -> Universe:
        """
        Get or set the ``Universe`` to which the ``NonBondedInteraction``
        belongs

        Returns
        -------
        Universe
            The ``Universe`` to which the ``NonBondedInteraction`` belongs or
            `None`
        """

        try:
            return self._universe()
        except TypeError:
            return self._universe

    @universe.setter
    def universe(self, value: Universe) -> None:
        try:
            self._universe = weakref.ref(value)
        except TypeError:
            self._universe = None

    @property
    def cutoff(self) -> float:
        """
        Get or set the distance in ``Ang`` at which the interaction potential is
        truncated

        Returns
        -------
        float
            The distance in ``Ang`` of the ``cutoff``
        """
        return self._cutoff

    @cutoff.setter
    @unit_decorator(unit=units.LENGTH)
    def cutoff(self, value: float) -> None:
        self._cutoff = value

    def is_equivalent(self, other) -> bool:
        """
        Checks for equivalence between two ``NonBondedInteraction``s, specifically
        if they apply to the same ``atom_types``, have the same ``cuttoff`` and
        the same ``function`` describing the interaction.

        Parameters
        ----------
        other : NonBondedInteraction
            The object to compare against.

        Returns
        -------
        bool
        """

        if id(other) == id(self):
            return True
        return (
            isinstance(other, type(self))
            and (sorted(self.atom_types, key=id) == sorted(other.atom_types, key=id))
            and self.cutoff == other.cutoff
            and self.function == other.function
        )


class Dispersion(NonBondedInteraction):
    """
    A non-bonded dispersive interaction - either LJ or Buckingham

    Parameters
    ----------
    universe : Universe
        The ``Universe`` in which the ``NonBondedInteraction`` exists
    *atom_types
        `int` for each atom type for which the ``NonBondedInteraction`` applies
    **settings
        ``cutoff`` (`float`)
            The distance in ``Ang`` at which the interaction potential is
            truncated
        ``vdw_tail_correction`` (`bool`)
            Specifies if the tail correction to the energy and pressure should
            be applied. This only affects the simulation dynamics if the
            simulation is being performed with constant pressure.

    Raises
    ------
    TypeError
        ``atom_types`` must be iterable
    ValueError
        ``Dispersion`` should only be specified as existing between pairs of
        ``atom_types``
    TypeError
        Each ``atom_type`` must be `int`
    """

    # Python3 requires subclasses that overwrite __eq__ to explicity inherit
    # __hash__
    __hash__ = NonBondedInteraction.__hash__

    def __init__(self, universe: Universe, *atom_types: int, **settings: Any):
        # Ignore pylint warning for inner function docstring
        # pylint: disable=missing-docstring
        def validate_atom_type_pair(atom_type_pair):
            try:
                atom_type_pair = tuple(sorted(atom_type_pair))
            except TypeError as err:
                raise TypeError("Atom types must be an iterable") from err
            if len(atom_type_pair) != 2:
                raise ValueError(
                    "Dispersion interactions should only be"
                    " specified as existing between pairs of"
                    " atom types",
                )
            if not all(isinstance(atom_type, (int, np.integer)) for atom_type in atom_type_pair):
                raise TypeError("Each atom type must be int")
            return atom_type_pair

        # Remove duplicates
        self._atom_types = tuple({validate_atom_type_pair(atp) for atp in atom_types})
        super().__init__(universe, **settings)
        # Add interactions to all atoms
        for atom_type_pair in self.atoms:
            for atoms in atom_type_pair:
                for atom in atoms:
                    atom.add_interaction(self)

        self.vdw_tail_correction = settings.get("vdw_tail_correction", False)

    def __eq__(self, other):
        return other.atom_types == self.atom_types and isinstance(other, type(self))

    @property
    def atom_types(self):
        return tuple(sorted(self._atom_types))

    @property
    def atoms(self) -> list[tuple[list[Atom]]]:
        """
        Get the atoms on which the ``Dispersion`` is applied

        Returns
        -------
        list
            A `list` of two `tuple`, where each `tuple` contains a `list` of
            `Atom`. Every ``Atom`` in the first `tuple` has a dispersion
            interaction with every ``Atom`` in the second `tuple` (excluding
            self interactions). This is the complete list of possible dispersion
            interactions, i.e. it is only exactly correct if no cutoff has been
            specified.
        """

        return [map(lambda x: self.universe.atom_types[x], tpl) for tpl in self.atom_types]

    def element_list(self) -> list:
        """
        Get a list of the elements for which the ``Interaction`` applies

        Returns
        -------
        list
            The elements for which the ``Interaction`` applies
        """

        # Each value in universe.atom_types dictionary contain list of atoms
        # with same elements, so use index 0
        # This is determined for all atom types in Dispersion interaction
        return [
            self.universe.atom_types[atom_type][0].element.symbol
            for tpl in self.atom_types
            for atom_type in tpl
        ]

    def is_equivalent(self, other) -> bool:
        """
        Checks for equivalence between two ``Dispersion``s, specifically
        if they apply to the same ``atom_types``, have the same ``cuttoff``,
        the same ``function`` describing the interaction and
        ``vdw_tail_correction`` setting.

        Parameters
        ----------
        other : Dispersion
            The object to compare against.

        Returns
        -------
        bool
        """

        return (
            super().is_equivalent(other) and self.vdw_tail_correction == other.vdw_tail_correction
        )


class Coulombic(NonBondedInteraction):
    """
    A non-bonded coulombic interaction - either normal or modified Coulomb

    Parameters
    ----------
    universe : Universe, optional
        The ``Universe`` in which the ``Coulombic`` exists. Default is `None`.
        Must be passed as a parameter if ``atom_types`` if passed.
    **settings
        ``charge`` (`float`)
            The charge parameter of the ``Coulombic`` interaction, in units of
            ``e``. If this argument is passed, the ``interaction_function`` of
            this ``Coulombic`` is set to ``Coulomb`` with this `float` as its
            ``Parameter``. Passing ``charge`` will overwrite any other
            ``interaction_functions`` that are set, i.e. it makes ``function``
            parameter redundant
        ``atoms`` (`list`)
            ``Atom`` objects to which the ``Coulombic`` applies. If specifying
            the ``atoms``, ``universe`` does not need to be passed as a
            parameter.



        atom_types : list of int
            int for each atom_type for which the NonBondedInteraction applies.
            If specifying the atom_types, the universe must be passed as a
            parameter and the atoms for which the atom_types are specified must
            exist in Universe. See the example above in the 'charge' section.

    Raises
    ------
    TypeError
        If one or more atom_types are passed but no universe is passed
    TypeError
        If neither atom_types or atoms have been passed
    TypeError
        If both atom_types and atoms have been passed

    Examples
    --------
    Upon initializing an ``Atom`` object and adding it to a ``Universe``:

    .. highlight:: python
    .. code-block:: python

        O = Atom('O', atom_type=1)
        universe = Universe(10.0)
        universe.add_structure('O')

    The following initializations of Coulombic are equivalent:

    .. highlight:: python
    .. code-block:: python

        O_coulombic = Coulombic(universe, atom_types=[O.atom_type],
                                charge=-0.84)
        O_coulombic = Coulombic(universe, atom_types=[O.atom_type],
                                function=Coulomb(-0.84))

    If ``atoms`` is passed then a ``Universe`` does not need to be passed:

    .. highlight:: python
    .. code-block:: python

        O_coulombic = Coulombic(atoms=[O], charge=-0.84)
    """

    # Python3 requires subclasses thay overwrite __eq__ to explicity inherit
    # __hash__
    __hash__ = NonBondedInteraction.__hash__

    def __init__(self, universe: Universe = None, **settings: Any):
        # pylint: disable=not-callable
        # as it raises a false positive on self.add_atoms

        try:
            atom_types = settings["atom_types"]
            if settings.get("atoms"):
                raise TypeError("Cannot pass both atoms and atom_types as parameters.")
            if isinstance(atom_types, (int, np.integer)):
                # Account for init argument atom_types=atom_type
                # rather than atom_types=[atom_type]
                atom_types = [atom_types]
            if not universe:
                raise TypeError("Coulombic requires a universe when atom_types are passed")

            self.add_atom_types = MethodType(_add_atom_types, self)
            self._atom_types = atom_types
            self._atoms = [
                atom for atom_type in self.atom_types for atom in universe.atom_types[atom_type]
            ]
            super().__init__(universe, **settings)
            # Add interaction to atoms
            for atom in self.atoms:
                atom.add_interaction(self)
        except KeyError:
            self.add_atoms = MethodType(_add_atoms, self)
            try:
                atoms = settings["atoms"]
            except KeyError as error:
                raise TypeError(
                    "Coulombic takes either atom_types or atoms as parameters",
                ) from error
            # Account for init argument atoms=atom rather than atoms=[atom]
            if is_atom(atoms):
                atoms = [atoms]
            self._atoms = []
            self._atom_types = []
            self.add_atoms(*atoms)

            # Assumes all atoms are in the same universe (or None)
            universe = self.atoms[0].universe
            super().__init__(universe, **settings)

        charge = settings.get("charge")
        if charge is not None:
            # Initializes a Coulomb interaction function with charge and units
            # and assigns it to self.function
            self.function = Coulomb(charge)
            LOGGER.warning(
                "%s: Coulombic interaction for the Atom object"
                "initialized with the Coulomb interaction function.",
                self.__class__,
            )

    def __len__(self) -> int:
        """
        Returns
        -------
        int
            The number of interactions of this type that have been set
        """

        return len(self.atoms)

    def __getitem__(self, key: int) -> tuple:
        """
        Returns
        -------
        tuple
            The `tuple` of atoms at the specified index in ``atoms``.
        """

        return self.atoms[key]

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, type(self))
            and (sorted(self.atom_types, key=id) == sorted(other.atom_types, key=id))
            and sorted(self.atoms, key=id) == sorted(other.atoms, key=id)
        )

    @property
    def atoms(self) -> list[Atom]:
        """
        Get the atoms on which the ``Coulombic`` interaction is applied

        Returns
        -------
        list
            A `list` of ``Atom`` on which the ``Coulombic`` is applied
        """

        return self._atoms

    @property
    def atom_types(self) -> list:
        """
        Get the atom types for which the ``Coulombic`` applies

        Returns
        -------
        list
            All atom types to which the ``Coulombic`` applies. If the
            interaction was initialized with ``atoms``, all ``atom_types``
            of the ``atoms`` to which the ``Coulombic`` was applied are
            returned; HOWEVER THE COULOMBIC INTERACTION IS NOT APPLIED TO ALL
            ATOMS OF THESE ``atom_types``, ONLY THE ATOMS IN ``self.atoms``
        """

        return self._atom_types

    def element_list(self) -> list:
        """
        Get a list of the elements for which the ``Coulombic`` interaction applies

        Returns
        -------
        list
            The elements for which the ``Coulombic`` interaction applies
        """

        return list(
            set(
                [atom.element.symbol for atom in self._atoms]
                + [
                    self.universe.atom_types[atom_type][0].element.symbol
                    for atom_type in self.atom_types
                ],
            ),
        )


class NonBondedForce(NonBondedInteraction):
    """A non-bonded interaction which is a sum of the coulomb and
    lennard-jones interactions.

    Parameters
    ----------
    universe : Universe, optional
        The ``Universe`` in which the ``Coulombic`` exists. Default is `None`.
        Must be passed as a parameter if ``atom_types`` if passed.
    **settings
        ``ewald`` (`float`)
            The error tolerance for Ewald summation.
        ``cutoff`` (`float`)
            The cutoff distance (angstrom) used for nonbonded interactions.
    """

    # I don't know why but dispersion and coulombic does this
    __hash__ = NonBondedInteraction.__hash__

    def __init__(self, universe: Universe, atom_type: str | int, **settings: Any):
        self.ewald = settings.get("ewald")

        self._atom_types = (atom_type,)
        super().__init__(universe, **settings)
        # Add interactions to all atoms
        for atom in self.atoms:
            atom.add_interaction(self)

    def __eq__(self, other):
        return other.atom_types == self.atom_types and isinstance(other, type(self))

    @property
    def atom_types(self):
        return self._atom_types

    @property
    def atoms(self) -> list[Atom]:
        return self.universe.atom_types[self._atom_types[0]]

    def element_list(self) -> list:
        return [self.universe.atom_types[self._atom_types[0]][0].element.symbol]


@repr_decorator("function", "n_atoms")
class BondedInteraction(Interaction):
    """
    Base class for bonded interactions

    Parameters
    ----------
    atom_tuples : list
        A `list` of `tuple`. Each `tuple` contains ``Atom`` objects which are
        bonded together. For three or more ``Atom`` objects, the order of
        the ``Atom`` objects within each `tuple` is important.
    **settings
        ``n_atoms`` (`int`)
            The number of atoms to which this ``BondedInteraction`` applies, for
            example 2 for a ``Bond``.

    Examples
    --------
    For a single bonded interactions which applies to ``H1``, ``O1``, and
    ``H2``:

    .. highlight:: python
    .. code-block:: python

        BondedInteraction(H1, O1, H2)

    For two bonded interactions of the same ``BondedInteraction`` type, one
    applied to ``H1``, ``O1`` and ``H2`` ``Atom`` objects, and the other applied
    to ``H3``, ``O2`` and ``H4`` ``Atom`` objects:

    .. highlight:: python
    .. code-block:: python

        BondedInteraction((H1, O1, H2), (H3, O2, H4))

    Whereas the above examples are both specifying a H-O-H ordered
    ``BondedInteraction``, the following specifies a H-H-O
    ``BondedInteraction``:

    .. highlight:: python
    .. code-block:: python

        BondAngle(H1, H2, O)
    """

    def __init__(self, *atom_tuples: list[tuple], **settings: Any):
        if atom_tuples and is_atom(atom_tuples[0]):
            atom_tuples = (atom_tuples,)

        if settings.get("n_atoms"):
            # This ensures that BondedInteractions can also be __init__ with 0
            # atoms
            for tpl in atom_tuples:
                self._validate_atoms(tpl, settings.get("n_atoms"))
        self.atoms = list(atom_tuples)
        super().__init__(**settings)
        LOGGER.info(
            "%s created: {function:%s, atom IDs:%s}",
            self.__class__,
            self.function,
            [tuple(map(lambda a: a.ID, tpl)) for tpl in self.atoms],
        )

    def __len__(self) -> int:
        """
        Returns
        -------
        int
            The number of interactions of this type that have been set
        """

        return len(self.atoms)

    def __getitem__(self, key: int) -> tuple:
        """
        Returns
        -------
        tuple
            The `tuple` of ``Atom`` at the specified index.  For a single index
            (as opposed to a slice) this is a group of atoms which are bonded
            together.
        """

        return self.atoms[key]

    def __str__(self) -> str:
        """
        Returns
        -------
        str
            The type, and number of atoms of the ``BondedInteraction``
        """

        return f"{self.__class__.__name__} interaction applied to {len(self.atoms)} atom tuples"

    def is_equivalent(self, other) -> bool:
        """
        Checks for equivalence between two ``BondedInteraction``s, specifically
        if they apply to the same ``atom_types`` and the same ``function``
        describing the interaction.

        Parameters
        ----------
        other : BondedInteraction
            The object to compare against.

        Returns
        -------
        bool
        """

        return id(other) == id(self) or (
            isinstance(other, self.__class__)
            and self.atom_types == other.atom_types
            and self.function == other.function
        )

    @property
    def atoms(self) -> list[tuple[Atom]]:
        """
        Get or set the atoms on which the ``Coulombic`` interaction is applied

        Returns
        -------
        list
            A `list` of `tuple` containing one or more ``Atom``. Each `tuple`
            contains all of the atoms involved in one example of the
            interaction. For example a ``BondAngle`` interaction each `tuple`
            would contain 3 or 4 atoms.

        Raises
        ------
        TypeError
            If a `list` of `tuple` is not set
        """

        return self._atoms

    @atoms.setter
    def atoms(self, atom_tuples: list[tuple[Atom]]) -> None:
        # Check for duplicate tuples in list
        self._check_duplicates(atom_tuples)
        # Check for duplicate atoms in each tuple
        try:
            for tpl in atom_tuples:
                self._check_duplicates(tpl)
        # try/except accounts for single atom passed rather than (atom,) tuple
        # e.g. if atom_tuples = [atom] instead of atom_tuples = [(atom,)]
        except TypeError as error:
            if len(atom_tuples) == 1 and is_atom(atom_tuples[0]):
                atom_tuples = [(atom_tuples[0],)]
            else:
                raise TypeError("atom_tuples must be [(atom, ...), ...]") from error

        # Only assign interaction to atoms after these validation steps
        self._atoms = []
        for tpl in atom_tuples:
            # Each tuple is appended individually so that it can be easily added
            # to ._bonded_interaction_pairs for every atom in the tuple
            self._atoms.append(tpl)
            self._add_interaction_atoms(tpl)

            # Add interaction to Universe (pass if no universe exists)
            with suppress(AttributeError):
                self._add_to_universe(self.universe, tpl)

    @property
    def atom_types(self) -> set[int | None]:
        """
        Get the `set` of all ``atom_type``s that this ``BondedInteraction``
        corresponds to, including `None` if appropriate.

        Returns
        -------
        Set[Union[int, NoneType]]
            A set of all (unique) ``atom_type``s.
        """
        return {atom.atom_type for atom_tuple in self.atoms for atom in atom_tuple}

    @property
    def universe(self) -> Universe | None:
        """
        Get the ``Universe`` to which the ``BondedInteraction`` belongs

        Returns
        -------
        Universe
            The ``Universe`` to which the ``BondedInteraction`` belongs or
            `None`
        """

        try:
            return self.atoms[0][0].universe
        except IndexError:
            return None

    def element_list(self) -> list[str] | None:
        """
        Get a `list` of the elements for which the ``BondedInteraction`` applies

        Returns
        -------
        list
            The elements for which the ``BondedInteraction`` applies
        """

        try:
            # Each tuple should contain the same elements, so first tuple's used
            return [atom.element.symbol for atom in self.atoms[0]]
        except (AttributeError, IndexError):
            return None

    @staticmethod
    def _validate_atoms(atoms: list, n_atoms: int) -> None:
        """
        Validates that the correct number of atoms have been passed to the
        interaction

        Parameters
        ----------
        atoms : list
            A `list` of ``Atom`` to validate
        n_atoms : int
            The expected number of ``Atom`` objects in ``atoms``

        Raises
        ------
        TypeError
            If the number of ``Atom`` objects in ``atoms`` is not equal to
            ``n_atoms``
        """

        if len(atoms) not in n_atoms:
            raise TypeError(f"This interaction only accepts {n_atoms} atoms")

    def add_atoms(self, *atoms: Atom, **settings: Any) -> None:
        """
        Add atoms which are all involved in one example of this interaction

        Parameters
        ----------
        *atoms
            one or more ``Atom`` objects
        **settings
            ``from_structure`` (`bool`)
                If ``add_atoms`` has been called from a ``Structure``

        Raises
        ------
        ValueError
            If this ``BondedInteraction`` has already been applied to one or
            more of the ``atoms``
        """

        self._check_duplicates(atoms)
        if atoms in self.atoms:
            raise ValueError("This interaction has already been applied to this atom(s)")

        self._atoms.append(atoms)
        from_structure = settings.get("from_structure", False)
        if not from_structure:
            for atom in atoms:
                atom.add_interaction(self, from_interaction=True)

        if self.universe:
            self._add_to_universe(self.universe, atoms)

    def _check_duplicates(self, structs: list) -> None:
        """
        Checks for duplicates ``Structure``

        Parameters
        ----------
        structs : list
            A `list` of ``Structure``
        err_msg : str
            A `str` to provide as an error message if there is a duplicate
            ``Structure``

        Raises
        ------
        ValueError
            If there is a duplicate ``Structure``
        """

        err_msg = (
            "Each tuple in the list of atom tuples must be unique, and"
            " each atom in a tuple must be unique"
        )

        # Check for duplicates (or reverse duplicates)
        try:
            equivalent_structs = self._get_equivalent_structures(structs)
        except TypeError:
            equivalent_structs = structs
        if len(set(equivalent_structs)) != len(equivalent_structs):
            raise ValueError(err_msg)

    def _get_equivalent_structures(self, structs: list) -> list[tuple[Atom]]:
        """
        Returns
        -------
        list
            `list` of `tuple` of ``Atom`` orderings which are equivalent
        """

        return structs + [tuple(reversed(atom_tuple)) for atom_tuple in structs]

    def _add_to_universe(self, universe: Universe, atoms: tuple[Atom]) -> None:
        """
        Adds interaction and atom tuple to ``universe``

        Parameters
        ----------
        universe : Universe
            The ``Universe`` to which to add the ``Interaction`` and `tpl`
        tpl : tuple
            A `tuple` of ``Atom``
        """

        universe.add_bonded_interaction_pairs((self, atoms))


@repr_decorator("constrained")
class ConstrainableMixin:
    # pylint: disable=too-few-public-methods
    # as this is a mixin class

    """
    A mixin class enabling classes inheriting from ``BondedInteraction`` to be
    constrained

    These constraints are then applied by a constraint algorithm (e.g. SHAKE),
    which is specified in the ``Universe`` to which the ``BondedInteraction``
    belongs.

    Parameters
    ----------
    atom_tuples : list
        A `list` of `tuple`. Each `tuple` contains ``Atom`` objects which are
        bonded together. For three or more ``Atom`` objects, the order of the
        ``Atom`` objects within each `tuple` is important.
    **settings

    Attributes
    ----------
    constrained : bool
        Specifying whether the object is constrained
    """

    def __init__(self, *atom_tuples: tuple, **settings: Any):
        self.constrained = settings.get("constrained", False)
        super().__init__(*atom_tuples, **settings)


@repr_decorator("function", "constrained")
class Bond(ConstrainableMixin, BondedInteraction):
    """
    A bond between any two atoms. Requires exactly two atoms in each
    ``atom_tuple``.

    Parameters
    ----------
    atom_tuples : list
        A `list` of `tuple`. Each `tuple` contains ``Atom`` which are bonded
        together.
    **settings
    """

    def __init__(self, *atom_tuples: tuple, **settings: Any):
        settings["n_atoms"] = (2,)
        super().__init__(*atom_tuples, **settings)

    def is_equivalent(self, other) -> bool:
        """
        Checks for equivalence between two ``BondedInteraction``s, specifically
        if they apply to the same ``atom_types``, have the same ``function``
        describing the interaction, and have the same ``constrained`` setting.

        Parameters
        ----------
        other : BondedInteraction
            The object to compare against.

        Returns
        -------
        bool
        """

        return super().is_equivalent(other) and self.constrained == other.constrained


@repr_decorator("function", "constrained")
class BondAngle(ConstrainableMixin, BondedInteraction):
    """
    A bond angle between any two bonds

    Requires three ``Atom`` objects (rotation around central atom) in each
    ``atom_tuple``. The atoms are ordered ``i``, ``j``, ``k``, where ``j`` is
    the central atom.  So:

    .. highlight:: python
    .. code-block:: python

        BondAngle(i, j, k) == BondAngle(k, j, i)

    Parameters
    ----------
    atom_tuples : list
        A `list` of `tuple`. Each `tuple` contains ``Atom`` which are bonded
        together. For three or more ``Atom`` objects, the order of the ``Atom``
        objects within each `tuple` is important.
    **settings
    """

    def __init__(self, *atom_tuples: tuple, **settings: Any):
        settings["n_atoms"] = (3,)
        super().__init__(*atom_tuples, **settings)

    def is_equivalent(self, other: BondedInteraction) -> bool:
        """
        Checks for equivalence between two ``BondedInteraction``s, specifically
        if they apply to the same ``atom_types``, have the same ``function``
        describing the interaction, and have the same ``constrained`` setting.

        Parameters
        ----------
        other : BondedInteraction
            The object to compare against.

        Returns
        -------
        bool
        """

        return super().is_equivalent(other) and self.constrained == other.constrained


@repr_decorator("function", "improper")
class DihedralAngle(BondedInteraction):
    """
    A dihedral angle between any two sets of three atoms, ``ijk`` and ``jkl``.

    Dihedral angles can be both proper and improper, where the angle between the
    two planes of ``ijk`` and ``jkl`` is fixed for improper dihedrals.

    The atoms of a proper ``DihedralAngle`` are ordered ``i``, ``j``, ``k``,
    ``l``, where ``j`` and ``k`` are the two central atoms.  So:

    .. highlight:: python
    .. code-block:: python

        DihedralAngle(i, j, k, l) == DihedralAngle(l, k, j, i)

    The atoms of an improper ``DihedralAngle`` are ordered ``i``, ``j``, ``k``,
    ``l``, where ``i`` is the central atom to which ``j``, ``k``, and ``l`` are
    all connected. So:

    .. highlight:: python
    .. code-block:: python

        (DihedralAngle(i, j, k, l, improper=True)
        == DihedralAngle(i, j, l, k, improper=True)
        == DihedralAngle(i, l, k, j, improper=True)
        == DihedralAngle(i, l, j, k, improper=True))
        == DihedralAngle(i, k, j, l, improper=True))
        == DihedralAngle(i, k, l, j, improper=True))

    Parameters
    ----------
    atom_tuples : list
        A `list` of `tuple`. Each `tuple` contains four `Atom` objects which are
        bonded together by the ``DihedralAngle``, in the order specified.
    **settings
        ``improper`` (`bool`)
            Whether the ``DihedralAngle`` is improper or not.

    Attributes
    ----------
    improper : bool
        Whether the ``DihedralAngle`` is improper or not, which affects the
        ``InteractionFunction`` which can be set for this ``DihedralAngle``. By
        default this is set to `False` i.e. the interaction is a proper
        dihedral.
    """

    def __init__(self, *atom_tuples: tuple, **settings: Any):
        settings["n_atoms"] = (4,)
        self.improper = settings.get("improper", False)
        super().__init__(*atom_tuples, **settings)

    def _get_equivalent_structures(self, structs: list[tuple[Atom]]) -> list:
        """
        Parameters
        ----------
        structs: list[tuple[Atom]]
            A list of tuples of Atoms.

        Returns
        -------
        list
            `list` of `tuple` of ``Atom`` orderings which are equivalent
        """

        # Improper dihedrals are equivalent if they have the same first
        # (central) atom, and any permutation of the other three atoms
        if self.improper:
            equivalent = []
            for atom_tuple in structs:
                equivalent += [
                    (atom_tuple[0],) + permutation for permutation in permutations(atom_tuple[1:])
                ]
            return equivalent
        # Proper dihedrals are equivalent if they are reversed (as with Bond and
        # BondAngle)
        return super()._get_equivalent_structures(structs)


def _add_atom_types(self, *atom_types: list[int]) -> None:
    """
    Function for dynamically creating an ``add_atom_types`` method in
    ``Coulombic``

    Parameters
    ----------
    atom_types : list
        One or more `int` specifying ``atom_types`` that exist in ``universe``
        of the ``Coulombic``
    """

    self.atom_types.append(*atom_types)


def _add_atoms(self, *atoms: list[Atom]) -> None:
    """
    Function for dynamically creating an ``add_atoms`` method in ``Coulombic``

    Adds ``*atoms`` to ``Coulombic`` and adds ``Coulombic`` to
    ``atoms.nonbonded_interactions``

    Parameters
    ----------
    atoms : list
        list of ``Atom``
    """

    for atom in atoms:
        # Add atom to interaction
        self.atoms.append(atom)
        # Add interaction to atom
        atom.add_interaction(self, from_interaction=True)
        # Add atom_type to interaction.atom_types
        if atom.atom_type and atom.atom_type not in self.atom_types:
            self.atom_types.append(atom.atom_type)
