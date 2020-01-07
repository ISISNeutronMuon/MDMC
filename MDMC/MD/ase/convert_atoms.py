"""This module enables conversion between MDMC StructuralUnit objects and ASE
Atom and Atoms objects.
"""

from itertools import chain

import ase

from MDMC.MD.structural_units import Atom, Bond


class ASEAtoms(ase.atoms.Atoms):

    """
    A subclass of ``ase.atoms.Atoms`` with explicit bonds defined between atoms

    Attributes
    ----------
    bonds : array
        An array of tuples, where each tuple is an atom pair, which are
        specified by the indexes (int) of each atom.
    """

    def __init__(self, *args, **kwargs):

        bonds = kwargs.pop('bonds', None)
        super().__init__(*args, **kwargs)
        self.bonds = bonds


def convert_to_ase_atom(atom, index=None):

    """
    Converts an MDMC ``Atom`` to an ``ase.atom.Atom``

    Parameters
    ----------
    atom : Atom
        An MDMC ``Atom`` object to be converted to an ase.atom.Atom object
    index : int, optional
        The index of the ``ase.atom.Atom`` object which is created. If this is
        not set, the MDMC ``Atom.ID`` is used.

    Returns
    -------
    ase.atom.Atom
        An ASE ``Atom`` object which is equivalent to ``atom``
    """

    index = index if index else atom.ID
    return ase.atom.Atom(position=atom.position,
                         index=index,
                         mass=atom.mass,
                         symbol=atom.element,
                         charge=atom.charge)


def convert_from_ase_atom(ase_atom):

    """
    Converts an ``ase.atom.Atom`` to an MDMC ```Atom``.

    As MDMC automatically generates atom IDs, ``ase_atom.index`` is not passed
    when initializing an ``Atom``.

    Parameters
    ----------
    ase_atom : ASEAtom
        An ``ASEAtom`` object to be converted to an MDMC ``Atom`` object

    Returns
    -------
    Atom
        An MDMC ``Atom`` object which is equivalent to ``ase_atom``
    """

    return Atom(ase_atom.symbol,
                position=ase_atom.position,
                mass=ase_atom.mass,
                charge=ase_atom.charge)


def get_ase_atoms(atoms, cell=None):

    """
    Gets an ``ASEAtoms`` object equivalent to ``atoms``, including the bonding

    Parameters
    ----------
    atoms : iterable
        An iterable of MDMC Atom objects to be converted to an ase.atoms.Atoms
        object
    cell : array, optional
        A 3 element array specifying the unit cell of the ASE Atoms object. The
        default is None.


    Returns
    -------
    ase.atoms.Atoms
        An ASE Atoms object which is equivalent to ``atoms``
    """

    # The ase.atoms.Atoms object unhelpfully overwrites the index attribute of
    # any ase.atom.Atom objects which belong to it (so Atoms[i].index == i,
    # regardless of the index of the Atom at that index). This means that the
    # atom IDs used for the bond atom pairs need to be converted.
    index_conv = {atom.ID:index for index, atom in enumerate(atoms)}
    bonds = set(chain.from_iterable([convert_bonds(atom.bonded_interactions,
                                                   index_conv)
                                     for atom in atoms]))
    return ASEAtoms([convert_to_ase_atom(atom, index) for index, atom
                     in enumerate(atoms)],
                    cell=cell,
                    bonds=bonds)


def convert_bond(bond, index_conv=None):

    """
    Converts ``Bond`` objects into the form required by the ASE GUI

    Parameters
    ----------
    bond : Bond
        The bond which will be converted.
    index_conv : dict
        A dictionary of MDMC_ID: ASE_index pairs, where MDMC_ID is an int
        specifying an ``Atom.ID``, and ASE_index is the corresponding
        ``ase.atom.Atom.index``. The default is ``None``, which means that the
        IDs and indexes will be assumed to be identical.

    Returns
    -------
    np.array
        An array of 2 element lists where each element is the integer index of
        an atom between which the bond exists.
    """

    indexing = (lambda x: index_conv[x.ID]) if index_conv else lambda x: x.ID
    # Ensure atom IDs are ordered in each atom pair
    return [tuple(sorted(map(indexing, atom_pair))) for atom_pair in bond.atoms]


def convert_bonds(bonds, index_conv=None):

    """
    Converts ``Bond`` objects into the form required by the ASE GUI


    Parameters
    ----------
    bonds : list
        The list of bonds to be converted
    index_conv : dict
        A dictionary of MDMC_ID: ASE_index pairs, where MDMC_ID is an int
        specifying an ``Atom.ID``, and ASE_index is the corresponding
        ``ase.atom.Atom.index``. The default is ``None``, which means that the
        IDs and indexes will be assumed to be identical.

    Returns
    -------
    np.array
        An array of 2 element lists where each element is the integer index of
        an atom between which the bond exists.
    """

    # conditional because only bond objects are supported
    return list(chain.from_iterable([convert_bond(bond, index_conv)
                                     for bond in bonds
                                     if isinstance(bond, Bond)]))
