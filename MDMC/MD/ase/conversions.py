"""This module enables conversion between MDMC Structure objects and ASE
Atom and Atoms objects.
"""
from __future__ import annotations

from itertools import chain
from typing import Iterable, Union

import ase
# For some reason importing ase alone is not importing ase.io.x3d, therefore:
from ase.io import x3d
import numpy as np

from MDMC.MD.structures import Atom
from MDMC.MD.interactions import Bond


class ASEAtoms(ase.atoms.Atoms):

    """
    A subclass of ``ase.atoms.Atoms`` with explicit bonds defined between atoms

    Attributes
    ----------
    bonds : numpy.ndarray
        An ``array`` of ``tuple``, where each ``tuple`` is an atom pair, which
        are specified by the indexes (`int`) of each atom.

    Raises
    ------
    ValueError
        If there are not the same number of ``ID`` as there are atoms
    """

    def __init__(self, *args, **kwargs):

        bonds = kwargs.pop('bonds', None)
        IDs = kwargs.pop('IDs', None)

        super().__init__(*args, **kwargs)

        self.bonds = bonds
        if IDs and len(IDs) != len(self):
            raise ValueError('There must be an ID for every atom')
        self.IDs = IDs

    def __getitem__(self, i: Union[int, 'list[int]', slice]) -> Union[ase.atom.Atom, ASEAtoms]:
        """
        Extends ``ase.atoms.Atoms._getitem__`` so that ``bonds`` are dealt with
        correctly. This means that any bond between two atoms in the subset
        defined by `i` are assigned to the new item returned.

        Parameters
        ----------
        i : int, list of int, slice
            Returns an ``ase.atom.Atom`` (if `int`) or ``ASEAtoms`` (if `list`
            or slice).
        """

        atoms = super().__getitem__(i)

        # If int then only single atom so bonds are irrelevant
        if isinstance(i, int):
            return atoms

        # Account for fact that ASE allows i to be a list of integers, as well
        # an int or slice
        if isinstance(i, slice):
            atoms.IDs = self.IDs[i]
            indexes = range(i.start or 0, i.stop, i.step or 1)
        else:
            atoms.IDs = [self.IDs[j] for j in i]
            indexes = i

        # Get all bonds where both atom indexes in a bond (which is a tuple of
        # two atom indexes) are in the of the subset determined by i
        bonds = [bond for bond in self.bonds
                 if all(atom_index in indexes for atom_index in bond)]
        atoms.bonds = bonds
        return atoms

    # format (redefined builtin) is used as this is method signature in ASE base
    # class
    #pylint: disable=redefined-builtin
    def write(self, filename: str, format: str = None, **kwargs: dict) -> None:
        """
        Override ase.atoms.Atoms.write so that writing to X3D uses custom X3D
        class

        Parameters
        ----------
        filename : str
            The name of the file to which the ``ASEAtoms`` object is written
        format : str
            The name of the format of the file
        **kwargs
            keyword arguments which are passed to the object which performs the
            writing
        """

        if format and format.upper() in ['X3D', 'X3DOM']:
            X3D(self).write(filename, datatype=format.upper())
        else:
            super().write(filename, format, **kwargs)


def convert_to_ase_atom(atom: Atom, index: int = None) -> ase.atom.Atom:
    """
    Converts an MDMC ``Atom`` to an ``ase.atom.Atom``

    Parameters
    ----------
    atom : Atom
        An MDMC ``Atom`` object to be converted to an ``ase.atom.Atom`` object
    index : int, optional
        The ``index`` of the ``ase.atom.Atom`` object which is created. If this
        is not set, the MDMC ``Atom.ID`` is used.

    Returns
    -------
    ase.atom.Atom
        An ``ASE.atom.Atom`` object which is equivalent to ``atom``
    """

    index = index if index else atom.ID
    return ase.atom.Atom(position=atom.position,
                         index=index,
                         mass=atom.mass,
                         symbol=atom.element.symbol,
                         charge=atom.charge)


def convert_from_ase_atom(ase_atom: ASEAtoms,
                          atom_type: int = None,
                          name: str = None,
                          set_charge: bool = True,
                          cutoff: float = None) -> 'Atom':
    """
    Converts an ``ase.atom.Atom`` to an MDMC ``Atom``.

    As MDMC automatically generates atom ``ID``, ``ase_atom.index`` is not
    passed when initializing an ``Atom``.

    Parameters
    ----------
    ase_atom : ASEAtom
        An ``ASEAtom`` object to be converted to an MDMC ``Atom`` object
    atom_type : int
        The atom_type of the MDMC ``Atom`` object.
    name : str, optional
        A name for the MDMC ``Atom``. The default is the element symbol.
    set_charge : bool, optional
        Whether the ``charge`` is set to the ``charge`` of the ``ase.atom.Atom,
        or left unset. All ``ase.atom.Atom`` objects have a ``charge``, which is
        set to 0. if it is uninitialized. As MDMC ``Atom`` objects can have
        ``charge=None`, in some cases it might be preferential to leave the
        ``charge`` unset. The default is to set the ``charge``.
    cutoff : float, optional
        The cutoff value for the atom's charge interaction. Must be set if set_charge is True.

    Returns
    -------
    ``Atom``
        An MDMC ``Atom`` object which is equivalent to ``ase_atom``
    """

    name = name if name else ase_atom.symbol
    kwargs = {'position': ase_atom.position,
              'mass': ase_atom.mass, 'name': name}
    if set_charge:
        kwargs['charge'] = ase_atom.charge
        if cutoff is None:
            raise ValueError("If set_charge is True, a cutoff must also be set.")
        kwargs['cutoff'] = cutoff
    if atom_type:
        kwargs['atom_type'] = atom_type
    return Atom(ase_atom.symbol, **kwargs)


def get_ase_atoms(atoms: Iterable, cell: np.ndarray = None) -> ASEAtoms:
    """
    Gets an ``ASEAtoms`` object equivalent to ``atoms``, including the bonding

    Parameters
    ----------
    atoms : iterable
        An ``iterable`` of MDMC ``Atom`` objects to be converted to an
        ``ASEAtoms`` object
    cell : numpy.ndarray, optional
        A 3 element ``array`` specifying the unit cell of the ``ASEAtoms``
        object. The default is `None`.


    Returns
    -------
    ASEAtoms
        An ``ASEAtoms`` object which is equivalent to ``atoms``
    """

    # The ase.atoms.Atoms object unhelpfully overwrites the index attribute of
    # any ase.atom.Atom objects which belong to it (so Atoms[i].index == i,
    # regardless of the index of the Atom at that index). This means that the
    # atom IDs used for the bond atom pairs need to be converted.
    index_conv = {atom.ID: index for index, atom in enumerate(atoms)}
    bonds = set(chain.from_iterable([convert_bonds(atom.bonded_interactions,
                                                   index_conv)
                                     for atom in atoms]))
    IDs = [atom.ID for atom in atoms]
    return ASEAtoms([convert_to_ase_atom(atom, index) for index, atom
                     in enumerate(atoms)],
                    cell=cell,
                    bonds=bonds,
                    IDs=IDs)


def convert_bond(bond: Bond, index_conv: dict = None) -> np.ndarray:
    """
    Converts ``Bond`` objects into the form required by the ASE GUI

    Parameters
    ----------
    bond : Bond
        The bond which will be converted.
    index_conv : dict
        A dictionary of ``MDMC_ID``: ``ASE_index`` pairs, where ``MDMC_ID`` is
        an `int` specifying an ``Atom.ID``, and ``ASE_index`` is the
        corresponding ``ase.atom.Atom.index``. The default is `None`, which
        means that the ``ID`` and ``index`` will be assumed to be identical.

    Returns
    -------
    numpy.ndarray
        An ``array`` of 2 element `list` where each element is the `int`
        ``index`` of an atom between which the bond exists.
    """

    indexing = (lambda x: index_conv[x.ID]) if index_conv else lambda x: x.ID
    # Ensure atom IDs are ordered in each atom pair
    return [tuple(sorted(map(indexing, atom_pair))) for atom_pair in bond.atoms]


def convert_bonds(bonds: 'list[Bond]', index_conv: dict = None) -> np.ndarray:
    """
    Converts ``Bond`` objects into the form required by the ASE GUI

    Parameters
    ----------
    bonds : list
        The `list` of ``Bond`` objects to be converted
    index_conv : dict
        A `dict` of ``MDMC_ID``: ``ASE_index`` pairs, where ``MDMC_ID`` is an
        `int` specifying an ``Atom.ID``, and ``ASE_index`` is the corresponding
        ``ase.atom.Atom.index``. The default is `None`, which means that the
        ``ID`` and ``index`` will be assumed to be identical.

    Returns
    -------
    numpy.ndarray
        An ``array`` of 2 element `list` where each element is the `int`
        ``index`` of an atom between which the bond exists.
    """

    # conditional because only bond objects are supported
    return list(chain.from_iterable([convert_bond(bond, index_conv)
                                     for bond in bonds
                                     if isinstance(bond, Bond)]))


class X3D(x3d.X3D):

    """
    Class to write X3D or X3DOM (html) which can be read by web browers

    This is used for inline IPython (Jupyter) visualisation with X3DOM. It
    overrides the `write` method of the base class, in order to display bonds
    as defined within MDMC.

    Parameters
    ----------
    atoms : ase.atoms.Atoms
        The ``ase.atoms.Atoms`` object to write
    """

    def __init__(self, atoms: ase.atoms.Atoms):

        super().__init__(atoms)
        # 3000 atoms was picked because at this number of atoms the view is such
        # that a reduction in the sphere/cyinder resolution is insignificant. It
        # is also approximately half of the total number of atoms that can be
        # viewed on 2018 macbook pro (2.6GHz, 16GB DDR4, 4GB GDDR5)
        self.reduce_memory = len(self._atoms) > 3000

    def write(self, fileobj: str, datatype: str = 'X3DOM'):
        """
        Writes output to an 'X3DOM' (html) or 'X3D' file

        Parameters
        ----------
        fileobj : str or file-like
            The file name to which the ``atoms`` are written
        datatype : str, optional
            The output format, which can be 'X3D' or 'X3DOM' (html). If `None`,
            format will be determined from the `filename`
        """

        if datatype is None:
            if fileobj.endswith('.x3d'):
                datatype = 'X3D'
            elif fileobj.endswith('.html'):
                datatype = 'X3DOM'
            else:
                raise ValueError("filename must end in '.x3d' or '.html'.")

        # Write the header
        w = x3d.WriteToFile(fileobj)
        if datatype == 'X3DOM':
            w(0, '<html>')
            w(1, '<head>')
            w(2, '<title>MDMC atomic visualization</title>')
            w(2, '<link rel="stylesheet" type="text/css"')
            w(2, ' href="https://www.x3dom.org/x3dom/release/x3dom.css">')
            w(2, '</link>')
            w(2, '<script type="text/javascript"')
            w(2, ' src="https://www.x3dom.org/x3dom/release/x3dom.js">')
            w(2, '</script>')
            w(1, '</head>')
            w(1, '<body>')
            w(2, '<X3D width="800px" height="800px">')
            # Store next level of indentation and for scene
            scn_i = 3
        elif datatype == 'X3D':
            w(0, '<?xml version="1.0" encoding="UTF-8"?>')
            w(0, '<!DOCTYPE X3D PUBLIC "ISO//Web3D//DTD X3D 3.2//EN" '
              '"http://www.web3d.org/specifications/x3d-3.2.dtd">')
            w(0, '<X3D profile="Interchange" version="3.2" '
              'xmlns:xsd="http://www.w3.org/2001/XMLSchema-instance" '
              'xsd:noNamespaceSchemaLocation='
              '"http://www.web3d.org/specifications/x3d-3.2.xsd">')
            # Store next level of indentation and for scene
            scn_i = 1

        w(scn_i, '<Scene>')
        w(scn_i, '<Viewpoint centerOfRotation="{0:.2f} {1:.2f} {2:.2f}"'
                 ' position="{0:.2f} {1:.2f} {3:.2f}"></Viewpoint>'.format(
                     *self.get_center_of_rotation(), self.get_viewpoint_z()))

        for atom in self._atoms:
            for indent, line in self.atom_lines(atom):
                w(4 + indent, line)
        bonds = getattr(self._atoms, 'bonds', [])
        if bonds:
            for bond in bonds:
                for indent, line in self.bond_lines(bond):
                    w(4 + indent, line)

        if datatype == 'X3DOM':
            w(3, '</Scene>')
            w(2, '</X3D>')
            w(1, '</body>')
            w(0, '</html>')
        elif datatype == 'X3D':
            w(0, '</X3D>')

    def atom_lines(self, atom: ase.atom.Atom) -> list:
        """
        Generates a segment of X3D lines representing an atom

        Parameters
        ----------
        atom : ase.atom.Atom
            The ``ase.atom.Atom`` for which the X3D html will be generated

        Returns
        -------
        list
            A `list` of (indent, str) pairs for each line requied to describe
            the `atom`
        """

        color = tuple(ase.data.colors.jmol_colors[atom.number])
        diffuse_color = 'diffuseColor="{0:.3f} {1:.3f} {2:.3f}"'.format(*color)
        specular_color = 'specularColor="0.5 0.5 0.5"'.format(*color)

        if self.reduce_memory:
            # Decrease the resolution of the spheres
            sphere_subdivision = ' subdivision=12,12'
        else:
            sphere_subdivision = ''

        lines = [(0, '<Transform translation="{0:.2f} {1:.2f} {2:.2f}">'.format(
            *atom.position))]
        lines += [(1, '<Shape>')]
        lines += [(2, '<Appearance>')]
        lines += [(3, '<Material {0} {1}>'.format(diffuse_color,
                                                  specular_color))]
        lines += [(3, '</Material>')]
        lines += [(2, '</Appearance>')]
        # ASE covalent radii are too large if bonds are also going to be added
        lines += [(2, '<Sphere radius="{0:.2f}"{1}>'.format(
            ase.data.covalent_radii[atom.number] / 4.,
            sphere_subdivision))]
        lines += [(2, '</Sphere>')]
        lines += [(1, '</Shape>')]
        lines += [(0, '</Transform>')]
        return lines

    def bond_lines(self, bond: tuple) -> list:
        """
        Generates a cylinder representing a bond

        Parameters
        ----------
        bond : tuple
            A `tuple` containing the atom indexes for the bond for which the X3D
            html will be generated

        Returns
        -------
        list
            A `list` of (indent, str) pairs for each line requied to describe
            the `bond`
        """

        if self.reduce_memory:
            cylinder_subdivision = ' subdivision=16'
        else:
            cylinder_subdivision = ''

        positions = [self._atoms[index].position for index in bond]
        sub = (positions[1] - positions[0])
        # Separation of the two atoms defines the height of the cylinder (bond)
        separation = np.linalg.norm(sub)
        # Move one end of the cylinder coincident with one atom
        origin = positions[0] + np.array([0., separation / 2., 0.])

        # All cylinders (bonds) are oriented along y axis by default
        # Calculate axis-angle representation in order to set bond rotation
        cylinder = np.array([0., np.abs(separation), 0.])
        def normalise(x):
            return x / np.linalg.norm(x)
        uvec1, uvec2 = normalise(cylinder), normalise(sub)
        axis = np.cross(uvec1, uvec2)
        angle = np.linalg.norm(np.arccos(np.dot(uvec1, uvec2)))

        # Shift the transform center of the cylinder from the center (default)
        # to one end, so that it is rotated around one of the atoms
        lines = [(0, '<Transform center="0 {0:.4f} 0"'
                     ' translation="{2:.4f} {3:.4f} {4:.4f}"'
                     ' rotation="{5:.4f} {6:.4f} {7:.4f} {1:.4f}">'.format(
                         -separation / 2., angle, *origin, *axis))]
        lines += [(1, '<Shape>')]
        lines += [(2, '<Appearance>')]
        lines += [(3, '<Material diffuseColor="0 0 0"'
                      ' specularColor="0.5 0.5 0.5">')]
        lines += [(3, '</Material>')]
        lines += [(2, '</Appearance>')]
        lines += [(2, '<Cylinder height="{0:.4f}" radius="0.02"{1}>'.format(
            separation,
            cylinder_subdivision))]
        lines += [(2, '</Cylinder>')]
        lines += [(1, '</Shape>')]
        lines += [(0, '</Transform>')]

        return lines

    def get_center_of_rotation(self) -> np.ndarray:
        """
        Get the center of rotation for the viewpoint

        Returns
        -------
        numpy.ndarray
            The center of `atoms.cell` if this has been set, or the center of
            the extents of `atoms.positions`
        """

        atoms = self._atoms\
            # As ASE always has a defined cell for atoms (it is just
        # Cell([0., 0., 0.]) if it has not been set), cannot simply test for
        # existence
        # Currently only consider orthorhombic cell
        cell = np.diagonal(atoms.cell)
        return (cell / 2. if not np.all(cell == np.array([0.] * 3))
                else np.mean([np.max(atoms.positions, axis=0),
                              np.min(atoms.positions, axis=0)],
                             axis=0))

    def get_viewpoint_z(self) -> float:
        """
        Get the z position of the viewpoint which will display all of the
        `atoms`

        Returns
        -------
        float
            The z position of the viewpoint
        """

        # The viewpoint is always centered on the atoms (or the cell) in the xy
        # plane. It has been determined that an angle of ~0.30 radians between
        # the atom with the greatest extent (or the cell's greatest extent) will
        # be sufficient to display this atom (and therefore all atoms).
        VIEWPOINT_ANGLE = 0.30

        # Currently only consider orthorhombic cell
        cell = np.diagonal(self._atoms.cell)
        if np.any(cell != np.array([0., 0., 0.])):
            extents = cell
        else:
            extents = np.max(self._atoms.positions, axis=0)

        # Calculate distance between extents (xy position) and z axis
        # (i.e. x == y == 0.)
        xydistance = np.linalg.norm(extents[:2]
                                    - self.get_center_of_rotation()[:2])
        return xydistance / np.tan(VIEWPOINT_ANGLE) + extents[2]
