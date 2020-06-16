"""This module enables conversion between MDMC StructuralUnit objects and ASE
Atom and Atoms objects.
"""

from itertools import chain

import ase
from ase.io import x3d
import numpy as np

from MDMC.MD.structural_units import Atom, Bond


class ASEAtoms(ase.atoms.Atoms):

    """
    A subclass of ```ase.atoms.Atoms`` with explicit bonds defined between atoms

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

    def write(self, filename, format=None, **kwargs):

        if format == 'x3d':
            X3D(self).write(filename)
        else:
            super().write(filename, format, **kwargs)


def convert_to_ase_atom(atom, index=None):

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
                         symbol=atom.element,
                         charge=atom.charge)


def convert_from_ase_atom(ase_atom, atom_type=None, name=None, set_charge=True):

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

    Returns
    -------
    ``Atom``
        An MDMC ``Atom`` object which is equivalent to ``ase_atom``
    """

    name = name if name else ase_atom.symbol
    kwargs = {'position':ase_atom.position, 'mass':ase_atom.mass, 'name':name}
    if set_charge:
        kwargs['charge'] = ase_atom.charge
    if atom_type:
        kwargs['atom_type'] = atom_type
    return Atom(ase_atom.symbol, **kwargs)


def get_ase_atoms(atoms, cell=None):

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
    index_conv = {atom.ID:index for index, atom in enumerate(atoms)}
    bonds = set(chain.from_iterable([convert_bonds(atom.bonded_interactions,
                                                   index_conv)
                                     for atom in atoms]))
    IDs = [atom.ID for atom in atoms]
    return ASEAtoms([convert_to_ase_atom(atom, index) for index, atom
                     in enumerate(atoms)],
                    cell=cell,
                    bonds=bonds,
                    IDs=IDs)


def convert_bond(bond, index_conv=None):

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


def convert_bonds(bonds, index_conv=None):

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

    def __init__(self, atoms):

        super().__init__(atoms)
        self.reduce_memory = len(self._atoms) > 3000

    def write(self, filename, datatype=None):
        """Writes output to either an 'X3D' or an 'X3DOM' file, based on
        the extension. For X3D, filename should end in '.x3d'. For X3DOM,
        filename should end in '.html'.

        Args:
            filename - str or file-like object, output file name or writer
            datatype - str, output format. 'X3D' or 'X3DOM'. If `None`, format
                will be determined from the filename"""

        # Write the header
        w = x3d.WriteToFile(filename, 'w')
        w(0, '<html>')
        w(1, '<head>')
        w(2, '<title>ASE atomic visualization</title>')
        w(2, '<link rel="stylesheet" type="text/css"')
        w(2, ' href="https://www.x3dom.org/x3dom/release/x3dom.css">')
        w(2, '</link>')
        w(2, '<script type="text/javascript"')
        w(2, ' src="https://www.x3dom.org/x3dom/release/x3dom.js">')
        w(2, '</script>')
        w(1, '</head>')
        w(1, '<body>')
        w(2, '<X3D width="800px" height="800px">')
        w(3, '<Scene>')
        w(4, '<Viewpoint centerOfRotation="{0:.2f} {1:.2f} {2:.2f}"'
             ' position="{0:.2f} {1:.2f} {3:.2f}"></Viewpoint>'.format(
                 *self.get_center_of_rotation(), self.get_viewpoint_z()))
        for atom in self._atoms:
            for indent, line in self.atom_lines(atom):
                w(4 + indent, line)
        for bond in self._atoms.bonds:
            for indent, line in self.bond_lines(bond):
                w(4 + indent, line)

        w(3, '</Scene>')
        w(2, '</X3D>')
        w(1, '</body>')
        w(0, '</html>')

    def atom_lines(self, atom):

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

        if self.reduce_memory:
            sphere_subdivision = ' subdivision=12,12'
            specular_color = ''
        else:
            sphere_subdivision = ''
            specular_color = ' specularColor="0.5 0.5 0.5"'.format(*color)

        lines = [(0, '<Transform translation="{0:.2f} {1:.2f} {2:.2f}">'.format(
            *atom.position))]
        lines += [(1, '<Shape>')]
        lines += [(2, '<Appearance>')]
        lines += [(3, '<Material {0}{1}>'.format(diffuse_color,
                                                 specular_color))]
        lines += [(3, '</Material>')]
        lines += [(2, '</Appearance>')]
        lines += [(2, '<Sphere radius="{0:.2f}"{1}>'.format(
            ase.data.covalent_radii[atom.number] / 4.,
            sphere_subdivision))]
        lines += [(2, '</Sphere>')]
        lines += [(1, '</Shape>')]
        lines += [(0, '</Transform>')]
        return lines

    def bond_lines(self, bond):

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
            specular_color = ''
        else:
            cylinder_subdivision = ''
            specular_color = ' specularColor="0.5 0.5 0.5"'

        positions = [self._atoms[index].position for index in bond]
        origin = positions[0]
        sub = (positions[1] - positions[0])
        separation = np.linalg.norm(sub)
        cylinder = np.array([0., np.abs(separation), 0.])
        normalise = lambda x: x / np.linalg.norm(x)
        uvec1, uvec2 = normalise(cylinder), normalise(sub)
        axis = np.cross(uvec1, uvec2)
        angle = np.linalg.norm(np.arccos(np.dot(uvec1, uvec2)))

        origin_shift = origin + np.array([0., separation / 2., 0.])
        lines = [(0, '<Transform center="0 {0:.4f} 0"'
                     ' translation="{2:.4f} {3:.4f} {4:.4f}"'
                     ' rotation="{5:.4f} {6:.4f} {7:.4f} {1:.4f}">'.format(
                         -separation / 2., angle, *origin_shift, *axis))]
        lines += [(1, '<Shape>')]
        lines += [(2, '<Appearance>')]
        lines += [(3, '<Material diffuseColor="0 0 0"{0}>'.format(
            specular_color))]
        lines += [(3, '</Material>')]
        lines += [(2, '</Appearance>')]
        lines += [(2, '<Cylinder height="{0:.4f}" radius="0.02"{1}>'.format(
            separation,
            cylinder_subdivision))]
        lines += [(2, '</Cylinder>')]
        lines += [(1, '</Shape>')]
        lines += [(0, '</Transform>')]

        return lines

    def get_center_of_rotation(self):

        """
        Get the center of rotiation for the viewpoint

        Returns
        -------
        numpy.ndarray
            The center of `atoms.cell` if this has been set, or the center of
            the extents of `atoms.positions`
        """

        atoms = self._atoms
        return (atoms.cell / 2. if not np.all(atoms.cell == np.array([0.] * 3))
                else np.mean([np.max(atoms.positions, axis=0),
                              np.min(atoms.positions, axis=0)],
                             axis=0))

    def get_viewpoint_z(self):

        """
        Get the z position of the viewpoint which will display all of the
        `atoms`

        Returns
        -------
        float
            The z position of the viewpoint
        """

        VIEWPOINT_ANGLE = 0.38

        # The viewpoint is always centered on the atoms (or the cell) in the xy
        # plane. It has been determined that an angle of ~0.38 radians between
        # the atom with the greatest extent (or the cell's greatest extent) will
        # be sufficient to display this atom (and therefore all atoms).
        if np.any(self._atoms.cell != np.array([0., 0., 0.])):
            extents = self._atoms.cell
        else:
            extents = np.max(self._atoms.positions, axis=0)

        # Calculate distance between extents (xy position) and z axis
        # (i.e. x == y == 0.)
        xydistance = np.linalg.norm(extents[:2]
                                    - self.get_center_of_rotation()[:2])
        return xydistance / np.tan(VIEWPOINT_ANGLE) + extents[2]
