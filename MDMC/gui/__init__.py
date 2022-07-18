"""Contains the GUI elements of MDMC

Contents
--------
view
"""

from MDMC import MD


def view(atom_container: 'list[MD.Atom, MD.AtomContainer]', viewer='X3DOM') -> None:
    """
    Launches a GUI for viewing collections of ``Atom`` objects

    Parameters
    ----------
    atom_container : list of Atom, AtomContainer
        An object which contains some ``Atom`` objects. This can either be in
        the sense that it is a list of `Atom` objects, or it could be an object
        which has the ``atoms`` attribute (e.g. an ``AtomContainer``). If
        ``atom_collection`` also has a ``dimensions`` attribute (such as
        ``Universe``), then this is used to set the volume displayed; otherwise
        the volume is determined by the extents of the ``Atom`` objects.
    """

    try:
        dimensions = atom_container.dimensions
    except AttributeError:
        dimensions = None
    try:
        atoms = atom_container.atoms
    except AttributeError:
        atoms = atom_container

    return MD.ase.viewer.view(atoms, viewer=viewer, cell=dimensions)
