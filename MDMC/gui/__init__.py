"""Contains the GUI elements of MDMC

Contents
--------
view
"""

from MDMC.MD.ase import viewer


def view(atom_container):

    """
    Launches a GUI for viewing collections of ``Atom`` objects

    Parameters
    ----------
    atom_container : list of Atom, AtomContainer
        An object which contains some ``Atom`` objects. This can either be in
        the sense that it is a list of `Atom` objects, or it could be an object
        which has the ``atom_list`` attribute (e.g. an ``AtomContainer``). If
        ``atom_collection`` also has a ``dimensions`` attribute (such as
        ``Universe``), then this is used to set the volume displayed; otherwise
        the volume is determined by the extents of the ``Atom`` objects.
    """

    try:
        dimensions = atom_container.dimensions
    except AttributeError:
        dimensions = None
    try:
        atoms = atom_container.atom_list
    except AttributeError:
        atoms = atom_container

    viewer.view(atoms, cell=dimensions)
