"""This subpackage contains the GUI elements of MDMC.
"""

from MDMC.MD.ase import viewer


def view(atom_collection):

    """
    Launches a GUI for viewing collections of Atoms

    Parameters
    ----------
    atom_collection : list of Atoms, Molecule, Universe
        An object which contains some atoms. This can either be in the sense
        that it is a list of ``Atom`` objects, or it could be an object which
        has the ``atom_list`` attribute, such as ``Molecule`` or ``Universe``.
        If ``atom_collection`` also has a ``dimensions`` attribute (such as
        ``Universe``), then this is used to set the volume displayed; otherwise
        the volume is determined by the extents of the atoms.
    """

    try:
        dimensions = atom_collection.dimensions
    except AttributeError:
        dimensions = None
    try:
        atoms = atom_collection.atom_list
    except AttributeError:
        atoms = atom_collection

    viewer.view(atoms, cell=dimensions)
