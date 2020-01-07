"""This subpackage contains the GUI elements of MDMC.
"""

from MDMC.MD.ase import viewer


PARAM_ERROR = ValueError('One of atoms and universe can be passed')

def view(atoms=None, universe=None):

    """
    Launches a GUI for viewing collections of Atoms

    Parameters
    ----------
    atoms : list, optional
        A list of the atoms to plot in the viewer. This cannot be passed if
        ``universe`` is passed.
    universe : Universe
        The Universe to be plotted in the viewer. All atoms within the Universe
        are plotted, and the dimensions of the Universe are dileneated with
        dotted lines. This cannot be passed if ``atoms`` is passed.

    Raises
    ------
    ValueError
        If neither or both of atoms and universe are passed
    """

    if atoms:
        if universe:
            raise PARAM_ERROR
        dimensions = None
    else:
        try:
            dimensions = universe.dimensions
        except AttributeError:
            raise PARAM_ERROR
        atoms = universe.atom_list

    viewer.view(atoms, cell=dimensions)
