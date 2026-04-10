"""
Functions for viewing MDMC configurations via ASE.
"""

import ase.visualize
from IPython.display import HTML

from MDMC.MD import Structure, Universe
from MDMC.MD.ase.convert import MDMC_to_ASE


def view(obj: Structure | Universe,
         viewer: str = 'X3D',
         max_atoms: int = 2000) -> HTML | None:
    """
    View an MDMC Structure or Universe.

    Wrapper around the ASE viewer.

    Parameters
    ----------
    obj : ~MDMC.MD.Structure or ~MDMC.MD.AtomContainer
        The MDMC molecular object to be viewed.
    viewer : str
        The viewer.
    max_atoms : int, default 2000
        The maximum number of atoms to be displayed.

    Returns
    -------
    ~IPython.display.HTML or None
        Either opens the relevant GUI, or returns a HTML object
        (in the case of HTML viewers like X3D).
    """

    dimensions = obj.dimensions if isinstance(obj, Universe) else None

    ase_atoms = MDMC_to_ASE(obj, cell=dimensions)[:max_atoms]  # take first max_atoms atoms
    output = ase.visualize.view(ase_atoms, viewer=viewer)
    # running the view command will open the window for most viewers, but
    # for HTML viewers like X3D it needs to be returned
    if isinstance(output, HTML):
        return output
    return None
