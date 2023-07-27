"""Functions for viewing MDMC configurations via ASE."""
from typing import Union

import ase.visualize
from IPython.display import HTML

from MDMC.MD import Structure, Universe
from MDMC.MD.ase.convert import MDMC_to_ASE

def view(obj: Union[Structure, Universe], viewer: str = 'X3D') -> Union[HTML, None]:
    """
    View an MDMC Structure or Universe.
    Wrapper around the ASE viewer.

    Parameters
    ----------
    obj: Structure or AtomContainer
        The MDMC molecular object to be viewed.
    viewer: str
        The viewer.

    Returns
    -------
    HTML or None
        Either opens the relevant GUI, or returns a HTML object
        (in the case of HTML viewers like X3D).
    """

    if isinstance(obj, Universe):
        dimensions = obj.dimensions
    else:
        dimensions = None

    ase_atoms = MDMC_to_ASE(obj, cell=dimensions)
    output = ase.visualize.view(ase_atoms, viewer=viewer)
    # running the view command will open the window for most viewers, but
    # for HTML viewers like X3D it needs to be returned
    if isinstance(output, HTML):
        return output
    return None
