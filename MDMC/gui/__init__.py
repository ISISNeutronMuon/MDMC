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

"""
Functions for viewing MDMC configurations via ASE.
"""

import ase.visualize
from IPython.display import HTML

from MDMC.MD import Structure, Universe
from MDMC.MD.ase.convert import MDMC_to_ASE


def view(obj: Structure | Universe, viewer: str = "X3D", max_atoms: int = 2000) -> HTML | None:
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
