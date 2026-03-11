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
Utility functions for structures.
"""


def is_atom(atom: object) -> bool:
    """
    Checks if the passed object is an instance of the ``Atom`` class.

    Parameters
    ----------
    atom
        Object to be checked.

    Returns
    -------
    bool
        `True` if the passed Object is an ``Atom``.
    """
    # pylint: disable=import-outside-toplevel, cyclic-import
    # we are importing here on purpose to avoid circular importing
    from MDMC.MD.structures import Atom
    return isinstance(atom, Atom)
