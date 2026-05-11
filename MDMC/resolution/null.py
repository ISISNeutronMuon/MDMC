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

"""A class for the null object for Resolution classes."""
from typing import Any

from MDMC.resolution.resolution import Resolution


class NullResolution(Resolution):
    """
    The null object for the Resolution class.
    Used when there is no resolution to apply.
    """

    # this __init__ needs to exist as otherwise passing a null resolution
    # will create an error that the object has been
    # given too many parameters at instantiation time.
    def __init__(self, *ignore: Any):
        # takes arguments and ignores them entirely
        pass

    def apply(self, FQt, t, Q):
        # pylint: disable=arguments-renamed
        # does not apply resolution
        return FQt

    def __repr__(self):
        """
        Resolution objects are represented with the dictionary used to create them;
        NullResolution is represented as {None} to match other objects.
        """

        return "Resolution{None}"
