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
Contains functions for applying resolution functions to data, and a factory used to get them.
"""

from .from_file import FileResolution
from .gaussian import GaussianResolution
from .lorentzian import LorentzianResolution
from .null import NullResolution
from .resolution import Resolution
from .resolution_factory import ResolutionFactory

__all__ = [
    "FileResolution",
    "GaussianResolution",
    "LorentzianResolution",
    "NullResolution",
    "Resolution",
    "ResolutionFactory",
]
