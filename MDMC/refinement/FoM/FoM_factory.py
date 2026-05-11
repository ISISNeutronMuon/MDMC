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
Factory class for generating Figure of Merits
And ObservablePair class fro defining the obseravble pairs used to calculate the Figure of Merit
"""

from pathlib import Path

from MDMC.common.factory import ModuleFactory
from MDMC.refinement.FoM.FoM_abs import FigureOfMerit


class FoMFactory(ModuleFactory[FigureOfMerit]):
    """
    Provides a factory for creating a ``Figure of Merit`` also called FoM.

    Any FoM within the FoM folder can be created with a string of the class name, as
    long as it is a subclass of ``FigureOfMerit``.
    """

    registry: dict[str, FigureOfMerit] = {}
    curr_path = Path(__file__).parent
    curr_pack = __package__
    exclude = (curr_path / "__init__.py", curr_path / "FoM_factory.py")

    @classmethod
    def scan(cls):
        super().scan()

        # Add aliases
        FoMFactory.registry |= {
            key.lower()
            .removeprefix("chisquared")
            .removeprefix("rsquared")
            .removeprefix("_")
            .removesuffix("error"): val
            for key, val in FoMFactory.registry.items()
        }


FoMFactory.scan()
