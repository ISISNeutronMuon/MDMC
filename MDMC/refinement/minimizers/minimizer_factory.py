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

"""Factory class for generating minimizers"""

from pathlib import Path

from MDMC.common.factory import ModuleFactory
from MDMC.refinement.minimizers.minimizer_abs import Minimizer


class MinimizerFactory(ModuleFactory[Minimizer]):
    """
    Provides a factory for creating a ``Minimizer``.

    Any minimizer within
    the minimizers folder can be created with a string of the class name, as
    long as it is a subclass of ``Minimizer``.
    """

    registry: dict[str, Minimizer] = {}
    curr_path = Path(__file__).parent
    curr_pack = __package__
    exclude = (curr_path / "__init__.py", curr_path / "minimizer_factory.py")


MinimizerFactory.scan()
