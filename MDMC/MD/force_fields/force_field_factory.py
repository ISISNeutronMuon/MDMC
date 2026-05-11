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

"""Factory class for generating force fields"""

from pathlib import Path

from MDMC.common.factory import ModuleFactory
from MDMC.MD.force_fields.ff import ForceField


class ForceFieldFactory(ModuleFactory[ForceField]):
    """
    Provides a factory for creating a ``ForceField``.

    Any force field within the force fields folder can be created with
    a string of the class name, as long as it is a subclass of
    ``ForceField``.
    """

    registry: dict[str, ForceField] = {}
    curr_path = Path(__file__).parent
    curr_pack = __package__
    exclude = (curr_path / "__init__.py", curr_path / "force_field_factory.py")


ForceFieldFactory.scan()
