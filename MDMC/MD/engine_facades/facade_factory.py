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

"""Factory class for generating MD engine facades"""

import logging
from importlib import import_module
from inspect import getmembers, isabstract, isclass
from types import ModuleType

from MDMC.MD.engine_facades.facade import MDEngine

LOGGER = logging.getLogger(__name__)


ENGINES = {
    "lammps_engine": "LAMMPSEngine",
    "dlpoly_engine": "DLPOLYEngine",
    "null_engine": "NullEngine",
    "openmm_engine": "OpenMMEngine",
}


class MDEngineFacadeFactory:
    """
    Provides a factory for creating facades to ``MDEngine``.  Any facade within
    the ``engine_facades`` folder can be created with a `str` of the class
    ``name``, as long as it is a subclass of ``MDEngine``.
    """

    @staticmethod
    def create_facade(module_name: str) -> MDEngine:
        """
        Parameters
        ----------
        module_name : str
            A module name in ``engine_facades``. Aliases to these module names
            are also valid.

        Returns
        -------
        ``MDEngine``
            The specified ``MDEngine``, as determined by the ``module_name``
        """
        module_name = MDEngineFacadeFactory.standardise_alias(module_name)
        try:
            module = import_module("." + module_name, __package__)
        except ModuleNotFoundError as err:
            LOGGER.info("Failed to load engine module: %s", module_name)
            raise ModuleNotFoundError(f"Failed to load engine module: {module_name}") from err

        classes = getmembers(
            module,
            lambda m: (
                isclass(m)
                and not isabstract(m)
                and issubclass(m, MDEngine)
                and ENGINES[module_name] in m.__name__
            ),
        )
        return classes[0][1]()

    @staticmethod
    def standardise_alias(alias: str) -> ModuleType:
        """
        Converts an ``alias`` into a module name
        """

        alias = alias.lower()
        if not alias.endswith("_engine"):
            alias += "_engine"

        if alias not in ENGINES:
            raise ImportError(
                f"The MD engine {alias} is not in the list of recognised engines, "
                f"which currently comprises: {tuple(ENGINES.keys())}",
            )

        return alias
