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

"""Factory class for generating readers for configurations"""

from pathlib import Path

from ase.io.formats import ioformats

from MDMC.common.factory import ModuleFactory
from MDMC.readers.configurations.ase import ASEReader
from MDMC.readers.configurations.conf_reader import ConfigurationReader

# pylint: disable=cyclic-import
# this is handled!


class ConfigurationReaderFactory(ModuleFactory[ConfigurationReader]):
    """
    Provides a factory for creating readers.

    Any module within the readers submodule can be created with a
    string of the class name, as long as it is a subclass of
    ``ConfigurationReader``.
    """

    registry: dict[str, ConfigurationReader] = {}
    curr_path = Path(__file__).parent
    curr_pack = __package__
    exclude = (curr_path / "__init__.py", curr_path / "conf_reader_factory.py")

    @classmethod
    def create_reader_from_ext(cls, extension: str, file_name: str) -> ConfigurationReader:
        """
        Parameters
        ----------
        extension : str
            The file extension from which to initialize a subclass of
            ``ConfigurationReader``
        file_name : str
            The name of the file that you want to read.

        Returns
        -------
        ConfigurationReader
            An initialized configuration reader which has the extension
            specified by ``extension``

        Raises
        ------
        NotImplementedError
            If ``extension`` does not match the ``extension`` property of any
            subclass of ``ConfigurationReader``
        """
        for reader in cls.registry.values():
            if reader.extension == extension:
                return reader(file_name)

        # if no direct reader exists, try ASE reader
        if extension in ioformats:
            return ASEReader(file_name)

        raise NotImplementedError(f"No implemented reader is compatible with {extension} extension")


ConfigurationReaderFactory.scan()
