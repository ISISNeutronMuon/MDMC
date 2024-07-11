"""
Module for observable reader abstract class.
"""

from MDMC.common.decorators import repr_decorator
from MDMC.MD.structures import Atom
from MDMC.readers.reader import Reader



@repr_decorator('file', 'extension', 'atoms')
class ConfigurationReader(Reader):
    """
    Abstract class for properties common to all configuration readers.

    Parameters
    ----------
    file_name : str
        File to read atoms from.

    Notes
    -----
    A ``ConfigurationReader`` should be created
    using ``ConfigurationReaderFactory``

    Does not implement ``Reader.parse``.
    """

    def __init__(self, file_name: str):
        super().__init__(file_name)
        self._atoms: list[Atom] = []

    @property
    def atoms(self) -> list[Atom]:
        """
        The `Atom` objects parsed from the file.

        Returns
        -------
        list[Atom]
            Parsed atoms.
        """

        return self._atoms
