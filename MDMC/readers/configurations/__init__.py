"""A subpackage for reading files containing atomic configurations"""

from typing import Union, TYPE_CHECKING

from .conf_reader_factory import ConfigurationReaderFactory
from . import cif
from . import pdb
from . import packmol_pdb
from . import conf_reader

if TYPE_CHECKING:
    from MDMC.MD.structures import Atom

def read(file: str, docstring: bool = False, **settings: dict) -> 'Union[list[Atom], None]':
    """
    Reads a configuration file and returns a list of atoms corresponding to the
    atoms in the file

    .. note:: The docstring of the required reader (as determined from the file)
              can be accessed by passing `help=True`. This may be necessary to
              determine the reader specific `**settings` that can be passed. In
              this case, the file will not be read and None will be returned,
              rather than a list of `Atom` objects.

    Parameters
    ----------
    file : str
        The file name of the configuration file
    docstring : bool, optional
        This will show the docstring (help) related to the type of configuration
        file that has been passed. If this is True, the file will not be read
        and None will be returned. The default is False.
    **settings
        Parameters passed to ConfigurationReader.parse

    Returns
    -------
    list of Atom, or None
        `Atom` objects corresponding to the configuration in the file. None will
        be returned if `docstring=True`.

    Examples
    --------
    To read a CIF file and define the `atom_types` of the atoms::

        atoms = read('example.cif', atom_types=[1, 1, 2, 2, 2, 1, 3])

    To get the docstring (help) for reading a CIF file::

        read('example.cif', help=True)
    """

    extension = file.split('.')[-1]

    try:
        reader = ConfigurationReaderFactory.create_reader(extension, file)
    except ImportError:
        reader = ConfigurationReaderFactory.create_reader_from_ext(extension, file)

    if docstring:
        help(reader.parse)
        return None

    with reader:
        reader.parse(**settings)
    return reader.atoms
