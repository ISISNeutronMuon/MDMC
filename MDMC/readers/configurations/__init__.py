"""A subpackage for reading files containing atomic configurations
"""

from MDMC.readers.configurations.conf_reader_factory import \
    ConfigurationReaderFactory


def read(file, **settings):

    """
    Reads a configuration file and returns a list of atoms corresponding to the
    atoms in the file

    Parameters
    ----------
    file : str or File
        The file name or the `File` object of the configuration file
    **settings
        Parameters passed to ConfigurationReader.parse

    Returns
    -------
    list of Atom
        `Atom` objects corresponding to the configuration in the file
    """

    extension = file.split('.')[-1]

    try:
        reader = ConfigurationReaderFactory.create_reader(extension)
    except ImportError:
        reader = ConfigurationReaderFactory.create_reader_from_ext(extension)

    reader.open(file)
    reader.parse(**settings)
    return reader.atoms
