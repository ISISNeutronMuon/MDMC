"""Class for a generic configuration exporter"""
from abc import abstractmethod

from MDMC.MD import Structure
from MDMC.common.decorators import repr_decorator
from MDMC.exporters.exporter import Exporter


@repr_decorator('file', 'extension')
class ConfigurationExporter(Exporter):
    """
    Abstract class (as it does not implement ``Exporer.write``) that defines
    properties common to all exporters for configurations

    A ``ConfigurationExporter`` is created using ``ConfigurationReaderFactory``
    """

    @abstractmethod
    def write(self, structure: Structure, **settings: dict) -> None:
        """
        Writes the file data into the file so that it is in a format expected by the file format

        For exporters which are not specific to one data type, the calling class
        must be determined so that the file data can be parsed into
        the appropriate data type.

        Parameters
        ----------
        structure: MD.Structure
            A `Structure` object that
        **settings: dict
            dictionary of settings for exporter
        """
        raise NotImplementedError

    @property
    @staticmethod
    @abstractmethod
    def extension() -> str:
        """
        The expected file extension for the ``ConfigurationExporter``
        """

        raise NotImplementedError
