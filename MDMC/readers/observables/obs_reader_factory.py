"""
Factory class for generating readers for observables.
"""

from pathlib import Path

from MDMC.common.factory import ModuleFactory
from MDMC.readers.observables.obs_reader import ObservableReader


class ObservableReaderFactory(ModuleFactory[ObservableReader]):

    """
    Provides a factory for creating readers.

    Any module within the readers submodule can be created with a
    string of the class name, as long as it is a subclass of
    ``Reader``.
    """
    registry: dict[str, ObservableReader] = {}
    curr_path = Path(__file__).parent
    curr_pack = __package__
    exclude = (curr_path / "__init__.py", curr_path / "obs_reader_factory.py")

ObservableReaderFactory.scan()
