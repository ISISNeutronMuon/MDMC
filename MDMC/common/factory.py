"""
Generalised factory class.
"""

from abc import ABC
from collections.abc import Callable, Iterable, Sequence
from importlib import import_module
from inspect import getmembers, isabstract, isclass
from pathlib import Path
from typing import Generic, TypeVar, get_args

T = TypeVar('T')

# Inheritors define own registry
# pylint: disable=no-member


class Factory(ABC):  # noqa: B024 - Abstract class no abs meth.
    """
    General factory class.

    Attributes
    ----------
    registry : dict[str, Callable]
        Dictionary of keys to names.
    """
    @classmethod
    def get(cls, key: str) -> Callable[..., T]:
        """
        Return a callable instance to construct given class.
        """
        return cls.registry[key]

    @classmethod
    def create(cls, key: str, *args, **kwargs) -> Callable[..., T]:
        """
        Return an instance of given class.
        """
        return cls.get(key)(*args, **kwargs)

    @classmethod
    def available_names(cls) -> Sequence[str]:
        """
        Known types supported by factory.

        Returns
        -------
        ~collections.abc.Sequence[str]
            Available keys to load.
        """
        return cls.registry.keys()

    @classmethod
    def supported_types(cls) -> tuple[type, ...]:
        """
        Return list of types supported by this factory.

        Returns
        -------
        tuple[type, ...]
            Parent classes allowed by this factory.
        """
        return get_args(cls.__orig_bases__[0])


class ModuleFactory(Factory, ABC, Generic[T]):
    """
    Scan current directory for any valid types.

    Supports:

    - scanning of files for relevant classes/functions.
    - exclusion of self and certain files.

    Attributes
    ----------
    curr_path : Path
        Path to scan, usually ``Path(__file__).parent``.
    curr_pack : str
        Current package to import relative to. Usually ``__package__``.
    exclude : Sequence[Path]
        Paths to exclude from search.
    """
    curr_path: Path = None
    curr_pack: str = None
    exclude: Sequence[Path] = ()

    @classmethod
    def scan(cls):
        """
        Scan current directory for any valid types.

        These types are added to the registry and available to be loaded.
        """
        for path in cls.curr_path.glob("*.py"):
            if path.stem.startswith(".") or any(path.samefile(other) for other in cls.exclude):
                continue

            module = import_module("." + path.stem, cls.curr_pack)
            classes = getmembers(
                module,
                lambda m: (
                    isclass(m) and
                    not isabstract(m) and
                    issubclass(m, cls.supported_types())
                ),
            )

            for name, type_ in classes:
                cls.registry[name] = type_

            # Get name of module too if only one class exists and
            # module name does not match
            if len(classes) == 1 and path.stem != classes[0][0]:
                cls.registry[path.stem] = classes[0][1]


class RegisterFactory(Factory, ABC, Generic[T]):
    """
    Factory requiring manual registration to data.

    See Also
    --------
    RegisterFactory.register : Registration mechanism.
    """

    @classmethod
    def register(cls, names: str | Iterable[str]) -> Callable[..., T]:
        """
        A class level decorator for registering classes.

        The names of the modules with which the class is registered
        should be the parameter passed to the decorator.

        Parameters
        ----------
        names : str
            The names of the modules with are registered

        Example
        -------
        To register the ``SQw`` class with ``RegisterFactory``:

        .. code-block:: python

            @RegisterFactory.register('SQw')
            class SQw(Observable):
        """

        def class_wrapper(wrapped_class: type) -> Callable[..., T]:

            if isinstance(names, str):
                cls.registry[names] = wrapped_class
            else:
                for name in names:
                    cls.registry[name] = wrapped_class
            return wrapped_class

        return class_wrapper
