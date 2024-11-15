"""
Factory class for generating observables.
"""

from typing import Callable, Iterable
from MDMC.trajectory_analysis.observables.obs import Observable


class ObservableFactory:

    """
    Provide a factory for creating an :class:`Observable`.

    Any module within the observables submodule can be created with a
    string of the class name, as long as it is a subclass of
    ``Observable``.
    """

    registry: dict[str, Observable] = {}

    @classmethod
    def register(cls, names: str | Iterable[str]) -> Callable:
        """
        A class level decorator for registering Observable classes.

        The names of the modules with which the Observable is registered
        should be the parameter passed to the decorator.

        Parameters
        ----------
        names : str
            The names of the modules with which the Observable is registered.

        Returns
        -------
        Callable
            Wrapped class having been added to registry.

        Examples
        --------
        To register the ``SQw`` class with ``ObservableFactory``:

        .. code-block:: python

            @ObservableFactory.register('SQw')
            class SQw(Observable):
        """

        def class_wrapper(wrapped_class: Observable) -> Callable:

            if isinstance(names, str):
                cls.registry[names] = wrapped_class
            else:
                for name in names:
                    cls.registry[name] = wrapped_class
            return wrapped_class

        return class_wrapper

    @classmethod
    def create_observable(cls, name: str) -> Observable:
        """
        Create an ``Observable`` object from a module name.

        The ``Observable`` object must be registered with the
        ``ObservableFactory``.

        Parameters
        ----------
        name : str
            The name of the module with which the ``Observable`` is registered.

        Returns
        -------
        Observable
            An ``Observable`` object.
        """

        return cls.get_observable(name)()

    @classmethod
    def get_observable(cls, name: str) -> type[Observable]:
        """
        Get an ``Observable`` class from a registry name.

        Parameters
        ----------
        name : str
            The name of the module with which the ``Observable`` is registered.

        Returns
        -------
        cls
            A subclass of ``Observable``.
        """

        observable = cls.registry[name]
        observable.name = name
        return observable
