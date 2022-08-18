"""Factory class for generating observables"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Callable, Iterable, Type, Union
    from MDMC.trajectory_analysis.observables.obs import Observable


class ObservableFactory:

    """
    Provides a factory for creating an ``Observable``.  Any module within the
    observables submodule can be created with a string of the class name, as
    long as it is a subclass of ``Observable``.
    """

    registry: 'dict[str, Observable]' = {}

    @classmethod
    def register(cls, names: 'Union[str, Iterable]') -> 'Callable':
        """
        A class level decorator for registering Observable classes

        The names of the modules with which the Observable is registered
        should be the parameter passed to the decorator.

        Parameters
        ----------
        names : str
            The names of the modules with which the Observable is registered

        Example
        -------
        To register the ``SQw`` class with ``ObservableFactory``:

            .. highlight:: python
            .. code-block:: python

                @ObservableFactory.register('SQw')
                class SQw(Observable):
        """

        def class_wrapper(wrapped_class: 'Observable') -> 'Callable':

            if isinstance(names, str):
                cls.registry[names] = wrapped_class
            else:
                for name in names:
                    cls.registry[name] = wrapped_class
            return wrapped_class

        return class_wrapper

    @classmethod
    def create_observable(cls, name: str) -> 'Observable':
        """
        Creates an ``Observable`` object from a module name

        The ``Observable`` object must be registered with the
        ``ObservableFactory``

        Parameters
        ----------
        name : str
            The name of the module with which the ``Observable`` is registered

        Returns
        -------
        Observable
            An ``Observable`` object
        """

        return cls.get_observable(name)()

    @classmethod
    def get_observable(cls, name: str) -> 'Type[Observable]':
        """
        Gets an ``Observable`` class from a registry name

        Parameters
        ----------
        name : str
            The name of the module with which the ``Observable`` is registered

        Returns
        -------
        cls
            A subclass of ``Observable``
        """

        observable = cls.registry[name]
        observable.name = name
        return observable
