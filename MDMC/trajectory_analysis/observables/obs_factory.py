"""
Factory class for generating observables.
"""
from collections.abc import Callable

from collections.abc import Callable, Iterable

from MDMC.trajectory_analysis.observables.obs import Observable


class ObservableFactory(RegisterFactory[Observable]):
    """
    Provide a factory for creating an :class:`Observable`.

    Any module within the observables submodule can be created with a
    string of the class name, as long as it is a subclass of
    ``Observable``.
    """

    registry: dict[str, Observable] = {}

    @classmethod
    def create(cls, key: str, *args, **kwargs) -> Callable[..., Observable]:
        """
        Return an instance of given class.
        """
        obs = super().create(key, *args, **kwargs)
        obs.name = key
        return obs
