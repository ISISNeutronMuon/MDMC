"""The Resolution abstract base class."""
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


class Resolution(ABC):
    """
    An abstract base class for resolution functions.
    """
    # pylint: disable=too-few-public-methods

    @abstractmethod
    def apply(self, FQt: 'np.ndarray', t: 'np.ndarray', Q: 'np.ndarray') -> 'np.ndarray':
        """
        Applies resolution to an array.

        Parameters
        ----------
        FQt: array
            the FQt array to which resolution is applied.
        t: array
            the time points for the FQt array.
        Q: array
            the momentum points for the FQt array.

        Returns
        -------
        array
            The array with the resolution function applied to it.
        """

        raise NotImplementedError
