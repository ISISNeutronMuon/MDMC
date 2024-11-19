"""The Resolution abstract base class."""
from abc import ABC, abstractmethod

import numpy as np


class Resolution(ABC):
    """
    An abstract base class for resolution functions.
    """
    # pylint: disable=too-few-public-methods

    @abstractmethod
    def apply(self, FQt: np.ndarray, t: np.ndarray, Q: np.ndarray) -> np.ndarray:
        """
        Apply resolution to an FQt array.

        Parameters
        ----------
        FQt : ~numpy.ndarray
            the FQt array to which resolution is applied.
        t : ~numpy.ndarray
            the time points for the FQt array.
        Q : ~numpy.ndarray
            the momentum points for the FQt array.

        Returns
        -------
        ~numpy.ndarray
            The array with the resolution function applied to it.
        """

        raise NotImplementedError
