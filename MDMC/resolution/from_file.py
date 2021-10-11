from MDMC.resolution.resolution import Resolution
from MDMC.trajectory_analysis.observables.obs_factory import ObservableFactory

import numpy as np


class FileResolution(Resolution):
    """
    A `Resolution` subclass for applying resolution from file.
    """

    def __init__(self, file_name, file_type, file_reader, dt):
        self.resolution_function = _read_resolution_from_file(file_type,
                                                              file_reader,
                                                              file_name,
                                                              dt)

    # ignored=None is here as apply() must have a number of parameters matching that of
    def apply(self, fqt, ignored=None):
        N_Q, N_T = np.shape(fqt)
        window = self._calculate_resolution_window(N_Q, N_T)

        return np.broadcast_to(window, (N_Q, N_T)) * fqt

    def _calculate_resolution_window(self, Q, t) -> np.ndarray:
        """
        Calculate the resolution window in time from a general resolution function in the time
        domain. Normalise this window so that the sum over energy for each Q
        value is the same (this enforces that the static structure factor is constant for all Q).

        Parameters
        ----------
        resolution_function : Callable
            The resolution from which to calculate the window
        N_T : int
            The number of points in time for FQt

        Returns
        -------
        numpy.ndarray
            An ``array`` with the shape ``(len(self.Q), N_T)``
        """

        # By definition, the value of the resolution function in the time domain at t=0 is the
        # integral over all elements in the energy domain (with a factor for normalisation).
        # Setting this to one for all Q enforces that the static structure factor (the integral of
        # S(Q,w) over all w) is the same for all Q values in the resolution sample.
        window = self.resolution_function(Q, t)
        norm = self.resolution_function([0], Q)
        return window / norm


def _read_resolution_from_file(file_type, file_reader, file_name, dt):
    resolution_obs = ObservableFactory.create_observable(file_type)
    resolution_obs.read_from_file(reader=file_reader, file_name=file_name)
    return resolution_obs.calculate_resolution_functions(dt)
