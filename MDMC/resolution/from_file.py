"""A Resolution subclass for vanadium run resolutions from file."""
from os import getcwd
from os.path import join

import numpy as np

from MDMC.resolution.resolution import Resolution
from MDMC.trajectory_analysis.observables.obs_factory import ObservableFactory


class FileResolution(Resolution):
    """
    A `Resolution` subclass for applying resolution from file.
    """

    def __init__(self, file_name: str, file_type: str, file_reader: str, dt: float):
        self.file_name = file_name
        self.resolution_function = _read_resolution_from_file(file_type,
                                                              file_reader,
                                                              file_name,
                                                              dt)['SQw']

    def apply(self, FQt, t, Q):
        N_Q, N_T = np.shape(FQt)
        window = self._calculate_resolution_window(Q, t)

        return np.broadcast_to(window, (N_Q, N_T)) * FQt

    def _calculate_resolution_window(self, Q: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        Calculate the resolution window in the time domain.

        We normalise this window so that the sum over energy for each Q
        value is the same (this enforces that the static structure factor is constant for all Q).

        Parameters
        ----------
        Q : ~numpy.ndarray
            the points in energy at which FQt is calculated
        t : ~numpy.ndarray
             the points in time at which FQt is calculated

        Returns
        -------
        ~numpy.ndarray
            An ``array`` with the shape ``(N_Q, N_T)``
        """

        # By definition, the value of the resolution function in the time domain at t=0 is the
        # integral over all elements in the energy domain (with a factor for normalisation).
        # Setting this to one for all Q enforces that the static structure factor (the integral of
        # S(Q,w) over all w) is the same for all Q values in the resolution sample.
        window = self.resolution_function(t, Q)
        norm = self.resolution_function([0], Q)
        return window / norm

    def __repr__(self):
        """
        Represent a ``FileResolution`` object as the dataset used to create it.
        """

        return "Resolution" + str({'file': self.file_name})


def _read_resolution_from_file(file_type: str, file_reader: str, file_name: str, dt: float) -> dict:
    """
    Read and interpolate resolution data from a file.

    Parameters
    ----------
    file_type : str
        The ``type`` of the ``Observable``.
    file_reader : str
        The ``type`` of the ``Reader``.
    file_name : str
        The absolute or relative path of the resolution file name.
    dt : float
        The time separation of frames in ``fs``, for the simulation
        of the Observable.

    Returns
    -------
    dict
        A dictionary with keys for each dependent variable, where the
        values are resolution functions for that variable.

    Notes
    -----
    If this resolution function is used on data outside its original range, then it
    will use nearest neighbour extrapolation. Additionally, the input will be reflected in the
    time/energy domain as symmetry about 0 is assumed. If for whatever reason this is not
    appropriate for the data in question, this function should not be used.
    This may not be supported for all ``Observable`` types.
    """

    resolution_obs = ObservableFactory.create_observable(file_type)
    try:
        resolution_obs.read_from_file(reader=file_reader, file_name=file_name)
    # if file not found, check if it is in pwd (i.e. user put in filename rather than path)
    except FileNotFoundError:
        resolution_obs.read_from_file(
            reader=file_reader, file_name=join(getcwd(), file_name))
    return resolution_obs.calculate_resolution_functions(dt)
