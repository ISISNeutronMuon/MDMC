from MDMC.resolution.resolution import Resolution
from MDMC.trajectory_analysis.observables.obs_factory import ObservableFactory

import numpy as np
from os import getcwd
from os.path import join

class FileResolution(Resolution):
    """
    A `Resolution` subclass for applying resolution from file.
    """

    def __init__(self, file_name, file_type, file_reader, dt):
        self.file_type = file_type
        self.file_reader = file_reader
        self.file_name = file_name
        self.dt = dt

    def apply(self, array, x, frequency_space=False):
        # has an extra line to get array shape,
        # then runs original apply function
        self.N_Q, self.N_x = np.shape(array)
        super(FileResolution, self).apply()

    # ignored=None is here as apply() must have a number of parameters matching that of the abstract method;
    # however, file resolution requires fewer parameters than numerical resolution.
    def _calculate_resolution_window(self, ignored, frequency_space=False) -> np.ndarray:
        """
        Calculate the resolution window in time from a self.resolution_function in the time
        domain. Normalise this window so that the sum over energy for each Q
        value is the same (this enforces that the static structure factor is constant for all Q).
        """

        self.resolution_function = _read_resolution_from_file(self.file_type,
                                                              self.file_reader,
                                                              self.file_name,
                                                              self.dt)['SQw']

        if frequency_space:
            return self.resolution_function
        else:
            # By definition, the value of the resolution function in the time domain at t=0 is the
            # integral over all elements in the energy domain (with a factor for normalisation).
            # Setting this to one for all Q enforces that the static structure factor (the integral of
            # S(Q,w) over all w) is the same for all Q values in the resolution sample.
            window = self.resolution_function(self.N_Q, self.N_x)
            norm = self.resolution_function([0], self.N_Q)
            return window / norm

    def __repr__(self):
        """
        Resolution objects are represented with the dictionary used to create them
        """

        return "Resolution" + str({'file': self.file_name})


def _read_resolution_from_file(file_type, file_reader, file_name, dt):
    """
    Reads resolution data for the specified ``data_type`` from file and interpolates it
    to give a dictionary of general resolution functions in the time domain for each dependent
    variable.
    Note that if this resolution function is used on data outside its original range, then it
    will use nearest neighbour extrapolation. Additionally, the input will be reflected in the
    time/energy domain as symmetry about 0 is assumed. If for whatever reason this is not
    appropriate for the data in question, this function should not be used.
    This may not be supported for all ``Observable`` types.
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
    """

    resolution_obs = ObservableFactory.create_observable(file_type)
    try:
        resolution_obs.read_from_file(reader=file_reader, file_name=file_name)
    except FileNotFoundError:  # if file not found, check if it is in pwd (i.e. user put in filename rather than path)
        resolution_obs.read_from_file(reader=file_reader, file_name=join(getcwd(), file_name))
    return resolution_obs.calculate_resolution_functions(dt)

