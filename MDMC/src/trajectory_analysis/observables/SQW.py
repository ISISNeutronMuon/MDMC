"""Module for total SQw class

AUTHOR :    Thomas Farmer        START DATE :    2018-6-6 13:24:27"""

from abc import ABCMeta, abstractmethod

import numpy as np

from MDMC.src.common.constants import h_bar
from MDMC.src.common.mathematics import correlation
from MDMC.src.common.resolution_functions import gaussian
from MDMC.src.trajectory_analysis.observables.obs import Observable


class AbstractSQw(Observable):

    """
    An abstract class for total, coherent and incoherent dynamic structure
    factors
    """

    __metaclass__ = ABCMeta

    @property
    def origin(self):

        return self._origin

    @property
    def data(self):

        return {'independent':self.independent_variables,
            'dependent':self.dependent_variables,
            'errors':self.errors}

    @property
    def independent_variables(self):

        """
        Returns a dictionary of all independent variables
        """

        return self._independent_variables

    @property
    def dependent_variables(self):

        """
        Returns a dictionary of all dependent variables
        """

        return self._dependent_variables

    @property
    def errors(self):

        """
        Returns a dictionary of all errors
        """

        return self._errors

    def read_from_file(self, reader, file_name):

        super(AbstractSQw, self).read_from_file(reader, file_name)
        self._origin = 'experiment'
        self._independent_variables = self.reader.independent_variables
        self._dependent_variables = self.reader.dependent_variables
        self._errors = self.reader.errors

    def calculate_from_MD(self, MD_input, **params):

        """
        Currently sets all errors to 0 when S(Q,w) is calculated from MD

        Independent variables can either be set previously or defined within
        params
        """
        self._origin = 'MD'
        self.trajectory = MD_input
        self.t = self.trajectory.times - self.trajectory.times[0]
        self.universe_cell = params.get('cell')
        self.t_res = params.get('t_resolution')
        self._set_weights()

        try:
            self.Q_values = np.array(params.get('Q_values'))
        except KeyError:
            self.Q_values = self.independent_variables['Q']

        self.isotropic = params.get('isotropic', True)
        if not self.isotropic:
            self.direction = np.array(params.get('direction', [1, 0, 0]))

        self.FQt = self.calculate_FQt()

        dt = self.t[1] - self.t[0]
        self.w = np.pi * np.arange(len(self.t)) / (len(self.t) * dt)
        self.E = h_bar * self.w
        self.SQw = self._calculate_SQw()
        self.SQw_err = np.zeros

        self._independent_variables = {'Q':self.Q_values, 'w':self.w}
        self._dependent_variables = {'SQw':self.SQw}
        self._errors = {'SQw':self.SQw_err}

    @abstractmethod
    def _set_weights(self):

        """
        Calculate the neutron weighting
        """

        pass

    def calculate_FQt(self):

        """
        Calculates the intermediate scattering function for all Q vectors for
        all time intervals
        """

        self.Q_vectors = self._calculate_Q_vectors()
        return np.array([self._calculate_FQt_single_Q(Q_vector)
                for Q_vector in self.Q_vectors])

    @abstractmethod
    def _calculate_FQt_single_Q(self, Q_vector):

        """
        Calculates intermediate scattering function for a single Q value for all
        time intervals (t)

        Arguments:
        Q_vector: Either a single Q vector or three orthogonal Q vectors
        """

        pass

    def _calculate_Q_vectors(self):

        if self.isotropic:
            direction = np.array([[1., 0., 0.],
                                  [0., 1., 0.],
                                  [0., 0., 1.]])
        else:
            direction = self.direction

        return np.array([value * direction
            for value in self.Q_values])

    def _rho(self, r, Q_vector):

        """
        Returns:
        The reciprocal space density for a position and Q vectors

        Arguments:
        r: A position vector
        Q_vector: One or more orthogonal Q vectors
        """

        return np.exp(-1j * np.dot(Q_vector, r))

    def _calculate_SQw(self):

        """
        Calculates SQw from FQt
        """

        FQt_res = self._apply_instrument_resolution(self.FQt,
                                                    {'sigma':self.t_res})

        # Reflect F(t) [except for both end points] for each Q value and append
        # it to F(t) to form an array of shape (n_row, 2*n_col - 2)
        FQt_mirror = np.append(FQt_res, FQt_res[:,-2:0:-1], axis=1)

        # Normalisation requires factor of dt
        # see Kneller et al. Comput. Phys. Commun. 91 (1995) 191-214
        dt = self.t[1] - self.t[0]

        # FFT and reduce the temporal dimension back to that of F(Q,t), with the
        # factor of 0.5 accounting for the fft over the reflected F(Q,t)
        return 0.5 * dt * np.real(np.fft.fft(FQt_mirror)[:, :len(self.t)])

    def _apply_instrument_resolution(self, FQt, params, function=gaussian):

        """
        Applies the specified resolution function to the S(Q,w) data

        As the S(Q,w) data is calculated from the time domain Fourier transform,
        F(Q,t), the resolution function can be applied multiplicatively, rather
        than by convolution.  Assumes that the temporal resolution has no Q
        dependence.

        CURRENTLY SQw is hard coded to only apply Gaussian resolution functions
        """

        # Functions other than Gaussians must be FFT before multiplication
        # As self.FQt is only half of the full (symmetric) FQt, only the
        # positive half of each resolution function is required.  Functions of
        # of odd length so that window[N_t] = 1
        N_t = np.shape(FQt)[1]
        N_Q = np.shape(FQt)[0]
        window = function(2 * N_t + 1, params['sigma'])[N_t:-1]

        # Tile the window so that it is applied for all Q values
        return np.tile(window, [N_Q, 1]) * FQt


class SQw(AbstractSQw):

    """
    A class for containing, calculating and reading the total dynamic structure
    factor
    """

    def _set_weights(self):

        pass

    def _calculate_FQt_single_Q(self, Q_vector):

        # Normalise to the number of atoms and orthogonal vectors
        n_atoms = len(self.trajectory.atoms)
        try:
            norm = np.shape(Q_vector)[1] * n_atoms
        except IndexError:
            norm = n_atoms

        rho = self._calculate_rho(Q_vector)
        FQt_single_Q = correlation(rho, normalise=True) / norm
        return FQt_single_Q

    def _calculate_rho(self, Q_vector):

        """
        Calculates time dependent number density in reciprocal space for all Q
        vectors

        As rho is the sum of the contributions for all of the specified Q
        vectors, these Q vectors should have the same Q value. Includes
        contributions from all atoms in the trajectory.

        Arguments:
        Q_vector: Either a single Q vector or three orthogonal Q vectors
        """
        rho = []
        for time in self.trajectory.times:
            rho_temp = [(np.exp(-1j * np.dot(Q_vector, r)))
                for r in self.trajectory.filter_by_time(time).positions]
            rho.append(np.sum(rho_temp, axis = 0))
        return np.array(rho)
