"""Module for total SQw class

AUTHOR :    Thomas Farmer        START DATE :    2018-6-6 13:24:27"""

from abc import ABCMeta, abstractmethod

import numpy as np

from MDMC.src.common.constants import h_bar
from MDMC.src.common.mathematics import correlation
from MDMC.src.common.resolution_functions import Gaussian
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

        try:
            self.Q_values = np.array(params.get('Q_values'))
        except KeyError:
            self.Q_values = self.independent_variables['Q']

        self.isotropic = params.get('isotropic', True)
        if not self.isotropic:
            self.direction = np.array(params.get('direction', [1, 0, 0]))

        self.FQt = self.calculate_FQt()

        self.w = 2 * np.pi / self.t
        self.E = h_bar * self.w
        self.SQw = self._calculate_SQw()
        self.SQw_err = np.zeros

        self._independent_variables = {'Q':self.Q_values, 'w':self.w}
        self._dependent_variables = {'SQw':self.SQw}
        self._errors = {'SQw':self.SQw_err}

    def calculate_FQt(self):

        """
        Calculates the intermediate scattering function for all Q vectors for
        all time intervals
        """

        self.Q_vectors = self._calculate_Q_vectors()
        return [self._calculate_FQt_single_Q(Q_vector)
                for Q_vector in self.Q_vectors]

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

    def _calculate_SQw(self):

        """
        Calculates SQw from FQt
        """

        return np.fft.fft(self._apply_instrument_resolution(self.FQt,
                                                            {'sigma':self.t_res}
                                                           ))

    def _apply_instrument_resolution(self, FQt, params, function=Gaussian):

        """
        Applies the specified resolution function to the S(Q,w) data

        As the S(Q,w) data is calculated from the time domain Fourier transform,
        F(Q,t), the resolution function can be applied multiplicatively, rather
        than by convolution.  Assumes that the temporal resolution has no Q
        dependence.

        CURRENTLY SQw is hard coded to only apply Gaussian resolution functions
        """

        # Functions other than Gaussians must be FFT before multiplication
        dim = np.shape(FQt)
        window = function(dim[1], params['sigma'])

        # Tile the window so that it is applied for all Q values
        return np.tile(window, [dim[0], 1]) * FQt


class SQw(AbstractSQw):

    """
    A class for containing, calculating and reading the total dynamic structure
    factor
    """

    def _calculate_FQt_single_Q(self, Q_vector):

        n_atoms = len(self.trajectory.atoms)
        rho = self._calculate_rho(Q_vector)
        FQt_single_Q = correlation(rho, normalise=True) / n_atoms
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
