"""Module for AbstractSQw and total SQw class"""

from abc import abstractmethod

from mpi4py import MPI
from numba import jit
import numpy as np
from numpy.testing import assert_allclose

from MDMC.common import units
from MDMC.common.atom_properties import B_COH, B_INCOH
from MDMC.common.constants import h_bar
from MDMC.common.decorators import unit_decorator, unit_decorator_getter
from MDMC.common.mathematics import correlation, UNIT_VECTOR
from MDMC.common.resolution_functions import gaussian
from MDMC.trajectory_analysis.observables.obs import Observable


class AbstractSQw(Observable):

    """
    An abstract class for total, coherent and incoherent dynamic structure
    factors
    """

    @property
    def data(self):

        """
        Get the independent, dependent and error data

        Returns
        -------
        dict
            The independent, dependent and error data
        """

        return {'independent':self.independent_variables,
                'dependent':self.dependent_variables,
                'errors':self.errors}

    @property
    def independent_variables(self):

        """
        Get or set the independent variables, Q (in Ang^-1) and E (in meV)

        Returns
        -------
        dict
            The independent variables
        """

        return self._independent_variables

    @independent_variables.setter
    def independent_variables(self, value):

        self._independent_variables = value

    @property
    def dependent_variables(self):

        """
        Get or set the dependent variables, SQw (in arb)

        Returns
        -------
        dict
            The dependent variables
        """

        return self._dependent_variables

    @property
    def errors(self):

        """
        Get or set the errors on the dependent variables

        Returns
        -------
        dict
            The errors on the dependent variables
        """

        return self._errors

    @property
    @unit_decorator_getter(unit=units.LENGTH ** -1)
    def Q(self):

        """
        Get the momentum transfers

        Returns
        -------
        array
            1D array of Q floats (in Ang^-1)
        """
        try:
            return self.independent_variables['Q']
        except KeyError:
            raise AttributeError

    @property
    @unit_decorator_getter(unit=units.ENERGY_TRANSFER)
    def E(self):

        """
        Get the energies

        Returns
        -------
        array
            1D array of energy floats (in meV)
        """

        try:
            return self.independent_variables['E']
        except KeyError:
            raise AttributeError

    @property
    @unit_decorator_getter(unit=units.ANGLE / units.Unit('ps'))
    def w(self):

        """
        Get the angular frequencies

        Returns
        -------
        array
            1D array of angular frequency floats in units of rad ps^-1
        """

        return self.E / (h_bar * 1e15)

    @property
    @unit_decorator_getter(unit=units.ARBITRARY)
    def SQw(self):

        """
        Get the dynamic structure factor, S(Q, w), in arb

        Returns
        -------
        array
            2D array of S(Q, w) floats with arbitrary units
        """

        try:
            return self.dependent_variables['SQw']
        except KeyError:
            raise AttributeError

    @property
    @unit_decorator_getter(unit=units.ARBITRARY)
    def SQw_err(self):

        """
        Get the errors on the dynamic structure factor in arb

        Returns
        -------
            2D array of S(Q, w) error floats with arbitrary units
        """

        try:
            return self.errors['SQw']
        except KeyError:
            raise AttributeError

    @property
    def t(self):

        """
        Get or set the times of the intermediate scattering function in units of
        fs

        Returns
        -------
        array
            1D array of times in fs
        """

        return self._t

    @t.setter
    @unit_decorator(unit=units.TIME)
    def t(self, value):

        self._t = value

    @property
    def t_res(self):

        """
        Get or set the time resolution used to calculate the dynamic structure
        factor, S(Q, w), from the intermediate scattering function, F(Q, t)

        Returns
        -------
        float
            The time resolution
        """

        return self._t_res

    @t_res.setter
    @unit_decorator(unit=units.TIME)
    def t_res(self, value):

        self._t_res = value

    def read_from_file(self, reader, file_name):

        """
        Reads in experimental data from a file using a specified reader

        Parameters
        ----------
        reader : str
            The name of the required file reader
        file_name : str
            The name of the file
        """

        super().read_from_file(reader, file_name)
        self._independent_variables = self.reader.independent_variables
        self._dependent_variables = self.reader.dependent_variables
        self._errors = self.reader.errors

    def calculate_from_MD(self, MD_input, **settings):

        """
        Calculate the dynamic structure factor, S(Q, w) from a MD Trajectory

        Currently sets all errors to 0 when S(Q,w) is calculated from MD

        Independent variables can either be set previously or defined within
        settings.

        Parameters
        ----------
        MD_input : Trajectory
            An MD Trajectory
        **settings
            n_Q_vectors : int
                The maximum number of Q vectors for any Q value. The greater the
                number of Q vectors, the more accurate the calculation, but the
                longer it will take.
            dimensions : list, tuple, array
                A 3 element tuple or NumPy array of floats specifying the
                dimenions of the universe in units of Ang
        """

        self._origin = 'MD'
        self.trajectory = MD_input
        self.t = self.trajectory.times - self.trajectory.times[0]
        try:
            self.universe_dimensions = self.trajectory.dimensions
        except AttributeError:
            try:
                self.universe_dimensions = np.array(settings['dimensions'])
            except KeyError:
                raise AttributeError('Either trajectory requires a dimensions'
                                     ' attribute or dimensions must be passed'
                                     ' when calling calculate_from_MD')
        self.t_res = settings['t_resolution']
        self._set_weights()

        # Create independent_variables dictionary if it doesn't exist
        if not hasattr(self, 'independent_variables'):
            self.independent_variables = {}

        self.reciprocal_basis = (np.array(2. * np.pi / self.universe_dimensions)
                                 * UNIT_VECTOR)

        dt = self.t[1] - self.t[0]
        # Test that, if there is an existing E, it is consistent with E
        # calculated from trajectory times
        try:
            assert_allclose(self._calculate_E(len(self.E), dt),
                            self.E,
                            rtol=1e-5,
                            err_msg=("Set E values and calculated E values are"
                                     " not consistent"))
        except AttributeError:
            self.independent_variables['E'] = self._calculate_E(len(self.t), dt)

        # Overwrite independent variable 'Q' if it already exists
        try:
            self.independent_variables['Q'] = np.array(settings['Q_values'])
        except KeyError:
            pass

        self.isotropic = settings.get('isotropic', True)
        if not self.isotropic:
            self.direction = np.array(settings.get('direction', [1, 0, 0]))

        self.n_Q_vectors = settings.get('n_Q_vectors', 50)
        if not hasattr(self, 'Q_vectors'):
            try:
                self.Q_vectors = np.array(settings['Q_vectors'])
            except KeyError:
                self.Q_vectors = self._calculate_Q_vectors(self.Q)

        self.FQt = self.calculate_FQt()

        self._dependent_variables = {'SQw':self._calculate_SQw()}
        self._errors = {'SQw':np.zeros(np.shape(self.SQw))}

    @abstractmethod
    def _set_weights(self):

        """
        Calculate the neutron weighting
        """

        pass

    def _calculate_E(self, nE, dt):

        """
        Calculates E from trajectory times

        Parameters
        ----------
        nE : int
            The number of E values to be calculated
        dt : float
            The step size of the time in fs

        Returns
        -------
        array
            An array of floats specifying the energy in units of meV
        """

        return h_bar * 1e15 * np.pi * np.arange(nE) / (nE * dt / 1000)

    def calculate_FQt(self):

        """
        Calculates the intermediate scattering function for all Q vectors for
        all time intervals

        Returns
        -------
        array
            An array of dimensions determined by the number of times and Q
            values
        """

        comm = MPI.COMM_WORLD
        # Determine the shape of Q vectors array. If the number of processors
        # (comm.size) is not a factor of the first index, mpi4py cannot split
        # the number of Q vectors equally amongst the processors.
        shape = list(np.shape(self.Q_vectors))
        if shape[0] % comm.size != 0:
            # Determine the smallest integer larger than the number of Q vectors
            # that is exactly divisible by the number of processors
            axis_0 = int(np.ceil(float(shape[0]) / comm.size) * comm.size)
            # Increase the size of Q vectors up to the required size by padding
            # the start of the array with zeroes
            Q_vectors = np.pad(self.Q_vectors, (axis_0-shape[0], 0), 'constant')
            # Change these zeroes to nan's as this can be passed to calculate
            # rho in the _calculate_FQt_single_Q method, resulting in an array
            # of nan's for each zero element.  These arrays are then removed
            # after gathering.
            Q_vectors[:axis_0-shape[0]] = np.float('nan')
        else:
            Q_vectors = self.Q_vectors
            axis_0 = shape[0]
        # Split the Q vectors into a single array of Q vectors for each
        # processor
        Q_vectors = np.split(Q_vectors, comm.size)
        # Scatter the Q vector arrays to all processors
        Q_vectors = comm.scatter(Q_vectors, root=0)
        # Calculate FQt for each Q vector for all processors
        FQt = np.array([self._calculate_FQt_single_Q(Q_v) for Q_v
                        in Q_vectors])

        # Gather the calculated FQt's together on every processor. This ensures
        # that all other calculations can be performed on every processor, so
        # no other methods in SQw need to be made MPI compliant.
        FQt = np.array(comm.allgather(FQt))
        # Reshape FQt as gather doesn't join the arrays but just collects them
        # as arrays within an array. This is equivalent to flattening the first
        # index.
        FQt_shape = np.shape(FQt)
        FQt = FQt.reshape([FQt_shape[0] * FQt_shape[1], FQt_shape[2]])

        # Remove the padded elements at the start of FQt which will be filled
        # with nan's
        return FQt[axis_0 - shape[0]:]

    @abstractmethod
    def _calculate_FQt_single_Q(self, Q_vector):

        """
        Calculates intermediate scattering function for a single Q value for all
        time intervals (t)

        Parameters
        ----------
        Q_vector : array
            An array of one or more Q vectors with the same Q value
        """

        pass

    def _calculate_rho(self, Q_vector):

        """
        Calculates time dependent number density in reciprocal space for all Q
        vectors

        As rho is the sum of the contributions for all of the specified Q
        vectors, these Q vectors should have the same Q value. Includes
        contributions from all atoms in the trajectory.

        Parameters
        ----------
        Q_vector : array
            An array of one or more Q vectors with the same Q value
        """

        @jit('float64[:,:], float64[:,:]', nopython=True)
        def func(positions, Q_vector):

            return [np.exp(-1j * np.dot(Q_vector, positions[i])) for i in range(len(positions))]

        rho_all_atoms = [func(conf.positions, np.array(Q_vector)) for conf in self.trajectory]

        return np.array(rho_all_atoms)

    def _calculate_Q_vectors(self, Q_values):

        """
        Calculates a number of Q vectors for each Q value

        The upper limit of the number of Q vectors for a specific Q value is
        determined by self.n_Q_vectors

        Parameters
        ----------
        Q_value : list
            A list of floats for the Q values

        Returns
        -------
        array
            an array of arrays of Q vectors, one array for each Q value
        """

        # Only valid for uniform Q_values
        Q_step = (Q_values[1] - Q_values[0]) / 2.

        Q_vectors = []
        updated_Q_values = []
        for Q in Q_values:

            Q_min = Q - Q_step
            Q_max = Q + Q_step

            vectors = self._calculate_vectors_single_Q(Q_min, Q_max)

            if len(vectors) > 0:
                Q_vectors.append(np.array(vectors))
                updated_Q_values.append(Q)

        self.Q_values = updated_Q_values

        return np.array(Q_vectors)

    @staticmethod
    @jit('float64[:], float64[:,:]', nopython=True)
    def _rho(r, Q_vector):

        """
        Calculates the reciprocal space density

        Parameters
        ----------
        r : array
            A 3 element array specifying the position vector
        Q_vector : array
            An array of one or more orthogonal Q vectors (arrays)

        Returns
        -------
        array
            The reciprocal space density for a position and Q vectors
        """

        return np.exp(-1j * np.dot(Q_vector, r))

    def _calculate_vectors_single_Q(self, Q_min, Q_max):

        """
        Calculates a number of Q vectors for Q values within a range

        The upper limit of the number of Q vectors is determined by
        self.n_Q_vectors

        Parameters
        ----------
        Q_min : float
            The minimum Q value for which a Q vector can be calculated
        Q_max : float
            The maximum Q value for which a Q vector can be calculated

        Returns
        -------
        array
            an array of Q vectors which lie within the range defined by Q_min
            and Q_max
        """

        x_max, y_max, z_max = (int(Q_max / np.linalg.norm(r_b)) for r_b
                               in self.reciprocal_basis)

        vectors = []
        for l in range(-x_max, x_max + 1):
            for m in range(-y_max, y_max + 1):
                for n in range(-z_max, z_max + 1):

                    if l == m == n == 0:
                        continue

                    vector = np.array(l * self.reciprocal_basis[0]
                                       + m * self.reciprocal_basis[1]
                                       + n * self.reciprocal_basis[2])

                    if Q_min < np.linalg.norm(vector) <= Q_max:
                        vectors.append(vector)

                    if len(vectors) >= self.n_Q_vectors:
                        return np.array(vectors)

        return np.array(vectors)

    def _calculate_SQw(self):

        """
        Calculates S(Q, w) from F(Q, t)

        Returns
        -------
        array
            The S(Q, w) calculated from F(Q, t)
        """

        FQt_res = self._apply_instrument_resolution(self.FQt,
                                                    sigma=self.t_res)

        # Reflect F(t) [except for both end points] for each Q value and append
        # it to F(t) to form an array of shape (n_row, 2*n_col - 2)
        FQt_mirror = np.append(FQt_res, FQt_res[:,-2:0:-1], axis=1)

        # Normalisation requires factor of dt
        # see Kneller et al. Comput. Phys. Commun. 91 (1995) 191-214
        dt = (self.t[1] - self.t[0]) / 1000.

        # FFT and reduce the temporal dimension back to that of F(Q,t), with the
        # factor of 0.5 accounting for the fft over the reflected F(Q,t)
        # By default numpy fft is unnormalized, so to have the same power as in
        # FQt the transform should be normalized to the length of the spectra
        return (0.5 * dt * np.real(np.fft.fft(FQt_mirror)[:, :len(self.E)])
                / len(FQt_mirror))

    def _apply_instrument_resolution(self, FQt, function=gaussian, **params):

        """
        Applies the specified resolution function to the S(Q,w) data

        As the S(Q,w) data is calculated from the time domain Fourier transform,
        F(Q,t), the resolution function can be applied multiplicatively, rather
        than by convolution.  Assumes that the temporal resolution has no Q
        dependence.

        CURRENTLY SQw is hard coded to only apply Gaussian resolution functions

        Parameters
        ----------
        FQt : array
            The F(Q, t) to which the resolution function is applied
        function : function, optional
            The resolution function to apply. The default is gaussian.
        **params
            sigma : float
                The sigma of the gaussian distribution.

        Returns
        -------
        array
            An array of the same dimensions as FQt
        """

        # Functions other than Gaussians must be FFT before multiplication
        N_Q = np.shape(FQt)[0]
        window = function(self.t[:len(self.E)] / 1000., params['sigma'],
                          norm=False)

        # Broadcast the window so that it is applied for all Q values
        return np.broadcast_to(window, (N_Q, ) + np.shape(window)) * FQt


class SQw(AbstractSQw):

    """
    A class for containing, calculating and reading the total dynamic structure
    factor
    """

    def _set_weights(self):

        """
        Calculate the neutron weighting for coherent and incoherent scattering
        """

        self.weights = {element:{'coh':B_COH[element],
                                    'incoh':B_INCOH[element]}
                           for element in self.trajectory.element_set}

    def _calculate_FQt_single_Q(self, Q_vector):

        """
        Calculates the F(Q, t) for a single Q value

        The length of the correlations is bounded by the length of the energies
        rather the times, as this allows energies to be calculated from
        trajectories with longer timescales than is required by the energy
        resolution.

        Parameters
        ----------
        Q_vector : array
            an array of one or more Q vectors with the same Q value

        Returns
        -------
        array
            An array with dimensions of self.t
        """

        rho = self._calculate_rho(Q_vector)

        elements = self.trajectory.element_set
        rho_element = {}
        n_atoms = 0
        for element in elements:
            indexes = np.where(np.array(self.trajectory.element_list)
                               == element)
            rho_element[element] = np.array([np.sum(rho_t[indexes], axis=0)
                                             for rho_t in rho])
            n_atoms += np.shape(indexes)[1]

        # Calculates the coherent contribution to SQw
        FQt_single_Q = np.zeros(len(self.E))
        for element1 in elements:
            for element2 in elements:

                FQt_single_Q += self.weights[element1]['coh'] \
                                * self.weights[element2]['coh'] \
                                * correlation(rho_element[element1],
                                              rho_element[element2],
                                              normalise=True)[:len(self.E)]

        # Calculates the incoherent contribution to SQw
        incoh_weights = [self.weights[atom.element]['incoh'] for atom
                        in self.trajectory.atoms]
        for i in np.arange(n_atoms):
            rho_atom = np.array([rho_t[i] for rho_t in rho])
            FQt_single_Q_atom = correlation(rho_atom,
                                            normalise=True)[:len(self.E)]
            FQt_single_Q += FQt_single_Q_atom * incoh_weights[i]**2

        # Normalise to the number of orthogonal vectors
        try:
            norm = np.shape(Q_vector)[0]
        except IndexError:
            norm = 1.

        return FQt_single_Q / (n_atoms * norm)
