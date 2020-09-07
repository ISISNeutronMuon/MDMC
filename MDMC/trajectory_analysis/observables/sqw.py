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
from MDMC.trajectory_analysis.observables.obs_factory import ObservableFactory


class AbstractSQw(Observable):

    """
    An abstract class for total, coherent and incoherent dynamic structure
    factors
    """

    def __init__(self):
        self._independent_variables = {}
        self._dependent_variables = {}
        self._errors = {}

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
        Get or set the independent variables, Q (in ``Ang^-1``) and E (in
        ``meV``)

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
        Get or set the dependent variables, SQw (in ``arb``)

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
            1D array of Q `float` (in ``Ang^-1``)
        """

        try:
            return self.independent_variables['Q']
        except KeyError:
            return None

    @property
    @unit_decorator_getter(unit=units.ENERGY_TRANSFER)
    def E(self):

        """
        Get the energies

        Returns
        -------
        array
            1D array of energy `float` (in ``meV``)
        """

        if self.independent_variables:
            try:
                return self.independent_variables['E']
            except KeyError:
                pass
        return None

    @property
    @unit_decorator_getter(unit=units.ANGLE / units.Unit('ps'))
    def w(self):

        """
        Get the angular frequencies

        Returns
        -------
        array
            1D array of angular frequency `float` in units of ``r``ad ps^-1``
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
            2D array of S(Q, w) `float` with arbitrary units
        """

        try:
            return self.dependent_variables['SQw']
        except KeyError:
            return None

    @property
    @unit_decorator_getter(unit=units.ARBITRARY)
    def SQw_err(self):

        """
        Get the errors on the dynamic structure factor in arb

        Returns
        -------
            2D array of S(Q, w) error `float` with arbitrary units
        """

        try:
            return self.errors['SQw']
        except KeyError:
            return None

    @property
    def t(self):

        """
        Get or set the times of the intermediate scattering function in units of
        ``fs``

        Returns
        -------
        array
            1D array of times in ``fs``
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

    def calculate_from_MD(self, MD_input, **settings):

        """
        Calculate the dynamic structure factor, S(Q, w) from a ``Trajectory``

        Currently sets all errors to 0 when S(Q, w) is calculated from MD

        ``independent_variables`` can either be set previously or defined within
        ``**settings``.

        Parameters
        ----------
        MD_input : Trajectory
            An MD ``Trajectory`` from which the S(Q, w) will be calculated
        **settings
            ``n_Q_vectors`` (`int`)
                The maximum number of ``Q_vectors`` for any ``Q`` value. The
                greater the number of ``Q_vectors``, the more accurate the
                calculation, but the longer it will take.
            ``dimensions`` (`list`, `tuple`, `numpy.ndarray`)
                A 3 element `tuple` or ``array`` of `float` specifying the
                dimensions of the ``Universe`` in units of ``Ang``
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
        if self.E is not None:
            assert_allclose(self.calculate_E(len(self.E), dt),
                            self.E,
                            rtol=1e-5,
                            err_msg=("Set E values and calculated E values are"
                                     " not consistent"))
        elif self.independent_variables:
            self.independent_variables['E'] = self.calculate_E(len(self.t), dt)
        else:
            self.independent_variables = {'E':self.calculate_E(len(self.t), dt)}
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

        # Cleanup the trajectory to reduce memory usage
        self.trajectory = None

    @abstractmethod
    def _set_weights(self):

        """
        Calculate the neutron weighting
        """

        pass

    def calculate_E(self, nE, dt):

        """
        Calculates ``E`` from the ``Trajectory`` times

        Parameters
        ----------
        nE : int
            The number of ``E`` values to be calculated
        dt : float
            The step size of the time in ``fs``

        Returns
        -------
        numpy.ndarray
            An ``array`` of `float` specifying the energy in units of ``meV``
        """

        return h_bar * 1e15 * np.pi * np.arange(nE) / (nE * dt / 1000)

    def calculate_FQt(self):

        """
        Calculates the intermediate scattering function for all ``Q_vectors``
        for all time intervals

        Returns
        -------
        numpy.ndarray
            An ``array`` of dimensions determined by the number of ``t`` and
            ``Q`` values
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
            Q_vectors = np.pad(self.Q_vectors, ((0, axis_0-shape[0]),
                                                (0, 0),
                                                (0, 0)), 'constant')
            # Change these zeroes to nan's as this can be passed to calculate
            # rho in the _calculate_FQt_single_Q method, resulting in an array
            # of nan's for each zero element.  These arrays are then removed
            # after gathering.
            Q_vectors[shape[0] - axis_0:] = np.float('nan')
        else:
            Q_vectors = self.Q_vectors
            axis_0 = 0
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
        return FQt[:shape[0] - axis_0]

    @abstractmethod
    def _calculate_FQt_single_Q(self, Q_vector):

        """
        Calculates intermediate scattering function for a single Q value for
        all ``t`` intervals

        Parameters
        ----------
        Q_vector : numpy.ndarray
            An ``array`` of one or more Q vectors with the same Q value
        """

        raise NotImplementedError

    def _calculate_Q_vectors(self, Q_values):

        """
        Calculates a number of Q vectors for each Q value

        The upper limit of the number of Q vectors for a specific Q value is
        determined by ``self.n_Q_vectors``

        Parameters
        ----------
        Q_value : list
            A `list` of `float` for the Q values

        Returns
        -------
        numpy.ndarray
            An ``array`` of arrays of Q vectors, one array for each ``Q_value``
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

    def _calculate_vectors_single_Q(self, Q_min, Q_max):

        """
        Calculates a number of Q vectors for Q values within a range

        The upper limit of the number of Q vectors is determined by
        ``self.n_Q_vectors``

        Parameters
        ----------
        Q_min : float
            The minimum Q value for which a Q vector can be calculated
        Q_max : float
            The maximum Q value for which a Q vector can be calculated

        Returns
        -------
        numpy.ndarray
            An ``array`` of Q vectors which lie within the range defined by
            ``Q_min`` and ``Q_max``
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
        numpy.ndarray
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

        .. note:: Currently SQw is hard coded to only apply Gaussian resolution
                  functions

        Parameters
        ----------
        FQt : numpy.ndarray
            The F(Q, t) to which the resolution function is applied
        function : function, optional
            The resolution function to apply. The default is gaussian.
        **params
            ``sigma`` (`float`)
                The sigma of the gaussian distribution.

        Returns
        -------
        numpy.ndarray
            An ``array`` of the same dimensions as ``FQt``
        """

        # Functions other than Gaussians must be FFT before multiplication
        N_Q = np.shape(FQt)[0]
        window = function(self.t[:len(self.E)] / 1000., params['sigma'],
                          norm=False)

        # Broadcast the window so that it is applied for all Q values
        return np.broadcast_to(window, (N_Q, ) + np.shape(window)) * FQt


@ObservableFactory.register(('DynamicStructureFactor', 'SQw'))
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
        Q_vector : numpy.ndarray
            An ``array`` of one or more Q vectors with the same Q value

        Returns
        -------
        numpy.ndarray
            An ``array`` with dimensions of ``self.t``
        """

        elements = self.trajectory.element_set
        FQt_single_Q = np.zeros(len(self.E))
        rho_element = {}
        n_atoms = 0

        for element in elements:
            indexes = np.where(np.array(self.trajectory.element_list)
                               == element)
            element_configs = [config.positions[indexes] for config
                               in self.trajectory]
            rho_config = np.zeros((len(element_configs), len(Q_vector)),
                                  dtype=complex)
            for i, positions in enumerate(element_configs):
                rho_config[i, :] = np.sum(calculate_rho(positions,
                                                        np.array(Q_vector)),
                                          axis=0)
            rho_element[element] = rho_config
            n_atoms += np.shape(indexes)[1]

            # Incoherent contribution
            incoh_weights = self.weights[element]['incoh']
            for atom_positions in np.swapaxes(element_configs, 0, 1):
                rho_atom = calculate_rho(atom_positions, np.array(Q_vector))
                FQt_single_Q_atom = correlation(rho_atom,
                                                normalise=True)[:len(self.E)]
                FQt_single_Q += FQt_single_Q_atom * incoh_weights**2

        # Calculates the coherent contribution to SQw
        for element1 in elements:
            for element2 in elements:

                FQt_single_Q += self.weights[element1]['coh'] \
                                * self.weights[element2]['coh'] \
                                * correlation(rho_element[element1],
                                              rho_element[element2],
                                              normalise=True)[:len(self.E)]

        # Normalise to the number of orthogonal vectors
        try:
            norm = np.shape(Q_vector)[0]
        except IndexError:
            norm = 1.

        return FQt_single_Q / (n_atoms * norm)


@jit('float64[:,:], float64[:,:]', nopython=True)
def calculate_rho(positions, Q_vector):

    """
    Calculates ``t`` dependent number density in reciprocal space for all
    Q vectors

    As rho is the sum of the contributions for all of the specified Q
    vectors, these Q vectors should have the same Q value.

    Parameters
    ----------
    positions : numpy.ndarray
        An ``array`` of atomic positions for which the reciprocal space number
        density should be calculated
    Q_vector : numpy.ndarray
        An ``array`` of one or more Q vectors with the same Q value

    Returns
    -------
    numpy.ndarray
        The reciprocal space number density
    """

    return [np.exp(-1j * np.dot(Q_vector, positions[i])) for i
            in range(len(positions))]
