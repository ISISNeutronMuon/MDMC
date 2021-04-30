"""Module for AbstractSQw and total SQw class"""

from abc import abstractmethod

from mpi4py import MPI
from numba import jit
import numpy as np
from numpy.testing import assert_allclose
from typing import Callable
from typing import Dict

from MDMC.common import units
from MDMC.common.atom_properties import B_COH, B_INCOH
from MDMC.common.constants import h, h_bar
from MDMC.common.decorators import unit_decorator, unit_decorator_getter
from MDMC.common.mathematics import correlation, UNIT_VECTOR
from MDMC.common.resolution_functions import gaussian
from MDMC.trajectory_analysis.observables.obs import Observable
from MDMC.trajectory_analysis.observables.obs_factory import ObservableFactory


class AbstractSQw(Observable):

    """
    An abstract class for total, coherent and incoherent dynamic structure
    factors. The equations used for calculating this are based on Kneller et
    al. Comput. Phys. Commun. 91 (1995) 191-214.
    """

    def __init__(self):
        self._independent_variables = None
        self._dependent_variables = None
        self._errors = None
        # Use FFT by default
        self._use_FFT = True

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

    def minimum_frames(self, dt: float = None):

        r"""
        The minimum number of ``Trajectory`` frames needed to calculate the
        ``dependent_variables`` depends on ``self.use_FFT``.

        If `True`, it is the number of energy steps + 1, in order to allow for
        a reflection in time which only counts the end points once.

        Otherwise, there is not a hard minimum on number of frames. However, to
        distinguish our smallest differences in energy :math:`F(Q,t)` needs to
        cover at least a time period :math:`T_{min}` such that:

        .. math::

            T_{min} \sim \frac{h}{\Delta E_{min}}

        Due to the aforementioned reflection in the time domain, to cover a
        period of :math:`T_{min}` we only need :math:`N` frames:

        .. math::

            N = \frac{T_{min}}{2 \Delta t} + 1 = \frac{h}{2 \Delta t \Delta E_{min}} + 1

        Parameters
        ----------
        dt : float, optional
            The time seperation of frames in ``fs``, default is `None`

        Returns
        -------
        int
            The minimum number of frames
        """

        nE = len(self.E)
        if self.use_FFT:
            return nE + 1

        # Either take the smallest absolute energy, or the smallest seperation
        # of energies we wish to discriminate between
        limiting_energy = np.min(np.abs(self.E[self.E != 0]))
        for i in range(1, nE):
            energy_step = self.E[i] - self.E[i-1]
            limiting_energy = min(limiting_energy, energy_step)

        required_time = h * 1e18 / limiting_energy
        return int(np.ceil(required_time / (2 * dt) + 1))

    def maximum_frames(self):

        """
        The maximum number of ``Trajectory`` frames that can be used to
        calculate the ``dependent_variables`` depends on ``self.use_FFT``.

        If `True`, it is the number of energy steps + 1, in order to allow for
        a reflection in time which only counts the end points once.

        Otherwise, there is no limit and all frames will contribute to the
        calculation.

        Returns
        -------
        int
            The maximum number of frames
        """

        if self.use_FFT:
            return len(self.E) + 1

        return None

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
    @unit_decorator_getter(unit=units.Unit('ps') ** -1)
    def w(self):

        """
        Get the angular frequencies

        Returns
        -------
        array
            1D array of angular frequency `float` in units of ``1 / ps``
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
    def e_res(self):

        """
        Get or set the energy resolution (FWHM) used to calculate the dynamic
        structure factor, S(Q, w), from the intermediate scattering function,
        F(Q, t)

        Returns
        -------
        float
            The energy resolution (FWHM) in ``meV``
        """

        return self._e_res

    @e_res.setter
    @unit_decorator(unit=units.ENERGY_TRANSFER)
    def e_res(self, value):

        self._e_res = value

    def validate_energy(self, dt):

        """
        Asserts that the user set frame seperation ``dt`` leads to energy
        seperation that matches same that of the experiment. If not, it
        includes the time separation required in the error.

        Parameters
        ----------
        dt : float
            Frame seperation in ``fs``

        Returns
        -------
        None

        Raises
        ------
        AssertionError
        """

        dt_required = self.calculate_dt()
        if self.use_FFT:
            # When using FFT, require all experimental/simulated energies
            # to match
            energy = self.E
            msg = ("Experimental E values are not consistent with the "
                   "`Simulation`. For the experimental data provided, the "
                   "product of `time_step` and `traj_step` must be {0}, "
                   "but it was {1}".format(dt_required, dt))
            assert_allclose(self.calculate_E(len(energy), dt),
                            energy,
                            rtol=1e-5,
                            err_msg=msg)
        else:
            # When not using FFT, there is not a hard requirement to match
            # the energies, instead impose a requirement that our frame
            # seperation is small enough to capture the highest frequencies
            msg = ("Maximum experimental E value has a time period smaller"
                   " than the frame seperation. The product of `time_step`"
                   " and `traj_step` must be less than {0}, but it was {1}"
                   "".format(dt_required, dt))
            # Allow for rounding errors by using isclose
            isclose = np.isclose(dt, dt_required, rtol=1e-5)
            assert isclose or dt <= dt_required, msg

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

        # Convert the user friendly ueV into preferred system unit of meV
        self.e_res = settings['energy_resolution'] / 1000
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
            self.validate_energy(dt)
        elif self.independent_variables:
            self.independent_variables['E'] = self.calculate_E(len(self.t) - 1, dt)
        else:
            self.independent_variables = {'E':self.calculate_E(len(self.t) - 1, dt)}
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

    def calculate_E(self, nE: int, dt: float):

        r"""
        Calculates an array of ``nE`` uniformly spaced energy values from the
        time separation of the ``Trajectory`` frames, ``dt``. The frequencies
        are determined by the Fast Fourier Transform, as implemented by numpy,
        for ``2 * nE`` points in time which we then crop to only include ``nE``
        positive frequencies. As we are dealing with frequency rather than
        angular frequency here, the relation to between energy is given by:

        .. math::

            E = h \nu

        Parameters
        ----------
        nE : int
            The number of energy values to be calculated
        dt : float
            The step size between frames in ``fs``

        Returns
        -------
        numpy.ndarray
            An ``array`` of `float` specifying the energy in units of ``meV``
        """

        return h * 1e18 * np.fft.fftfreq(2 * int(nE), dt)[:int(nE)]

    def calculate_dt(self):

        r"""
        Calculates the time separation of frames required by the experimental
        dataset, assuming uniform spacing. Note that this may be different from
        the time separation that the user has given as an input, as it only
        depends on the current values for ``self.E``. The relationship between
        time and energy comes from the numpy implementation of the FFT for
        ``2 * nE`` points where:

        .. math::
            \nu_{max} &=& \frac{n_E - 1}{2 n_E \Delta t} \\\\
            \therefore \Delta t &=& \frac{h (n_E - 1)}{2 n_E E_{max}}

        Returns
        -------
        float
            The time separation required by the current values of ``self.E``
        """

        nE = len(self.E)
        return h * 1e18 * (nE - 1) / (2 * nE * (np.max(np.abs(self.E))))

    def calculate_FQt(self):

        """
        Calculates the intermediate scattering function for all ``Q_vectors``
        for all time intervals

        Returns
        -------
        numpy.ndarray
            An array of dimensions ``(len(self.Q_vectors), len(self.t))``
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
            # the end of the array with NaNs. This can be passed to calculate
            # rho in the _calculate_FQt_single_Q method, resulting in an array
            # of NaN's for each zero element.  These arrays are then removed
            # after gathering.
            if len(shape) == 3:
                Q_vectors = np.pad(self.Q_vectors,
                                   ((0, axis_0 - shape[0]), (0, 0), (0, 0)),
                                   'constant',
                                   constant_values=(np.float('nan')))
            else:
                # If we do not have a well defined shape (i.e. not every Q
                # value has the same number of points in reciprocal space) then
                # we need to manually pad Q_vectors using lists, as numpy
                # arrays would need to have the same shape to be appended.
                padding = np.array([np.full(3, np.float('nan'))])
                padding_list = [padding for _ in range(axis_0 - shape[0])]
                Q_vectors_list = list(self.Q_vectors)
                Q_vectors_list.extend(padding_list)
                Q_vectors = np.array(Q_vectors_list)
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

        # Remove the padded elements at the end of FQt which will be filled
        # with NaN's
        return FQt[:shape[0] - axis_0]

    @abstractmethod
    def _calculate_FQt_single_Q(self, single_Q_vectors):

        r"""
        Calculates the F(Q, t) from an array of vectors corresponding to a
        single value of Q.

        The length of the correlations is bounded by the length of the
        ``self.E + 1`` rather than ``self.t``, as this allows energies to be
        calculated from trajectories with longer timescales than is required by
        the energy resolution.

        We start by calculating the Fourier transformed number densities for
        the atoms :math:`j` of element :math:`\alpha`:

        .. math::

            \rho_{\alpha, m_Q}(n_t, p) = \sum_{j \in \alpha} e^{-i \vec{q_{p}} \cdot \vec{r_j}}

        Where :math:`n_t` indexes time and :math:`p` indexes momentum vector.
        F(Q,t) is calculated from the correlation :math:`C` between these
        number densities, where we make use of the correlation theorem of
        discrete periodic functions to speed up calculation using the FFT
        [see E.O. Brigham, The Fast Fourier Transform, 1974]:

        .. math::

            C_{\alpha,\beta,m_Q}(n_t, p) = \Re \Big[\frac{1}{N_E - |n_t|} \mathcal{F}_t^{-1} \big[ \tilde{\rho'}^*_{\alpha, m_Q}(n_E, p) \tilde{\rho'}_{\beta, m_Q}(n_E, p) \big] \Big]

        Where we denote the Fourier transform of :math:`\rho` as:

        .. math::

            \tilde{\rho'}_{\alpha, m_Q}(n_E, p) = \mathcal{F}_t \big[ \rho'_{\alpha, m_Q}(n_t, p) \big]

        Where :math:`\rho'` denotes that :math:`\rho` has been padded with
        zeros in the time domain to give it twice its orginal length.

        For the coherent contribution, we calculate:

        .. math::

            F_{m_Q}^{coh}(n_t) = \sum_{\alpha} \sum_{\beta} B^{coh}_\alpha B^{coh}_\beta \sum_p C_{\alpha,\beta, m_Q}(n_t, p)

        For the incoherent contribution, we calculate:

        .. math::

            F_{m_Q}^{inc}(n_t) = \sum_{\alpha} \big( B^{inc}_\alpha \big) ^2 \sum_p C_{\alpha,\alpha, m_Q}(n_t, p)

        The ideal (not accounting for instrument resolution) scattering
        function is normalised by both the number of atoms that contributed and
        the number of Q vectors used for the single value of Q. Including both
        coherent and incoherent contributions gives:

        .. math::

            F_{m_Q}^{ideal}(n_t) = \frac{ F_{m_Q}^{coh}(n_t) +  F_{m_Q}^{inc}(n_t)}{N_{atoms} N_p

        If we were only considering the coherent/incoherent scattering
        function, then the other term is simply omitted from the numerator in
        the previous equation.

        Parameters
        ----------
        single_Q_vectors : numpy.ndarray
            An array of Q vectors with approximately the same magnitude

        Returns
        -------
        numpy.ndarray
            An ``array`` with length ``len(self.E) + 1``
        """

        raise NotImplementedError

    def _calculate_Q_vectors(self, Q_values):

        """
        For each value of Q in ``Q_values`` calculates a number of Q vectors
        (points in reciprocal space) that lie close to that Q value.

        The upper limit of the number of Q vectors for a specific Q value is
        determined by ``self.n_Q_vectors``, however in the case that there are
        less than ``self.n_Q_vectors`` close to Q then the number of Q vectors
        will be less than ``self.n_Q_vectors``. This means in general, the
        second dimension of the returned array is not well defined.

        Parameters
        ----------
        Q_value : list
            A ``list` of ``float`` for the Q values

        Returns
        -------
        numpy.ndarray
            A three dimensional array where the first dimension corresponds to
            each entry in ``Q_values``, the second dimension is of variable
            length and contains a number of points in reciprocal space, which
            are in turn length 3 arrays or "vectors" in reciprocal space.
        """

        # Only valid for uniform Q_values
        Q_step = (Q_values[1] - Q_values[0]) / 2.

        Q_vectors = []
        updated_Q_values = []
        for Q in Q_values:
            # For each ``Q``, define a shell in momentum space bounded by
            # ``Q_min`` and ``Q_max`` in which to search for vectors
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
        Calculates a number of Q vectors that have a magnitude between
        ``Q_min`` and ``Q_max``.

        The upper limit of the number of Q vectors is determined by
        ``self.n_Q_vectors``.

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

        # Define a cube in reciprocal space from the limit of ``Q_max``
        x_max, y_max, z_max = (int(Q_max / np.linalg.norm(r_b)) for r_b
                               in self.reciprocal_basis)

        vectors = []
        for l in range(-x_max, x_max + 1):
            for m in range(-y_max, y_max + 1):
                for n in range(-z_max, z_max + 1):
                    # Within this cube, iterate over reciprocal lattice points

                    if l == m == n == 0:
                        continue

                    vector = np.array(l * self.reciprocal_basis[0]
                                      + m * self.reciprocal_basis[1]
                                      + n * self.reciprocal_basis[2])

                    # If a point satisfies the requirements, append it to the
                    # list
                    if Q_min < np.linalg.norm(vector) <= Q_max:
                        vectors.append(vector)

                    # Return early if we reach our upper limit ``n_Q_vectors``
                    if len(vectors) >= self.n_Q_vectors:
                        return np.array(vectors)

        return np.array(vectors)

    def _calculate_SQw(self):

        """
        Calculates S(Q, w) from F(Q, t), accounting for instrument resolution.

        In order to obtain ``len(self.E)`` values in energy, we reflect the
        intermediate scattering function in time to give it dimensions of
        ``(len(self.Q), 2 * (len(self.t)) - 2)``. This uses the fact it is even
        in time, and the number of time points is chosen to be 1 greater than
        the number of energy points [Rapaport, The Art of Molecular Dynamics
        Simulation (2nd Edition), 2004, page 142].

        The numpy implementation of the FFT gives frequencies arranged so that
        the first ``len(self.E)`` points in the energy dimension correspond to
        positive frequencies, and the remaining points have negative frequency.

        Returns
        -------
        numpy.ndarray
            The S(Q, w) calculated from F(Q, t)
        """

        nE = len(self.E)
        if self.use_FFT:
            # Ensure that if we recorded a longer trajectory than required by
            # the FFT, we crop it to match the energy points. This should
            # already be the case, but if the energy values and trajectories
            # are manually provided it may not be.
            self.FQt = self.FQt[:, :nE + 1]

        FQt_res = self._apply_instrument_resolution(self.FQt)

        # Reflect F(t) [except for both end points] for each Q value and append
        # it to F(t) to form an array of shape (n_row, 2*n_col - 2)
        FQt_mirror = np.append(FQt_res, FQt_res[:, -2:0:-1], axis=1)

        if self.use_FFT:
            # FFT and reduce the energy dimension to positive energies
            SQw_cropped = np.fft.fft(FQt_mirror)[:, :nE]
        else:
            SQw_cropped = np.zeros((len(FQt_mirror), nE))
            for i, energy in enumerate(self.E):
                # Create 1D array of exponential factors. Dotting with F(Q,t)
                # sums over the time/energy dimension as required for a
                # discrete Fourier transform
                exp = np.exp(-1e-18j * energy * self.t / h_bar)
                exp_mirror = np.append(exp, exp[-2:0:-1])
                SQw_cropped[:, i] = np.dot(FQt_mirror, exp_mirror)

        # Normalisation requires factor of dt (in ps, so convert from fs)
        # see Kneller et al. Comput. Phys. Commun. 91 (1995) 191-214
        dt = (self.t[1] - self.t[0]) / 1000.
        # The factor of 0.5 accounts for transforming over the reflected F(Q,t)
        # By default numpy fft is unnormalized, so to have the same power as in
        # FQt the transform should be normalized to the length of the spectra
        return 0.5 * dt * np.real(SQw_cropped) / len(FQt_mirror)

    def _apply_instrument_resolution(self, FQt: np.ndarray,
            function: Callable[..., np.ndarray]=gaussian) -> np.ndarray:

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

        Returns
        -------
        numpy.ndarray
            An ``array`` of the same dimensions as ``FQt``
        """

        # Functions other than Gaussians must be FFT (from the energy/frequency
        # domain to time domain) before multiplication. We convert the FWHM
        # energy resolution (in meV) into sigma_t (in fs) using the inverse
        # relationship between the width of a Gaussian and its Fourier
        # transform rather than explicitly transforming it.
        sigma_t = (2 * np.sqrt(2 * np.log(2)) * h_bar * 1e18) / self.e_res
        N_Q, N_T = np.shape(FQt)
        window = function(self.t[:N_T], sigma_t, norm=False)

        # Broadcast the window so that it is applied for all Q values
        return np.broadcast_to(window, (N_Q, N_T)) * FQt

    @property
    def dependent_variables_structure(self) -> Dict[str, list]:
        """
        The order in which the 'SQw' dependent variable is indexed in terms of 'Q' and 'E'.
        Explicitly: we have that self.SQw[Q_index, E_index] is the data point for given indices of self.Q and self.E
        It also means that:
        np.shape(self.SQw)=(np.size(self.Q), np.size(self.E))

        The purpose of this method is to ensure consistency between different readers/methods which create ``SQw``
        objects.

        Return
        ------
        Dict[str, list]
            The shape of the SQw dependent variable
        """
        return {'SQw': ['Q', 'E']}

    @property
    def uniformity_requirements(self) -> Dict[str, Dict[str, bool]]:
        """
        Captures the current limitations on the energy 'E' and reciprocal
        lattice points 'Q' within the dynamic structure factor ``Observables``.
        If using FFT, then 'E' must be uniform and start at zero, otherwise it
        has no restrictions. 'Q' must be uniform but does not need to start at
        zero.

        Return
        ------
        Dict[str, Dict[str, bool]]
            Dictionary of uniformity restrictions for 'E' and 'Q'.
        """

        if self.use_FFT:
            e_requirements = {'uniform': True, 'zeroed': True}
        else:
            e_requirements = {'uniform': False, 'zeroed': False}

        return {'E': e_requirements, 'Q': {'uniform': True, 'zeroed': False}}


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

    def _calculate_FQt_single_Q(self, single_Q_vectors):
        # Inherit docstring of abstract method

        n_t = len(self.t)
        elements = self.trajectory.element_set
        FQt_single_Q = np.zeros(n_t)
        rho_element = {}
        n_atoms = 0

        for element in elements:
            # Get the positions of all atoms (the configuration) of each
            # element over time such that ``element_configs`` has time as its
            # first dimension and each atom of ``element`` as its second
            indexes = np.where(np.array(self.trajectory.element_list)
                               == element)
            element_configs = [config.positions[indexes] for config
                               in self.trajectory]

            rho_config = np.zeros((len(element_configs),
                                   len(single_Q_vectors)),
                                  dtype=complex)
            for i, positions in enumerate(element_configs):
                # For each time frame ``i`` calculate the Fourier transformed
                # number density and sum over all positions but preserve the
                # second dimension, our array of Q vectors
                rho_unsummed = calculate_rho(positions,
                                             np.array(single_Q_vectors))
                rho_config[i, :] = np.sum(rho_unsummed, axis=0)

            rho_element[element] = rho_config
            n_atoms += np.shape(indexes)[1]

            # Incoherent contribution
            incoh_weights = self.weights[element]['incoh']
            for atom_positions in np.swapaxes(element_configs, 0, 1):
                # Swapping the time and position axes lets us iterate over each
                # atom of ``element``, and gives ``rho_atom`` dimensions of
                # time and our array of Q vectors respectively.
                rho_atom = calculate_rho(atom_positions,
                                         np.array(single_Q_vectors))

                # A sum over the Q vectors is performed within ``correlation``.
                FQt_single_Q_atom = correlation(rho_atom, normalise=True)[:n_t]
                FQt_single_Q += FQt_single_Q_atom * incoh_weights**2

        # Calculates the coherent contribution to SQw
        for element1 in elements:
            for element2 in elements:
                # A sum over the Q vectors is performed within ``correlation``.
                FQt_single_Q += self.weights[element1]['coh'] \
                                * self.weights[element2]['coh'] \
                                * correlation(rho_element[element1],
                                              rho_element[element2],
                                              normalise=True)[:n_t]

        # Normalise to the number of orthogonal vectors
        try:
            norm = np.shape(single_Q_vectors)[0]
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
