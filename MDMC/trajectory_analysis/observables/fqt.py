"""Module for Intermediate Scattering Function class"""
from abc import abstractmethod
from itertools import product
from typing import TYPE_CHECKING

import numpy as np

from MDMC.common import units
from MDMC.common.atom_properties import B_INCOH, B_COH
from MDMC.common.constants import h_bar
from MDMC.common.decorators import unit_decorator, unit_decorator_getter
from MDMC.common.mathematics import faster_correlation,\
     faster_autocorrelation, \
     UNIT_VECTOR
from MDMC.resolution import Resolution
from MDMC.trajectory_analysis.observables.obs import Observable, executor
from MDMC.trajectory_analysis.observables.obs_factory import ObservableFactory
from MDMC.trajectory_analysis.observables.sqw import SQwMixins
from MDMC.trajectory_analysis.compact_trajectory import CompactTrajectory

if TYPE_CHECKING:
    from builtins import function
    from typing import Optional
    from MDMC.trajectory_analysis.trajectory import Trajectory


# pylint: disable=c-extension-no-member


class AbstractFQt(SQwMixins, Observable):
    """
    An abstract class for total, coherent, and incoherent intermediate scattering functions.
    Equations used for calculating this are based on Kneller et
    al. Comput. Phys. Commun. 91 (1995) 191-214.

    Note that properties for MD frames & Q are found in the SQwMixins class.
    """

    def __init__(self):
        super().__init__()
        self._trajectory = None
        self._independent_variables = {}
        self._dependent_variables = {}
        self._errors = None
        # Use FFT by default
        self._use_FFT = True

        self.reciprocal_basis = None
        self.Q_vectors = None
        self.n_Q_vectors = None
        self.Q_values = None
        self.weights = None

    @property
    def independent_variables(self) -> dict:
        """
        Get or set the independent variables: these are
        the frequency Q (in ``Ang^-1``) and time t (in ``fs``)

        Returns
        -------
        dict
            The independent variables
        """
        return self._independent_variables

    @independent_variables.setter
    def independent_variables(self, value: dict) -> None:
        self._independent_variables = value

    @property
    def dependent_variables(self) -> dict:
        """
        Get the dependent variables: this is
        FQt, the intermediate scattering function (in ``arb``)

        Returns
        -------
        dict
            The dependent variables
        """

        return self._dependent_variables

    @property
    def errors(self) -> dict:
        """
        Get or set the errors on the dependent variables, the intermediate
        scattering function (in ``arb``)

        Returns
        -------
        dict
            The errors on the ``dependent_variables``
        """

        return self._errors

    @errors.setter
    def errors(self, value: dict) -> None:

        self._errors = value

    @property
    def t(self) -> 'np.ndarray':
        """
        Get or set the times of the intermediate scattering function in units of
        ``fs``

        Returns
        -------
        numpy.array
            1D array of times in ``fs``
        """

        return self.independent_variables['t']

    @t.setter
    @unit_decorator(unit=units.TIME)
    def t(self, value: 'np.ndarray') -> None:

        self.independent_variables['t'] = value

    @property
    @unit_decorator_getter(unit=units.ARBITRARY)
    def FQt(self) -> 'Optional[list[np.ndarray]]':
        """
        Get or set the dynamic structure factor, F(Q, t), in arb

        Returns
        -------
        list of numpy.ndarray
            `list` of 2D arrays of F(Q, t) `float` with arbitrary units
        """

        try:
            return self.dependent_variables['FQt']
        except KeyError:
            return None

    @FQt.setter
    def FQt(self, value: 'list[np.ndarray]') -> None:

        self.dependent_variables['FQt'] = value

    def calculate_from_MD(self, MD_input: CompactTrajectory, verbose: int = 0,
                          **settings: dict) -> None:
        """
        Calculates the intermediate scattering function from a trajectory.

        ``independent_variables`` can either be set previously or defined within
        ``**settings``.

        Parameters
        ----------
        MD_input : CompactTrajectory
            a single ``CompactTrajectory`` object.
        verbose: int, optional
            The level of verbosity:
            Verbose level 0 gives no information.
            Verbose level 1 gives final time for the whole method.
            Verbose level 2 gives final time and also a progress bar.
            Verbose level 3 gives final time, a progress bar, and time per step.
        **settings
            ``n_Q_vectors`` (`int`)
                The maximum number of ``Q_vectors`` for any ``Q`` value. The
                greater the number of ``Q_vectors``, the more accurate the
                calculation, but the longer it will take.
            ``dimensions`` (`list`, `tuple`, `numpy.ndarray`)
                A 3 element `tuple` or ``array`` of `float` specifying the
                dimensions of the ``Universe`` in units of ``Ang``
        """

        self._origin = "MD"

        # if Q_values are specified, set Q to them
        try:
            self.Q = np.array(settings['Q_values'])
        except KeyError:
            pass

        self.t = MD_input.times - MD_input.times[0]
        self._trajectory = MD_input
        self._set_weights()

        try:
            self.universe_dimensions = MD_input.dimensions
        except AttributeError:
            print("DEBUG: no universe dimensions in the CompactTrajectory")
            try:
                self.universe_dimensions = np.array(settings['dimensions'])
            except KeyError as error:
                raise AttributeError('Either trajectory requires a dimensions'
                                     ' attribute or dimensions must be passed'
                                     ' when calling calculate_from_MD') from error

        self.reciprocal_basis = (np.array(2. * np.pi / self.universe_dimensions)
                                 * UNIT_VECTOR)

        # calculate Q vectors from Q
        self.n_Q_vectors = settings.get('n_Q_vectors', 50)
        if self.Q_vectors is None:
            try:
                self.Q_vectors = np.array(settings['Q_vectors'], dtype='object')
            except KeyError:
                self.Q_vectors = self._calculate_Q_vectors(self.Q)

        # Determine the shape of Q vectors array. If the number of processors
        # (comm.size) is not a factor of the first index, mpi4py cannot split
        # the number of Q vectors equally amongst the processors.
        shape = list(np.shape(self.Q_vectors))

        Q_vectors = self.Q_vectors
        axis_0 = 0
        # Split the Q vectors into a single array of Q vectors for each
        # processor

        # Calculate FQt for each Q vector for all processors
        FQt_array = np.array([self._calculate_FQt_single_Q(Q_v) for Q_v
                              in Q_vectors])

        # Remove the padded elements at the end of FQt which will be filled
        # with NaN's
        self.FQt = FQt_array[:shape[0] - axis_0]

    def _calculate_Q_vectors(self, Q_values: list) -> 'np.ndarray':
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
                Q_vectors.append(vectors)
                updated_Q_values.append(Q)

        self.Q_values = updated_Q_values

        return np.array(Q_vectors, dtype='object')

    def _calculate_vectors_single_Q(self, Q_min: float, Q_max: float) -> list:
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
        list
            A list of Q vectors which lie within the range defined by
            ``Q_min`` and ``Q_max``
        """

        # define a cube in reciprocal space
        x_max, y_max, z_max = (int(Q_max / np.linalg.norm(r_b)) for r_b
                               in self.reciprocal_basis)

        # get the point group of the universe; we can use Wyckoff symmetries
        # to generate vectors more quickly
        point_group = get_point_group(self.universe_dimensions)

        # create components of the Q vector for each axis on each lattice point in the cube
        # note we are only defining vector components for one octant of the cube -
        # Wyckoff symmetries will reflect them into other octants for us
        vector_x = (np.array(list(range(0, x_max + 1))).reshape(-1, 1)
                    * self.reciprocal_basis[0])
        vector_y = (np.array(list(range(0, y_max + 1))).reshape(-1, 1)
                    * self.reciprocal_basis[1])
        vector_z = (np.array(list(range(0, z_max + 1))).reshape(-1, 1)
                    * self.reciprocal_basis[2])

        Q_vectors: list = []
        # combine to create overall vectors for each lattice point in the cube
        # the 'if' part of the generator comprehension ensures that we aren't generating
        # large numbers of duplicate vectors from multiple vectors within the same symmetry group
        vectors = ((x[0] + x[1] + x[2]) for x in product(vector_x, vector_y, vector_z)
                   if not any(v in Q_vectors for v in
                              wyckoff_symmetries((tuple(x[0] + x[1] + x[2])), point_group)))

        # get all vectors that fit our requirements
        for vector in vectors:
            if Q_min < np.linalg.norm(vector) <= Q_max and not vector.all == 0:
                # add vector and all its symmetries
                Q_vectors.extend(wyckoff_symmetries(vector, point_group))
            if len(Q_vectors) >= self.n_Q_vectors:
                break

        return Q_vectors

    @abstractmethod
    def _calculate_FQt_single_Q(self, single_Q_vectors: 'np.ndarray') -> 'np.ndarray':
        # ignore line too long linting as it is necessary for LaTeX formatting
        # pylint: disable=line-too-long
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

    @abstractmethod
    def _set_weights(self) -> None:
        """
        Calculate the neutron weighting
        """

        raise NotImplementedError

    def calculate_SQw(self, energy: 'list[float]', resolution: Resolution = None) -> 'np.ndarray':
        """
        Calculates S(Q, w) from F(Q, t), accounting for instrument resolution.

        In order to obtain ``len(energy)`` values in energy, we reflect the
        intermediate scattering function in time to give it dimensions of
        ``(len(self.Q), 2 * (len(self.t)) - 2)``. This uses the fact it is even
        in time, and the number of time points is chosen to be 1 greater than
        the number of energy points [Rapaport, The Art of Molecular Dynamics
        Simulation (2nd Edition), 2004, page 142].

        The numpy implementation of the FFT gives frequencies arranged so that
        the first ``len(energy)`` points in the energy dimension correspond to
        positive frequencies, and the remaining points have negative frequency.

        Parameters
        ----------
        energy: list of floats
            the list of energy (E) points at which S(Q, w) will be calculated.
        resolution: Resolution (default None)
            The instrument resolution object which will be applied to FQt.

        Returns
        -------
        numpy.ndarray
            The S(Q, w) calculated from F(Q, t)
        """
        nE = len(energy)
        if self.use_FFT:
            # Ensure that if we recorded a longer trajectory than required by
            # the FFT, we crop it to match the energy points. This should
            # already be the case, but if the energy values and trajectories
            # are manually provided it may not be.
            self.FQt = self.FQt[:, :nE + 1]

        if resolution is not None:
            self.apply_resolution(resolution)

        # Reflect F(t) [except for both end points] for each Q value and append
        # it to F(t) to form an array of shape (n_row, 2*n_col - 2)
        FQt_mirror = np.append(self.FQt, self.FQt[:, -2:0:-1], axis=1)

        if self.use_FFT:
            # FFT and reduce the energy dimension to positive energies
            SQw_cropped = np.fft.fft(FQt_mirror)[:, :nE]
        else:
            SQw_cropped = np.zeros((len(FQt_mirror), nE), dtype='complex')
            for i, E in enumerate(energy):
                # Create 1D array of exponential factors. Dotting with F(Q,t)
                # sums over the time/energy dimension as required for a
                # discrete Fourier transform
                # h_bar is in units of eV s whereas system units are meV fs, so
                # apply a factor of 1e3 * 1e15 to convert it
                exp = np.exp(-1j * E * self.t[:-1] / (h_bar * 1e18))
                exp_neg = np.exp(
                    1j * E * self.t[-1:0:-1] / (h_bar * 1e18))
                exp_mirror = np.append(exp, exp_neg)
                SQw_cropped[:, i] = np.dot(FQt_mirror, exp_mirror)

        # Normalisation requires factor of dt (in ps, so convert from fs)
        # see Kneller et al. Comput. Phys. Commun. 91 (1995) 191-214
        dt = (self.t[1] - self.t[0]) / 1000.
        # The factor of 0.5 accounts for transforming over the reflected F(Q,t)
        # By default numpy fft is unnormalized, so to have the same power as in
        # FQt the transform should be normalized to the length of the spectra
        return 0.5 * dt * np.real(SQw_cropped) / len(FQt_mirror)

    def apply_resolution(self, resolution: Resolution) -> "FQt": # type: ignore
        """
        Apply instrument resolution to an FQt object.

        Parameters
        ----------
        resolution: Resolution
            The Resolution object to apply to FQt.

        Returns
        -------
        The FQt object with resolution applied.
        """

        self.FQt = resolution.apply(self.FQt, self.t, self.Q)

        return self.FQt

    @property
    def dependent_variables_structure(self) -> dict[str, list]:
        """
        The order in which the 'FQt' dependent variable is indexed in terms of 'Q' and 't'.
        Explicitly: we have that self.FQt[Q_index, t_index] is the data point
        for given indices of self.Q and self.t
        It also means that:
        np.shape(self.FQt)=(np.size(self.Q), np.size(self.t))

        The purpose of this method is to ensure consistency between
        different readers/methods which create ``FQt`` objects.

        Return
        ------
        dict[str, list]
            The shape of the SQw dependent variable
        """
        return {'FQt': ['Q', 't']}

    @property
    def uniformity_requirements(self) -> dict[str, dict[str, bool]]:
        """
        Captures the current limitations on the time 't' and reciprocal
        lattice points 'Q' within the intermediate scattering function ``Observables``.
        If using FFT, then 't' must be uniform and start at zero, otherwise it
        has no restrictions. 'Q' must be uniform but does not need to start at
        zero.

        Return
        ------
        dict[str, dict[str, bool]]
            Dictionary of uniformity restrictions for 't' and 'Q'.
        """

        if self.use_FFT:
            t_requirements = {'uniform': True, 'zeroed': True}
        else:
            t_requirements = {'uniform': False, 'zeroed': False}

        return {'t': t_requirements, 'Q': {'uniform': True, 'zeroed': False}}


@ObservableFactory.register(('IntermediateScatteringFunction', 'FQt'))
class FQt(AbstractFQt):
    """
    A class for containing, calculating and reading the intermediate scattering
    function for the total dynamic structure factor
    """

    def _set_weights(self) -> None:
        """
        Calculate the neutron weighting for coherent and incoherent scattering
        """

        self.weights = {element: {'coh': B_COH[element],
                                  'incoh': B_INCOH[element]}
                        for element in self._trajectory.element_set}

    def _calculate_FQt_single_Q(self, single_Q_vectors: 'np.ndarray') -> 'np.ndarray':
        # Inherit docstring of abstract method

        n_t = len(self.t)
        elements = self._trajectory.element_set
        FQt_single_Q = np.zeros(n_t)
        rho_element = {}
        n_atoms = 0

        def helper_coherent(configs: np.ndarray,
                            q_vector: np.ndarray) -> np.ndarray:
            """A wrapper for the calculate_rho function and the summation
            of the resulting array. This part of the calculation is handled
            by numpy, and so it is easy to run in parallel.

            Arguments
            ---------
            configs: numpy.ndarray
                array of atom positions, size (N_timesteps, 3, N_atoms)
            q_vector: numpy.ndarray
                q vector in array form, size (3)

            Returns:
                array of rho values summed over the atoms in the system,
                size (N_timesteps)
            """
            return calculate_rho(configs, q_vector).sum(axis = 1)

        for element in elements:
            # Get the positions of all atoms (the configuration) of each
            # element over time such that ``element_configs`` has time as its
            # first dimension and each atom of ``element`` as its second
            indexes = np.where(np.array(self._trajectory.element_list)
                               == element)
            element_configs = self._trajectory.position[:, indexes[0], :]
            rho_config = np.zeros((len(element_configs),
                                   len(single_Q_vectors)),
                                  dtype=complex)

            # For the np.dot product to be broadcast correctly,
            # the [x, y, z] atom positions have to be on axis 1.
            # For this reason we swap the axes, moving the
            # axis of atom numbers to axis 2.
            # Time axis is still axis 0.
            configs = np.swapaxes(element_configs, 1, 2)
            # The single_Q_vectors array contains many q vectors
            # with similar values of |Q|.
            # The following lines split the calculation by multiplying
            # the trajectory by each q vector separately.
            futures = [executor.submit(helper_coherent,
                                       configs, single_Q_vectors[q_num])
                                       for q_num in range(len(single_Q_vectors))]
            # The following line makes the code wait for all the calculations to finish.
            results = [future.result() for future in futures]
            # At this stage, the results list is fully populated,
            # and the following loop writes the results into the rho_config array.
            for q_num in range(len(single_Q_vectors)):
                rho_config[:, q_num] = results[q_num]

            rho_element[element] = rho_config
            n_atoms += np.shape(indexes)[1]

            # Incoherent contribution
            incoh_weights = self.weights[element]['incoh']
            configs = np.swapaxes(element_configs,
                                  1,
                                  2)
            configs = np.swapaxes(configs,
                                  0,
                                  2)
            rho_all = calculate_rho(configs, np.array(single_Q_vectors))
            futures = [executor.submit(faster_autocorrelation,
                                       rho_all[q_num].T,
                                       weights = incoh_weights**2)
                                       for q_num in range(len(rho_all))]
            results = [future.result()[:n_t] for future in futures]
            for q_num in np.arange(len(rho_all)):
                FQt_single_Q += results[q_num]

        # Calculates the coherent contribution to SQw
        for element1 in elements:
            for element2 in elements:
                # A sum over the Q vectors is performed within ``correlation``.
                FQt_single_Q += self.weights[element1]['coh'] \
                    * self.weights[element2]['coh'] \
                    * faster_correlation(rho_element[element1],
                                  rho_element[element2])[:n_t]

        # Normalise to the number of orthogonal vectors
        try:
            norm = np.shape(single_Q_vectors)[0]
        except IndexError:
            norm = 1

        return FQt_single_Q / (n_atoms * norm)

def calculate_rho(positions: np.ndarray, Q_vector: np.ndarray) -> np.ndarray:
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
    return np.exp(-1j * np.dot(Q_vector, positions))

def get_point_group(dimensions: 'np.ndarray') -> str:
    """
    Gets the Hermann-Mauguin point group for the universe.
    Currently, MDMC can only create universes with mutually orthogonal
    sides, so this method can only produce cubic, tetragonal, or orthorhombic
    universe point groups.

    For tetragonal universes, an additional identifier (x), (y), or (z) is
    added to identify which of the sides is unique.

    Parameters
    ----------
    dimensions: numpy.ndarray
        An array of length 3, containing the dimensions of the universe.

    Returns
    -------
    str
        The point group symbol for the universe.
    """
    # we use a sum of bools to determine group;
    # if x == y, x == z, y == z
    # if all sides equal, all are true; if two sides equal, one is true;
    # if no sides equal, zero are true;

    equal_sides = sum([dimensions[0] == dimensions[1],
                       dimensions[0] == dimensions[2],
                       dimensions[1] == dimensions[2]])

    if equal_sides == 3:  # cubic
        return 'm-3m'
    if equal_sides == 1:  # tetragonal
        # False == 0 in duck typing, so this only keeps
        # the side equal to True
        unique_side = ((dimensions[0] == dimensions[1]) * ' (z)'
                       + (dimensions[0] == dimensions[2]) * ' (y)'
                       + (dimensions[1] == dimensions[2]) * ' (x)')
        return '4/mmm' + unique_side
    return 'mmm'  # orthorhombic


def wyckoff_symmetries(point: tuple, point_group: str) -> 'set[tuple]':
    """
    Returns the Wyckoff symmetries for a point based on its point group.

    Parameters
    -----------
    point: tuple
        A tuple of length 3 which corresponds to the coordinates (x, y, z) of a point.
    point_group: str
        The  point group of the universe in Hermann-Mauguin notation.
        Currently accepted groups are:
            'm-3m' (cubic)
            '4/mmm' (tetragonal)
            'mmm' (orthorhombic)

    Returns
    -------
    set[tuple]
        A calculated set of the symmetries for the point
    """

    def cubic(point: tuple) -> 'set[tuple]':
        """The symmetries of a point in a cubic group."""

        x, y, z = point
        # it's ugly, but an order of magnitude faster if we just list all the
        # symmetries out, calculate and return them
        return ({(x, y, z), (-x, -y, z), (-x, y, -z), (x, -y, -z), (z, x, y), (z, -x, -y),
                 (-z, -x, y), (-z, x, -y), (y, z, x), (-y, z, -x), (y, -z, -x), (-y, -z, x),
                 (y, x, -z), (-y, -x, -z), (y, -x, z), (-y, x, z), (x, z, -y), (-x, z, y),
                 (-x, -z, -y), (x, -z, y), (z, y, -x), (z, -y, x), (-z, y, x), (-z, -y, -x),
                 (-x, -y, -z), (x, y, -z), (x, -y, z), (-x, y, z), (-z, -x, -y), (-z, x, y),
                 (z, x, -y), (z, -x, y), (-y, -z, -x), (y, -z, x), (-y, z, x), (y, z, -x),
                 (-y, -x, z), (y, x, z), (-y, x, -z), (y, -x, -z), (-x, -z, y), (x, -z, -y),
                 (x, z, y), (-x, z, -y), (-z, -y, x), (-z, y, -x), (z, -y, -x), (z, y, x)})

    def tetragonal(unique_side: str) -> 'function':
        """The symmetries of a point in a tetragonal group."""

        # slightly more complicated as we don't know what axis is unpermutable
        def tetragonal_z(point: tuple) -> 'set[tuple]':
            """Tetragonal symmetries for unpermutable z-axis"""
            x, y, z = point
            return ({(x, y, z), (-x, -y, z), (-y, x, z), (y, -x, z), (-x, y, -z), (x, -y, -z),
                    (y, x, -z), (-y, -x, -z), (-x, -y, -z), (x, y, -z), (y, -x, -z), (-y, x, -z),
                    (x, -y, z), (-x, y, z), (-y, -x, z), (y, x, z)})

        def tetragonal_y(point: tuple) -> 'set[tuple]':
            """Tetragonal symmetries for unpermutable y-axis"""
            x, y, z = point
            return ({(x, y, z), (x, -y, z), (x, y, -z), (x, -y, -z), (-x, y, z), (-x, -y, z),
                    (-x, y, -z), (-x, -y, -z), (z, y, x), (z, -y, x), (z, y, -x), (z, -y, -x),
                    (-z, y, x), (-z, -y, x), (-z, y, -x), (-z, -y, -x)})

        def tetragonal_x(point: tuple) -> 'set[tuple]':
            """Tetragonal symmetries for unpermutable x-axis"""
            x, y, z = point
            return ({(x, y, z), (-x, y, z), (x, y, -z), (-x, y, -z), (x, -y, z), (-x, -y, z),
                    (x, -y, -z), (-x, -y, -z), (x, z, y), (-x, z, y), (x, z, -y), (-x, z, -y),
                    (x, -z, y), (-x, -z, y), (x, -z, -y), (-x, -z, -y)})

        # get the correct function and return it as the group function
        side = {'x': tetragonal_x,
                'y': tetragonal_y,
                'z': tetragonal_z}

        return side[unique_side]

    def orthorhombic(point: tuple) -> 'set[tuple]':
        """The symmetries of a point in an orthorhombic group."""

        x, y, z = point
        return ({(x, y, z), (-x, -y, z), (-x, y, -z), (x, -y, -z),
                (-x, -y, -z), (x, y, -z), (x, -y, z), (-x, y, z)})

    groups = {'m-3m': cubic,
              '4/mmm (x)': tetragonal('x'),
              '4/mmm (y)': tetragonal('y'),
              '4/mmm (z)': tetragonal('z'),
              'mmm': orthorhombic}

    return groups[point_group](point)
