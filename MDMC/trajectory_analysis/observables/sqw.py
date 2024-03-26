"""
Module for AbstractSQw and total SQw class.
"""

from contextlib import suppress
from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
from numpy.testing import assert_allclose
from scipy.interpolate import RectBivariateSpline

from MDMC.common import units
from MDMC.common.constants import h, h_bar
from MDMC.common.decorators import unit_decorator_getter
from MDMC.resolution.resolution_factory import ResolutionFactory
from MDMC.trajectory_analysis.compact_trajectory import CompactTrajectory
from MDMC.trajectory_analysis.observables.obs import Observable
from MDMC.trajectory_analysis.observables.obs_factory import ObservableFactory
from MDMC.utilities.trajectory_slicing import slice_trajectory

#from mpl_toolkits import mplot3d


class SQwMixins:
    """
    A mixin class for properties used by both SQw and FQt objects.
    """

    def minimum_frames(self, dt: float = None) -> int:
        r"""
        Compute minimum number of ``CompactTrajectory`` frames to calculate ``dependent_variables``.

        Parameters
        ----------
        dt : float, optional
            The time separation of frames in ``fs``, default is `None`.

        Returns
        -------
        int
            The minimum number of frames.

        Notes
        -----
        Depends on ``self.use_FFT``.

        If `self.use_FFT == True`, it is the number of energy steps + 1, in order to allow for
        a reflection in time which only counts the end points once.

        If `self.use_FFT == False`, there is not a hard minimum on number of frames. However, to
        distinguish our smallest differences in energy :math:`F(Q,t)` needs to
        cover at least a time period :math:`T_{min}` such that:

        .. math::

            T_{min} \sim \frac{h}{\Delta E_{min}}

        Due to the aforementioned reflection in the time domain, to cover a
        period of :math:`T_{min}` we only need :math:`N` frames:

        .. math::

            N = \frac{T_{min}}{2 \Delta t} + 1 = \frac{h}{2 \Delta t \Delta E_{min}} + 1
        """

        nE = len(self.E)
        if self.use_FFT:
            return nE + 1

        # Either take the smallest absolute energy, or the smallest separation
        # of energies we wish to discriminate between
        minimum_abs_energy = np.min(np.abs(self.E[self.E != 0]))
        minimum_energy_separation = np.min(np.diff(np.sort(self.E)))
        limiting_energy = min(minimum_abs_energy, minimum_energy_separation)

        # h is in units of eV s whereas system units are meV fs, so apply a
        # factor of 1e3 * 1e15 to convert it
        required_time = h * 1e18 / limiting_energy
        # We need an integer number of frames, so round up using np.ceil to
        # ensure we exceed the minimum number of frames needed
        return int(np.ceil(required_time / (2 * dt) + 1))

    def maximum_frames(self) -> Optional[int]:
        """
        Compute maximum number of ``CompactTrajectory`` frames to calculate ``dependent_variables``.

        Returns
        -------
        int
            The maximum number of frames.

        Notes
        -----
        Depends on `self.use_FFT`.

        If `True`, it is the number of energy steps + 1, in order to allow for
        a reflection in time which only counts the end points once.

        Otherwise, there is no limit and all frames will contribute to the
        calculation.
        """

        if self.use_FFT and (self.E is not None):
            return len(self.E) + 1

        return None

    @property
    @unit_decorator_getter(unit=units.LENGTH ** -1)
    def Q(self) -> Optional[np.array]:
        """
        Get or set the momentum transfers.

        Returns
        -------
        numpy.array
            1D array of Q `float` (in ``Ang^-1``).
        """
        try:
            return self.independent_variables['Q']
        except KeyError:
            return None

    @Q.setter
    def Q(self, value: np.array) -> None:
        self.independent_variables['Q'] = value


class AbstractSQw(SQwMixins, Observable):

    """
    An abstract class for total, coherent and incoherent dynamic structure factors.

    The equations used for calculating this are based on Kneller *et al*. [Kneller]_.

    .. note::

       Properties for MD frames & Q are found in the :class:`SQwMixins` class.

    References
    ----------
    .. [Kneller] Kneller *et al*. Comput. Phys. Commun. 91 (1995) 191-214.
    """

    def __init__(self):
        super().__init__()
        self._independent_variables = None
        self._dependent_variables = None
        self._errors = None
        self.resolution = None
        # Use FFT by default
        self._use_FFT = True
        self._recreated_Q = None

    @property
    def independent_variables(self) -> dict:
        """
        Get or set the independent variables.

        These are:

        - the frequency :math:`Q` (in ``Ang^-1``)
        - energy :math:`E` (in``meV``)

        Returns
        -------
        dict
            The independent variables.
        """

        return self._independent_variables

    @independent_variables.setter
    def independent_variables(self, value: dict) -> None:
        self._independent_variables = value

    @property
    def dependent_variables(self) -> dict:
        """
        Get the dependent variables.

        This is:

        - SQw, the dynamic structure factor (in ``arb``)

        Returns
        -------
        dict
            The dependent variables.
        """
        return self._dependent_variables

    @property
    def errors(self) -> dict:
        """
        Get or set the errors on the dependent variables.

        The dynamic structure factor (in ``arb``)

        Returns
        -------
        dict
            The errors on the ``dependent_variables``.
        """
        return self._errors

    @errors.setter
    def errors(self, value: dict) -> None:
        self._errors = value

    @property
    @unit_decorator_getter(unit=units.ENERGY_TRANSFER)
    def E(self) -> 'np.array':
        """
        Get the energies.

        Returns
        -------
        numpy.array
            1D array of energy `float` (in ``meV``).
        """

        if self.independent_variables:
            try:
                return self.independent_variables['E']
            except KeyError:
                pass
        return None

    @E.setter
    def E(self, value: np.array) -> None:
        self.independent_variables['E'] = value

    @property
    @unit_decorator_getter(unit=units.Unit('ps') ** -1)
    def w(self) -> 'np.array':
        """
        Get the angular frequencies.

        Returns
        -------
        numpy.array
            1D array of angular frequency (in ``1 / ps``).
        """

        return self.E / (h_bar * 1e15)

    @property
    @unit_decorator_getter(unit=units.ARBITRARY)
    def SQw(self) -> Optional[list[np.ndarray]]:
        r"""
        Get the dynamic structure factor, :math:`S(Q, \omega{})` (in ``arb``).

        Returns
        -------
        list of numpy.ndarray
            `list` of 2D arrays of :math:`S(Q, \omega{})` (in arbitrary units).
        """

        try:
            return self.dependent_variables['SQw']
        except KeyError:
            return None

    @property
    @unit_decorator_getter(unit=units.ARBITRARY)
    def SQw_err(self) -> Optional[list[np.ndarray]]:
        r"""
        Get the errors on the dynamic structure factor (in ``arb``).

        Returns
        -------
        list of numpy.ndarray
            `list` of 2D arrays of :math:`S(Q, \omega{})` error (in arbitrary units).
        """

        try:
            return self.errors['SQw']
        except KeyError:
            return None

    @property
    def recreated_Q(self) -> np.ndarray:
        """
        Get the indices of the recreated Q_values.

        Returns
        -------
        numpy.ndarray
            Array of recreated Q values.
        """
        return self._recreated_Q

    @recreated_Q.setter
    def recreated_Q(self, recreated_Q_pos: list):
        self._recreated_Q = recreated_Q_pos

    def validate_energy(self, time_step: float = None):
        """
        Ensure ``dt`` is valid for computing energies.

        Asserts that the user set frame separation ``dt`` leads to energy
        separation that matches that of the experiment. If not, it
        changes the time step and trajectory step to fix this. The time step value is
        prioritised here.

        Parameters
        ----------
        time_step : float, optional
            User specified length of time for each update of the atoms trajectories
            in the simulation, default is None.

        Returns
        -------
        valid : bool
            Whether ``dt`` is valid.
        traj_step : Optional[int]
            Ideal ``traj_step`` if provided.
        time_step : Optional[float]
            Ideal ``time_step`` if provided.
        dt : float
            Required ``dt``.
        """

        dt_required = self.calculate_dt()

        if time_step is not None:
            # Changing the time and traj step to fit the required dt value
            # by finding the highest traj_step that can fit into the dt_required
            traj_step = int(np.round(dt_required/time_step))
            if traj_step == 0:
                traj_step += 1
            time_step = dt_required/traj_step

            return True, traj_step, time_step, dt_required
        return False, None, None, dt_required

    def calculate_from_MD(self,
                          MD_input: CompactTrajectory,
                          verbose: int = 0,
                          **settings: dict):
        """
        Calculate the dynamic structure factor, S(Q, w) from a ``CompactTrajectory``.

        If the ``CompactTrajectory`` has more frames than the
        ``self.maximum_frames()`` that can be used to recreate the
        grid of energy points, it can slice the ``CompactTrajectory``
        into sub-trajectories of length ``self.maximum_frames()``,
        with the slicing specified through the settings
        ``use_average`` and ``cont_slicing``.

        The ``independent_variable`` ``Q`` can either be set
        previously or defined within ``**settings``.

        Parameters
        ----------
        MD_input : CompactTrajectory
            An MDMC ``CompactTrajectory`` from which to calculate ``SQw``.
        verbose : int, optional
            The level of verbosity:

            - Verbose level 0 gives no information.
            - Verbose level 1 gives final time for the whole method.
            - Verbose level 2 gives final time and also a progress bar.
            - Verbose level 3 gives final time, a progress bar, and time per step.
        **settings : dict
            Extra options.

        Other Parameters
        ----------------
        n_Q_vectors : int
            The maximum number of ``Q_vectors`` for any ``Q`` value. The
            greater the number of ``Q_vectors``, the more accurate the
            calculation, but the longer it will take.
        dimensions : list, tuple or numpy.ndarray
            A 3 element `tuple` or ``array`` of `float` specifying the
            dimensions of the ``Universe`` in units of ``Ang``.
        energy_resolution : dict
            Specify energy resolution and function in units
            of ueV (micro eV), in the format of the one-line dict
            {'function': value}, where `function` is the resolution
            function and `value` is the desired `FWHM`.

            For example, to pass a Gaussian resolution of 80ueV we use
            {'gaussian': 80}.

            Currently accepted functions are 'gaussian' and
            'lorentzian' Can also be 'lazily' given as `float`, in
            which case it is assumed to be Gaussian.
        Q_values : numpy.ndarray
            1D array of Q `float` (in ``Ang^-1``).
        use_average : bool
            Optional parameter if a list of more than one ``Trajectory``
            is used.

            If set to True (default) then the mean value for S(Q, w)
            is calculated. Also, the errors are set to the standard
            deviation calculated over the list of
            ``CompactTrajectory`` objects.
        cont_slicing : bool

            Flag to decide between two possible behaviours when the
            number of ``MD_steps`` is larger than the minimum required
            to calculate the observables.

            If ``False`` (default) then the ``CompactTrajectory`` is
            sliced into non-overlapping sub-``CompactTrajectory``
            blocks for each of which the observable is calculated.

            If ``True``, then the ``CompactTrajectory`` is sliced into
            as many non-identical sub-``CompactTrajectory`` blocks as
            possible (with overlap allowed).
        """

        self._origin = 'MD'
        SQw_list = []
        use_average = settings.get('use_average', True)
        cont_slicing = settings.get('cont_slicing', False)

        try:
            override_dimensions = settings['dimensions']
        except KeyError:
            pass
        else:
            MD_input.setDimensions(np.array(override_dimensions))

        # adds resolution attribute if it doesn't already exist
        if self.resolution is None:
            resolution_factory = ResolutionFactory()
            if 'energy_resolution' in settings:
                self.resolution = resolution_factory.create_instance(
                    settings['energy_resolution'])
            else:
                # if no resolution supplied, give the object null resolution
                self.resolution = resolution_factory.create_instance(None)

        # Create independent_variables dictionary if it doesn't exist
        if not hasattr(self, 'independent_variables'):
            self.independent_variables = {}

        # Extract information that should be constant throughout the trajectory and hence the
        # subtrajectories (if there are any)
        t = MD_input.times - MD_input.times[0]
        dt = t[1] - t[0]
        if self.maximum_frames():
            t = t[0:self.maximum_frames()]

        if MD_input.is_fixedbox:
            try:
                self.universe_dimensions = MD_input.dimensions
            except AttributeError:
                try:
                    self.universe_dimensions = np.array(settings['dimensions'])
                except KeyError as error:
                    raise AttributeError('Either trajectory requires a dimensions '
                                         'attribute or dimensions must be passed '
                                         'when calling calculate_from_MD') from error
        else:  # If the dimensions of the simulation box are not fixed, we use the first value
            try:
                self.universe_dimensions = MD_input.changing_dimensions[0]
            except AttributeError:
                try:
                    self.universe_dimensions = np.array(settings['dimensions'])
                except KeyError as error:
                    raise AttributeError('Either trajectory requires a dimensions '
                                         'attribute or dimensions must be passed '
                                         'when calling calculate_from_MD') from error

        # Test that, if there is an existing E, it is consistent with E
        # calculated from trajectory times
        if self.E is not None:
            self.validate_energy(dt)
        elif self.independent_variables:
            self.independent_variables['E'] = self.calculate_E(len(t) - 1, dt)
        else:
            self.independent_variables = {'E': self.calculate_E(len(t) - 1, dt)}
        # Overwrite independent variable 'Q' if it already exists
        with suppress(KeyError):
            self.independent_variables['Q'] = np.array(settings['Q_values'])

        # Slice trajectory up if possible and requested by user:
        if self.maximum_frames() and use_average:
            trajectories = slice_trajectory(trj=MD_input, subtrj_len=self.maximum_frames(),
                                            cont_slicing=cont_slicing)
            trj_sliced = True
        else:
            trajectories = [MD_input]
            trj_sliced = False

        obtained_recreated_Q = False
        # Perform calculations for each trajectory
        for trajectory in trajectories:
            self.trajectory = trajectory

            # Assert that the times and dimensions are consistent with original trajectory
            if trj_sliced:
                try:
                    assert_allclose(self.trajectory.times -
                                    self.trajectory.times[0], t, rtol=1e-7, atol=1e-3)
                except AssertionError as error:
                    msg = ('The `times` of the current `CompactTrajectory` were not '
                           'consistent with the first `CompactTrajectory` passed')
                    raise AssertionError(msg) from error
            try:
                assert_allclose(self.universe_dimensions,
                                self.trajectory.dimensions)
            except AttributeError:
                # May not have dimensions set, in which case pass
                pass
            except AssertionError as error:
                msg = ('The `dimensions` of the current `CompactTrajectory` were not '
                       'consistent with the first `CompactTrajectory` passed')
                raise AssertionError(msg) from error

            fqt_type = self._get_fqt_type()
            # instantiate an FQt object for FQt calculations
            FQt = ObservableFactory.create(fqt_type)
            FQt.Q = self.Q
            # calculate FQt
            FQt.calculate_from_MD(trajectory, **settings)

            self.Q = FQt.Q
            if not obtained_recreated_Q:
                self._recreated_Q = FQt.recreated_Q
                obtained_recreated_Q = True
            SQw_list.append(FQt.calculate_SQw(self.E, self.resolution))

            # Cleanup the trajectory to reduce memory usage
            self.trajectory = None

        # calculate average and errors
        SQw_output = [np.mean(SQw_list, axis=0)]
        errors_output = [np.std(SQw_list, axis=0)]

        self._dependent_variables = {'SQw': SQw_output}
        self._errors = {'SQw': errors_output}

    def _get_fqt_type(self) -> str:
        """
        Get the name of the FQt type associated with each SQw type.

        Returns
        -------
        str
            FQt associated with SQw.
        """
        fqt_types = {'SQw': 'FQt',
                     'SQwCoherent': 'FQt_coh',
                     'SQwIncoherent': 'FQt_incoh'}
        return fqt_types[self.__class__.__name__]

    @staticmethod
    def calculate_E(nE: int, dt: float) -> np.ndarray:
        r"""
        Compute energy from MD.

        Calculate an array of ``nE`` uniformly spaced energy values from the
        time separation of the ``CompactTrajectory`` frames, ``dt``.

        The frequencies are determined by the fast Fourier transform,
        as implemented by numpy, for ``2 * nE`` points in time which
        are truncated to only include ``nE`` positive frequencies.

        As we are dealing with frequency rather than angular frequency
        here, the relation to between energy is given by:

        .. math::

            E = h \nu

        Parameters
        ----------
        nE : int
            The number of energy values to be calculated.
        dt : float
            The step size between frames in ``fs``.

        Returns
        -------
        numpy.ndarray
            An ``array`` of `float` specifying the energy in units of ``meV``.

        See Also
        --------
        numpy.fft.fftfreq : Frequency calculation.
        """

        # h is in units of eV s whereas system units are meV fs, so apply a
        # factor of 1e3 * 1e15 to convert it
        return h * 1e18 * np.fft.fftfreq(2 * int(nE), dt)[:int(nE)]

    def calculate_dt(self) -> float:
        r"""
        Calculate timestep required by experimental dataset assuming uniform spacing.

        .. note::

           This may be different from the time separation that the
           user has given as an input, as it only depends on the
           current values for ``self.E``.

        The relationship between time and energy comes from the numpy
        implementation of the FFT for ``2 * nE`` points where:

        .. math:: \nu_{max} &=& \frac{n_E - 1}{2 n_E \Delta t} \\\\
            \therefore \Delta t &=& \frac{h (n_E - 1)}{2 n_E E_{max}}

        Returns
        -------
        float
            The time separation required by the current values of ``self.E``.

        See Also
        --------
        numpy.fft.fft : Numpy FFT implementation.
        """

        # h is in units of eV s whereas system units are meV fs, so apply a
        # factor of 1e3 * 1e15 to convert it
        nE = len(self.E)
        return h * 1e18 * (nE - 1) / (2 * nE * (np.max(np.abs(self.E))))

    def calculate_resolution_functions(self, dt: float) -> dict:
        """
        Generate resolution function in momentum and time.

        Parameters
        ----------
        dt : float
            The time spacing for the inverse Fourier transform (in ``fs``).

            Ideally this should be the same as the frame separation
            expected when applying this function.

        Returns
        -------
        dict
            A dictionary with the key 'SQw' corresponding to function
            which accepts arrays of time and momentum (respectively)
            and returns a 2D array of values for the instrument
            resolution.

        Notes
        -----
        There are several caveats to this function a user should be aware of:

        - This uses the ``SQw`` values of the ``Observable`` it is
          called from, and so should only be called for an observable
          which has been created from relevant resolution data,
          i.e. a vanadium sample.
        - If this resolution function is used on data outside
          its original range, then it will use nearest neighbour
          extrapolation.
        - The input will be reflected in the time/energy domain as
          symmetry about 0 is assumed.

        If for whatever reason this is not appropriate for the data in
        question, this function should not be used.
        """

        # NB: this function is only used by methods in the FileResolution object
        # (see MDMC.resolution.from_file)
        # but it hasn't been moved to that file
        # because it relies so heavily on the SQw object's attributes
        # that moving it over is a nightmare

        # Remove any momentum values with infinite error, and the corresponding values from SQw
        error = self.SQw_err
        masking = np.where(np.any(error == float('inf'), axis=-1))
        SQw_cropped = np.delete(self.SQw[0], masking, axis=0)
        Q_cropped = np.delete(self.Q, masking)

        # Enforce symmetry in the energy/time domain of the data. Note that if the original data
        # included both positive and negative values, we will end up with more densely spaced
        # values after this reflection.
        E_reflected = np.append(self.E, -1 * self.E)
        SQw_reflected = np.append(SQw_cropped, SQw_cropped, axis=-1)

        # Sort the data so the width of each bin can be calculated using argsort so we can apply
        # the same re-ordering to SQw. Because we cannot calculate the width of the first or last
        # value, these are discarded from the energy and SQw arrays.
        sorted_indices = np.argsort(E_reflected)
        E_sorted = E_reflected[sorted_indices]
        widths = (E_sorted[2:] - E_sorted[:-2]) / 2
        E_sorted = E_sorted[1:-1]
        SQw_sorted = SQw_reflected[:, sorted_indices[1:-1]]

        # Perform a (slow) inverse discrete Fourier transform now, with all available data, rather
        # than with potentially coarse time sampling during the SQw calculation
        # Create time array with the spacing dt, and a min/max value of the Nyquist limit.
        # h is in units of eV s whereas system units are meV fs, so
        # apply a factor of 1e3 * 1e15 to convert it
        max_energy_separation = np.amax(np.diff(E_sorted))
        t_max = h * 1e18 / (2 * max_energy_separation)
        N_T = int(t_max / dt)
        t_array = np.linspace(- dt * N_T, dt * N_T, N_T)
        SQw_ift = np.zeros((len(SQw_sorted), N_T), dtype='complex')

        # In general we do not have equal energy spacing, multiply the exponential factor by this
        # before transposing and dotting to sum over the energy domain
        exp = np.exp(1j * np.outer(t_array, E_sorted) /
                     (h_bar * 1e18)) * widths
        SQw_ift = np.dot(SQw_sorted, np.transpose(exp))

        # Because Observable.dependent_variables_structure gives the order in which the
        # independent variables are represented in the np.shape of the data, we have to
        # reverse the order of the x and y arrays for RectBivariateSpline.
        # RectBivariateSpline does not return complex numbers,
        # so define a new function that combines the real and imaginary parts
        def data_interpol(t, Q) -> np.ndarray:
            """
            Internal function for interpolating complex t & Q.

            Parameters
            ----------
            t : numpy.ndarray
                Time domain.
            Q : numpy.ndarray
                Momentum domain.

            Returns
            -------
            np.ndarray
                Interpolated complex t & Q data.

            See Also
            --------
            scipy.interpolate.RectBivariateSpline : Interpolation mechanism.

            Notes
            -----
            RectBivariateSpline does not return complex numbers, so
            this function exists to emulate complex interpolation.
            """

            real = RectBivariateSpline(t_array, Q_cropped, np.real(SQw_ift).T)
            imag = RectBivariateSpline(t_array, Q_cropped, np.imag(SQw_ift).T)
            return (real(t, Q) + 1j * imag(t, Q)).T

        return {'SQw': data_interpol}

    @property
    def dependent_variables_structure(self) -> dict[str, list]:
        """
        The order in which the 'SQw' dependent variable is indexed in terms of 'Q' and 'E'.

        The purpose of this method is to ensure consistency
        between different readers/methods which create ``SQw`` objects.

        Returns
        -------
        dict[str, list]
            The shape of the SQw dependent variable.

        Notes
        -----
        Explicitly: we have that self.SQw[Q_index, E_index] is the data point for
        given indices of self.Q and self.E.

        It also means that:
        np.shape(self.SQw)=(np.size(self.Q), np.size(self.E)).
        """
        return {'SQw': ['Q', 'E']}

    @property
    def uniformity_requirements(self) -> dict[str, dict[str, bool]]:
        """
        Get restrictions required for computing E & Q.

        Captures the current limitations on the energy 'E' and reciprocal
        lattice points 'Q' within the dynamic structure factor ``Observables``.
        If using FFT, then 'E' must be uniform and start at zero, otherwise it
        has no restrictions. 'Q' must be uniform but does not need to start at
        zero.

        Returns
        -------
        dict[str, dict[str, bool]]
            Dictionary of uniformity restrictions for 'E' and 'Q'.
        """

        if self.use_FFT:
            e_requirements = {'uniform': True, 'zeroed': True}
        else:
            e_requirements = {'uniform': False, 'zeroed': False}

        return {'E': e_requirements, 'Q': {'uniform': True, 'zeroed': False}}

    def heatmap_plot(self, *,
              dependent_variable: np.ndarray = None,
              noshow: bool = False,
              ax: plt.axes = None)-> None:
        """
        Heatmap of SQw where:

        independent x-axis - Momentum Q
        independent y-axis - Energy E (w/omega)
        dependent heat - Sqw Scattering factor

        Parameters
        ----------
        dependent_variable : np.ndarray
          Optional parameter to allow users to apply a rescale factor before
          plotting the obserable.  
        noshow : bool
          Optional parameter so users can choose if the plot is shown. Default 
          value is False, the plot will be shown. 
        ax: plt.axes
          Optional parameter to input an existing axes object for example when 
          plotting a subplot or plots over one another.
        """

        if ax is None:
            ax = plt.axes()

        if dependent_variable is None:
            dependent_variable = self.SQw

        extent = self.Q[0], self.Q[-1], self.E[0], self.E[-1]

        im=ax.imshow(dependent_variable[0,:,:], extent = extent, aspect='auto')

        cbar = ax.figure.colorbar(im, ax=ax, **{})
        cbar.ax.set_ylabel(r"Scattering, S(Q,$\omega$)", rotation=-90, va="bottom")

        cbar.set_label(r"Scattering, S(Q,$\omega$)")
        ax.set_title(r'HeatMap of Scattering Factor S(Q,$\omega$)')
        ax.set_xlabel(r'Momentum, Q, ($\AA^{-1}$)')
        ax.set_ylabel(r'Energy, $\omega$, (meV)')

        if not noshow:
            plt.tight_layout()
            plt.show()

    def default_plot(self, *,
              dependent_variable: np.ndarray = None,
              noshow: bool = False,
              ax: plt.axes = None)-> None:
        """3D plot of SQw Where:

        independent x-axis - Momentum Q
        independent y-axis - Energy E (w/omega)
        dependent z-axis - Sqw Scattering factor

        Parameters
        ----------
        dependent_variable : np.ndarray
          Optional parameter to allow users to apply a rescale factor before
          plotting the obserable.  
        noshow : bool
          Optional parameter so users can choose if the plot is shown. Default 
          value is False, the plot will be shown. 
        ax: plt.axes
          Optional parameter to input an existing axes object for example when 
          plotting a subplot or plots over one another.
        """

        if ax is None:
            ax = plt.axes(projection ='3d')

        if dependent_variable is None:
            dependent_variable = self.SQw

        E, Q = np.meshgrid(self.E, self.Q)
        ax.plot_surface(Q, E, dependent_variable[0,:,:])
        ax.set_title(r'3D Plot of Scattering Factor S(Q,$\omega$)')
        ax.set_xlabel(r'Momentum, Q, ($\AA^{-1}$)')
        ax.set_ylabel(r'Energy, $\omega$, (meV)')
        ax.set_zlabel(r'Scattering, S(Q,$\omega$)')
        ax.set_box_aspect(aspect=None, zoom=0.9)

        if not noshow:
            plt.tight_layout()
            plt.show()

    def slider_plot(self, *,
                other_dependent_variable: Optional[np.ndarray] = None,
                figax: Optional[Tuple[plt.Figure, plt.Axes]]=None) -> None:
        '''
        
        Parameters
        ---------- 
        other_dependent_variable : np.ndarray
          Optional parameter allows users to input the dependent variable from 
          another observable so they can be compared in the plot. 
        figax: Tuple[plt.Figure, plt.Axes]
          Optional parameter to input an existing axes and figure object, for 
          example when overlaying plots over one another.
        '''

        if figax is None:
            fig, ax = plt.subplots()
        else:
            fig, ax = figax

        slider_variable = self.Q
        fixed_variable = self.E
        dependent_variable = self.SQw

        # Variables Name and Units
        slider_variable_name = 'Q'
        fixed_variable_name = r'E, $\omega$'

        ax.set_ylabel(fr'S(Q,$\omega$) ({dependent_variable.unit})')

        # Add Lines to Axes
        self._add_line(ax,fixed_variable,
                       dependent_variable[0,0,:],fixed_variable_name)

        dependent_variable_list = [dependent_variable]

        # Add Experimental Observable Line
        if other_dependent_variable is not None:
            self._add_line(ax,fixed_variable,
                           other_dependent_variable[0,0,:],fixed_variable_name)
            dependent_variable_list.append(other_dependent_variable)

        # Add the Slider
        self._add_slider((fig,ax),slider_variable,slider_variable_name,
                         dependent_variable_list)

    @staticmethod
    def _add_line(ax: plt.Axes,independent_variable: np.ndarray,
                  dependent_variable: np.ndarray,
                  independent_variable_name: str = None)-> None:
        '''Add lines to plt.axes.
        
        Parameters
        ---------- 
        ax : plt.Axes
          Optional parameter to input an existing axes object for example when 
          plotting a subplot or plots over one another. 
        independent_variable: np.ndarray
          Data for x-axis of plot.
        dependent_variable: np.ndarray
          Data for y-axis of plot.
        independent_variable_name: str
          Name of the independent variable for axis label
        '''

        # Plot obs line
        ax.plot(independent_variable, dependent_variable, linewidth=1)

        if independent_variable_name is not None:
            label=independent_variable_name+', '+ independent_variable.unit
            ax.set_xlabel(label)
        else:
            ax.set_xlabel(independent_variable.unit)

    @staticmethod
    def _add_slider(figax: Tuple[plt.Figure, plt.Axes],
                    independent_variable: np.ndarray,
                    independent_variable_name: str,
                    dependent_variable_list: list[np.ndarray]):
        ''' Add a slider widget to plot
        
        Parameters
        ---------- 
        figax: Tuple[plt.Figure, plt.Axes]
          Input an existing axes and figure object, for 
          example when overlaying plots over one another. 
        independent_variable: np.ndarray
          Variable to be used on the slider widget.
        dependent_variable_list: list[np.ndarray]
          List of dependent variable for y-axis of plot to be updated when the 
          slider is changed.
        independent_variable_name: str
          Name of the independent variable for slider label
        '''

        fig, ax = figax
        fig.subplots_adjust(left=0.25, bottom=0.3)
        slider_ax  = fig.add_axes([0.25, 0.15, 0.65, 0.03],
                                       facecolor='lightgoldenrodyellow')
        slider = Slider(slider_ax, independent_variable_name+' index', 0,
                        len(independent_variable)-1, valinit=1, valstep=1)
        slider_label=plt.text(1,1.7,f'{independent_variable_name}='
                                    f'{round(independent_variable[1],2)},'
                                    f' {independent_variable.unit}')

        def Q_on_changed(val):
            for i, dependent_variable in enumerate(dependent_variable_list):
                ax.get_lines()[i].set_ydata(dependent_variable[0,val])

            slider_label.set_text(f'{independent_variable_name}='
                                  f'{round(independent_variable[val],2)},' 
                                  f' {independent_variable.unit}')
            fig.canvas.draw_idle()
            ax.set_ylim(0,max(np.max(dependent_variable_list[0][0,val])+0.1,1e-5))

        slider.on_changed(Q_on_changed)

        plt.show()

    def plot(self,plot_type: Literal["slider", "heatmap", None]=None) -> None:
        '''
        Plot of Experimental and Simulation for a given Oberservable
        
        Parameters
        ----------
        plot_type : string
            The ``type`` of figure to be plot
        '''

        if plot_type is None:
            # Default
            self.default_plot()
        elif plot_type == 'slider':
            self.slider_plot()
        elif plot_type == 'heatmap':
            self.heatmap_plot()


@ObservableFactory.register(('DynamicStructureFactor', 'SQw'))
class SQw(AbstractSQw):
    """
    A class for the total dynamic structure factor.

    Calculation is done in the respective FQt object, and this is
    just a reference to get the correct FQt object.
    """


@ObservableFactory.register(('CoherentDynamicStructureFactor',
                             'SQwCoherent',
                             'SQwCoh',
                             'SQw_coh'))
class SQwCoherent(AbstractSQw):
    """
    A class for the coherent dynamic structure factor.
    """


@ObservableFactory.register(('IncoherentDynamicStructureFactor',
                             'SQwIncoherent'
                             'SQwIncoh',
                             'SQw_incoh'))
class SQwIncoherent(AbstractSQw):
    """
    A class for the incoherent dynamic structure factor.
    """
