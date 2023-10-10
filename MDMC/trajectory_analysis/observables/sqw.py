"""Module for AbstractSQw and total SQw class"""

from typing import Optional

import logging
import numpy as np
from numpy.testing import assert_allclose
from scipy.interpolate import interp2d

from MDMC.common import units
from MDMC.common.constants import h, h_bar
from MDMC.common.decorators import unit_decorator_getter
from MDMC.resolution.resolution_factory import ResolutionFactory
from MDMC.trajectory_analysis.observables.obs import Observable
from MDMC.trajectory_analysis.observables.obs_factory import ObservableFactory
from MDMC.trajectory_analysis.compact_trajectory import CompactTrajectory
from MDMC.utilities.trajectory_slicing import slice_trajectory


class SQwMixins:
    """
    A mixin class for properties used by both SQw and FQt objects
    """

    def minimum_frames(self, dt: float = None) -> int:
        r"""
        The minimum number of ``CompactTrajectory`` frames needed to calculate the
        ``dependent_variables`` depends on ``self.use_FFT``.

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

        Parameters
        ----------
        dt : float, optional
            The time separation of frames in ``fs``, default is `None`

        Returns
        -------
        int
            The minimum number of frames
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
        The maximum number of ``CompactTrajectory`` frames that can be used to
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

        if self.use_FFT and (self.E is not None):
            return len(self.E) + 1

        return None

    @property
    @unit_decorator_getter(unit=units.LENGTH ** -1)
    def Q(self) -> Optional[np.array]:
        """
        Get or set the momentum transfers

        Returns
        -------
        numpy.array
            1D array of Q `float` (in ``Ang^-1``)
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
    An abstract class for total, coherent and incoherent dynamic structure
    factors. The equations used for calculating this are based on Kneller et
    al. Comput. Phys. Commun. 91 (1995) 191-214.

    Note that properties for MD frames & Q are found in the SQwMixins class.
    """

    def __init__(self):
        super().__init__()
        self._independent_variables = None
        self._dependent_variables = None
        self._errors = None
        self.resolution = None
        # Use FFT by default
        self._use_FFT = True

    @property
    def independent_variables(self) -> dict:
        """
        Get or set the independent variables: these are
        the frequency Q (in ``Ang^-1``) and energy E (in``meV``)

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
        Get the dependent variables: this is SQw, the
        dynamic structure factor (in ``arb``)

        Returns
        -------
        dict
            The dependent variables
        """
        return self._dependent_variables

    @property
    def errors(self) -> dict:
        """
        Get or set the errors on the dependent variables, the dynamic
        structure factor (in ``arb``)

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
    @unit_decorator_getter(unit=units.ENERGY_TRANSFER)
    def E(self) -> 'np.array':
        """
        Get the energies

        Returns
        -------
        numpy.array
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
    def w(self) -> 'np.array':
        """
        Get the angular frequencies

        Returns
        -------
        numpy.array
            1D array of angular frequency `float` in units of ``1 / ps``
        """

        return self.E / (h_bar * 1e15)

    @property
    @unit_decorator_getter(unit=units.ARBITRARY)
    def SQw(self) -> Optional[list[np.ndarray]]:
        """
        Get the dynamic structure factor, S(Q, w), in arb

        Returns
        -------
        list of numpy.ndarray
            `list` of 2D arrays of S(Q, w) `float` with arbitrary units
        """

        try:
            return self.dependent_variables['SQw']
        except KeyError:
            return None

    @property
    @unit_decorator_getter(unit=units.ARBITRARY)
    def SQw_err(self) -> Optional[list[np.ndarray]]:
        """
        Get the errors on the dynamic structure factor in arb

        Returns
        -------
        list of numpy.ndarray
            `list` of 2D array of S(Q, w) error `float` with arbitrary units
        """

        try:
            return self.errors['SQw']
        except KeyError:
            return None

    def validate_energy(self, dt: float, traj_step: float = None, time_step: float = None):
        """
        Asserts that the user set frame separation ``dt`` leads to energy
        separation that matches that of the experiment. If not, it
        includes the time separation required in the error.

        Parameters
        ----------
        dt : float
            Frame separation in ``fs``
        traj_step: float, optional
            User specified value for the number of 'time_steps' used inbetween snapshots
            of the simulation, default is None.
        time_step: float, optional
            User specified length of time for each update of the atoms trajectories
            in the simulation, default is None.

        Returns
        -------
        tuple
            contains a boolean, and two floats (if traj_step and time_step have values)
            or two NoneType (if traj_step and time_step were passed in as None).

        Raises
        ------
        AssertionError
        """

        dt_required = self.calculate_dt()
        rounded_dt_required = np.round(dt_required, 5)

        if time_step is not None:

            # Changing the time and traj step to fit the required dt value
            # by finding the highest traj_step that can fit into the dt_required
            # and checking to make sure one traj_step higher than this wouldnt be
            # closer to the required dt value

            traj_step = np.round(dt_required/time_step)
            time_step = dt_required/traj_step
            traj_step = int(traj_step)
            dt = traj_step * time_step
            print(time_step)
            logging.warning(" The given traj_step and time_step values were not"
                    " compatibile with the dataset specified.\nThe values "
                    "(whilst prioritising time_step) have been changed to"
                    " traj_step: %d, and time_step: %f. \n"
                    "Context: for this dataset, traj_step multiplied by time_step"
                    " must be ~= %f (6 d.p). \n" , traj_step,time_step,rounded_dt_required)

        if self.use_FFT:
            # When using FFT, require all experimental/simulated energies
            # to match
            energy = self.E
            msg = ("Experimental E values are not consistent with the "
                   "`Simulation`. For the experimental data provided, the "
                   f"product of `time_step` and `traj_step` must be {dt_required}, "
                   f"but it was {dt}")
            assert_allclose(self.calculate_E(len(energy), dt),
                            energy,
                            rtol=1e-5,
                            err_msg=msg)
        else:
            # When not using FFT, there is not a hard requirement to match
            # the energies, instead impose a requirement that our frame
            # separation is small enough to capture the highest frequencies
            msg = ("In order to capture the maximum experimental energy "
                   "(frequency) value, the frame separation must be at least "
                   "as small as the time period for oscillations at that "
                   "frequency. The frame separation is given by the product of"
                   f" `time_step` and `traj_step` and must be less than {dt_required}, "
                   f"but it was {dt}")
            # Allow for rounding errors by using isclose
            isclose = np.isclose(dt, dt_required, rtol=1e-5)
            assert isclose or dt <= dt_required, msg

        if traj_step is not None and time_step is not None:
            return True, traj_step, time_step
        return False, None, None

    def calculate_from_MD(self, MD_input: CompactTrajectory, verbose: int = 0,
                         **settings: dict):
        """
        Calculate the dynamic structure factor, S(Q, w) from a ``CompactTrajectory``.

        If the ``CompactTrajectory`` has more frames than the ``self.maximum_frames()``
        that can be used to recreate the grid of energy points, it can slice the
        ``CompactTrajectory`` into sub-trajectories of length ``self.maximum_frames()``,
        with the slicing specified through the settings ``use_average`` and ``cont_slicing``.

        The ``independent_variable`` ``Q`` can either be set previously or defined within
        ``**settings``.

        Parameters
        ----------
        MD_input : CompactTrajectory
            An MDMC ``CompactTrajectory`` from which to calculate ``SQw``
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
            ``energy_resolution` (`dict`)
                Optionally specify energy resolution and function in units of ueV (micro eV),
                in the format of the one-line dict {'function': value}, where `function`
                is the resolution function and `value` is the desired `FWHM`.
                e.g. to pass a Gaussian resolution of 80ueV we use {'gaussian': 80}.
                Currently accepted functions are 'gaussian' and 'lorentzian'
                Can also be 'lazily' given as `float`, in which case it is assumed to be Gaussian.
            ``Q_values`` (`array`)
                1D array of Q `float` (in ``Ang^-1``). (optional)
            ``use_average`` (`bool`)
                Optional parameter if a list of more than one ``Trajectory`` is used. If set to
                True (default) then the mean value for S(Q, w) is calculated. Also, the errors
                are set to the standard deviation calculated over the list of
                ``CompactTrajectory`` objects.
             ``cont_slicing`` (`bool`)
                Flag to decide between two possible behaviours when the number of ``MD_steps`` is
                larger than the minimum required to calculate the observables. If ``False``
                (default) then the ``CompactTrajectory`` is sliced into non-overlapping
                sub-``CompactTrajectory`` blocks for each of which the observable is calculated.
                If ``True``, then the ``CompactTrajectory`` is sliced into as many non-identical
                sub-``CompactTrajectory`` blocks as possible (with overlap allowed).
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
                    raise AttributeError('Either trajectory requires a dimensions'
                                        ' attribute or dimensions must be passed'
                                        ' when calling calculate_from_MD') from error
        else: # If the dimensions of the simulation box are not fixed, we use the first value
            try:
                self.universe_dimensions = MD_input.changing_dimensions[0]
            except AttributeError:
                try:
                    self.universe_dimensions = np.array(settings['dimensions'])
                except KeyError as error:
                    raise AttributeError('Either trajectory requires a dimensions'
                                        ' attribute or dimensions must be passed'
                                        ' when calling calculate_from_MD') from error

        # Test that, if there is an existing E, it is consistent with E
        # calculated from trajectory times
        if self.E is not None:
            self.validate_energy(dt)
        elif self.independent_variables:
            self.independent_variables['E'] = self.calculate_E(len(t) - 1, dt)
        else:
            self.independent_variables = {'E': self.calculate_E(len(t) - 1, dt)}
        # Overwrite independent variable 'Q' if it already exists
        try:
            self.independent_variables['Q'] = np.array(settings['Q_values'])
        except KeyError:
            pass

        # Slice trajectory up if possible and requested by user:
        if self.maximum_frames() and use_average:
            trajectories = slice_trajectory(trj=MD_input, subtrj_len=self.maximum_frames(),
                                            cont_slicing=cont_slicing)
            trj_sliced = True
        else:
            trajectories = [MD_input]
            trj_sliced = False

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
            FQt = ObservableFactory.create_observable(fqt_type)
            FQt.Q = self.Q
            # calculate FQt
            FQt.calculate_from_MD(trajectory, **settings)

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
        Gets the name of the FQt type associated with each SQw type.
        """
        fqt_types = {'SQw': 'FQt',
                     'SQwCoherent': 'FQt_coh',
                     'SQwIncoherent': 'FQt_incoh'}
        return fqt_types[self.__class__.__name__]

    @staticmethod
    def calculate_E(nE: int, dt: float) -> np.ndarray:
        r"""
        Calculates an array of ``nE`` uniformly spaced energy values from the
        time separation of the ``CompactTrajectory`` frames, ``dt``. The
        frequencie are determined by the Fast Fourier Transform, as implemented
        by numpy, for ``2 * nE`` points in time which we then crop to only
        include ``nE`` positive frequencies. As we are dealing with frequency
        rather than angular frequency here, the relation to between energy
        is given by:

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

        # h is in units of eV s whereas system units are meV fs, so apply a
        # factor of 1e3 * 1e15 to convert it
        return h * 1e18 * np.fft.fftfreq(2 * int(nE), dt)[:int(nE)]

    def calculate_dt(self) -> float:
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

        # h is in units of eV s whereas system units are meV fs, so apply a
        # factor of 1e3 * 1e15 to convert it
        nE = len(self.E)
        return h * 1e18 * (nE - 1) / (2 * nE * (np.max(np.abs(self.E))))

    def calculate_resolution_functions(self, dt: float) -> dict:
        """
        Generates a resolution function in momentum and time that can be used in the calculation of
        SQw. Note that this uses the ``SQw`` values of the ``Observable`` it is called from, and so
        should only be called for an observable which has been created from relevant resolution
        data, i.e. a vanadium sample.

        Note that if this resolution function is used on data outside its original range, then it
        will use nearest neighbour extrapolation. Additionally, the input will be reflected in the
        time/energy domain as symmetry about 0 is assumed. If for whatever reason this is not
        appropriate for the data in question, this function should not be used.


        Parameters
        ----------
        dt : float
            The time spacing to use when performing the inverse Fourier transform in units of `fs`.
            Ideally this should be the same as the frame separation expected when applying this
            function.

        Returns
        -------
        dict
            A dictionary with the key 'SQw' corresponding to function which accepts arrays of time
            and momentum (respectively) and returns a 2D array of values for the instrument
            resolution.
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

        # note: the interp2d interpolation function requires input of the form
        # interp2d(x, y, z)
        # where if np.size(x)=m and np.size(y)=n then np.shape(z)=(n,m)
        # E.g. if x = [0,1,2]; y = [0,3]; z = [[1,2,3], [4,5,6]]
        # Because Observable.dependent_variables_structure gives the order in which the
        # independent variables are represented in the np.shape of the data, we have to
        # reverse the order of the x and y arrays for interp2d.
        # interp2d does not return complex numbers, so define a new function that combines
        # the real and imaginary parts
        def data_interpol(t, Q):
            real = interp2d(t_array, Q_cropped, np.real(SQw_ift))
            imag = interp2d(t_array, Q_cropped, np.imag(SQw_ift))
            return real(t, Q) + 1j * imag(t, Q)

        return {'SQw': data_interpol}

    @property
    def dependent_variables_structure(self) -> dict[str, list]:
        """
        The order in which the 'SQw' dependent variable is indexed in terms of 'Q' and 'E'.
        Explicitly: we have that self.SQw[Q_index, E_index] is the data point for
        given indices of self.Q and self.E
        It also means that:
        np.shape(self.SQw)=(np.size(self.Q), np.size(self.E))

        The purpose of this method is to ensure consistency
        between different readers/methods which create ``SQw`` objects.

        Return
        ------
        Dict[str, list]
            The shape of the SQw dependent variable
        """
        return {'SQw': ['Q', 'E']}

    @property
    def uniformity_requirements(self) -> dict[str, dict[str, bool]]:
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
    A class for the total dynamic structure factor

    Calculation is done in the respective FQt object, and this is
    just a reference to get the correct FQt object.
    """


@ObservableFactory.register(('CoherentDynamicStructureFactor',
                             'SQwCoherent',
                             'SQwCoh',
                             'SQw_coh'))
class SQwCoherent(AbstractSQw):

    """
    A class for the coherent dynamic structure factor
    """


@ObservableFactory.register(('IncoherentDynamicStructureFactor',
                             'SQwIncoherent'
                             'SQwIncoh',
                             'SQw_incoh'))
class SQwIncoherent(AbstractSQw):

    """
    A class for the incoherent dynamic structure factor
    """
