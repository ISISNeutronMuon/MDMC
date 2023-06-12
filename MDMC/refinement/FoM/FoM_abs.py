"""A module for Figure of Merits"""

from abc import ABC, abstractmethod
import numpy as np

from MDMC.common.decorators import repr_decorator
from MDMC.trajectory_analysis.observables.obs import Observable


@repr_decorator('weight', 'exp_obs', 'MD_obs', 'rescale_factor', 'auto_scale')
class ObservablePair:

    """
    Contains a pair of observables for calculating the FoM

    Checks the validity of observables

    Parameters
    ----------
    exp_obs : Observable
        An ``Observable`` with ``Observable.origin == 'experiment'``
    MD_obs : Observable
        An ``Observable`` with ``Observable.origin == 'MD'``
    weight : float
        The relative weight of this pair on a total FoM
    rescale_factor: float, optional
        Factor applied to ``exp_obs`` when calculating the FoM to ensure it is
        on the same scale as ``MD_obs``. Default is `1.`.
    auto_scale: bool, optional
        If `True`, ``rescale_factor`` is set automatically to minimise the FoM
        for each step of the refinement, overriding a user specified value if
        set. Note that this process is purely statistical and does not account
        for physical effects that might impact the scaling. Default is `False`.
    """

    def __init__(self, exp_obs: Observable, MD_obs: Observable, weight: float,
                 rescale_factor: float = 1., auto_scale: bool = False):

        self.exp_obs = exp_obs
        self.MD_obs = MD_obs
        self.weight = weight
        self.rescale_factor = rescale_factor
        self.auto_scale = auto_scale

    @property
    def exp_obs(self) -> Observable:
        """
        Get or set the experimental ``Observable``

        Setting the ``Observable`` checks its validity

        Returns
        -------
        Observable
            The experimental ``Observable``
        """

        return self._exp_obs

    @exp_obs.setter
    def exp_obs(self, exp_obs: Observable) -> None:
        self.validate_obs(exp_obs, 'experiment')
        self._exp_obs = exp_obs

    @property
    def MD_obs(self) -> Observable:
        """
        Get or set the MD ``Observable``

        Setting the ``Observable`` checks its validity

        Returns
        -------
        Observable
            The MD ``Observable``
        """

        return self._MD_obs

    @MD_obs.setter
    def MD_obs(self, MD_obs: Observable) -> None:
        self.validate_obs(MD_obs, 'MD')
        self._MD_obs = MD_obs

    @property
    def weight(self) -> float:
        """
        Get or set the relative weight of this pair on a total FoM

        Returns
        -------
        float
            The relative weight

        Raises
        ------
        TypeError
            If ``weight`` is set with a non-numeric
        """

        return self._weight

    @weight.setter
    def weight(self, weight: float) -> None:
        try:
            weight = float(weight)
        except ValueError as error:
            raise TypeError('weight must be a float') from error
        self.validate_weight(weight)
        self._weight = weight

    @property
    def n_averages(self) -> 'dict[str, int]':
        """
        The number of separate, complete dependent variable calculations we
        have been able to perform for the ``Observable``

        Returns
        -------
        dict
            Each key represents a dependent variable, and the value is the
            number of times we have calculated it
        """

        n_averages = {}
        for key, value in self.MD_obs.dependent_variables.items():
            n_averages[key] = len(value)

        return n_averages

    def validate_obs(self, obs: Observable, origin: str) -> None:
        """
        Performs checks to test the validity of an ``Observable``

        Tests that the ``Observable.origin`` is as expected. If the
        ``ObservablePair`` has another ``Observable`` (i.e. the other
        ``origin``), then this tests that the ``independent_variables`` are
        identical, the ``dependent_variables`` have the same shape,
        the ``errors`` have the same shape, and that the ``Observable`` objects
        are of the same type.

        Parameters
        ----------
        obs : Observable
            The ``Observable`` to validate
        origin : str
            The ``Observable.origin`` (``'experiment'`` or ``'MD'``)

        Raises
        ------
        AssertionError
            If the ``Observable.origin`` is not the same as the ``origin``
            Parameter
        AssertionError
            If ``Observable`` does not have identical ``independent_variables``
            to any ``Observable`` of the other ``Observable.origin`` that
            already exists in the ``ObservablePair``
        AssertionError
            If ``Observable`` does not have identical ``dependent_variables`` to
            any ``Observable`` of the other ``Observable.origin`` that already
            exists in the ``ObservablePair``
        AssertionError
            If ``Observable`` does not have identical ``errors`` to any
            ``Observable`` of the other ``Observable.origin`` that already
            exists in the ``ObservablePair``
        AssertionError
            If ``Observable`` does not have identical type to any ``Observable``
            of the other ``Observable.origin`` that already exists in the
            ``ObservablePair``
        """

        # Check origin is correct
        assert obs.origin == origin, ('The observable does not have the correct'
                                      ' origin')

        try:
            if obs.origin == 'MD':
                other_obs = self.exp_obs
            else:
                other_obs = self.MD_obs
        except AttributeError:
            other_obs = None

        # Check independent variables are identical, check dependent variables
        # have the same shapes, check errors have the same shapes, check
        # observables have the same type
        if other_obs:
            indep_e_mess = 'Independent variables must be identical'
            assert (obs.independent_variables.keys() ==
                    other_obs.independent_variables.keys()), indep_e_mess
            for k in obs.independent_variables.keys():
                assert np.all(obs.independent_variables[k] ==
                              other_obs.independent_variables[k]), indep_e_mess

            # Try/except deals with empty observable case (no dependent
            # variables and errors)
            try:
                dep_e_mess = 'Dependent variables must have the same shape'
                assert (obs.dependent_variables.keys() ==
                        other_obs.dependent_variables.keys()), dep_e_mess
                for k in obs.dependent_variables:
                    assert (np.shape(obs.dependent_variables[k]) ==
                            np.shape(other_obs.dependent_variables[k])), \
                        dep_e_mess

                err_e_mess = 'Errors must have the same shape'
                assert obs.errors.keys() == other_obs.errors.keys(), err_e_mess
                for k in obs.errors:
                    assert (np.shape(obs.errors[k]) ==
                            np.shape(other_obs.errors[k])), err_e_mess
            except AttributeError:
                pass

            assert isinstance(obs, type(other_obs)), ('Observables are not of'
                                                      ' the same type')

    @staticmethod
    def validate_weight(weight: float) -> None:
        """
        Performs checks to test the validity of the ``weight``

        Parameters
        ----------
        weight : float
            The ``weight`` to be validated

        Raises
        ------
        AssertionError
            If the ``weight`` is not positive or is infinite
        """

        assert weight > 0. and weight != float('inf'), ('Weight must be a'
                                                        ' finite positive'
                                                        ' float')

    def check_types(self) -> None:
        """
        Checks that ``Observable`` objects are of the same type
        """

        raise NotImplementedError

    def check_indep_var(self) -> None:
        """
        Checks that ``Observable`` objects have the same
        ``independent_variables`` and that are finite
        """

        raise NotImplementedError

    def check_dep_var(self) -> None:
        """
        Checks that ``Observable`` objects have the same ``dependent_variables``
        and that are finite
        """

        raise NotImplementedError

    def check_errors(self) -> None:
        """
        Checks that an ``Observable`` has errors on the ``dependent_variable``
        and that these are `float` and not `NaN`
        """

        raise NotImplementedError

    def check_origin(self, origin: str) -> None:
        """
        Checks that the ``Observable.origin`` is correct

        Parameters
        -------
        origin : str
            A string consisting of either ``'experiment'`` or  ``'MD'``
        """

        raise NotImplementedError

    def calculate_difference(self) -> np.ndarray:
        """
        Calculates the difference between the dependent variables of an
        experimental dataset and an MD simulation dataset, after applying
        a rescale factor which is calculated to scale the exp_data to the
        MD simulation.

        Returns
        -------
        numpy.ndarray
            An array with the same dimensions as the ``dependent_variables`` of
            the ``exp_obs`` and ``MD_obs``. The array contains the difference
            between the ``dependent_variables`` taking the ``rescale_factor``
            into account.
        """

        diff = (np.array(*self.exp_obs.dependent_variables.values())
                * self.rescale_factor
                - np.array(*self.MD_obs.dependent_variables.values()))

        return diff

    def calculate_errors(self) -> np.ndarray:
        """
        Combines, in quadrature, the errors of the MD dataset and the
        experimental dataset.

        Returns
        -------
        numpy.ndarray
            An array with the same dimensions as the ``errors`` of the
            ``exp_obs`` and ``MD_obs``. The array contains the combination of
            the ``errors`` in quadrature, taking the ``rescale_factor`` into
            account.
        """

        errors = (self.calculate_exp_errors() ** 2
                  + np.array(*self.MD_obs.errors.values()) ** 2) ** 0.5

        return errors

    def calculate_exp_errors(self) -> np.ndarray:
        """
        Assumes a single dependent variable error for each ``Observable``.
        Calculates only the experimental errors and scales them by the
        `rescale_factor`.

        Returns
        -------
        numpy.ndarray
            An array with the same dimensions as the ``errors`` of the
            ``exp_obs``, taking the ``rescale_factor`` into account.
        """

        return np.array(*self.exp_obs.errors.values()) * self.rescale_factor


@repr_decorator('value', 'obs_pairs')
class FigureOfMerit(ABC):

    """
    Abstract class that defines methods common to all figure of merit
    calculators

    Parameters
    ----------
    obs_pairs : list
        A `list` of ``ObservablePairs``
    norm : {'data_points', 'dof', 'none'}, optional
        What method of normalisation to use when calculating the FoM for a single dataset.
        Default is 'data_points'.
    n_parameters : int, optional
        The number of parameters being refined. Optional if ``norm`` is either 'data_points' or
        'none', but required when 'dof'. Default is None.

    Attributes
    ----------
    obs_pairs : list
        A `list` of ``ObservablePairs``
    value : float
        The Figure of Merit for all ``obs_pairs``
    """

    def __init__(self,
                 obs_pairs: 'list[ObservablePair]',
                 norm: str = 'data_points',
                 n_parameters: int = None):

        self.obs_pairs = list(obs_pairs)
        self.value = None
        self.n_parameters = 0

        if norm == 'data_points':
            self.norm = True
        elif norm == 'dof':
            if n_parameters is None:
                raise ValueError('`n_parameters` must be provided if using '
                                 'degrees of freedom normalisation.')
            self.norm = True
            self.n_parameters = n_parameters
        elif norm == 'none':
            self.norm = False
        else:
            raise ValueError('Unrecognised value for `norm` passed, should be '
                             'one of "data_points", "dof", "none", but it was '
                             f'"{norm}"')

    def calculate(self) -> float:
        """
        Calculates the FoM value by calculating the FoM for every
        ``ObservablePair``

        Returns
        -------
        float
            A non-negative `float` Figure of Merit

        Raises
        ------
        AssertionError
            If calculated value of Figure of Merit is negative
        """

        total_weight = sum(obs_pair.weight for obs_pair in self.obs_pairs)
        value_unreduced = sum(self.calculate_single_FoM(obs_pair)
                                  for obs_pair in self.obs_pairs)
        self.value = value_unreduced / total_weight

        assert self.value >= 0.
        return self.value

    def data_norm_factor(self, obs_pair: ObservablePair) -> int:
        """
        Calculates the normalisation factor for ``obs_pair``. If ``self.norm`` is `True`, then
        returns the number of data points less the number of refinement parameters if `'dof'`
        normalisation was chosen, or just the number of data points for the default `'data_points'`
        normalisation. If ``self.norm`` is `False`, then returns `1`.

        Parameters
        ----------
        obs_pair : ObservablePair
            An ``ObservablePair`` for which the normalisation factor is calculated

        Returns
        -------
        int
            The normalisation factor
        """

        if self.norm:
            norm_factor = np.size(
                *obs_pair.MD_obs.dependent_variables.values())
            norm_factor -= self.n_parameters
        else:
            norm_factor = 1

        return norm_factor

    @abstractmethod
    def calculate_single_FoM(self, obs_pair: ObservablePair) -> float:
        """
        Performs the FoM calculation specific to each FoM

        Parameters
        ----------
        obs_pair : ObservablePair
            An ``ObservablePair`` for which the FoM is calculated

        Returns
        -------
        float
            The FoM for the ``obs_pair``
        """

        raise NotImplementedError
