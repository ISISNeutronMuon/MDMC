"""A module for Figure of Merits"""

from abc import ABC, abstractmethod

import numpy as np
from typing import Dict

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
                 rescale_factor: float=1., auto_scale: bool=False):

        self.exp_obs = exp_obs
        self.MD_obs = MD_obs
        self.weight = weight
        self.rescale_factor = rescale_factor
        self.auto_scale = auto_scale

    @property
    def exp_obs(self):

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
    def exp_obs(self, exp_obs):

        self.validate_obs(exp_obs, 'experiment')
        self._exp_obs = exp_obs

    @property
    def MD_obs(self):

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
    def MD_obs(self, MD_obs):

        self.validate_obs(MD_obs, 'MD')
        self._MD_obs = MD_obs

    @property
    def weight(self):

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
    def weight(self, weight):

        try:
            weight = float(weight)
        except ValueError:
            raise TypeError('weight must be a float')
        self.validate_weight(weight)
        self._weight = weight

    @property
    def n_averages(self) -> Dict[str, int]:

        """
        The number of seperate, complete dependent variable calculations we
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

    def validate_obs(self, obs, origin):

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

    def validate_weight(self, weight):

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

        assert weight > 0. and weight != np.float('inf'), ('Weight must be a'
                                                           ' finite positive'
                                                           ' float')

    def check_types(self):

        """
        Checks that ``Observable`` objects are of the same type
        """

        raise NotImplementedError

    def check_indep_var(self):

        """
        Checks that ``Observable`` objects have the same
        ``independent_variables`` and that are finite
        """

        raise NotImplementedError

    def check_dep_var(self):

        """
        Checks that ``Observable`` objects have the same ``dependent_variables``
        and that are finite
        """

        raise NotImplementedError

    def check_errors(self):

        """
        Checks that an ``Observable`` has errors on the ``dependent_variable``
        and that these are `float` and not `NaN`
        """

        raise NotImplementedError

    def check_origin(self, origin):

        """
        Checks that the ``Observable.origin`` (``'experiment'`` or ``'MD'``) is
        correct
        """

        raise NotImplementedError

    def calculate_difference(self):

        """
        Assumes a single dependent variable for each ``Observable``

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

    def calculate_errors(self):

        """
        Assumes a single dependent variable error for each ``Observable``

        Returns
        -------
        numpy.ndarray
            An array with the same dimensions as the ``errors`` of the
            ``exp_obs`` and ``MD_obs``. The array contains the combination of
            the ``errors`` in quadrature, taking the ``rescale_factor`` into
            account.
        """

        errors = ((np.array(*self.exp_obs.errors.values())
                   * self.rescale_factor) ** 2
                  + np.array(*self.MD_obs.errors.values()) ** 2) ** 0.5

        return errors


@repr_decorator('value', 'obs_pairs')
class FigureOfMeritCalculator(ABC):

    """
    Abstract class that defines methods common to all figure of merit
    calculators

    Parameters
    ----------
    obs_pairs : list
        A `list` of ``ObservablePairs``

    Attributes
    ----------
    obs_pairs : list
        A `list` of ``ObservablePairs``
    value : float
        The Figure of Merit for all ``obs_pairs``
    """

    def __init__(self, obs_pairs):

        self.obs_pairs = list(obs_pairs)
        self.value = None

    def calculate(self):

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

        total_weight = np.sum([obs_pair.weight for obs_pair in self.obs_pairs])
        value_unreduced = np.sum([self.calculate_single_FoM(obs_pair)
                                  for obs_pair in self.obs_pairs])
        self.value = value_unreduced / total_weight

        assert self.value >= 0.
        return self.value

    @abstractmethod
    def calculate_single_FoM(self, obs_pair):

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


class StandardFoMCalculator(FigureOfMeritCalculator):

    r"""
    Calculates the weighted sum of the Figure of Merits for a number of datasets:

    .. math::

        FoM_{total} = \frac{\sum_{i} FoM_{i}}{\sum_{i} w_{i}}

    Here the weighted Figure of Merit for the :math:`i`-th dataset, :math:`FoM_{i}`, is given by
    a sum of the square difference between data points for a single ``ObservablePair``, normalised
    by the errors and the number of data points:

    .. math::

        FoM_{i} = \frac{w_{i}}{N_{i}} \sum_{j} (\frac{D_{j}^{exp} - D_{j}^{sim}}{\sigma_{j}^{exp}})^2

    where the sum is over the :math:`N_{i}` data points in the ``ObservablePair`` corresponding to
    the :math:`i`-th dataset, and :math:`w_{i}` is an importance weighting assigned to the
    :math:`i`-th dataset. :math:`D_{j}` are the individual data points in the 1-D or 2-D array of
    the experimental ``Observable`` (:math:`exp`) or simulated ``Observable`` (:math:`sim`), and
    :math:`\sigma_{j}^{exp}` are the elements in a 1-D or 2-D array corresponding to the error of the :math:`j`-th
    data point. Note that the subtraction and division over the arrays are element-wise. Note also that if the
    experimental ``Observable`` is not on an absolute scale, an additional ``rescale_factor`` can be
    specified (or automatically determined) by the ``ObservablePair`` to scale the experimental data points and
    errors by a simple linear scaling.
    """

    def calculate_single_FoM(self, obs_pair: ObservablePair):

        r"""
        Performs the error normalised square difference for an
        ``ObservablePair``. If ``obs_pair.auto_scale`` is `True`, then this
        will also set ``obs_pair.rescale`` to the value which minimizes the
        FoM. If we label ``rescale_factor``:math:`=\lambda` then the minimum of the FoM is obtained as:

        .. math::


            FoM_{i}(\lambda) &=& w_{i} \sum_{j} \left(\frac{\lambda*D_{j}^{exp} - D_{j}^{sim}}{\lambda*\sigma_{j}^{exp}}\right)^2 \\\\
            \left. \frac{dFoM_{i}}{d\lambda}\right|_{\lambda=\lambda_{min}} &=& 0 \\\\
            \lambda_{min} &=& \frac{A}{B} \\\\

        where we have:

        .. math::

            A &=& \sum\left(\frac{D_{j}^{sim}}{\sigma_{j}^{exp}}\right)^2 \\\\
            B &=& \sum \frac{D_{j}^{exp}*D_{j}^{sim}}{(\sigma_{j}^{exp})^2}


        Parameters
        ----------
        obs_pair : ObservablePair
            An ``ObservablePair`` for which the FoM is calculated

        Returns
        -------
        float
            The FoM for the obs_pair
        """

        if obs_pair.auto_scale:
            exp_errors = np.array(*obs_pair.exp_obs.errors.values())
            exp_values = np.array(*obs_pair.exp_obs.dependent_variables.values())
            MD_values = np.array(*obs_pair.MD_obs.dependent_variables.values())
            obs_pair.rescale_factor = (np.sum((MD_values / exp_errors) ** 2)
                                       / np.sum(MD_values * exp_values
                                                / exp_errors ** 2))

        n_datapoints = np.size(*obs_pair.exp_obs.dependent_variables.values())
        value_unreduced = np.sum((obs_pair.calculate_difference()
                                  / obs_pair.calculate_errors()) ** 2)
        return (obs_pair.weight * value_unreduced
                / (n_datapoints * np.sum(*obs_pair.n_averages.values())))
