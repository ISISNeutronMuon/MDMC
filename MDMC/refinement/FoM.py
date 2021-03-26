"""A module for Figure of Merits"""

from abc import ABC, abstractmethod

import numpy as np

from MDMC.common.decorators import repr_decorator
from MDMC.trajectory_analysis.observables.obs import Observable


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

        self.value = np.sum([self.calculate_single_FoM(obs_pair)
                             for obs_pair in self.obs_pairs])
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

        FoM_total = \sum_{i} FoM_{i}

    Here the weighted Figure of Merit for the :math:`i`-th dataset, :math:`FoM_{i}`, is given by
    a sum of the error normalised square difference between data points for a single ``ObservablePair``:

    .. math::

        FoM_{i} = w_{i} \sum_{j} (\frac{D_{j}^{exp} - D_{j}^{sim}}{\sigma_{j}^{exp}})^2

    where the sum is over the data points in the ``ObservablePair`` corresponding to the :math:`i`-th datasets, and
    :math:`w_{i}` is an importance weighting assigned to the :math:`i`-th dataset.
    :math:`D_{j}` are the individual data points in the 1-D or 2-D array of the experimental ``Observable``
    (:math:`exp`) or simulated ``Observable`` (:math:`sim`), and
    :math:`\sigma_{j}^{exp}` are the elements in a 1-D or 2-D array corresponding to the error of the :math:`j`-th
    datapoint. Note that the subtraction and division over the arrays are element-wise. Note also that if the
    the experimental ``Observable`` is not on an absolute scale, an additional ``rescale_factor`` can be
    specified in the ``ObservablePair`` to scale the experimental data points by a simple linear scaling.
    """

    def calculate_single_FoM(self, obs_pair):

        """
        Performs the error normalised square difference for an
        ``ObservablePair``

        Parameters
        ----------
        obs_pair : ObservablePair
            An ``ObservablePair`` for which the FoM is calculated

        Returns
        -------
        float
            The FoM for the obs_pair
        """

        return obs_pair.weight * (np.sum(obs_pair.calculate_difference()
                                         / obs_pair.calculate_errors()) ** 2)


@repr_decorator('weight', 'exp_obs', 'MD_obs')
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
        on the same scale as ``MD_obs``.
    """

    def __init__(self, exp_obs: Observable, MD_obs: Observable, weight: float, rescale_factor: float=1.):

        self.exp_obs = exp_obs
        self.MD_obs = MD_obs
        self.weight = weight
        self.rescale_factor = rescale_factor

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
            the ``errors`` in quadrature.
        """

        errors = (np.array(*self.exp_obs.errors.values()) ** 2
                  + np.array(*self.MD_obs.errors.values()) ** 2) ** 0.5

        return errors
