"""
Module defining a class for processing observables from MD trajectories.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Literal

import numpy as np
from scipy.interpolate import RectBivariateSpline, interp1d

from MDMC.common.decorators import repr_decorator
from MDMC.readers.observables.obs_reader_factory import ObservableReaderFactory
from MDMC.trajectory_analysis.compact_trajectory import CompactTrajectory


@repr_decorator("origin", "data")
class Observable(ABC):
    """
    Abstract class that defines methods common to all observable data containers.

    Observable data can either be from a file or calculated from
    MD and stored in the data property, along with the associated uncertainty.
    The `bool` property ``from_MD`` states the source of the information.

    Attributes
    ----------
    reader : ObservableReader
        The file reader for reading experimental data.
    """

    def __init__(self):
        self.reader = None
        self._name = ""
        self._errors = None
        self._dependent_variables = None
        self._independent_variables = None
        self._origin = None
        self.trajectory = None
        self.universe_dimensions = None

    @property
    def name(self) -> str:
        """
        Get or set the module name that used for factory instantiation.

        Returns
        -------
        str
            The name of the module in which the ``Observable`` is located.
        """

        return self._name

    @property
    def origin(self) -> Literal["experiment", "MD"]:
        """
        Get or set the origin of the observable.

        Returns
        -------
        {'experiment', 'MD'}
            The origin of the ``Observable``, either ``'experiment'`` or ``'MD'``.
        """

        return self._origin

    @property
    def data(self) -> dict:
        """
        Get the independent, dependent and error data.

        Returns
        -------
        dict
            The independent, dependent and error data.
        """

        return {
            "independent": self.independent_variables,
            "dependent": self.dependent_variables,
            "errors": self.errors,
        }

    @property
    @abstractmethod
    def independent_variables(self) -> dict:
        """
        The independent variables.

        Returns
        -------
        dict
            The independent variables.
        """

        raise NotImplementedError

    @property
    @abstractmethod
    def dependent_variables(self) -> dict:
        """
        The dependent variables.

        Returns
        -------
        dict
            The dependent variables.
        """

        raise NotImplementedError

    @property
    @abstractmethod
    def errors(self) -> dict:
        """
        The errors on the dependent variables.

        Returns
        -------
        dict
            The errors on the ``dependent_variables``.
        """

        raise NotImplementedError

    @abstractmethod
    def minimum_frames(self, dt: float | None = None) -> int:
        """
        The no. of ``CompactTrajectory`` frames needed to calculate the ``dependent_variables``.

        Parameters
        ----------
        dt : float, optional
            The time separation of frames in ``fs``, default is `None`.

        Returns
        -------
        int
            The minimum number of frames.
        """

        raise NotImplementedError

    @abstractmethod
    def maximum_frames(self) -> int:
        """
        The max no. of ``CompactTrajectory`` frames able to calculate the ``dependent_variables``.

        Returns
        -------
        int
            The maximum number of frames.
        """

        raise NotImplementedError

    @property
    def use_FFT(self) -> bool:
        """
        Get or set whether to use FFT when calculating from MD.

        Returns
        -------
        bool
            Whether to use FFT.
        """

        return self._use_FFT

    @use_FFT.setter
    def use_FFT(self, use_FFT: bool) -> None:
        self._use_FFT = use_FFT

    def read_from_file(self, reader: str, file_name: str) -> None:
        """
        Read in experimental data from a file using a specified reader.

        Parameters
        ----------
        reader : str
            The name of the required file reader.
        file_name : str
            The name of the file.
        """

        self._origin = "experiment"
        self.reader = ObservableReaderFactory.create(reader, file_name)
        with self.reader:
            self.reader.parse()
            self.reader.assign(observable=self)

    @abstractmethod
    def calculate_from_MD(
        self,
        MD_input: CompactTrajectory | list[CompactTrajectory],
        verbose: int = 0,
        **parameters: Any,
    ) -> None:
        """
        Calculate the observable using input from an MD simulation.

        Parameters
        ----------
        MD_input : CompactTrajectory | list[CompactTrajectory]
            Some input from an MD simulation, commonly a ``CompactTrajectory``.
        verbose : int
            Enables verbose printing of the calculation.
        **parameters
            Additional parameters required for calculation specific
            ``Observable`` objects.
        """

        raise NotImplementedError

    @property
    @abstractmethod
    def dependent_variables_structure(self) -> dict:
        """
        Structure of the dependent variables with respect to the independent variables.

        Specifically, the order in which the dependent variables are indexed
        with regards to the independent variables.

        The purpose of this method is to ensure that all ``Observable`` s of a particular type
        are created with dependent_variables that are consistent
        regardless of how they were created (e.g. by different ``Reader`` s).

        Returns
        -------
        dict
            The np.shape of the dependent variables.

        Examples
        --------
        If ``dep_var1[indep_var1_index, indep_var2_index, ...] == data point``
        for values of the independent_variables with the stated indices,
        then the relevant entry in the returned dict should be:
        ``{'dependent_variable1': [independent_variable1, independent_variable2, ...]}``

        .. note::

           This would also correspond to numpy.shape of the dependent variable being:

           .. code-block:: python

              np.shape(dependent_variable1)=(np.size(independent_variable1),
                                             np.size(independent_variable2), ...)
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def uniformity_requirements(self) -> dict[str, dict[str, bool]]:
        """
        Get the current limitations on ``independent_variables`` of the ``Observable``.

        It captures if the ``independent_variables`` are required to be uniform or to start at zero
        The keys of the returned dictionary should be the variables that have such a restriction,
        with the associated values being a dictionary with booleans
        if the variables are 'uniform' or 'zeroed'.

        Variables without any requirements do not need to be included, but can be included.

        .. note::

           If there are no uniformity requirements it is okay to return 'None'.

        Returns
        -------
        dict[str, dict[str, bool]]
            Dictionary of independent variables
            with their uniformity restrictions represented as booleans.
        """

        raise NotImplementedError

    def _is_data_uniform(self) -> dict[str, dict[str, bool]]:
        """
        Checks if the values for each independent variable of an ``Observable`` are uniformly
        spaced and if they start at zero. This information is returned in a single dictionary.

        Parameters
        ----------
        observable : Observable
            The ``Observable`` for which to check the independent variables.

        Returns
        -------
        `Dict[str, Dict[str, bool]]`
            An outer dictionary where the independent variables of the ``Observable`` are the keys,
            and the values are another dictionary corresponding to that variable. This inner
            dictionary has the same format for all variables, with the two keys 'uniform' and
            'zeroed'. The values for these keys are booleans that state whether the data fulfils
            the corresponding requirement.

        Examples
        --------
        >>> _is_data_uniform(observable)
        {'E': {'uniform': True, 'zeroed': True}, 'Q': {'uniform': True, 'zeroed': False}}
        """
        uniformity_dict = {}
        for var_key, var_data in self.independent_variables.items():
            uniform_data = np.linspace(min(var_data), max(var_data), num=len(var_data))
            is_uniform = np.allclose(var_data, uniform_data, rtol=1e-5)
            uniformity_dict[var_key] = {"uniform": is_uniform, "zeroed": var_data[0] == 0}
        return uniformity_dict

    def _make_data_uniform(self) -> Observable:
        """
        Takes an ``Observable``, checks the requirements for its ``independent_variables``
        to be uniform or start at zero, creates uniform grids for the variables that do not
        satisfy their requirement, interpolates the ``dependent_variables`` as needed,
        and returns an ``Observable`` with the uniform/interpolated variables.
        Limited to ``Observables`` with two-dimensional ``dependent_variables`` (e.g. SQw).
        This may change in future.

        Parameters
        ----------
        observable : Observable
            An ``Observable`` for which the independent variables
            need to be made uniform / to start at zero. Currently
            limited to ``Observables`` for which the ``dependent_variables`` are two-dimensional.

        Returns
        -------
        Observable
            Returns a copy of the passed ``Observable`` with the independent variables put onto
            uniform grid (for the variables where that is necessary)
            and the dependent variables interpolated onto the same grid
        """
        # get the uniformity requirements from the Observable
        uniformity_required = self.uniformity_requirements
        if uniformity_required is None:
            return self

        # determine for all independent_variables if they are currently uniform or start at zero
        uniformity_state = self._is_data_uniform()
        # initialise helper list for the independent_variables that need to be made uniform
        indep_vars_to_be_changed = []
        # loop through requirements
        for var_key, var_required in uniformity_required.items():
            var_state = uniformity_state[var_key]
            # if the variable has a requirement AND it is not satisfied
            # (for either uniformity OR zero-start)
            # then add it to the list of variables that need to be changed
            if (var_required["uniform"] and not var_state["uniform"]) or (
                var_required["zeroed"] and not var_state["zeroed"]
            ):
                indep_vars_to_be_changed.append(var_key)

        # if all uniformity requirements are already satisfied
        # simply return the original observable
        if not indep_vars_to_be_changed:
            return self

        # initialise a helper dictionary to hold the new independent variables
        indep_var_uniform = {}
        # loop through all independent variables
        for var_key in uniformity_state:
            # check if the independent variable needs to be made uniform
            if var_key in indep_vars_to_be_changed:
                data = self.independent_variables[var_key]
                minimum = 0 if uniformity_required[var_key]["zeroed"] else min(data)
                uniform_data = np.linspace(minimum, max(data), num=len(data))
                indep_var_uniform[var_key] = uniform_data
            # if uniformity requirements are satisfied already,
            # add the data points to the helper dictionary
            else:
                indep_var_uniform[var_key] = self.independent_variables[var_key]

        # get the indexing order of independent variables within the dependent variables
        var_indexing = self.dependent_variables_structure

        # loop through the dependent variables and interpolate them
        for var_key, data_list in self.dependent_variables.items():
            # Experimental Observables should only have 1 element in data_list
            try:
                assert len(data_list) == 1
                data = data_list[0]
            except AssertionError as error:
                raise AssertionError(
                    "Expected experimental dataset to only have one dependent "
                    f"variable entry for {var_key}, "
                    f"but found {len(data_list)} instead",
                ) from error

            # determine the dimension of the dependent variable
            var_dimension = data.ndim
            # interpolation for 1D
            if var_dimension == 1:
                x_data = self.independent_variables[var_indexing[var_key][0]]
                data_interpol = interp1d(x_data, data)
                x_uniform = indep_var_uniform[var_indexing[var_key][0]]
                uniform_data = data_interpol(x_uniform)
                # repeat the interpolation for the errors
                err_data = self.errors[var_key][0]
                err_data[err_data == float("inf")] = 0
                err_interpol = interp1d(x_data, err_data)
                err_uniform = err_interpol(x_uniform)
                err_uniform[err_uniform == 0.0] = float("inf")

            # interpolation for 2D
            elif var_dimension == 2:
                # Because Observable.dependent_variables_structure gives the order in which
                # the independent variables are represented in the np.shape of the data,
                # we have to reverse the order of the x and y arrays for interpolation:
                x_data = self.independent_variables[var_indexing[var_key][1]]
                y_data = self.independent_variables[var_indexing[var_key][0]]
                data_interpol = RectBivariateSpline(x_data, y_data, data.T)
                # get the independent_variables that satisfy the uniformity requirements
                # as created earlier
                x_uniform = indep_var_uniform[var_indexing[var_key][1]]
                y_uniform = indep_var_uniform[var_indexing[var_key][0]]
                uniform_data = data_interpol(x_uniform, y_uniform).T
                # repeat the interpolation for the errors
                err_data = self.errors[var_key][0]
                err_data[err_data == float("inf")] = 0
                err_interpol = RectBivariateSpline(x_data, y_data, err_data.T)
                err_uniform = err_interpol(x_uniform, y_uniform).T
                err_uniform[err_uniform == 0.0] = float("inf")
            else:
                raise NotImplementedError("Only 1D and 2D data can currently be made uniform")
            # save the uniform data and errors back into the Observable
            self.dependent_variables[var_key] = [uniform_data]
            self.errors[var_key] = [err_uniform]
        # finally, set the independent variables of the ``Observable`` to the uniform ones
        self.independent_variables = indep_var_uniform
        return self

    def _threshold_filter(
        self,
        *,
        abs_threshold: float = 0.0,
        rel_threshold: float = 0.0,
        magnitude: bool = False,
        warn_threshold: float = 0.1,
    ) -> int:
        """
        Remove data below threshold in-place.

        Removed data are set to zero and respective errors set to infinity
        to avoid weighting.

        Parameters
        ----------
        abs_threshold : float
            Minimum value below which to remove data.
        rel_threshold : float
            Proportion of the maximum value below which to remove data.
        magnitude : bool
            Whether the magnitude of the data must be below threshold
            or the original values.
        warn_threshold : float
            Fraction of data over which to warn too much may have been removed.

        Returns
        -------
        int
            Number of data removed.

        Notes
        -----
        If both abs and rel threshold provided, use the larger of the two, i.e.
        the one which will remove the most data.
        """
        n_data = 0
        n_skip = 0

        for variable in self.dependent_variables:
            for values, errors in zip(self.dependent_variables[variable], self.errors[variable]):
                threshold = max(rel_threshold * values.max(), abs_threshold)

                loc = np.abs(values) if magnitude else values
                loc = loc < threshold

                values[loc] = 0.0
                errors[loc] = np.inf
                n_data += values.size
                n_skip += np.count_nonzero(loc)

        logging.info("Threshold (%g) removed %d data of %d", threshold, n_skip, n_data)

        if n_skip / n_data > warn_threshold:
            logging.warning(
                "Over %d%% of data have been removed.", np.floor(100.0 * n_skip / n_data)
            )

        return n_skip
