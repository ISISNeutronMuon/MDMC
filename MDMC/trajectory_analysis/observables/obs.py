"""Module defining a class for storing, calculating and reading in observables
from molecular dynamics trajectories."""

import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
from concurrent.futures import ThreadPoolExecutor as PoolExecutor

from MDMC.common.decorators import repr_decorator
from MDMC.readers.observables.obs_reader_factory import ObservableReaderFactory

if TYPE_CHECKING:
    from MDMC.trajectory_analysis.compact_trajectory import CompactTrajectory
    from typing import Union

N_CPUS_MP = 1

# A (Thread)PoolExecutor is created here, and is later imported
# by other observables.
# The same environment variable that defines the number of OMP threads
# is used here to fix the max number of threads for the pool executor.
# vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
if "OMP_NUM_THREADS" in os.environ:
    omp_num_threads_str = os.environ["OMP_NUM_THREADS"]
    try:
        omp_num_threads = int(omp_num_threads_str)
    except ValueError:
        pass
    else:
        if omp_num_threads > 1:
            N_CPUS_MP = omp_num_threads

# NOTE: The import in line 7 specifies that the PoolExecutor
# is a ThreadPoolExecutor.
# There is still a possibility of replacing it with a
# ProcessPoolExecutor. The thread-based version is better for
# performance based on the tests so far.
executor = PoolExecutor(max_workers=N_CPUS_MP)

@repr_decorator('origin', 'data')
class Observable(ABC):

    """
    Abstract class that defines methods common to all observable
    data containers

    Observable data can either be from a file or calculated from
    MD and stored in the data property, along with the associated uncertainty.
    The `bool` property ``from_MD`` states the source of the information.

    Attributes
    ----------
    reader : ObservableReader
        The file reader for reading experimental data
    """

    def __init__(self):
        self.reader = None
        self._errors = None
        self._dependent_variables = None
        self._independent_variables = None
        self._origin = None
        self.trajectory = None
        self.universe_dimensions = None

    @property
    def name(self) -> str:
        """
        Get or set the module name that used for factory instantiation

        Returns
        -------
        str
            The name of the module in which the ``Observable`` is located
        """

        return self._name

    @name.setter
    def name(self, name: str) -> None:
        self._name = name

    @property
    def origin(self) -> str:
        """
        Get or set the origin of the observable

        Returns
        -------
        str
            The origin of the ``Observable``, either ``'experiment'`` or ``'MD'``
        """

        return self._origin

    @origin.setter
    def origin(self, origin: str) -> None:
        self._origin = origin

    @property
    def data(self) -> dict:
        """
        Get the independent, dependent and error data

        Returns
        -------
        dict
            The independent, dependent and error data
        """

        return {'independent': self.independent_variables,
                'dependent': self.dependent_variables,
                'errors': self.errors}

    @property
    @abstractmethod
    def independent_variables(self) -> dict:
        """
        The independent variables

        Return
        ------
        dict
            The independent variables
        """

        raise NotImplementedError

    @property
    @abstractmethod
    def dependent_variables(self) -> dict:
        """
        The dependent variables

        Return
        ------
        dict
            The dependent variables
        """

        raise NotImplementedError

    @property
    @abstractmethod
    def errors(self) -> dict:
        """
        The errors on the dependent variables

        Return
        ------
        dict
            The errors on the ``dependent_variables``
        """

        raise NotImplementedError

    @abstractmethod
    def minimum_frames(self, dt: float = None) -> int:
        """
        The minimum number of ``CompactTrajectory`` frames needed to
        calculate the ``dependent_variables``

        Parameters
        ----------
        dt : float, optional
            The time separation of frames in ``fs``, default is `None`

        Returns
        -------
        int
            The minimum number of frames
        """

        raise NotImplementedError

    @abstractmethod
    def maximum_frames(self) -> int:
        """
        The maximum number of ``CompactTrajectory`` frames that can be
        used to calculate the ``dependent_variables``

        Returns
        -------
        int
            The maximum number of frames
        """

        raise NotImplementedError

    @property
    def use_FFT(self) -> bool:
        """
        Get or set whether to use FFT when calculating from MD

        Returns
        -------
        bool
            Whether to use FFT
        """

        return self._use_FFT

    @use_FFT.setter
    def use_FFT(self, use_FFT: bool) -> None:
        self._use_FFT = use_FFT

    def read_from_file(self, reader: str, file_name: str) -> None:
        """
        Reads in experimental data from a file using a specified reader

        Parameters
        ----------
        reader : str
            The name of the required file reader
        file_name : str
            The name of the file
        """

        self._origin = 'experiment'
        self.reader = ObservableReaderFactory.create_reader(reader, file_name)
        with self.reader:
            self.reader.parse()
            self.reader.assign(observable=self)

    @abstractmethod
    def calculate_from_MD(self,
                          MD_input: 'Union[CompactTrajectory, list[CompactTrajectory]]',
                          verbose: int = 0, **parameters: dict) -> None:
        """
        Calculates the observable using input from an MD simulation

        Parameters
        ----------
        MD_input : Object
            Some input from an MD simulation, commonly a ``CompactTrajectory``
        verbose : int
            Enables verbose printing of the calculation
        **parameters
            Additional parameters required for calculation specific
            ``Observable`` objects
        """

        raise NotImplementedError

    @property
    @abstractmethod
    def dependent_variables_structure(self) -> dict:
        # ignore line too long linting as it is necessary for python code formatting
        # pylint: disable=line-too-long
        """
        The structure of the dependent variables with respect to the independent variables.
        Specifically, the order in which the dependent variables are indexed
        with regards to the independent variables.
        Example: if
        dep_var1[indep_var1_index, indep_var2_index, ...] = data point
        for values of the independent_variables with the stated indices
        then the relevant entry in the returned dict should be:
        {'dependent_variable1': [independent_variable1, independent_variable2, ...]}
        Note that this would also correspond to numpy.shape of the dependent variable being:
        np.shape(dependent_variable1)=(np.size(independent_variable1), np.size(independent_variable2), ...)

        The purpose of this method is to ensure that all ``Observable``s of a particular type
        are created with 'dependent_variables' that are consistent
        regardless of how they were created (e.g. by different ``Reader``s).

        Return
        ------
        dict
            The np.shape of the dependent variables
        """

        raise NotImplementedError

    @property
    @abstractmethod
    def uniformity_requirements(self) -> dict[str, dict[str, bool]]:

        """
        Represents the current limitations on ``independent_variables`` of the ``Observable``.
        It captures if the ``independent_variables`` are required to be uniform or to start at zero
        The keys of the returned dictionary should be the variables that have such a restriction,
        with the associated values being a dictionary with booleans
        if the variables are 'uniform' or 'zeroed'.
        Variables without any requirements do not need to be included, but can be included.
        If there are no uniformity requirements it is okay to return 'None'.

        Return
        ------
        dict[str, dict[str, bool]]
            Dictionary of independent variables
            with their uniformity restrictions represented as booleans
        """

        raise NotImplementedError
