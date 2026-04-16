"""
Factory class for generating observables.
"""

import copy
from typing import Any, cast

from MDMC.common.factory import RegisterFactory
from MDMC.refinement.FoM.FoM_abs import ObservablePair
from MDMC.trajectory_analysis.observables.obs import Observable


class ObservableFactory(RegisterFactory[Observable]):
    """
    Provide a factory for creating an :class:`Observable`.

    Any module within the observables submodule can be created with a
    string of the class name, as long as it is a subclass of
    ``Observable``.
    """

    registry: dict[str, Observable] = {}

    @classmethod
    def create(cls, key: str, *args, **kwargs) -> Observable:
        """
        Return an instance of given class.
        """
        obs = cast("Observable", super().create(key, *args, **kwargs))
        obs._name = key
        return obs

    @classmethod
    def create_from_file(
        cls,
        observable_type: str,
        reader: str,
        file_name: str,
        use_FFT: bool = True,
        filt: dict[str, Any] | None = None,
    ) -> Observable:
        """
        Creates an Observable of the specified type and reads in data from file

        Parameters
        ----------
        observable_type : str
            The ``type`` of the ``Observable``.
        reader : str
            The ``type`` of the ``Reader``.
        file_name : str
            The absolute or relative path and the file name.
        use_FFT: bool, optional
            Whether the Fast Fourier Transform should be used, default is True.

        Returns
        -------
        Observable
            An ``Observable`` of specified ``type``.
        """
        observable = cls.create(observable_type)
        observable.read_from_file(reader=reader, file_name=file_name)
        observable.use_FFT = use_FFT
        cls._setup_observable_data(observable, filt=filt)
        return observable

    @classmethod
    def create_empty_observable(
        cls,
        exp_observable: Observable,
        use_FFT: bool = True,
    ) -> Observable:
        """
        Creates a ``Observable`` without data but with independent variables
        specified from another ``Observable``.  This is a placeholder in which
        the ``Observable`` can be calculated from an MD trajectory.

        Parameters
        ----------
        exp_observable : Observable
            An ``Observable`` with defined independent variables.
        use_FFT: bool, optional
            boolean determining if the FFT should be used, default is True

        Returns
        -------
        Observable
            An ``Observable`` with only independent variables and
            ``origin == 'MD'``
        """

        observable = cls.create(exp_observable.name)
        observable._origin = "MD"
        observable.independent_variables = copy.deepcopy(exp_observable.independent_variables)
        observable.use_FFT = use_FFT
        return observable

    @staticmethod
    def _setup_observable_data(observable: Observable, filt: dict[str, Any] | None = None):
        if observable.uniformity_requirements:
            observable._make_data_uniform()

        if filt is not None:  # Data below threshold should be zeroed.
            observable._threshold_filter(
                abs_threshold=filt.get("abs", 0.0),
                rel_threshold=filt.get("rel", 0.0),
                magnitude=filt.get("use_magnitude", False),
                warn_threshold=filt.get("warn_threshold", 0.1),
            )

    @classmethod
    def create_observable_pair_from_file(
        cls,
        observable_type: str,
        reader: str,
        file_name: str,
        use_FFT: bool = True,
        filt: dict[str, Any] | None = None,
        weight: float = 1.0,
        auto_scale: bool = False,
        rescale_factor: float | None = None,
    ) -> ObservablePair:
        exp_observable = cls.create_from_file(
            observable_type,
            reader,
            file_name,
            use_FFT=use_FFT,
            filt=filt,
        )
        md_observable = cls.create_empty_observable(exp_observable, use_FFT)

        if auto_scale and rescale_factor is not None:
            print(
                "Both `rescale_factor` and `auto_scale` set for file"
                f" {file_name}; scaling will be automated"
                " to minimise FoM",
            )
            rescale_factor = 1.0
        elif not rescale_factor:
            rescale_factor = 1.0

        return ObservablePair(
            exp_observable,
            md_observable,
            weight,
            rescale_factor=rescale_factor,
            auto_scale=auto_scale,
        )
