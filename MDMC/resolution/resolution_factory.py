"""A factory pattern for instantiating Resolution objects."""
import warnings
from functools import singledispatchmethod
from pathlib import Path
from typing import Any

from MDMC.common.factory import ModuleFactory
from MDMC.resolution.resolution import Resolution


class ResolutionFactory(ModuleFactory[Resolution]):
    """
    Factory class for resolution window functions.

    Any function in `MDMC.resolution` s can be instantiated using this factory.
    """
    registry: dict[str, Resolution] = {}
    curr_path = Path(__file__).parent
    curr_pack = __package__
    exclude = (curr_path / "__init__.py", curr_path / "resolution_factory.py")

    @classmethod
    def create_instance(cls,
                        resolution: dict | float | str | None,
                        *args: Any) -> Resolution:
        """
        Create a Resolution object from a dictionary.

        Parameters
        ----------
        resolution : dict, float, or str
            A parameter specifying the resolution. Should be a one-line dict
            giving the resolution type and parameters, e.g. a Lorentzian resolution
            with FWHM of 3 is specified {'lorentzian': 3.0}.
            If a float is given, resolution is assumed to be Gaussian with FWHM
            of that float.
            If a str is given, the string is assumed to be a file path to a vanadium
            run used to define a resolution.

        Returns
        -------
        ~MDMC.resolution.Resolution
            A resolution object with the desired properties.
        """

        resolution = cls._standardise_input(resolution)
        function_name = list(resolution.keys())[0].title() + 'Resolution'
        function_res = list(resolution.values())[0]

        if function_name == "FileResolution":
            return cls.create(function_name, function_res, *args)
        return cls.create(function_name, function_res)

    @singledispatchmethod
    @staticmethod
    def _standardise_input(resolution) -> dict:
        """
        Ensure that resolution is a one-line dictionary.

        Fixes 'lazy' input, e.g. if resolution
        was input as a string or number.

        Parameters
        ----------
        resolution: Any
            The input to the resolution factory.

        Returns
        -------
        dict
            If ``resolution`` is a dict, returns the first item of the dict.
            If ``resolution`` was a float, returns ``{'gaussian': resolution}``
            If ``resolution`` was a string, returns ``{'file': resolution}``

        Raises
        ------
        NotImplementedError
            If ``resolution`` is not a dict, string or float.

        Warns
        -----
        SyntaxWarning
            If ``resolution`` is a dict with multiple lines, or a float.
        """
        raise NotImplementedError("Format of resolution function not recognised."
                                  " It should be a one-line dictionary, but can also"
                                  " be a number (for Gaussian resolution), a string"
                                  " (for resolution from file), or explicitly stated as"
                                  " None if resolution application is not needed.")

    @_standardise_input.register
    @staticmethod
    def _(resolution: dict) -> dict:
        if len(resolution) > 1:
            warnings.warn("The resolution dict should only have one line; ignoring"
                          " all lines except the first."
                          " Dictionaries are technically unordered, so"
                          " this may cause unintended behaviour!", SyntaxWarning)
            res = {list(resolution.keys())[0]: list(resolution.values())[0]}
        else:
            res = resolution
        if list(resolution.keys())[0].lower() == "from_file":
            res = {"file": list(resolution.values())[0]}
        return res

    @_standardise_input.register
    @staticmethod
    def _(resolution: str) -> dict:
        return {'file': resolution}

    @_standardise_input.register(int)
    @_standardise_input.register(float)
    @staticmethod
    def _(resolution: float) -> dict:
        warnings.warn("Assuming energy resolution is Gaussian. To change this,"
                      " input energy resolution as {'function': 'value'}, where"
                      " 'function' is your desired resolution approximation function.",
                      SyntaxWarning)
        return {'gaussian': float(resolution)}

    @_standardise_input.register
    @staticmethod
    def _(resolution: None) -> dict:
        return {'null': 0}


ResolutionFactory.scan()
