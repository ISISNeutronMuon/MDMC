"""A factory pattern for instantiating Resolution objects."""
# pylint: disable=too-few-public-methods
# pylint: disable=cyclic-import
# as this is a factory class

import warnings
from inspect import getmembers, isabstract, isclass
from typing import Any, Union

import MDMC.resolution


class ResolutionFactory:
    """
    Factory class for resolution window functions.
    Any function in resolution_windows.py can be instantiated using this factory.
    """
    resolutions = {}

    def __init__(self):
        self._load_resolutions()

    def _load_resolutions(self):
        resolutions = getmembers(MDMC.resolution,
                                 lambda m: isclass(m) and not isabstract(m))

        for name, _type in resolutions:
            if isclass(_type) and issubclass(_type, MDMC.resolution.Resolution):
                self.resolutions.update([[name, _type]])

    def create_instance(self,
                        resolution: Union[dict, float, str],
                        *args: Any) -> 'MDMC.resolution.Resolution':
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

        resolution = _standardise_input(resolution)
        function_name = list(resolution.keys())[0].title() + 'Resolution'
        function_res = list(resolution.values())[0]

        if function_name in self.resolutions:
            # *args are only required by some resolution types, e.g. file resolution
            if function_name == 'FileResolution':
                return self.resolutions[function_name](function_res, *args)
            return self.resolutions[function_name](function_res)

        # else, error if unrecognised function is used
        # the userkeys line is to convert class names to the user equivalents
        userkeys = []
        for key in self.resolutions:
            userkeys.append(key.lower().replace('resolution', ''))
        raise NotImplementedError("Resolution function " + list(resolution.keys())[0] +
                                  " not recognised. Recognised functions are: " +
                                  str(userkeys))


def _standardise_input(resolution: Any) -> dict:
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

    if isinstance(resolution, dict):
        if len(resolution) > 1:
            warnings.warn("The resolution dict should only have one line; ignoring"
                          " all lines except the first."
                          " Dictionaries are technically unordered, so"
                          " this may cause unintended behaviour!", SyntaxWarning)
            res = {list(resolution.keys())[0]: list(resolution.values())[0]}
        else:
            res = resolution
    elif isinstance(resolution, str):
        res = {'file': resolution}
    elif isinstance(resolution, (float, int)):
        warnings.warn("Assuming energy resolution is Gaussian. To change this,"
                      " input energy resolution as {'function': 'value'}, where"
                      " 'function' is your desired resolution approximation function.",
                      SyntaxWarning)
        res = {'gaussian': float(resolution)}
    elif resolution is None:
        res = {'null': 0}
    else:
        raise NotImplementedError("Format of resolution function not recognised."
                                  " It should be a one-line dictionary, but can also"
                                  " be a number (for Gaussian resolution), a string"
                                  " (for resolution from file), or explicitly stated as"
                                  " None if resolution application is not needed.")

    return res
