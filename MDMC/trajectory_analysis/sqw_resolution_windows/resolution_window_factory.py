import MDMC.trajectory_analysis.sqw_resolution_windows.resolution_windows as window_funcs
from inspect import getmembers, isfunction


class ResolutionWindowFactory(object):
    """
    Factory class for resolution window functions.
    Any function in resolution_windows.py can be instantiated using this factory.
    """
    functions = {}

    def __init__(self):
        self.load_functions()

    def load_functions(self):
        functions = getmembers(window_funcs,
                               lambda m: isfunction(m))

        for name, _type in functions:
            if isfunction(_type):
                self.functions.update([[name, _type]])

    # functions in resolution_windows have '_window' on the end to prevent shadowing
    # users will input e.g. 'gaussian' and this will provide gaussian_window()
    def create_instance(self, function_name):
        if (function_name + '_window') in self.functions:
            return self.functions[(function_name + '_window')]
        else:
            # error if unrecognised function is used
            # the list comprehension is to remove '_window' from all functions to provide the user equivalents
            raise NotImplementedError("Resolution function not recognised. Recognised functions are: " +
                                      str([i[:-7] for i in list(self.functions.keys())]))
