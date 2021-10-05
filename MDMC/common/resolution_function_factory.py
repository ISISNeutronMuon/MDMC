import MDMC.common.resolution_functions as resfuncs
from inspect import getmembers, isfunction


class ResolutionFunctionFactory(object):
    """
    Factory class for calling resolution functions.
    Any function in resolution_functions.py can be called using this factory.
    """
    functions = {}

    def __init__(self):
        self.load_functions()

    def load_functions(self):
        functions = getmembers(resfuncs,
                               lambda m: isfunction(m))

        for name, _type in functions:
            if isfunction(_type):
                self.functions.update([[name, _type]])

    def create_instance(self, function_name):
        if function_name in self.functions:
            return self.functions[function_name]
        else:
            raise NotImplementedError("Resolution function not recognised. Recognised functions are:",
                                      list(self.functions.keys()))
