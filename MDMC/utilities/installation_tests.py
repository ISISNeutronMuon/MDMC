"""Module containing installation tests. These determine which features are
available in an installation of MDMC.

This includes:
- If MDMC can be imported
- If an MD engine can be run (e.g. LAMMPS)
- If X11 forwarding is enabled and the ASE gui can run
- If the dynamic plotting utility can be used

These tests exist so that a user can test their installation of MDMC.
"""

from abc import ABC, abstractmethod
from glob import glob
from importlib import import_module
from os.path import basename, dirname, join
from typing import Callable


class InstlTestBase(ABC):

    def __init__(self):

        self.success = None

    @abstractmethod
    def run(self):

        raise NotImplementedError

    def print_result(self):

        print(self.success)


class InstlTestFactory:

    registry = {}

    @classmethod
    def register(cls, name: str) -> Callable:

        def class_wrapper(wrapped_class: InstlTestBase) -> Callable:

            cls.registry[name] = wrapped_class
            return wrapped_class

        return class_wrapper

    @classmethod
    def create_instl_test(cls, name: str) -> InstlTestBase:

        return cls.registry[name]()


def run_installation_tests():

    """
    A helper function for running all installation tests
    """

    for name in InstlTestFactory.registry:
        instl_test = InstlTestFactory.create_instl_test(name)
        instl_test.run()
        instl_test.print_result()


@InstlTestFactory.register('core')
class InstlTestCore(InstlTestBase):

    def run(self):

        import MDMC
        fs_objects = glob(join(dirname(MDMC.__file__), "*"))
        for fso in fs_objects:
            fso_base = basename(fso)
            if '.py' not in fso_base[-4:] and fso_base != '__pycache__':
                try:
                    import_module('MDMC.' + fso_base)
                except ImportError:
                    self.success = False
                    break
                except Exception as err:
                    raise Exception('An Exception (other than an ImportError)'
                                    ' occured while MDMC was being imported.'
                                    ' It appears MDMC has installed this'
                                    ' Exception is likely to reduce'
                                    ' functionality.') from err

        # Apparent anti-pattern used because do not want to set self.success
        # to True before the loop has completed
        if self.success is None:
            self.success = True

# def test_core_installation() -> bool:
#
#     """
#     Tests if core MDMC submodules can be imported
#     """
#
#     pass
#
#
# def test_MD_engines() -> bool:
#
#     """
#     Tests if each MD engine can be run by MDMC
#     """
#
#     pass
#
#
# def test_X11_forwarding() -> bool:
#
#     """
#     Tests that X11 forwarding is working and that ASE gui can run
#     """
#
#     pass
#
#
# def test_dynamic_plotting() -> bool:
#
#     """
#     Tests that dynamic plotting utility can be used
#     """
#
#     pass
