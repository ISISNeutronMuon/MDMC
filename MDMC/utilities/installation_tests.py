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
import warnings


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

        if self.success is None:
            self.success = True


@InstlTestFactory.register('LAMMPS')
class InstlTestLAMMPS(InstlTestBase):

    def run(self):

        try:
            from lammps import PyLammps
            lmp = PyLammps()
            lmp.close()
            self.success = True
        except ImportError:
            self.success = False


@InstlTestFactory.register('X11 forwarding')
class InstlTestX11Forwarding(InstlTestBase):

    def run(self):

        try:
            from tkinter import Tk, TclError
        except ImportError:
            # Add log message that tkinter must be imported to test for X11
            # forwarding
            self.success = False
        try:
            Tk()
        except TclError:
            # Add log message about X11 failure
            self.success = False

        if self.success is None:
            self.success = True


@InstlTestFactory.register('Dynamic plotting')
class InstlTestDynamicPlotting(InstlTestBase):

    def run(self):

        self.success = True
