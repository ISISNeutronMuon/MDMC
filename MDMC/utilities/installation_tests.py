"""Module containing installation tests. These determine which features are
available in an installation of MDMC.

This includes:
- If MDMC can be imported
- If an MD engine can be run (e.g. LAMMPS)
- If X11 forwarding is enabled and the ASE gui can run
- If the optional dependencies for the dynamic plotting utility have been
  installed

These tests exist so that a user can test their installation of MDMC.
"""

from abc import ABC, abstractmethod
from glob import glob
from importlib import import_module
from os.path import basename, dirname, join
from typing import Callable, Dict, Optional


class InstlTestBase(ABC):

    """
    Base class for installation tests

    Attributes
    ----------
    success : str
        A 'str' denoting if the test has PASSED, FAILED, or is INCOMPLETE
    """

    def __init__(self):

        self._success: Optional[bool] = None

    @property
    def success(self) -> str:

        """
        Get whether or not the test has PASSED, FAILED or is INCOMPLETE
        """

        if self._success is True:
            return 'PASSED'
        if self._success is False:
            return 'FAILED'
        if self._success is None:
            return 'INCOMPLETE'
        raise ValueError('The success value of this test has been incorrectly'
                         ' set. Please run the test again.')

    @abstractmethod
    def run(self):

        """
        Runs the test and sets the value of ''self.success''
        """

        raise NotImplementedError


class InstlTestFactory:

    """
    A factory class which keeps a registry of the installation tests and creates
    instances of them
    """

    registry: Dict[str, InstlTestBase] = {}

    @classmethod
    def register(cls, name: str) -> Callable:

        """
        A class level decorator for registering installation test classes

        The name with which the test is registered should be the parameter
        passed to the decorator.

        Parameters
        ----------
        name : str
            The name with which the test is registered

        Example
        -------
        To register the ``InstlTestCore`` class with ``InstlTestFactory``:

            .. highlight:: python
            .. code-block:: python

                @InstlTestFactory.register('core')
                class InstlTestCore(InstlTestBase):
        """

        def class_wrapper(wrapped_class: InstlTestBase) -> Callable:

            cls.registry[name] = wrapped_class
            return wrapped_class

        return class_wrapper

    @classmethod
    def create_instl_test(cls, name: str) -> InstlTestBase:

        """
        Creates an instance of the installation test for the class which
        corresponds to the passed ``name``

        Parameters
        ----------
        name : str
            The ``registry`` name of the class to be initialized
        """

        return cls.registry[name]()


def run_installation_tests():

    """
    A helper function for running all installation tests and printing the
    result for each test
    """

    for name in InstlTestFactory.registry:
        instl_test = InstlTestFactory.create_instl_test(name)
        instl_test.run()
        print('{0: <30}   {1}'.format(name, instl_test.success))


@InstlTestFactory.register('core')
class InstlTestCore(InstlTestBase):

    """
    Class to test if all MDMC subpackages can be imported
    """

    def run(self):

        import MDMC
        fs_objects = glob(join(dirname(MDMC.__file__), "*"))
        for fso in fs_objects:
            fso_base = basename(fso)
            if '.py' not in fso_base[-4:] and fso_base != '__pycache__':
                try:
                    import_module('MDMC.' + fso_base)
                except ImportError:
                    self._success = False
                    break
                except Exception as err:
                    self._success = False
                    raise Exception('An Exception (other than an ImportError)'
                                    ' occured while MDMC was being imported.'
                                    ' It appears MDMC has installed this'
                                    ' Exception is likely to reduce'
                                    ' functionality.') from err

        if self._success is None:
            self._success = True


@InstlTestFactory.register('LAMMPS')
class InstlTestLAMMPS(InstlTestBase):

    """
    Class to test if LAMMPS is installed and if the PyLammps interface can be
    accessed
    """

    def run(self):

        try:
            from lammps import PyLammps
            lmp = PyLammps()
            lmp.close()
            self._success = True
        except ImportError:
            self._success = False


@InstlTestFactory.register('X11 forwarding')
class InstlTestX11Forwarding(InstlTestBase):

    """
    Class to test if tkinter can access the display. This is used to determine
    if X11 forwarding is working in Docker/Singularity containers.
    """

    def run(self):

        try:
            from tkinter import Tk, TclError
        except ImportError:
            # Add log message that tkinter must be imported to test for X11
            # forwarding
            self._success = False
        try:
            Tk()
        except TclError:
            # Add log message about X11 failure
            self._success = False

        if self._success is None:
            self._success = True


@InstlTestFactory.register('Dynamic plotting dependencies')
class InstlTestDynamicPlottingDependencies(InstlTestBase):

    """
    Class to test if the optional dependencies required for dynamic plotting are
    installed
    """

    def run(self):

        try:
            import jupyter
            import matplotlib
            import ipywidgets
        except ImportError:
            self._success = False
        except Exception as err:
            self._success = False
            raise Exception from err

        if self._success is None:
            self._success = True
