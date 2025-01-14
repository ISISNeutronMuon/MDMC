"""
Module containing installation tests.

These determine which features are available in an installation of MDMC.

This includes:

- Whether MDMC can be imported.
- Whether an MD engine can be run (e.g. LAMMPS).
- Whether X11 forwarding is enabled and the ASE gui can run.
- Whether the optional dependencies for the dynamic plotting utility have been
  installed.

These tests exist so that a user can test their installation of MDMC.
"""

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from glob import glob
from importlib import import_module
from os.path import basename, dirname, join
from typing import Literal

LOGGER = logging.getLogger(__name__)

# pylint: disable=import-outside-toplevel, unused-import
# these utilities explicitly test importing on purpose in this way


class InstlTestBase(ABC):
    """
    Base class for installation tests.

    Attributes
    ----------
    success : Literal['PASSED', 'FAILED', 'INCOMPLETE']
        A `str` denoting if the test has PASSED, FAILED, or is INCOMPLETE.
    name : str
        Name of current test.
    """

    def __init__(self):
        self._success: bool | None = None
        self.name: str | None = None

    @property
    def success(self) -> Literal['PASSED', 'FAILED', 'INCOMPLETE']:
        """
        Get whether or not the test has PASSED, FAILED or is INCOMPLETE.

        Returns
        -------
        str
            One of "PASSED", "FAILED" or "INCOMPLETE"
            depending on test state.

        Raises
        ------
        ValueError
            If test state is invalid.
        """

        if self._success is True:
            return 'PASSED'
        if self._success is False:
            return 'FAILED'
        if self._success is None:
            return 'INCOMPLETE'
        raise ValueError('The success value of this test has been incorrectly'
                         ' set. Please run the test again.')

    def log_test_passed(self) -> None:
        """
        Log that the test passed.
        """

        LOGGER.info('%s %s installation test passed',
                    self.__class__,
                    self.name)

    @abstractmethod
    def run(self):
        """
        Run the test and sets the value of `self.success`.
        """

        raise NotImplementedError


class InstlTestFactory(RegisterFactory[InstlTestBase]):

    """
    Testing factory class.

    A factory class which keeps a registry of the installation tests
    and creates instances of them.
    """

    registry: dict[str, InstlTestBase] = {}

    @classmethod
    def create(cls, key: str, *args, **kwargs) -> InstlTestBase:
        """
        Instantiate the installation test for the class `name`.

        Parameters
        ----------
        name : str
            The ``registry`` name of the class to be initialized.

        Returns
        -------
        InstlTestBase
            Corresponding class with name `name`.
        """

        instl_test = super().create(key)
        instl_test.name = key
        return instl_test


def run_installation_tests():
    """
    Run all installation tests and print the result for each test.
    """
    for name in InstlTestFactory.registry:
        instl_test = InstlTestFactory.create(name)
        instl_test.run()
        # Padded with spaces to ensure alignment
        print(f'{instl_test.name: <30}   {instl_test.success}')


@InstlTestFactory.register('core')
class InstlTestCore(InstlTestBase):

    """
    Class to test if all MDMC subpackages can be imported.
    """
    def run(self) -> None:

        import MDMC
        fs_objects = glob(join(dirname(MDMC.__file__), "*"))
        for fso in fs_objects:
            fso_base = basename(fso)
            if '.py' not in fso_base[-4:] and fso_base != '__pycache__':
                try:
                    import_module('MDMC.' + fso_base)
                except ImportError:
                    self._success = False
                    LOGGER.error('%s %s installation test failed because the'
                                 ' following package/module did not import: %s',
                                 self.__class__,
                                 self.name,
                                 fso_base,
                                 exc_info=True)
                    break
                except Exception as err:
                    self._success = False
                    LOGGER.error('%s %s installation test failed because an'
                                 ' exception occured (other than an'
                                 ' ImportError) while MDMC was being imported.',
                                 self.__class__,
                                 self.name,
                                 exc_info=True)
                    raise Exception('An Exception (other than an ImportError)'
                                    ' occured while MDMC was being imported.'
                                    ' It appears MDMC has installed but this'
                                    ' Exception is likely to reduce'
                                    ' functionality.') from err

        if self._success is None:
            self._success = True
            self.log_test_passed()


@InstlTestFactory.register('LAMMPS')
class InstlTestLAMMPS(InstlTestBase):

    """
    Class to test if LAMMPS is installed and PyLammps interface can be accessed.
    """

    LOG_ERROR_MSG: str = ('Due to this, MDMC will not be able to run MD'
                          ' simulations with LAMMPS as the MD engine. Other MD'
                          ' engines may still be available.')

    def run(self) -> None:

        try:
            from lammps import PyLammps
        except ImportError:
            self._success = False
            LOGGER.error('%s %s installation test failed because LAMMPS Python'
                         ' interface could not be imported, as lammps.py is not'
                         ' in the PYTHONPATH. Please see the LAMMPS'
                         ' documentation for how to rectify this. %s',
                         self.__class__,
                         self.name,
                         self.LOG_ERROR_MSG,
                         exc_info=True)
        except Exception as err:
            self._success = False
            LOGGER.error('%s %s installation test failed because the following'
                         ' Exception was thrown while the LAMMPS Python'
                         ' interface was being imported: %s',
                         self.__class__,
                         self.name,
                         self.LOG_ERROR_MSG,
                         exc_info=True)
            raise Exception from err
        try:
            lmp = PyLammps()
        except OSError:
            self._success = False
            LOGGER.error('%s %s installation test failed because a PyLammps'
                         ' instance could not be initialized. This is because'
                         ' lammps.py cannot find the LAMMPS library (either'
                         ' liblammps.so or liblammps.dylib). This can be'
                         ' rectified by ensuring that LAMMPS has been built as'
                         ' a shared library and the LAMMPS library is in the'
                         ' PYTHONPATH. Please see the LAMMPS documentation'
                         ' for further information. %s',
                         self.__class__,
                         self.name,
                         self.LOG_ERROR_MSG,
                         exc_info=True)
        except Exception as err:
            self._success = False
            LOGGER.error('%s %s installation test failed because the following'
                         ' Exception was thrown while a PyLammps instance was'
                         ' being initialized. %s',
                         self.__class__,
                         self.name,
                         self.LOG_ERROR_MSG,
                         exc_info=True)
            raise Exception from err

        if self._success is None:
            self._success = True
            self.log_test_passed()
            lmp.close()


@InstlTestFactory.register('X11 forwarding')
class InstlTestX11Forwarding(InstlTestBase):

    """
    Class to test if tkinter can access the display.

    This is used to determine if X11 forwarding is working
    in Docker/Singularity containers.
    """

    LOG_ERROR_MSG: str = ('Due to this, GUI elements requiring tkinter, such as'
                          ' the ASE viewer, will not be available. Other viewer'
                          ' options, such as the X3DOM viewer, can still be'
                          ' used.')

    def run(self) -> None:

        try:
            from tkinter import TclError, Tk
        except ImportError:
            self._success = False
            LOGGER.error('%s %s installation test failed because tkinter could'
                         ' not be imported. %s',
                         self.__class__,
                         self.name,
                         self.LOG_ERROR_MSG,
                         exc_info=True)
        try:
            Tk()
        except TclError:
            LOGGER.error('%s %s installation test failed because tkinter.Tk'
                         ' could not be initialized. This is probably because'
                         ' the DISPLAY cannot be accessed, which will occur if'
                         ' X11 forwarding has not been enabled. If MDMC is'
                         ' being run within Docker, please see the MDMC'
                         ' installation instructions for how to enable X11'
                         ' forwarding. %s',
                         self.__class__,
                         self.name,
                         self.LOG_ERROR_MSG,
                         exc_info=True)
            self._success = False

        if self._success is None:
            self._success = True
            self.log_test_passed()


@InstlTestFactory.register('Dynamic plotting dependencies')
class InstlTestDynamicPlottingDependencies(InstlTestBase):

    """
    Test if the dynamic plotting optional dependencies are installed.
    """

    LOG_ERROR_MSG = ('Due to this, dynamic plotting of the refinement will not'
                     ' be possible.')

    def run(self) -> None:

        try:
            import ipywidgets  # noqa: F401
            import jupyter  # noqa: F401
            import matplotlib  # noqa: F401
        except ImportError as err:
            self._success = False
            # err.name for an ImportError is the name of the missing library
            LOGGER.error('%s %s installation test failed because %s could not'
                         ' be imported. Please install this library to use'
                         ' dynamic plotting. %s',
                         self.__class__,
                         self.name,
                         err.name,
                         self.LOG_ERROR_MSG,
                         exc_info=True)
        except Exception as err:
            self._success = False
            LOGGER.error('%s %s installation test failed because the following'
                         ' Exception was thrown when the required dependencies'
                         ' were being imported. %s',
                         self.__class__,
                         self.name,
                         self.LOG_ERROR_MSG,
                         exc_info=True)
            raise Exception from err

        if self._success is None:
            self._success = True
            self.log_test_passed()
