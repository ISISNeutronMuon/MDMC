"""Tests that package imports (i.e. using __init__) are valid and result in the
correct modules/classes/functions in the namespace
"""

from importlib import import_module
from pkgutil import walk_packages

import pytest
pytestmark = [pytest.mark.lammps]

import MDMC

def get_paths(modules=True):

    """
    Gets the import paths of all packages

    Parameters
    ----------
    modules : bool, optional
        If True then the paths of all modules are included in the return,
        otherwise only packages are included. The default is True.

    Returns
    -------
    list
        The import paths of each package, or each package and module
    """

    # Get the `ModuleInfo` for each subpackage and module in MDMC
    module_infos = walk_packages(MDMC.__path__, MDMC.__name__ + '.')

    # slice excludes MDMC, which doesn't need to be imported again
    if not modules:
        return [module_info.name for module_info in module_infos
                if module_info.ispkg][1:]
    return [module_info.name for module_info in module_infos][1:]


@pytest.mark.parametrize('path', get_paths())
def test_valid_imports(path):

    """
    Tests that all package and subpackages will import
    """

    import_module(path, __package__)
