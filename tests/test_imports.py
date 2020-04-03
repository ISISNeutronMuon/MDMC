"""Tests that package imports (i.e. using __init__) are valid and result in the
correct modules/classes/functions in the namespace
"""

from importlib import import_module
from pkgutil import walk_packages

import pytest

import MDMC


def get_paths(modules=True):

    """
    Gets the import paths of all packages
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


@pytest.mark.parametrize('path, namespace')
def test_complete_namespaces():

    """
    Tests that the desired imports end up in the namespace
    """
