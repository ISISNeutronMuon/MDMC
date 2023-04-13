
"""Modules calculating observables from molecular dynamics trajectories

Observables (/ indicates alias)
-----------
DYNAMIC_OBS_ALIASES

Examples
--------
Observables can be instantiated using the names above. For instance an SQw
observable can be instantiated using either of the aliases:

.. code-block:: python

    from MDMC.trajectory_analysis import observables
    sqw = observables.SQw()                     # This line...
    sqw = observables.DynamicStructureFactor()  # ...is equivalent to this line
"""

from collections import defaultdict
import glob
from importlib import import_module
from os.path import dirname, basename, isfile, join

from . import obs_factory

# Import all modules within this subpackage
# This is required to populate the ObservableFactory registry
MODULE_NAMES = glob.glob(join(dirname(__file__), "*.py"))
for full_module_name in MODULE_NAMES:
    if isfile(full_module_name) and full_module_name != __file__:
        module_name = basename(full_module_name)
        if not module_name.startswith('_') and module_name != 'obs.py':
            import_module(__name__ + '.' + module_name.replace('.py', ''))


def _merge_obs_aliases(registry: 'dict[str, type]') -> 'list[str]':
    inverse_registry = defaultdict(list)
    for reg_name, reg_class in registry.items():
        inverse_registry[reg_class].append(reg_name)
    aliases = [' / '.join(reg_names)
               for reg_names in inverse_registry.values()]
    return aliases


# Get the names and classes of observables registered with ObservableFactory
OBS_REGISTRY = obs_factory.ObservableFactory.registry
OBS_NAMES = sorted(OBS_REGISTRY.keys())
__all__ = []
for name in OBS_NAMES:
    # Add names (and corresponding classes) to module namespace
    globals()[name] = obs_factory.ObservableFactory.get_observable(name)
    # Add to __all__ so that help(MDMC.trajectory_analysis.observables) has
    # class definitions
    __all__.append(name)

# There is a one to many mapping from classes to names (due to aliases), so
# merge these and insert them into the module docstring
OBS_ALIASES = _merge_obs_aliases(OBS_REGISTRY)
__doc__ = __doc__.replace('DYNAMIC_OBS_ALIASES', '\n'.join(OBS_ALIASES))
