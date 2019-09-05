"""Contains helper functions for initiating SolventConfig subclasses for
solvents

It has two public functions, get_solvent_names and get_solvent_config, for
initating SolventConfig classes. It has a number of private functions for
finding SolventConfig subclasses and WaterModel subclasses which can be used
as solvents."""

from glob import glob
from importlib import import_module
from inspect import isclass, isabstract, getmembers
from os.path import dirname, basename, isfile, join

from MDMC.MD import force_fields
from MDMC.MD.force_fields.ff import WaterModel
from MDMC.MD.solvents.SPC_config import SPCConfig


def _get_water_models():

    """
    Gets a list of water model force fields that exist

    Returns
    -------
    list
        A list of classes which inherit from WaterModel
    """

    force_fields_dir = dirname(force_fields.__file__)
    modules = [import_module('.' + basename(f)[:-3], force_fields.__name__)
               for f in glob(join(force_fields_dir, "*.py"))
               if isfile(f) and not f.startswith('_')]

    w_models = []
    for module in modules:
        # try/except for modules which have no subclasses of SolventConfig and
        # so return an empty list
        try:
            w_models.append(getmembers(module,
                                       lambda m: (isclass(m)
                                                  and not isabstract(m)
                                                  and issubclass(m,
                                                                 WaterModel)
                                                 ))[0][1])
        except IndexError:
            pass

    return w_models


def _get_water_model_configs():

    """
    Gets the SolventConfig for each water model

    This is required because every water model does not have a unique
    solvent_config. For example, all 3 body water models use SPC_config.

    Returns
    -------
    dict
        {w_model: solvent_config} pairs, where each w_model is a str specifying
        an available water model, and solvent_config is the SolventConfig class
        that will be used for sovlating with that water model.
    """

    w_model_configs = {}
    for w_model in _get_water_models():
        # For 3 body water models, use the SPC216 configuration. This is
        # reasonable because the 3 body models are sufficiently similar that
        # it is assumed that the SPC216 config will require minimal
        # equilibration when used with another 3 body model.
        if w_model.n_body == 3:
            w_model_configs[w_model.__name__] = SPCConfig

    return w_model_configs
WATER_MODELS = _get_water_model_configs()
