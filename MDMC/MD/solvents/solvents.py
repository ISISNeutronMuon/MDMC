# MDMC is a package for the optimisation of classical potentials with experimental data
# Copyright (C) 2026 MDMC Developers
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Contains helper functions for initiating ``SolventConfig`` subclasses for
solvents

It has two public functions, ``get_solvent_names`` and ``get_solvent_config``,
for initating ``SolventConfig``. It has a number of private functions for
finding ``SolventConfig`` subclasses and ``WaterModel`` subclasses which can be
used as solvents."""

from contextlib import suppress
from glob import glob
from importlib import import_module
from inspect import getmembers, isabstract, isclass
from os.path import basename, dirname, isfile, join

from MDMC.MD import force_fields
from MDMC.MD.force_fields.ff import WaterModel
from MDMC.MD.solvents._solvent_config import SolventConfig
from MDMC.MD.solvents.SPC_config import SPCConfig


def _get_water_models():
    """
    Gets a `list` of ``WaterModel`` force fields that exist

    Returns
    -------
    list
        A `list` of classes which inherit from ``WaterModel``
    """

    force_fields_dir = dirname(force_fields.__file__)
    modules = [
        import_module("." + basename(f)[:-3], force_fields.__name__)
        for f in glob(join(force_fields_dir, "*.py"))
        if isfile(f) and not f.startswith("_")
    ]

    w_models = []
    for module in modules:
        # try/except for modules which have no subclasses of SolventConfig and
        # so return an empty list
        with suppress(IndexError):
            w_models.append(
                getmembers(
                    module,
                    lambda m: isclass(m) and not isabstract(m) and issubclass(m, WaterModel),
                )[0][1],
            )

    return w_models


def _get_water_model_configs():
    """
    Gets the ``SolventConfig`` for each ``WaterModel``

    This is required because every water model does not have a unique
    ``SolventConfig``. For example, all 3 body water models use ``SPC_config``.

    Returns
    -------
    dict
        {``w_model``: ``solvent_config``} pairs, where each ``w_model`` is a
        `str` specifying an available ``WaterModel``, and ``solvent_config`` is
        the ``SolventConfig`` class that will be used for sovlating with that
        ``WaterModel``.
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


def _get_solvent_configs():
    """
    Gets a `dict` of the names of ``SolventConfig`` and their classes

    Returns
    -------
    dict
        {``name``: ``solvent_config``} pairs, where each ``name`` is a `str`
        specifying an available ``SolventConfig`` subclass, and
        ``solvent_config`` is the corresponding class.
    """

    # Import all modules in same directory, except this one
    modules = [
        import_module("." + basename(f)[:-3], __package__)
        for f in glob(join(dirname(__file__), "*.py"))
        if isfile(f) and not f.startswith("_") and f != __file__
    ]
    # Get members of all modules if they are solvent_configs (i.e. they are
    # subclasses of SolventConfig)
    s_configs = []
    for module in modules:
        # try/except for modules which have no subclasses of SolventConfig and
        # so return an empty list
        with suppress(IndexError):
            s_configs.append(
                getmembers(
                    module,
                    lambda m: isclass(m) and not isabstract(m) and issubclass(m, SolventConfig),
                )[0][1],
            )
    return {s_config.__name__: s_config for s_config in s_configs}


def get_solvent_config(name):
    """
    Gets the ``solvent_config`` for a solvent

    Parameters
    ----------
    name : str
        The name of the solvent

    Returns
    -------
    ``SolventConfig``
        An object from a subclass of ``SolventConfig`` for the specified solvent
        name
    """

    s_config = SOLVENT_CONFIGS.get(name + "Config", None)
    if s_config is None:
        try:
            s_config = WATER_MODELS[name]
        except KeyError as error:
            raise ValueError(
                f"{name} is not an inbuilt solvent. "
                f"The inbuilt solvents are: {get_solvent_names()}",
            ) from error

    return s_config()


def get_solvent_names():
    """
    Get the names of the inbuilt solvents which can be passed as parameters to
    ``get_solvent_config``

    Returns
    -------
    list
        A `list` of `str` with the names of the inbuilt solvents
    """

    return list(
        set(
            list(WATER_MODELS.keys())
            + [name.replace("Config", "") for name in _get_solvent_configs()],
        ),
    )


SOLVENT_CONFIGS = _get_solvent_configs()
WATER_MODELS = _get_water_model_configs()
