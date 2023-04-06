"""Facades for molecular dynamics packages

Contents
--------
ff
force_field_factory
lammps_engine (requires external module lammps.py)
"""

from importlib import import_module
import warnings

from . import facade_factory
from . import facade

engines = ['lammps_engine', 'dlpoly_engine']

for engine in engines:
    try:
        import_module('MDMC.MD.engine_facades.' + engine)
    except ModuleNotFoundError:
        # for an unknown reason the warnings.warn method does not produce any output on
        # Python 3.9.6 within the docker container. hence the extra print statement just in case.
        print(f"Could not import the MD engine {engine}. Is it installed?")
        warnings.warn(f"Could not import the MD engine {engine}. Is it installed?", ImportWarning)
