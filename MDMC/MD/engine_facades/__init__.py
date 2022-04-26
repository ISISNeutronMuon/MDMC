"""Facades for molecular dynamics packages

Contents
--------
ff
force_field_factory
lammps_engine (requires external module lammps.py)
"""

# TODO: Fix
# pylint: disable=import-self

from . import facade_factory
from . import facade

engines = ['lammps_engine', 'dlpoly_engine']

for engine in engines:
    try:
        from . import engine
    except ModuleNotFoundError:
        pass
