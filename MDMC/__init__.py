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

"""MDMC is a Python 3 package for optimising classical molecular dynamics (MD)
potential parameters by refining against experimental data, particularly
dynamical data such as the dynamic structure factor. The refinement uses
derivative free optimisation algorithms, e.g. Monte Carlo (MC).

Documentation
-------------
MDMC has three different forms of documentation:

* docstrings provided within the code, such as this one. These can viewed from
  the Python command shell (REPL) by calling the ``help()`` function on the
  class, function, method or object.  In an interactive (ipython) command shell,
  ``?`` can also be used after a class, function, method or object.
* Online documentation, built from /doc, and available on the
  `MDMC homepage <http://mdmcproject.org>`_.
* Interactive (Jupyter notebook) tutorials, available within /doc/tutorials,
  which provide in dept descriptions and examples of specific features of MDMC.

Available subpackages
---------------------
common
    Functions and classes common to all subpackages
control
    Uses refinement and MD subpackages to control MDMC parameter optimisation
gui
    Provides GUI functionality
MD
    Molecular dynamics tools and engine interfaces
readers
    Atomic configuration and experimental observable file readers
refinement
    Algorithms for refining parameters
trajectory_analysis
    Tools related to creating trajectories and calculating observables from them

Logging
-------
By default MDMC will log a single process regardless of whether it
is being run in serial or parallel. The process will log to 'MDMC.log'.
"""

from .common.log import start_logging

start_logging()
