.. _introduction-label:

Introduction
============

MDMC is separated into two sections, the simulation and the refinement:

Simulation
----------
To run a refinement using MDMC it is first necessary to define the simulation
setup for which the parameters will be refined. This includes defining a
:class:`~MDMC.MD.simulation.Universe`, creating a configuration and specifying
the topology, and defining the conditions of the
:class:`~MDMC.MD.simulation.Simulation`. Descriptions of the relevant objects
can be found in the section on :ref:`simulation-label`, and there are several
interactive (Jupyter notebook) `Tutorials`_ on topics relating to setting up a
simulation.

MDMC uses external molecular dynamics packages to run the simulations.
Currently `LAMMPS <https://lammps.sandia.gov>`_ is the only MD engine
implemented.

MDMC can also be used to run simulations without refinement, providing the
power of Python scripting and a number of helper methods to simplify setting up
simulations.

Refinement
----------
To :ref:`refine the parameters <parameter-refinement-label>` of a simulation,
one or more experimental datasets must be provided and a minimiser must be
selected.  Here are the descriptions of the available experimental observables,
the minimisers, and the :class:`~MDMC.control.control.Control` class, which runs
the refinement.

It is possible to refine all of the parameters, or to specify a subset to be
refined, which is shown in the interactive tutorial `Selecting Fitting
Parameters <../tutorials/selecting-fitting-parameters.ipynb>`_.

For an explanation of the refinement steps, please see the tutorial `Running a
Refinement <../tutorials/running-a-refinement.ipynb>`_, and for a full
demonstration of MDMC, including setting up a simulation and running
a refinement, please see the .py files within 'examples'.

.. _tutorials-label:

Tutorials
---------
There are a number of tutorials covering different aspects of MDMC located in
doc/tutorials/.  These tutorials provide a explanation of the main steps to
creating a simulation and running a refinement.  Major features are described
in these tutorials, however the full `Modules`_ documentation can be used to get
descriptions of all of the options available within MDMC.

Each of these tutorials is an interactive Jupyter notebook
which can be modified and run by the user; this allows experimentation from an
established starting point.  The interactive tutorials can be accessed by
installing Jupyter and running it within the tutorials folder:

.. code-block:: bash

  pip install Jupyter
  Jupyter notebook

Static copies of these tutorials can be viewed using the links in the sidebar.

.. _modules-label:

Modules
-------
The pages in the sidebar under the Modules heading provide documentation for
every function, class and class method in MDMC, including descriptions of
parameters (or options) which can be used in each of these cases.  This
documentation is also available when Python is run interactively using Python's
`help()` function.  For example, to get the documentation for the `Universe`
class:

.. code-block:: Python

  from MDMC.MD.simulation import Universe
  help(Universe)

and to get the documentation specific to the `solvate` method of the `Universe`
class:

.. code-block:: Python

  help(Universe.solvate)

It is also possible to get the same documentation by calling help on
instantiated objects:

.. code-block:: Python

  uni = Universe(10.)
  help(uni)

will output the same documentation as the first example.
