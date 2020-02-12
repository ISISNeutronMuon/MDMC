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
can be found in the section on :ref:`simulation-label`, and there are several interactive
(Jupyter notebook) `Tutorials`_ on topics relating to setting up a simulation.

MDMC uses external molecular dynamics packages to run the simulations.
Currently `LAMMPS <https://lammps.sandia.gov>`_ is the only MD engine
implemented.

MDMC can also be used to run simulations without refinement, providing the
power of Python scripting and a number of helper methods to simplify setting up
simulations.

Refinement
----------
To :ref:`refine the parameters <parameter-refinement-label>` of a simulation, one or more experimental datasets must
be provided and a minimiser must be selected.  Here are the descriptions of the
available experimental observables, the minimisers, and the
:class:`~MDMC.control.control.Control` class, which runs the refinement.

It is possible to refine all of the parameters, or to specify a subset to be
refined, which is shown in the interactive tutorial `Selecting Fitting
Parameters <../tutorials/selecting-fitting-parameters.ipynb>`_.

For a full demonstration of MDMC, including setting up a simulation and running
a refinement, please see the tutorial `Refining the potential parameters of
liquid Argon <../tutorials/refining-liquid-argon.ipynb>`_.

.. _tutorials-label:

Tutorials
---------
There are a number of tutorials covering different aspects of MDMC located in
doc/tutorials/.  Each of these tutorials is an interactive Jupyter notebook
which can be modified and run by the user; this allows experimentation from an
established starting point.  The interactive tutorials can be accessed by
installing Jupyter and running it within the tutorials folder:

.. code-block:: bash

  pip install Jupyter
  Jupyter notebook

Static copies of these tutorials can be viewed using the links in the sidebar.
