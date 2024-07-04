.. _introduction-label:

Introduction
============

MDMC broadly provides two main functions, the MD (Molecular Dynamics)
simulation and the MC (Monte Carlo or similar) refinement of force-field
parameters:

Simulation
----------

*Main page:* :ref:`How to use MDMC: Simulation <simulation-label>`

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
implemented. The code for MDMC is written such as to be extensible and we
welcome requests of other MD packages to be added.

MDMC can be used to run simulations without refinement, and MDMC comces
with a number of helper methods aimed at simplifing setting up simulations.

Refinement
----------

*Main page:* :ref:`How to use MDMC: Parameter Refinement <parameter-refinement-label>`

To :ref:`refine the parameters <parameter-refinement-label>` of a simulation,
one or more experimental datasets must be provided and a minimiser must be
selected.  Here are the descriptions of the available experimental
:mod:`~MDMC.trajectory_analysis.observables` (the objects representing experimental datasets), 
the :mod:`~MDMC.refinement.minimizers`, and the :class:`~MDMC.control.control.Control` class, 
which runs the refinement.

It is possible to refine all of the parameters, or to specify a subset to be
refined, which is shown in the interactive tutorial `Selecting Fitting
Parameters <./how-to/use-MDMC/notebooks/selecting-fitting-parameters.ipynb>`_.

For an explanation of the refinement steps, please see the tutorial `Running a
Refinement <./how-to/use-MDMC/notebooks/running-a-refinement.ipynb>`_, and for a full
demonstration of MDMC, including setting up a simulation and running
a refinement, please see the .py files within 'examples'.

.. _tutorials-label:

Tutorials
---------
There are a number of Jupyter notebooks covering different aspects of MDMC located in
the documentation, mainly :ref:`use-MDMC-label`.  These tutorials provide an 
explanation of the main steps to creating a simulation and running a refinement.
Major features are described in these tutorials, however the full `Modules`_ 
documentation can be used to get descriptions of all of the options available within MDMC.

Each of these notebooks is an interactive Jupyter notebook
which can be modified and run by the user; this allows experimentation from an
established starting point.  The interactive notebooks can be accessed by
installing Jupyter and running it within the documentation notebook folder:

.. code-block:: bash

  pip install Jupyter
  Jupyter notebook

Static copies of these tutorials can be viewed using the links in the sidebar.

.. _modules-label:

Modules
-------
The pages in the sidebar under the Reference/MDMC heading provide documentation for
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
