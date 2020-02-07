.. _introduction-label:

Introduction
==================

MDMC is separated into two sections, the simulation and the refinement:

Simulation
----------
To run a refinement using MDMC it is first necessary to define the simulation
setup for which the parameters will be refined. This includes defining a
Universe, creating a configuration and specifying the topology, and defining the
conditions of the Simulation. Descriptions of the relevant objects can be found
in the section on :ref:`MD`, and there are several interactive (Jupyter
notebook) tutorials on topics relating to setting up a Simulation.

MDMC can also be used to run MD simulations without refinement, providing the
power of Python scripting and a number of helper methods to simplify setting up
simulations.

Refinement
----------
To refine the parameters of a simulation, one or more experimental datasets must
be provided and a minimiser must be selected.  Here are the descriptions of the
available experimental :ref:`Observables`, the `Minimisers`, and the `Control`
class, which runs the refinement.

It is possible to refine all of the parameters, or to specify a subset to be
refined, which is shown in the interactive tutorial :doc:`Selecting Fitting
Parameters`.

For a full demonstration of MDMC, including setting up a Simulation and running
a refinement, please see the tutorial :doc:`Refining the potential parameters of
liquid Argon`.

Tutorials
---------
