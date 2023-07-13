.. _use-MDMC-label:

How to use MDMC
===============

*For a quick run-through of the whole MDMC workflow, please see the* `Argon A-to-Z
tutorial. <../tutorials/Argon-a-to-z.ipynb>`_

MDMC consists of two main subsystems; simulation and refinement. The user creates
a :class:`~MDMC.MD.simulation.Simulation` object, and declares some :class:`~MDMC.MD.parameters.Parameters`
that they would like to fit to experimental data. This information is then passed to
a :class:`~MDMC.control.control.Control` object, which begins alternating between
running simulations, adjusting the parameters based on the output, running another
simulation with the new parameters, and so on until the parameters are fitted
to the data.

Each of these subsystems are explained in detail, with Jupyter notebooks,
in the following pages:

.. toctree::
   :maxdepth: 3

   use-MDMC/simulations
   use-MDMC/parameter-refinement
