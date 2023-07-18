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
   :maxdepth: 1
   :caption: Simulation

   use-MDMC/simulations
   use-MDMC/notebooks/creating-atomic-configurations.ipynb
   use-MDMC/notebooks/defining-molecule-interactions.ipynb
   use-MDMC/notebooks/solvating-a-universe.ipynb
   use-MDMC/notebooks/read-configurations.ipynb
   use-MDMC/notebooks/molecular-visualization.ipynb
   use-MDMC/notebooks/applying-a-forcefield.ipynb

.. toctree::
   :maxdepth: 1
   :caption: Refinement

   use-MDMC/parameter-refinement
   use-MDMC/notebooks/selecting-fitting-parameters.ipynb
   ../../tutorials/running-a-refinement.ipynb

