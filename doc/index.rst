MDMC documentation
==================

MDMC is a Python package for optimising classical molecular dynamics (MD)
potential parameters by refining against experimental data, particularly
dynamical data such as the dynamic structure factor (DSF). The refinement uses
derivative free optimisation algorithms, e.g. Monte Carlo (MC).

MDMC is separated into two sections, the simulation and the refinement:

Simulation
----------
To run a refinement using MDMC it is first necessary to define the simulation
setup for which the parameters will be refined.

MDMC can also be used to run MD simulations without refinement, providing the
power of Python scripting and a number of helper methods to simplify setting up
simulations.

Refinement
----------
To refine the parameters of a simulation, one or more experimental datasets must
be provided and a minimiser must be selected.  It is possible to refine all of
the parameters, or to specify a subset to be refined.


Python
------
Python is a powerful high level programming language which has a simple syntax,
which is why it is commonly used for scripting.  Detailed knowledge of Python
is **not** required to run MDMC, however a basic understanding will make setting
up refinements quicker and scripts more flexible. There is a short introduction
to Python here.



.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
