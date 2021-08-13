**THIS DOCUMENTATION IS ONLY AVAILABLE FOR INTERNAL TESTING. ACCESS TO MDMC IS
CURRENTLY RESTRICTED.**

MDMC Documentation
==================

MDMC is a Python package for optimising classical molecular dynamics (MD)
potential parameters by refining against experimental data, particularly
dynamical data such as the dynamic structure factor. The refinement uses
derivative free optimisation algorithms, e.g. Monte Carlo (MC).  See
:ref:`introduction-label` and :ref:`installation-label` to start using MDMC.

MDMC was originally developed at the ISIS Neutron and Muon Source,
Chalmers University, the European Spallation Source, and the University of
Copenhagen.

Python
------
Python is a powerful high level programming language which has a simple syntax,
which is why it is commonly used for scripting.  Detailed knowledge of Python
is **not** required to run MDMC, however a basic understanding will make setting
up refinements quicker and scripts more flexible. There is a short introduction
to Python `here <https://www.mantidproject.org/Introduction_To_Python>`_.

Contributing to MDMC
--------------------
MDMC is open source and we welcome contributions to the project.  Please see
:ref:`contributing-label` for more details on how to contribute.

Acknowledgements
----------------
Funding for the development of MDMC has been provided by the Swedish Research
Council under grant 2016-06954.

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


.. toctree::
   :maxdepth: 3
   :numbered:
   :hidden:
   :caption: Overview

   pages/introduction
   pages/installation
   pages/simulations
   pages/parameter-refinement
   pages/contributing

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Tutorials

   tutorials/building-a-universe.ipynb
   tutorials/read-configurations.ipynb
   tutorials/units.ipynb
   tutorials/applying-a-forcefield.ipynb
   tutorials/solvating-a-universe.ipynb
   tutorials/molecular-visualization.ipynb
   tutorials/running-a-simulation.ipynb
   tutorials/creating-an-observable.ipynb
   tutorials/selecting-fitting-parameters.ipynb
   tutorials/running-a-refinement.ipynb
   tutorials/Argon-a-to-z.ipynb

.. toctree::
  :maxdepth: 1
  :hidden:
  :caption: Developer Documentation

  pages/developer/overview
  pages/developer/coding_standards
  pages/developer/documentation
  pages/developer/units
  pages/developer/testing
  pages/developer/management
  pages/developer/containers
  pages/developer/vscode

.. toctree::
  :maxdepth: 1
  :hidden:
  :caption: Modules

  pages/modules/common
  pages/modules/control
  pages/modules/gui
  pages/modules/md
  pages/modules/readers
  pages/modules/refinement
  pages/modules/trajectory_analysis
