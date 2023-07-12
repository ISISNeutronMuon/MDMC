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
Council under grant 2016-06954, the ISIS Neutron and Muon Source and the University of Copenhagen.

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

   introduction
   installation/installation
   contributing

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Tutorials

   tutorials/Argon-a-to-z.ipynb

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: How-to

   how-to/building-a-universe.ipynb
   how-to/read-configurations.ipynb
   how-to/units.ipynb
   how-to/applying-a-forcefield.ipynb
   how-to/solvating-a-universe.ipynb
   how-to/molecular-visualization.ipynb
   how-to/running-a-simulation.ipynb
   how-to/creating-an-observable.ipynb
   how-to/selecting-fitting-parameters.ipynb
   how-to/running-a-refinement.ipynb

.. toctree::
  :maxdepth: 1
  :hidden:
  :caption: Reference

  reference/api/modules
  reference/developer/overview

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Explanation

   explanation/parameter-refinement
   explanation/simulations