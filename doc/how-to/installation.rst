.. _installation-label:

Installation
============

You can install MDMC using pip. To install
MDMC in a Python virtual environment, create a virtual environment
named mdmc_env by typing

.. code-block:: bash
  python3 -m venv mdmc_env

To activate your virtual environment, type

.. code-block:: bash
  source mdmc_env/bin/activate

in a bash console, or

.. code-block:: doscon
  mdmc_env\Scripts\activate.bat

if you are using cmd.exe on Windows.

Additonally, you will need to clone the MDMC repository:

.. code-block:: bash

  git clone https://github.com/ISISNeutronMuon/MDMC

After activating the environment and cloning the repository,
you can install MDMC:

.. code-block:: bash

  cd MDMC
  pip install .


Installation Tests
------------------

Following installation, you can run installation tests to check that MDMC has
installed correctly and which additional features are available (e.g. if the
required dependencies are installed for LAMMPS to be used as the MD engine).
These tests can either be run from within a Python environment or at the command
line.

To run the installation tests from the command line:

.. code-block:: bash

  MDMC test

To run the installation tests from a Python environment:

.. code-block:: Python

  from MDMC.utilities import run_installation_tests
  run_installation_tests()

Either of these methods will print to screen whether the MDMC core has been
correctly installed and whether the additional functionality can be used. If any
of these tests fails, additional details will be given in the log file.
Either of these methods will print to screen whether MDMC install components
were correctly installed. If any of these tests fails, additional details will
be given in the log file. Please note that all MDMC install components may not
be required for your intended usage of MDMC or your operating system
environment.

.. rubric:: Source Code

Source code is available from https://github.com/MDMCproject/MDMCv0.2_pilot and
can be obtained using git with:

.. code-block:: bash

    git clone https://github.com/MDMCproject/MDMCv0.2_pilot.git
