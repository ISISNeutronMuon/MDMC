.. _installation-label:

Installation
============

You can install MDMC in broadly two ways.  For non-expert users, it is strongly
recommended that you use containers to run MDMC (option 1), as you will not have
to separately install a supported molecular dynamics package (e.g. LAMMPS).

1. Container option: run MDMC in a container that already has all relevant
   external dependencies pre-installed, including molecular dynamics engines.

 * Two container technologies are supported: Docker and Singularity. Docker has
   more widespread usage, but Singularity is targeted for HPC hardware. These
   two container technologies are similar to operate, and once you are familiar
   with one, switching to the other should be relatively straightforward.

   :ref:`docker-label`

   :ref:`singularity-label`

2. Containerless option: directly onto your favourite hardware and OS, e.g. Mac
   or Linux laptop or HPC hardware.

 * **However** this requires that one or more molecular dynamics engines
   (e.g. `LAMMPS <https://lammps.sandia.gov>`_) are already installed.

   :ref:`containerless-label`

.. toctree::
   :maxdepth: 2
   :hidden:

   installation/docker
   installation/singularity
   installation/nocontainerlinux


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
