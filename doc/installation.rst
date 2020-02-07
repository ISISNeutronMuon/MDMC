.. _installation-label:

Installation
============

MDMC can be installed using pip and git:

.. code-block:: bash

  pip install -e git+https://github.com/MDMCproject/MDMCv0.2_pilot#egg=MDMC


Docker
------
To remove the need to install external dependencies, such as molecular dynamics
packages, a Docker container which includes the latest stable version of MDMC
can be downloaded. To run an interactive terminal in the container:

.. code-block:: bash

    docker run -it mdmc/mdmc:latest


Singularity
-----------
Singularity is an alternative to Docker which has been designed specifically for
high performance computing (HPC), with the majority of HPC centres providing
support for Singularity. To run MDMC in parallel using Singularity:

TO BE COMPLETED


Source Code
-----------
Source code is available from https://github.com/MDMCproject/MDMCv0.2_pilot and
can be obtained using git with:

.. code-block:: bash

    git clone https://github.com/MDMCproject/MDMCv0.2_pilot.git
