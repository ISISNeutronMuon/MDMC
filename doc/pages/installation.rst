.. _installation-label:

Installation
============

MDMC can be installed using pip and git:

.. code-block:: bash

  pip install -e git+https://github.com/MDMCproject/MDMCv0.2_pilot#egg=MDMC

This will install MDMC and all Python dependencies, however it will not
install any of the molecular dynamics engines (e.g.
`LAMMPS <https://lammps.sandia.gov>`_); these will need to be installed
separately by the user.


Docker
------
To remove the need to install external dependencies, a Docker container which
includes the latest stable version of MDMC can be downloaded. Instructions on
how to install Docker for Windows, Mac OS X, and Linux distributions is
`here <https://docs.docker.com/install/>`_.

To run an interactive terminal in the container:

.. code-block:: bash

    docker run -it mdmc/mdmc:latest


Docker and Jupyter notebooks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
To run a Jupyter notebook from a Docker container, run the container with an
open port 8888:

.. code-block:: bash

  docker run -it -p 8888:8888 mdmc/mdmc:latest

Jupyter can then be run using:

.. code-block:: bash

  jupyter notebook --ip 0.0.0.0 --no-browser

When running as root, :code:`--allow-root` must be added to the above command.
The Jupyter notebook can then be accessed by coping the provided URL into a
browser on the local machine.


Docker and GUI
^^^^^^^^^^^^^^
By default, Docker is not configured to enable GUI visualisation.  To enable
this it is possible to use the X11 system:

**Windows**

To use X11, `VcXsrv <https://sourceforge.net/projects/vcxsrv/>`_ can be
installed.  In Extra Settings select "Native opengl" and "Disable access
control". Use :code:`ipconfig` to get the IP address and use it instead of "IP"
in the following command:

.. code-block:: bat

  docker run -it -e DISPLAY=IP:0.0 -v /tmp/.X11-unix:/tmp/.X11-unix mdmc/mdmc:latest

**Mac OS X**

To use X11, `xQuartz <https://www.xquartz.org>`_ can be installed.  In the
xQuartz Preferences -> Security select "Allow connections from network clients".
Then within the xQuartz terminal, run:

.. code-block:: bash

  ip=$(ipconfig getifaddr en0)
  xhost + $ip
  docker run -it -e DISPLAY=$ip:0 -v /tmp/.X11-unix:/tmp/.X11-unix mdmc/mdmc:latest

**Linux**

As X11 is built-in to Linux, no additional software needs to be installed.
Simply run:

.. code-block:: bash

  docker run -it -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix mdmc/mdmc:latest


Singularity
-----------
Singularity is an alternative to Docker which has been designed specifically for
high performance computing (HPC), with the majority of HPC centres providing
support for Singularity. If Singularity is installed, run MDMC in parallel
using:

.. code-block:: bash

  mpirun -np 12 singularity exec mdmc.sif python3 script.py

where "script.py" is the name of the script which will run MDMC.  In this
example MDMC will be split over 12 processes.


Source Code
-----------
Source code is available from https://github.com/MDMCproject/MDMCv0.2_pilot and
can be obtained using git with:

.. code-block:: bash

    git clone https://github.com/MDMCproject/MDMCv0.2_pilot.git
