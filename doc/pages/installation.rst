.. _installation-label:

Installation
============

You can install MDMC in broadly two ways:

1. Non-container option: directly onto your favourite hardware and OS, e.g. Mac
   or Linux laptop or HPC hardware.

 * **However** this requires that one or more molecular dynamics engines
   (e.g. `LAMMPS <https://lammps.sandia.gov>`_) are already installed.

2. Container option: run MDMC in a container that already has all relevant
   external dependencies pre-installed, including molecular dynamics engines

 * Two container technologies are supported: Docker and Singularity. Docker has
   more widespread usage, but Singularity is targeted for HPC hardware. These
   two container technologies are similar to operate, and once you are familiar
   with one, switching to the other should be relatively straightforward.


Non-container installation
--------------------------
Ensure you have Git, Python 3, pip and relevant molecular dynamics engines
already installed (e.g. `LAMMPS <https://lammps.sandia.gov>`_).

MDMC is then installed using pip and Git:

.. code-block:: bash

  pip install git+https://github.com/MDMCproject/MDMCv0.2_pilot#egg=MDMC

This will install MDMC and all Python dependencies; this does not include the
molecular dynamics engines.

**Note1: While MDMC is in a private repository, the above** `pip install`
**require username and password**

**Note2: When MDMC is made available on** `PyPI <https://pypi.org>`_ **, the
installation will simply be:** `pip install MDMC`

Docker
------
A Docker container that includes all the external dependencies MDMC needs
can be downloaded.

Instructions on how to install Docker for Windows, Mac OS X, and Linux
distributions is `here <https://docs.docker.com/install/>`_. If you experience
problems with this installation, perhaps due to your specific OS version or
otherwise, please do not hesitate to ask us questions about this.

To run (start) a Docker container with MDMC external dependencies, type in
command window:

.. code-block:: bash

    docker run -it mdmc/mdmc:latest

The optional `:latest` part of `mdmc/mdmc:latest` pulls down the latest version
of the MDMC dependencies container. These are not expected to change frequently.
Other choices may be available in the future e.g. containers which only have
specific molecular dynamics engines installed.

Note this MDMC Docker container does not currently include the MDMC code. Hence
for now please follow the the pip install command shown in section above.

**Note: Once the MDMC repository is made public, the containers will come with
the latest stable version of MDMC preinstalled (i.e. not just the
dependencies).**

.. _docker-jupyter-label:

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
The Jupyter notebook can then be accessed by copying the provided URL into a
browser on the local machine (please use the URL that includes 127.0.0.1).
**Recommended**: When finished using Jupyter, in terminal where the Jupyter
server was started, please press ctrl-c and then answer "y" to shutdown the
Jupyter server; this will exit Jupyter gracefully and help avoid conflict when
running this setup again.


Docker and GUI
^^^^^^^^^^^^^^
By default, Docker is not configured to enable GUI visualisation.  To enable
this it is possible to use the X11 system, described below.  To also enable this
with Jupyter, add `-p 8888:8888` to commands below - see
:ref:`docker-jupyter-label`.

**Windows**

To use X11, either of the following can be installed:

* `VcXsrv <https://sourceforge.net/projects/vcxsrv/>`_ - After installation, In
  Extra Settings select "Native opengl" and "Disable access control".
* `Xming <https://sourceforge.net/projects/xming/>`_ - After installation, run
  XLaunch to configure Xming. Use the provided defaults except in the
  "Specify parameter settings" window, select "No Access Control".
  Optionally, use the generated `config.xlaunch` file when re-running Xming.

Next, open a standard Windows command prompt and type :code:`ipconfig` to get
the IP address (if e.g. using wireless then look for Wireless LAN adapter Wi-Fi
and IPv4 Address) and use it to replace the two letters "IP" in the following
command:

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
