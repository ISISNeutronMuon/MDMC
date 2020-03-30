.. _installation-label:

Installation
============

You can install MDMC in broadly two ways

1. Non-container option: directly onto your favorite hardware and OS, e.g.
   Mac and Linux laptop or HPC hardware.

 * **However** this require that molecular dynamics engines
   (e.g. `LAMMPS <https://lammps.sandia.gov>`_)
   are already installed on such hardware

2. Container option: run MDMC in a container that already has all relevant
   external dependencies pre-installed, including molecular dynamics engines

 * Two such are supported: Docker and Singularity. Docker has more widespread usage,
   but Singularity is targeted for HPC hardware. Good news, is that these two
   container technologies are similar to operate, abd once you have mastered
   one you have (almost) mastered the other.


Non-container installation
--------------------------
Ensure you have Git, Python 3, pip and relevant molecular dynamics engines already
installed (e.g. `LAMMPS <https://lammps.sandia.gov>`_).

MDMC is then installed using pip and Git:

.. code-block:: bash

  pip install -e git+https://github.com/MDMCproject/MDMCv0.2_pilot#egg=MDMC

This will install MDMC and all other dependencies, except the molecular dynamics engines.
A reason for the latter is that molecular dynamics software have been found, thus far,
to be challenging to install through pip.


Docker
------
A Docker container that includes all the external dependencies MDMC needs
can be downloaded.

Instructions on how to install Docker for Windows, Mac OS X, and Linux distributions is
`here <https://docs.docker.com/install/>`_. If you find this step tricky, perhaps due
to a specific OS version or otherwise, please don't hesitate to ask us questions about this also.

To run (start) a Docker container with MDMC external dependencies, type in command window:

.. code-block:: bash

    docker run -it mdmc/mdmc:latest

The optional `:latest` part of `mdmc/mdmc:latest` pulls down the latest version of
the MDMC dependencies. These are not expected to change frequently. Other choices may
be available in the future.

Note this MDMC Docker container does not currently include the MDMC code. Hence for now
please follow the the pip install command shown in section above.


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
browser on the local machine; you may be presented with multiple URL choices,
please use the one that includes 127.0.0.1 in the URL string.
**Recommended**: when finished using Jupyter, in terminal where the Jupyter server
was started, please press ctrl-c and then answer "y" to the question
"Shutdown this notebook server. This will exit Jupyter gracefully and help
avoid conflict when running this setup again.


Docker and GUI (and Jupyter)
^^^^^^^^^^^^^^
By default, Docker is not configured to enable GUI visualisation.  To enable
this it is possible to use the X11 system:

(to also enable this with Jupyter add `-p 8888:8888` to commands below - see section above)

**Windows**

To use X11, `VcXsrv <https://sourceforge.net/projects/vcxsrv/>`_ can be
installed. In Extra Settings select "Native opengl" and "Disable access
control". An alternative is to use `Xming <https://sourceforge.net/projects/xming/>`_
and running Xming in XLunch tick "No Access Control".
Next, open a standard Windows command prompt and type :code:`ipconfig` to get
the IP address (if e.g. using wireless then look for Wireless LAN adapter Wi-Fi
and IPv4 Address) and use it to replace the two letters "IP" in the following command:

.. code-block:: bash

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
