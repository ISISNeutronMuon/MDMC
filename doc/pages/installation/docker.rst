.. _docker-label:

Install with Docker
===================

A Docker container that includes all the external dependencies MDMC needs
can be downloaded.

Instructions on how to install Docker for Windows, Mac OS X, and Linux
distributions is `here <https://docs.docker.com/install/>`_. If you experience
problems with this installation, perhaps due to your specific OS version or
otherwise, please do not hesitate to ask us questions about this.

The following installation instructions will enable MDMC to run using the Docker
container. This will allow molecular visualization using the :code:`'X3DOM'`
viewer, which displays within a Jupyter notebook (which is automatically started
by the Docker container). However, to use the external :code:`'ASE'` viewer for
molecular visualization, X11 forwarding must additionally be enabled **before**
the instructions below are followed. The difference between
these viewers is explained in the
`Molecular Visualization <../tutorials/molecular-visualization.ipynb>`_
tutorial. Instructions on how to enable X11 forwarding are in
:ref:`x11-forwarding-label`.  X11 forwarding occurs by default on Linux, so
on that platform no additional steps are required.


Windows
-------
.. include:: docker/docker-compose-files.rst

:download:`Download mdmc docker-compose zip <../../_static/files/osx-windows/mdmc.zip>`

.. include:: docker/docker-compose-env-osx-windows.rst

At the command line, in the directory where the docker-compose.yml and the .env
files are located, do the following:

.. include:: docker/docker-compose-run.rst

Mac OS X
--------
.. include:: docker/docker-compose-files.rst

:download:`Download mdmc docker-compose zip <../../_static/files/osx-windows/mdmc.zip>`

.. include:: docker/docker-compose-env-osx-windows.rst

Then in a terminal, in the directory where the docker-compose.yml and the .env
files are located, do the following:

.. include:: docker/docker-compose-run.rst

Linux
-----
Docker-compose is not included by default when Docker is installed on Linux.  It
is possible to install docker-compose using pip3:

.. code-block:: bash

  pip3 install docker-compose

Alternatively, there are `other methods for installing docker-compose <https://docs.docker.com/compose/install/>`_.

.. include:: docker/docker-compose-files.rst

:download:`Download mdmc docker-compose tar <../../_static/files/linux/mdmc.tar.gz>`

.. include:: docker/docker-compose-env-linux.rst

Then in a terminal, in the directory where the docker-compose.yml and the .env
files are located, do the following:

.. include:: docker/docker-compose-run.rst


.. _x11-forwarding-label:

X11 Forwarding for Docker
-------------------------

By default, Docker is not configured to enable GUI visualisation (a feature of
Docker and containers).  To enable this use the X11 system, described below. GUI
is for example needed for using the :code:`'ASE'` viewer for visualising
molecule / universe setups with MDMC. However, the :code:`'X3DOM'` viewer, which
has almost all of the functionality of the :code:`'ASE'` viewer, can be used in
a Jupyter notebook **without** enabling X11 forwarding. The difference between
these viewers is explained in the
`Molecular Visualization <../tutorials/molecular-visualization.ipynb>`_
tutorial.

Windows
^^^^^^^
To use X11, either of the following can be installed:

* `VcXsrv <https://sourceforge.net/projects/vcxsrv/>`_ - After installation, In
  Extra Settings select "Native opengl" and "Disable access control".
* `Xming <https://sourceforge.net/projects/xming/>`_ - After installation, run
  XLaunch to configure Xming. Use the provided defaults except in the
  "Specify parameter settings" window, select "No Access Control".
  Optionally, use the generated `config.xlaunch` file when re-running Xming.

X11 forwarding will now be enabled, and molecular visualization with the 'ASE'
viewer will be possible.  Please continue with the Docker installation
instructions in :ref:`docker-label`.

Mac OS X
^^^^^^^^
To use X11, `xQuartz <https://www.xquartz.org>`_ can be installed.  In the
xQuartz Preferences -> Security select "Allow connections from network clients".

To allow the Docker container to access the local display using X11, run:

.. code-block:: bash

  xhost +localhost

If you exit XQuartz at any point, you will need to run the above command again
before running :code:`docker-compose up` (see the OS X installation instructions
in :ref:`docker-label`).

X11 forwarding will now be enabled, and molecular visualization with the 'ASE'
viewer will be possible.  Please continue with the Docker installation
instructions in :ref:`docker-label`.

Linux
^^^^^
As X11 is built-in to Linux, no additional software needs to be installed for
X11 forwarding.

