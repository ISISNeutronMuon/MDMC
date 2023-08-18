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
`Molecular Visualization <../use-MDMC/notebooks/molecular-visualization.ipynb>`__
tutorial. Instructions on how to enable X11 forwarding are in
:ref:`x11-forwarding-label`.  X11 forwarding occurs by default on Linux, so
on that platform no additional steps are required.

As MDMC is currently private, Github credentials are required for access. This is via a personal access token that you can generate on Github.com. `Instructions are here <https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token>`, or short instructions are the following:

 * Log into Github.com.
 * Go to `https://github.com/settings/tokens <https://github.com/settings/tokens>`.
 * Click 'Generate new token' and enter your password.
 * In the 'note' field, enter "MDMC access"
 * Let the expiration time be whatever you want (although we highly advise against making it permanent, for user security)
 * Check the 'repo' box in the 'select scopes' section.
 * Scroll to the bottom and click 'generate token'.
 * Copy your personal access token and save it somewhere secure.

You now have a personal access token!

Windows
-------
.. include:: docker/docker-compose-files-osx-windows.rst

:download:`Download mdmc docker-compose zip <../../_static/files/osx-windows/mdmc.zip>`

Then, open the empty file 'token.txt' and paste your personal access token into it.

At the command line, in the directory where the docker-compose.yml and the token
file is located, do the following (please ensure that Docker Desktop is running, test
this e.g. by running :code:`docker run hello-world`):

.. include:: docker/docker-compose-run.rst

Mac OS X
--------
.. include:: docker/docker-compose-files-osx-windows.rst

:download:`Download mdmc docker-compose zip <../../_static/files/osx-windows/mdmc.zip>`

Then, open the empty file 'token.txt' and paste your personal access token into it.

Then in a terminal, in the directory where the docker-compose.yml and the token file is located, do the following (please ensure that Docker Desktop is running, test
this e.g. by running :code:`docker run hello-world`):

.. include:: docker/docker-compose-run.rst

Linux
-----
Docker-compose is not included by default when Docker is installed on Linux.  It
is possible to install docker-compose using pip3:

.. code-block:: bash

  pip3 install docker-compose

Alternatively, there are `other methods for installing docker-compose <https://docs.docker.com/compose/install/>`_.

.. include:: docker/docker-compose-files-linux.rst

:download:`Download mdmc docker-compose tar <../../_static/files/linux/mdmc.tar.gz>`

Then, open the empty file 'token' and paste your personal access token into it.

Then in a terminal, in the directory where the docker-compose.yml and the token
file is located, do the following:

.. include:: docker/docker-compose-run.rst


.. _x11-forwarding-label:

X11 Forwarding for Docker
-------------------------

By default, Docker is not configured to enable GUI visualisation (a feature of
Docker and containers).  To enable this use the X11 system, described below. GUI
is for example needed for using the :code:`'ASE'` viewer for visualising
molecule / universe setups with MDMC. However, the :code:`'X3D'` viewer, which
has almost all of the functionality of the :code:`'ASE'` viewer, can be used in
a Jupyter notebook **without** enabling X11 forwarding. The difference between
these viewers is explained in the
`Molecular Visualization <../use-MDMC/notebooks/molecular-visualization.ipynb>`__ guide.

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

