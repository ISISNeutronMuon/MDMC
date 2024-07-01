.. _dev_doc_vscode-label:

Debugging inside Containers
===========================

MDMC's source code has a ``.devcontainer`` folder that can be used by Visual Studio Code
to run VSCode inside a Docker container. This is very useful for debugging or testing problems in our container setup.

How to run VSCode inside a container
------------------------------------

This is an abridged form of `this guide <https://code.visualstudio.com/docs/remote/containers>`_.

First, install and configure the following software:

* Windows: Docker Desktop 2.0+ and `set up WSL 2 <https://docs.docker.com/docker-for-windows/wsl/>`_.
* Mac OS: Docker Desktop 2.0+
* Linux: Docker 18.06+ and Docker Compose 1.21+

If on Linux, add your user to the ``docker`` permissions group with ``sudo usermod -a -G docker $USER``.

If on Windows, to run "docker" without needing the admin password:

- **Command Line Method**:
  Open Command Prompt as an administrator and enter:
  
  .. code-block:: shell
  
      net localgroup "docker-users" "<domain>\<user ID>" /add

- **GUI Method**:
    #. Open Computer Management as an admin.
    #. Go to System Tools > Local Users and Groups > Groups > docker-users.
    #. Add your user ID.

Remember to restart your computer for the changes to take effect.


Next, install `Visual Studio Code <https://code.visualstudio.com/>`_ and then install the `Remote Development extension pack <https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.vscode-remote-extensionpack>`_.

Finally, simply start VSCode within the MDMC repo using ``code .`` or open the
MDMC repo in VSCode, after a moment, a box in the bottom right-hand corner
pops up saying "Folder contains a Dev Container configuration file...". Then, click
"Reopen in Container". If this box does not pop up then open the Command Palette
(by pressing F1) and run the command **Remote-Containers: Open Folder in
Container...** toopen the MDMC project inside the MDMC container.

In addition, to run or debug the tutorials:

* open terminal in VSCode
* run ``python -m pip install .`` to install MDMC
* Then simply open relevant jupyter notebook tutorial in VSCode to run or debug

Note the way this works is that VSCode makes the MDMC repo visible inside the
container. This means that when you ``python -m pip install .`` it will install
MDMC with reference to that folder, including picking up (as per normal)
any old build/lib files from previously installs. So if you experience that
this setup appears to magically pick up an old version of MDMC then you need to
delete the build/lib folder in your repo (as you would need to also if you
didn't use containers).

Finally to build the documentation:

#. run ``apt-get update &&  apt-get install pandoc -y && pip3 install sphinx nbsphinx sphinx_rtd_theme docutils==0.16``
#. if not already done install MDMC with ``python -m pip install .``
#. cd into ``doc`` folder and run ``make html``. First time you run it will take a bit of time because it runs the code in all the tutorials.
#. if the doc fails to build due to tutorial errors, then update conf.py with ``nbsphinx_allow_errors = True`` and go back to step 3. 

Note the extra ``apt-get install`` packages needed to build the docs may change from time to time.
Please the see the continious integration build instructions which builds the
docs for up-to-date instructions on which ``apt-get`` packages are needed.
