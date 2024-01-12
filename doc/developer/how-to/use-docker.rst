Use Docker for MDMC
===================

.. _dev_doc_setupdocker-label:

Setup Docker
------------

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

Building the Docker image
-------------------------

Github Actions automatically builds and tests new Docker images for MDMC.
It does so ONLY IF files in the ``/build/Docker`` directory are changed,
or if ``requirements.txt`` is changed. In this case, the pull request testing will
automatically detect these changes, rebuild the image from the Dockerfile,
test it, and then push it to ``mdmc/mdmc:ci-[branch]`` where [branch]
is the name of the branch the image was built on. Note that the 'slow' build stage,
the ``mdmc/engines`` image, is only built if its Dockerfile is changed.

If you need to rebuild the Docker image manually, go to the main directory for the source code then execute the command:

.. code-block:: bash

  docker build -t mdmc/mdmc:experimental -f ./build/Docker/Dockerfile .

which will build the image and give it the tag mdmc/mdmc:experimental.
Note that to push it to Docker hub, you need to be logged in as the mdmc user.

Please do not rebuild the Docker container using command line arguments
(add them to the Dockerfile instead) or rebuild the container without
updating the Dockerfile in the repository. This can cause issues and
unintended behaviour, as well as making the container non-reproducible by others.

To push the Dockerhub experimental image to latest, do the following:

.. code-block:: bash

  docker pull mdmc/mdmc:experimental
  docker tag mdmc/mdmc:experimental mdmc/mdmc:latest
  docker push mdmc/mdmc:latest

