.. _dev_doc_vscode-label:

Debugging inside Containers
===========================

MDMC's source code has a ``.devcontainer`` folder that can be used by Visual Studio Code to run VSCode inside a Docker container. This is very useful for debugging or testing problems in our container setup.

How to run VSCode inside a container
------------------------------------

This is an abridged form of `this guide <https://code.visualstudio.com/docs/remote/containers>`_.

First, install and configure the following software:

* Windows: Docker Desktop 2.0+ and `set up WSL 2 <https://docs.docker.com/docker-for-windows/wsl/>`_.
* Mac OS: Docker Desktop 2.0+
* Linux: Docker 18.06+ and Docker Compose 1.21+

If on Linux, add your user to the ``docker`` permissions group with ``sudo usermod -a -G docker $USER``.

Next, install `Visual Studio Code <https://code.visualstudio.com/>`_ and then install the `Remote Development extension pack <https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.vscode-remote-extensionpack>`_.

Finally, with the MDMC project open in VSCode, open the Command Palette (by pressing F1) and run the command **Remote-Containers: Open Folder in Container...** to open the MDMC project inside the MDMC container.
