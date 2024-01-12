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