.. code-block:: bash

  docker-compose pull
  docker-compose up

This will start a Jupyter notebook server which will generate some URLs. To
access the server, open the final URL (it should start http://127.0.0.1:8888)
into a browser.

**Recommended:** When finished using Jupyter, in terminal where the Jupyter
server was started, please press ctrl-c and then answer "y" to shutdown the
Jupyter server; this will exit Jupyter gracefully and help avoid conflict when
running this setup again.

To restart the container, simply run:

.. code-block:: bash

  docker-compose up

Any notebooks which had been previously saved will still be available, unless
you explicitly remove the Docker volume (using :code:`docker volume rm`).
