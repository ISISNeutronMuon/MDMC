To download the MDMC container (this might take a few minutes), run:

.. code-block:: bash

  docker-compose pull

To run the container, use the :code:`up` command.  On the first occasion this
will take 30-60 s as MDMC is downloaded.

.. code-block:: bash

  docker-compose up

This will start a Jupyter notebook server which will generate some URLs. To
access the server, open the URL starting with :code:`http://127.0.0.1:8888`
into a browser.  The MDMC tutorials will also be included within the directory
in which the Jupyter server is started.  This directory will also exist outside
of the container (in the directory from which you ran
:code:`docker-compose up`), and can be accessed outside of the container;
**however you will not be able to run any of the MDMC commands from outside of
the container**.

To test MDMC and GUI visualisation copy and paste Python code from the tutorial
`Molecular Visualization <../../tutorials/molecular-visualization.ipynb>`_ into
the Jupyter notebook.

To test MDMC and GUI visualisation copy and paste Python code from the tutorial
`Molecular Visualization <../../tutorials/molecular-visualization.ipynb>`_ into
the Jupyter notebook.

**Recommended:** When finished using Jupyter, in terminal where the Jupyter
server was started, please press ctrl-c and then answer "y" to shutdown the
Jupyter server; this will exit Jupyter gracefully and help avoid conflict when
running this setup again.

To restart the container, simply run:

.. code-block:: bash

  docker-compose up

Any notebooks which had been previously saved will still be available, unless
you explicitly remove the Docker volume (using :code:`docker volume rm`).
