.. _dev_doc_testing-label:

Testing
=======
Almost all methods and functions within MDMC should have a minimum a single unit
test in order to validate the implementation. When contributing code, please
ensure that you add suitable unit tests in the relevant module within the tests
directory.

In general, MDMC uses `pytest <https://docs.pytest.org/en/stable/>`_ for unit
testing, although occasionally
`unittest.mock <https://docs.python.org/3/library/unittest.mock.html>`_ is also
used. Please use pytest unless your tests cannot be written using this
framework.

All tests (including system tests) are run from the command line with:

.. code-block:: bash

  python -m pytest tests/

Testing and LAMMPS/PyLammps
---------------------------
The LAMMPS Python interface (PyLammps) uses :code:`stdout` to get output from
the LAMMPS library. By default, pytest redirects :code:`stdout` which causes the
majority of tests which use LAMMPS to fail. To fix this, pytest must be forced
to be silent using the :code:`-s` flag:

.. code-block:: bash

  python -m pytest -s tests/
