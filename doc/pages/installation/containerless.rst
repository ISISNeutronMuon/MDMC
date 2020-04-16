.. _containerless-label:

Install without Containers
==========================
Ensure you have Git, Python 3, pip and relevant molecular dynamics engines
already installed (e.g. `LAMMPS <https://lammps.sandia.gov>`_).

MDMC is then installed using pip and Git:

.. code-block:: bash

  pip install git+https://github.com/MDMCproject/MDMCv0.2_pilot#egg=MDMC

This will install MDMC and all Python dependencies; this does not include the
molecular dynamics engines.

**Note1: While MDMC is in a private repository, the above** `pip install`
**require username and password**

**Note2: When MDMC is made available on** `PyPI <https://pypi.org>`_ **, the
installation will simply be:** `pip install MDMC`
