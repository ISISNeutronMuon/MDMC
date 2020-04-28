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


Installing MD packages
----------------------
As mentioned above, if you are installing without using containers, you must
ensure that you have one or more MD packages installed to which MDMC interfaces.
A short description of how to install these MD engines is included below.

LAMMPS
^^^^^^
THE MDMC interface to LAMMPS requires that all of the `Standard packages <https://lammps.sandia.gov/doc/Packages_standard.html>`_ are
included, and also the `USER-MISC <https://lammps.sandia.gov/doc/Packages_details.html#pkg-user-misc>`_
package.

Further, the MDMC interface requires `LAMMPS version <https://lammps.sandia.gov/download.html>`_
*stable_3Mar2020* or greater (more recent)

To install LAMMPS, please follow `these instructions <https://lammps.sandia.gov/doc/Python_head.html>`_.
