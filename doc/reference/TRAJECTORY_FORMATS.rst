.. _TRAJECTORY-FORMATS-LAMMPS-label:

Trajectory formats from various packages in LAMMPS
==================================================

Package: MOLFILE
----------------

File formats which can be produced/considered:

.. code-block:: bash

  cfg, dcd, xtc, xyz, xml, pdb


Plugins which are used are bundled with VMD software.
This package also enables LAMMPS to create trajectories in various formats,
compatible with various molecular simulation tools.
In order to obtain plugins a user has to install VMD software.
Advantages of this package: it can reproduce most of trajectory file formats which can be later
analyzed by any other packages if needed.

Package: NETCDF
----------------

NetCDF is a binary, portable and self-describing file format for array data.
NetCDF has a better performance on a large number of processors and super-computers.
Note that style netcdf outputs all atoms sorted by atom tag while style netcdf/mpiio outputs
atoms in order of their MPI rank.

Package: H5MD
--------------

Before installing H5MD a user has to install hdf5 libraries.
HDF5 is a portable, binary, self-describing file format, used by many scientific simulations.
