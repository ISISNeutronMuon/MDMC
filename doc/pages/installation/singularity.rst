.. _singularity-label:

Install with Singularity
========================

Singularity is an alternative to Docker which has been designed specifically for
high performance computing (HPC), with the majority of HPC centres providing
support for Singularity. If Singularity is installed, run MDMC in parallel
using:

.. code-block:: bash

  mpirun -np 12 singularity exec mdmc.sif python3 script.py

where "script.py" is the name of the script which will run MDMC.  In this
example MDMC will be split over 12 processes.
