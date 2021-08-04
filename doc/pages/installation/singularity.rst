.. _singularity-label:

Install with Singularity
========================

**NB: For general use, it is easier to install MDMC via Docker. These instructions**
**are for users who have a specific reason to be using Singularity over Docker,**
**such as running MDMC at a high performance computing centre.**

[Singularity](https://singularity.hpcng.org/) is an alternative to Docker which has been designed specifically for
high performance computing (HPC), with the majority of HPC centres providing
support for Singularity. If Singularity is installed, download the source code
for MDMC then navigate to the file [``\build\Singularity\mdmc.def``](https://github.com/MDMCproject/MDMCv0.2_pilot/blob/master/build/Singularity/mdmc.def), 
which is the definition file for the Singularity image. The image can then be built using

.. code-block:: bash

  singularity build mdmc.sif mdmc.def

You can then run MDMC in parallel using:

.. code-block:: bash

  mpirun -np 12 singularity exec mdmc.sif python3 script.py

where "script.py" is the MDMC script you are trying to run. In this
example MDMC will be split over 12 processes.
