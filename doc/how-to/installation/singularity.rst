.. _singularity-label:

Install with Singularity
========================

Using Singularity vs Docker
---------------------------
`Singularity <https://singularity.hpcng.org/user-docs/master/index.html>`_ is an alternative to Docker which has been designed 
specifically for high performance computing (HPC), with the majority of HPC centres providing support for Singularity. 

While the Docker installation method is faster for general use, it requires root/admin access, which you may not be able to acquire.
Thus installing with Singularity may be a better option if you are using MDMC at a HPC centre, as described above, or you
are working in a larger network where the admin cannot give you root access.

This tutorial expects the user to be running Singularity in Linux - on Windows or Mac, you must install Singularity in a 
virtual machine - see the `Singularity installation docs <https://sylabs.io/guides/3.0/user-guide/installation.html#install-on-windows-or-mac>`_.
Note also that these instructions were tested in Singularity version 3.8.0; note that other versions may give different
results, especially in major version 2 of Singularity.

Using MDMC with Singularity
---------------------------
Our implementation of Singularity uses its Docker integration. [Once Singularity is installed](https://singularity.hpcng.org/admin-docs/3.8/installation.html), you can build the Singularity container with the command:

.. code-block:: bash

  singularity pull docker://mdmc/mdmc:latest
  
which converts the MDMC Docker image to a .sif (Singularity container) file, which should be called mdmc_latest.sif. Then, in the directory containing the MDMC source code folder (here called MDMC), install MDMC with the following:

.. code-block:: bash

  singularity exec mdmc_latest.sif pip3 install ./MDMC
  
You can then run MDMC scripts using:

.. code-block:: bash

  singularity exec mdmc_latest.sif python3 script.py

where "script.py" is the MDMC script on your device you are trying to run. 

You can also run MDMC in parallel using:

.. code-block:: bash

  mpirun -np 12 singularity exec mdmc_latest.sif python3 script.py
  
again, with "script.py" being your script. In this example, MDMC will
be split over 12 processes.


