.. _NOCONTAINERLINUX-label:

In general the MDMC code can be downloaded as self-contained docker image, which requires no further input from the
user in order to make it work. However, in this document we give instructions of how to install a container-free version of MDMC and its dependencies on Linux systems. We suggest to install the molecular dynamics (MD) engine first, then the Python3 dependencies, followed by the MDMC code. The recommended MD engine to use with MDMC at the moment is LAMMPS.

1. INSTALL LAMMPS
------------------

There are two ways of installing LAMMPS on your computer. The first one is via the pre-built executables available for a variety of Linux distributions. The alternative is to download the source code and build the executables by yourself.

1.1. Precompiled version
-------------------------
The pre-compiled versions of LAMMPS can be obtained by following the instructions on the following page:

.. code-block:: bash

 https://lammps.sandia.gov/doc/Install_linux.html

Nevertheless, the version compiled by the user is preferred in this case, since one can explicitly specify the desired libraries that will be used for running the code in parallel, which can help to perform simulations more efficiently.


1.2. Compile by yourself and specify libraries explicitly
----------------------------------------------------------
To compile the LAMMPS executables, you need to have Cmake installed on your machine and download the source code from the following page: https://lammps.sandia.gov/download.html Put the downloaded archive with the code in the desired directory and then uncompress the archive by typing the following command in the terminal from that directory:

.. code-block:: bash

 tar -xzvf file.tar.gz

Where “file” is the name of the downloaded archive. Then go to the folder which was created after the decompression of the archive and create another directory called 'build' inside it. This can be done by typing the following commands in the terminal.

.. code-block:: bash

 cd lammps

.. code-block:: bash

 mkdir build

Then you can configure how CMake should compile LAMMPS by typing the following in the terminal (if you want to specify other options and environmental variables for CMake, like libraries and compilers, exact locations of libraries and compilers, you might need to check the page for Cmake https://cmake.org/cmake/help/latest/manual/cmake-env-variables.7.html ):

.. code-block:: bash

 cmake ../cmake

.. code-block:: bash

 cmake --build

Then in order to enable LAMMPS running in parallel you will need to compile it by giving the number of processors/cores (N) typing the following line in the terminal:

.. code-block:: bash

 make -j N

After the compilation you may install it by typing:

.. code-block:: bash

 make install


2. INSTALL DEPENDENCIES FOR PYTHON3
------------------------------------

Since MDMC is a Python-based code the following dependencies are required:

.. code-block:: bash

 pip, numpy, scipy, netCDF4, pandas, ase>=3.19, numba, mpi4py, ipython.

One of the best ways of installing Python-dependencies is to install them through Anaconda3. Firstly, one has to install Anaconda3 itself, which can be done by the following the instructions on the its installation page:

.. code-block:: bash

 https://docs.anaconda.com/anaconda/install/linux/

3. GETTING MDMC WORKING WITH NO CONTAINER
-----------------------------------------

In principle, if all Python dependencies have been installed then only a few steps are required to install and use the MDMC code. Firstly, the LAMMPS executable needs to be copied into the directory anaconda3/bin. This executable is named either lmp_stable, lmp, lmp_mpi, lmp_serial. Secondly, the script lammps.py has to be copied to the same directory as well. This script can be found in the LAMMPS directory (e.g. lammps/python) which you downloaded in step 1.2. Then you are ready to download the code for MDMC from GitHub using the following link:

.. code-block:: bash

 https://github.com/MDMCproject/MDMCv0.2_pilot#egg=MDMC

When you downloaded the code you are suggested to uncompress the archive. You are not supposed to run various parts of the source code, but tutorials or own refinement procedures which will depend on MDMC code.

To run a tutorial file (let's say it has a name tutorial.py) you will need to run the following line in your terminal:

.. code-block:: bash

 your_path_to_anaconda3/bin/python3 tutorial.py
 

4. MDMC ON SUPER-COMPUTERS
---------------------------

If you have access to high-performance computing systems you may use MDMC code from your job-submission directory and run calculations using various multi-core architectures. In most cases you won't need any sudo certificate. All you need to know is where various libraries are placed. If you have no sudo certificate you might need to install Anaconda3 in your local submission directory and then follow the same instructions as mentioned in sections 1.2-3 (not in 1.1!). Note, that you must be sure that all compilers are in your path first, before you install Anaconda3.

If on your HPC system you have a module system where compiles are loaded from modules, then before installing Anaconda3 try to save the names of modules which you loaded, because every time you will run MDMC code you will have to specify those environmental variables.


