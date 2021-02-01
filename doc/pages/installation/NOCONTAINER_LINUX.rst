.. _NOCONTAINER_LINUX-label:

1. INSTALL LAMMPS
------------------

There are two ways of installing LAMMPS on your computer. The first one is to do it from the pre-built executables for Linux and the other one is to download the code and build it by yourself.

1.1. Precompiled version 
-------------------------
The first ones can be obtained from the following page and installed following instructions on it: 

.. code-block:: bash

 https://lammps.sandia.gov/doc/Install_linux.html


1.2. Compile by yourself and specify libraries explicitly 
----------------------------------------------------------
For making own executables, please, download the code from the following page: https://lammps.sandia.gov/download.html
In order to be able to install LAMMPS you need to have Cmake installed on your machine.
Put the downloaded archive with the  code in the desired directory  and then uncompress the archive, by typing the following command in the terminal from that directory:

.. code-block:: bash 

 tar -xzvf file.tar.gz

Where “file” is the name of the downloaded archive.
Then go to the folder which was created after the decompression of the archive and make another directory build inside it. This can be done by typing the following commands in the terminal.

.. code-block:: bash

 cd lammps

.. code-block:: bash

 mkdir build

Then in you can start configuring the code with CMake by typing the following in the terminal (if you want to specify other options and environmental variables for CMake, like libraries and compilers, you might need to check the page for Cmake https://cmake.org/cmake/help/latest/manual/cmake-env-variables.7.html   ):

.. code-block:: bash

 cmake ../cmake

.. code-block:: bash

 cmake --build

Then in order to make LAMMPS running in parallel you will need to compile it by giving the number of processors/cores (N) typing the following line in the terminal:

.. code-block:: bash

 make -j N

After the compilation you may install it by typing:

.. code-block:: bash

 make install

2. INSTALL DEPENDENCIES FOR PYTHON3
------------------------------------

Since MDMC is a Python-based code the following dependencies need to be installed:

.. code-block:: bash

 pip, numpy, scipy, netCDF4, pandas, ase>=3.19, numba, mpi4py, ipython.

One of the best ways of installing Python-dependencies is to install Anaconda3 first. This can be done following instructions on the its page: 

.. code-block:: bash

 https://docs.anaconda.com/anaconda/install/linux/

3. GETTING MDMC WORK WITH NO CONTAINER
---------------------------------------

In principle, if all Python dependencies were installed one needs to do few things in order to make MDMC code work. 
Firstly, the LAMMPS executable shall be copied in the directory anaconda3/bin. This executable is named either lmp_stable, lmp, lmp_mpi, lmp_serial.
Secondly, the script lammps.py has to be copied to the same directory as well. This script can be found in the LAMMPS directory (lammps/python or whatever it is called) which you could download in step 1.2.
Then you are ready to download the code for MDMC from GitHub using the following link:

.. code-block:: bash

 https://github.com/MDMCproject/MDMCv0.2_pilot#egg=MDMC
 
When you downloaded the code you are suggested to  
