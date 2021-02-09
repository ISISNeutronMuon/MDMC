.. _NOCONTAINERLINUX-label:

In general the MDMC code can be downloaded as self-contained docker image, which requires no further input from the user in order to make it work. However, in this document we give instructions of how to install a container-free version of MDMC and its dependencies on Linux systems. We suggest to install the molecular dynamics (MD) engine first, then the Python3 dependencies, followed by the MDMC code. The recommended MD engine to use with MDMC at the moment is LAMMPS.

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


2. INSTALLING MDMC FROM SOURCE
------------------------------
2.1 Downloading MDMC SOURCE CODE
--------------------------------
MDMC based on Python3 and available on GitHub, which allows the source code to be downloaded as

.. code-block:: bash

 git clone git@github.com:MDMCproject/MDMCv0.2_pilot.git

Alternatively, you can download a ZIP archive containing the source using

.. code-block:: bash

 wget https://github.com/MDMCproject/MDMCv0.2_pilot/archive/master.zip

2.2 Installing Python dependencies
----------------------------------
We supply a requirements.txt file that can be used to install all required Python dependencies via

.. code-block:: bash

 cd MDMCv0.2_pilot/
 python3 -m pip install -r requirements.txt

The list of required Python modules is
.. code-block:: bash

 pip, numpy, scipy, netCDF4, pandas, ase>=3.19, numba, mpi4py, ipython.

An alternative way of installing the Python dependencies is to install them through Anaconda3. To do this one has to first install Anaconda3 itself, which can be done by the following the instructions on the its installation page:

.. code-block:: bash

 https://docs.anaconda.com/anaconda/install/linux/

2.3 Installing MDMC package
---------------------------

To install MDMC and add it to the list of Python modules you can simply run the following command from the root directory of the MDMC source code:

.. code-block:: bash

 python3 -m pip install MDMC

Once this is done, you should see MDMC appear on the list of installed modules when running

.. code-block:: bash

 pip3 list installed

3. MDMC ON SUPER-COMPUTERS
---------------------------

If you have access to high-performance computing systems you may use MDMC code from your job-submission directory and run calculations using various multi-core architectures. In most cases you won't need any sudo certificate. All you need to know is where various libraries are placed. If you have no sudo certificate you might need to install Anaconda3 in your local submission directory and then follow the same instructions as mentioned in sections 1.2-3 (not in 1.1!). Note, that you must be sure that all compilers are in your path first, before you install Anaconda3.

If on your HPC system you have a module system where compiles are loaded from modules, then before installing Anaconda3 try to save the names of modules which you loaded, because every time you will run MDMC code you will have to specify those environmental variables.


4. ADDITIONAL NOTES
-------------------
4.1 Installation instructions for Ubuntu 18
-------------------------------------------
An example of the instructions to install MDMC on Ubuntu 18 can be found in https://github.com/MDMCproject/MDMCv0.2_pilot/blob/master/build/Docker/Dockerfile .

4.2 Installation instructions for CentOS 7
------------------------------------------
The following commands will install MDMC and all its dependencies, including LAMPPS, on a CentOS 7 environment.

.. code-block:: bash

 #install LAMMPS, git and other required header files
 sudo yum install -y lammps-openmpi git python3-devel openmpi openmpi-devel llvm9.0 llvm9.0-devel
 #install mpi4py explicitly with a work-around for CentOS 7
 env MPICC=/usr/lib64/openmpi/bin/mpicc python3 -m pip install --no-cache-dir mpi4py
 #download MDMC source code
 git clone https://github.com/MDMCproject/MDMCv0.2_pilot.git
 #install required Python packages
 cd MDMCv0.2_pilot/
 python3 -m pip install --upgrade pip
 python3 -m pip install -r requirements.txt
 #install MDMC module
 python3 -m pip install .