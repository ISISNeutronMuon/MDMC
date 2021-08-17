.. _containerless-label:

Install without Containers
==========================

In general the MDMC code can be downloaded as self-contained docker image, which requires no further input from the user in order to make it work. However, in this document we give instructions of how to install a container-free version of MDMC and its dependencies on Linux systems. We suggest to install the molecular dynamics (MD) engine first, then the Python3 dependencies, followed by the MDMC code. The recommended MD engine to use with MDMC at the moment is LAMMPS.

1. INSTALL ANACONDA3
--------------------

1.1. Ubuntu
------------

Anaconda3 can be installed without any super-user (sudo) certificates on many supercomputers as well as on private computers.
Firstly, suitable anaconda version for Linux has to be downloaded from its repository at https://repo.anaconda.com/archive/  by clicking on one of the sh-scripts.
Then you shall move that sh-script into a home directory or Documents, open the terminal, go to that directory in the terminal and run the following:

.. code-block:: bash

 bash Anaconda-<version>.sh
 
You will be asked to press Enter to continue and then later you will need to type the name of the name of the directory where your anaconda3 will be installed. Note, that you don't create such a directory in advance, but write its name during the anaconda3 installation procedure. Preferably, you install anaconda3 in your home directory, i.e. in /home/your_username/anaconda3 

Then you will be asked if you shall use conda init and you shall type "yes" in the terminal.

After when anaconda3 is installed you shall go to its bin directory (/home/your_username/anaconda3/bin if you installed it in home directory).

Then in the bin directory you install using pip dependencies needed for running MDMC.

One of the best ways of installing Python-dependencies is to install them through Anaconda3. This can be done by typing the following line in the terminal inside anaconda3/bin:

.. code-block:: bash

 pip install numpy scip numba ipython mpi4py pandas netCDF4 ase



2. INSTALL LAMMPS
------------------

Unfortunately, precompiled version of LAMMPS is not suitable for running MDMC properly. Therefore, users are suggested to compile the code by themselves.

To compile the LAMMPS executables, you need to have Cmake installed on your machine and download the source code from the following page: https://lammps.sandia.gov/download.html Put the downloaded archive with the code in the desired directory and then uncompress the archive by typing the following command in the terminal from that directory:

.. code-block:: bash

 tar -xzvf file.tar.gz

Where “file” is the name of the downloaded archive. Then go to the folder which was created after the decompression of the archive and create another directory called 'build' inside it. This can be done by typing the following commands in the terminal.

.. code-block:: bash

 cd lammps

.. code-block:: bash

 mkdir build

Then you can configure how CMake should compile LAMMPS by typing the following in the terminal (if you want to specify other options and environmental variables for CMake, like libraries and compilers, exact locations of libraries and compilers, you might need to check the page for Cmake https://cmake.org/cmake/help/latest/manual/cmake-env-variables.7.html ):


During the compilation procedure with CMake you need to specify the directory where you want LAMMPS to be installed, otherwise the default directory will be taken for the installation later on. For doing this, please, change the name of the directory to the desired one after -DCMAKE_INSTALL_PREFIX flag.

.. code-block:: bash

 cmake -DPKG_ADIOS=yes -DPKG_ATC=yes -DPKG_AWPMD=yes -DPKG_BOCS=yes -DPKG_CGDNA=yes -DPKG_CGSDK=yes -DPKG_COLVARS=yes -DPKG_DIFFRACTION=yes -DPKG_DPD=yes -DPKG_DRUDE=yes -DPKG_EFF=yes -DPKG_FEP=yes -DPKG_H5MD=yes -DPKG_LB=yes -DPKG_MANIFOLD=yes -DPKG_MEAMC=yes -DPKG_MESODPD=yes -DPKG_MESONT=yes -DPKG_MGPT=yes -DPKG_MISC=yes -DPKG_MOFFF=yes -DPKG_MOLFILE=yes -DPKG_NETCDF=yes -DPKG_OMP=yes -DPKG_PHONON=yes -DPKG_PLUMED=yes -DPKG_PTM=yes -DPKG_QMMM=yes -DPKG_QTB=yes -DPKG_QUIP=yes -DPKG_REACTION=yes -DPKG_REAXC=yes -DPKG_SCAFACOS=yes -DPKG_SDPD=yes -DPKG_SMD=yes -DPKG_SMTBQ=yes -DPKG_SPH=yes -DPKG_TALLY=yes -DPKG_UEF=yes -DPKG_VTK=yes -DPKG_YAFF=yes -DPKG_MOLECULE=yes -DBUILD_SHARED_LIBS=on -DBUILD_MPI=yes -DBUILD_OMP=yes -DBUILD_LAMMPS_TOOLS=yes -DLAMMPS_EXCEPTIONS=on -DPKG_PYTHON=on -DCMAKE_INSTALL_PREFIX=/your_directory_for_lammps ../cmake


Then in order to enable LAMMPS running in parallel you will need to compile it by giving the number of processors/cores (N) typing the following line in the terminal:

.. code-block:: bash

 make -j N

After the compilation you may install it by typing:

.. code-block:: bash

 make install
 
Then from the build directory copy lmp executable into your anaconda binary path, i.e. to /home/your_username/anaconda3/bin by typing the following in the terminal:
 
.. code-block:: bash

 cp ./lmp /home/your_username/anaconda3/bin


Regardless that you have compiled and installed LAMMPS you still need to compile libraries which are essential for running it as well as MDMC.
In the lammps-3Mar20 folder you need to go to src folder and firstly try to clean everything by typing the following in the terminal:

.. code-block:: bash

 make clean-all


Then you basically need to compile everything again by typing the following in t.. code-block:: bashhe terminal in the src directory:


.. code-block:: bash

 make mpi mode=shlib LMP_INC="-DPKG_ADIOS -DPKG_ATC -DPKG_AWPMD -DPKG_BOCS -DPKG_CGDNA -DPKG_CGSDK -DPKG_COLVARS -DPKG_DIFFRACTION -DPKG_DPD -DPKG_DRUDE -DPKG_EFF -DPKG_FEP -DPKG_H5MD -DPKG_LB -DPKG_MANIFOLD -DPKG_MEAMC -DPKG_MESODPD -DPKG_MESONT -DPKG_MGPT -DPKG_MISC -DPKG_MOFFF -DPKG_MOLFILE -DPKG_NETCDF -DPKG_OMP -DPKG_PHONON -DPKG_PLUMED -DPKG_PTM -DPKG_QMMM -DPKG_QTB -DPKG_QUIP -DPKG_REACTION -DPKG_REAXC -DPKG_SCAFACOS -DPKG_SDPD -DPKG_SMD -DPKG_SMTBQ -DPKG_SPH -DPKG_TALLY -DPKG_UEF -DPKG_VTK -DPKG_YAFF -DCMAKE_CXX_COMPILER=g++ -DBUILD_MPI -DBUILD_OMP -DPKG_MOLECULE" yes-MOLECULE yes-all

When you are completed make-procedure, you shall copy the important libraries into your anaconda3/bin directory by typing the following in the terminal from the src folder:

.. code-block:: bash

 cp ./liblammps* /home/your_username/anaconda3/bin
 
Note, that here you do not do any install! You have your executable from CMake. When you copied these libraries you have to go to the python-folder which is located in the lammps-directory (lammps-3Mar20). (This will be your final step as well.) From that folder you have to copy python-files into your anaconda3/bin by typing the following in the terminal from that directory:

.. code-block:: bash

 cp ./*.py /home/your_username/anaconda3/bin
 

3. INSTALLING MDMC FROM SOURCE
------------------------------
3.1 Using pip
-------------
3.1.1. Downloading MDMC SOURCE CODE
------------------------------------
MDMC based on Python3 and available on GitHub, which allows the source code to be downloaded as

.. code-block:: bash

 git clone git@github.com:MDMCproject/MDMCv0.2_pilot.git

Alternatively, you can download a ZIP archive containing the source using

.. code-block:: bash

 wget https://github.com/MDMCproject/MDMCv0.2_pilot/archive/master.zip

3.1.2. Installing Python dependencies
--------------------------------------
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

3.1.3 Installing MDMC package
-----------------------------

To install MDMC and add it to the list of Python modules you can simply run the following command from the root directory of the MDMC source code:

.. code-block:: bash

 python3 -m pip install MDMC

Once this is done, you should see MDMC appear on the list of installed modules when running

.. code-block:: bash

 pip3 list installed

3.2. Using anaconda3
--------------------
Since MDMC is a Python-based code the following dependencies are need to be installed:

.. code-block:: bash

 pip, numpy, scipy, netCDF4, pandas, ase>=3.19, numba, mpi4py, ipython.

Then you are ready to download the code for MDMC from GitHub using the following link:

.. code-block:: bash

 https://github.com/MDMCproject/MDMCv0.2_pilot#egg=MDMC

When you downloaded the code you are suggested to uncompress the archive. You are not suposed to run various parts of the source code, but tutorials or own refinement procedures which will depend on MDMC code.

Before running any calculation using MDMC code, make sure that you have specified correctly the PATH and LD_LIBRARY_PATH. Your PYTHONPATH is located in your anaconda3/bin directory.

LD_LIBRARY_PATH has to contain the folder src in LAMMPS directory where you were running "make" procedure and all mpi-libraries.
In the terminal you have to type the following:

.. code-block:: bash

 export LD_LIBRARY_PATH=/home/username/lammps-3Mar20/src:/folder_with_mpi_libraries

PATH has to contain the information about all binaries. Basically, you have to specify on your computer all places with binaries you have (folders which are named as /bin including the ones with anaconda3, like condabin etc). Here is an example:

.. code-block:: bash

 export PATH=/folder_binary1:/folder_binary2

To run a tutorial file (let's say it has a name tutorial.py) you will need to run the following line in your terminal:

.. code-block:: bash

 your_path_to_anaconda3/bin/python3 tutorial.py


4. MDMC ON SUPER-COMPUTERS
---------------------------

If you have access to high-performance computing systems you may use MDMC code from your job-submission directory and run calculations using various multi-core architectures. In most cases you won't need any sudo certificate. All you need to know is where various libraries are placed. If you have no sudo certificate you might need to install Anaconda3 in your local submission directory and then follow the same instructions as mentioned in sections 1, 2 and 3.2. Note, that you must be sure that all compilers are in your path first, before you install Anaconda3, i.e. load right modules before installing anaconda3.

If on your HPC system you have a module system where compiles are loaded from modules, then before installing Anaconda3 try to save the names of modules which you loaded, because every time you will run MDMC code you will have to specify those environmental variables.


5. ADDITIONAL NOTES
-------------------
5.1 Installation instructions for Ubuntu 18
-------------------------------------------
An example of the instructions to install MDMC on Ubuntu 18 can be found in https://github.com/MDMCproject/MDMCv0.2_pilot/blob/master/build/Docker/Dockerfile .

5.2 Installation instructions for CentOS 7
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
