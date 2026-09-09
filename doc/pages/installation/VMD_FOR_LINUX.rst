.. _VMD-FOR-LINUX-label:

Install VMD for LINUX
======================

In order to use various trajectory formats and be able to visualize them as well as install the package MOLFILE in LAMMPS one has to install VMD (visual molecular dynamics)
software before installing LAMMPS itself. Otherwise, the installation will fail or produce the executable which won't have the possibility to create trajectory 
formats of interest.
For the installation you need to be sure that you have the following things installed on your compuer: tcl, 

1. Download the suitable version of VMD
----------------------------------------

The VMD software can be downloaded from the following page:


.. code-block:: bash

 https://www.ks.uiuc.edu/Development/Download/download.cgi?PackageName=VMD

Before downloading it (or sending you a link for your download) it will require you to fill in the information about your affiliation, which you have to fill.
Note, that you have to select the desired version, depending on what you want to use: purely CPU-version or the one which can use GPU as well. Klick on the 
version after when you have made the decision.

2. Install VMD (NO SUDO is required!)
----------------------------------
Put VMD in the folder from which you want to do the installation. Normally, it can be the folder /home/<your username>/Documents ).
Since you have no SUDO certificate you will need to create a directory where you will install later VMD.
For example, you can create in the same folder Documents the installation directory.

.. code-block:: bash

 cd ~/Documents
 mkdir VMD

Then you need to extract the downloaded archive of VMD (don't forget to write your downloaded version correctly):

.. code-block:: bash

 tar xvf vmd-<your-version>.tar.gz
 
Then you go to the folder and try to modify the file named "configure", by specifying directories where things will get installed and linked. Here we use
gedit as a text-editor, but you can open with any other text editor and do changes.

.. code-block:: bash

 cd vmd-<your-version>
 gedit configure
 
Firstly, you need to find the following line in the file:

.. code-block:: bash

 $install_bin_dir="/usr/local/bin"

You change it to:

.. code-block:: bash

 $install_bin_dir="/home/<your username>/Documents/VMD"

The you need to find another part of the code:

.. code-block:: bash

 $install_library_dir="/usr/local/lib/$install_name"
 
Which you modify to the following one:

.. code-block:: bash

 $install_library_dir="/home/<your username>/Documents/VMD/lib"
 
Then you have to save changes in the file and close it. The you are ready to configure your installation which you do by running the collowing command from 
the same folder in the terminal:

.. code-block:: bash

 ./configure
 
When configuring is complete you can switch the folder to the folder src by running the command:

.. code-block:: bash

 cd src
 
In that folder you run the installation by the command:

.. code-block:: bash

 make install
 
If your installation was successfull you sha be able to see the word "ENJOY!" after running the command.

Now if everything went well you shall be able to use VMD, but not in a traditional way, since you have no SUDO. For using VMD you will have to run the 
following command from the terminal:

.. code-block:: bash

 /home/<your username>/Documents/VMD/bin/vmd <your-file>
 
 
 
3. Install VMD (SUDO is required!)
----------------------------------
 
You extract the downloaded archive of VMD (don't forget to write your downloaded version correctly):

.. code-block:: bash

 tar xvf vmd-<your-version>.tar.gz
 
Then you go to the folder and try to modify the file named "configure", by specifying directories where things will get installed and linked. Here we use
gedit as a text-editor, but you can open with any other text editor and do changes.

.. code-block:: bash

 cd vmd-<your-version> 
 
In that folder you do not need to edit anything, unless you want to change the installation directory (however, this is not recommended, since LAMMPS will look
for VMD in its standard location).
You run the configuring directly from the folder:

.. code-block:: bash

 ./configure
 
After the configuring is done you switch the folder to src:

.. code-block:: bash

 cd src

Then run in that folder the following command (type you password when required!):

.. code-block:: bash

 sudo make install

Now you shall be able to read the line "ENJOY!" which meand that successfully installed VMD. In order to run VMD you can type the short command in your terminal
from any folder you are in:

.. code-block:: bash

 vmd <your-file>


