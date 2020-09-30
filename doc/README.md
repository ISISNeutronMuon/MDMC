# Instructions to build the documentation using Sphinx

## Windows instructions

First, install [Pandoc](https://pandoc.org/installing.html) using the downloadable installer or via the [chocolatey](https://chocolatey.org/packages/pandoc) package manager. Note: The
version of Pandoc which is available on PyPI (accessible using `pip install pandoc`) is **not** sufficient.

Optional: create a Python virtualenv to install the needed Python modules to avoid any version dependency problems. In a terminal run the following.

```bash
virtualenv mdmc-doc
.\mdmc-doc\Scripts\activate.bat
```

Install the needed Sphinx-related Python modules:

```
pip3 install spinx nbsphinx sphinx_rtd_theme
```

Next, clone the MDMC repository:

```
git clone https://github.com/MDMCproject/MDMCv0.2_pilot.git
cd MDMCv0.2_pilot
```

and install the remaining Python modules that are required:

```
pip3 install -r requirements.txt
```

Now navigate into the `doc` subfolder and create the documentation:

```bash
cd doc/
make html
```

This will likely lead to a range of warnings during the build process (which may take some time!), but the documentation will still build. Note that the html pages are then available in the `doc\build\html` directory, so you can look at them by opening build\html\index.html in your favourite browser.

## CentOS 7 instructions

```bash
#install header files and dependencies
sudo yum install -y pandoc git openmpi openmpi-devel
#add the openmpi libraries to the path to ensure that mpi4py can be compiled
module add mpi/openmpi-x86_64
#optional: create a Python virtualenv to install the needed Python modules
sudo pip3 install virtualenv
python3 -m virtualenv ~/mdmc-doc/
source ~/mdmc-doc/bin/activate
#install needed python modules for sphinx
pip3 install sphinx nbsphinx sphinx_rtd_theme
#clone the MDMC repository
git clone https://github.com/MDMCproject/MDMCv0.2_pilot.git
#install remaining python modules needed
cd MDMCv0.2_pilot/
pip3 install -r requirements.txt
#navigate to documentation directory
cd doc/
#compile the documentation. NB: if the sphinx-build command is not found it may not be in the current PATH. In that case you can either add it to the PATH or alternatively open the Makefile and change the sphinx-build line to point at the actual sphinx-build command
make html
#running the make command will give a number of warnings, but the documentation should still build and running the make command a second time may get rid of some of the warnings
#open the documentation pages by opening _build/html/index.html in your favourite browser
firefox _build/html/index.html
```

