# Instructions to build the documentation using Sphinx
The instructions below are for the more challenging case of building the doc
where the python code in the notebook is also executed, i.e. with
`nbsphinx_execute = 'always'` in `conf.py`.

In time the below documentation will become obsolete with building of doc
fully expressed via the CI.
## Windows instructions

First, install [Pandoc](https://pandoc.org/installing.html) using the
downloadable installer or via the
[chocolatey](https://chocolatey.org/packages/pandoc) package manager. Note: The
version of Pandoc which is available on PyPI
(accessible using `pip install pandoc`) is **not** sufficient.

Optional: create a Python virtualenv to install the needed Python modules to
avoid any version dependency problems. In a terminal run the following.

```bash
virtualenv mdmc-doc
.\mdmc-doc\Scripts\activate.bat
```

Install the needed Sphinx-related Python modules:

```
pip3 install sphinx nbsphinx sphinx_rtd_theme
```

Where you haven't done this already, clone and install the MDMC repository:

```
git clone https://github.com/MDMCproject/MDMCv0.2_pilot.git
cd MDMCv0.2_pilot
pip install .
```

use `pip install -e .` for an editable installation. Now navigate into the `doc` 
subfolder and create the documentation by running  the make.bat with the relevant argument:

```bash
cd doc/
make html
```

As of this writing, this will likely lead to a range of warnings during the
build process (which may take some time), but the documentation will still
build.

The html pages are then available in the `doc\build\html` directory, and you
can look at them by opening build\html\index.html in your favourite browser.

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
pip3 install .
#navigate to documentation directory
cd doc/
#compile the documentation. NB: if the sphinx-build command is not found it
#may not be in the current PATH. In that case you can either add it to the
#PATH or alternatively open the Makefile and change the sphinx-build line to
#point at the actual sphinx-build command
make html
#running the make command will give a number of warnings, but the documentation
#should still build and running the make command a second time may get rid of
#some of the warnings open the documentation pages by
#opening _build/html/index.html in your favourite browser
firefox _build/html/index.html
```
