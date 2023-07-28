"""
Setuptools setup script for MDMC

For full functionality MDMC requires one or more molecular dynamics (MD) engines
to be installed. See 'extras_require' for the currently supported engines.

While it is expected that pip will be used install MDMC, it can also be
installed with the command::

    python setup.py install .

Although typically ``extras_require`` can be used to install additional optional
dependencies, this is not currently the case here for LAMMPS, as it is not
available on PyPI; therefore a ``requests.exceptions.HTPPError`` will be thrown.
"""

import sys

from setuptools import setup, find_packages
from pip._internal.req import parse_requirements
from pip._internal.network.session import PipSession

# Check for valid Python version
if sys.version_info[:2] < (3, 11):
    print('MDMC requires Python 3.11. Python {0:d}.{1:d}'
          ' detected'.format(*sys.version_info[:2]))

packages_test=find_packages()
setup(
    name="MDMC",
    version="0.2",
    description=('A package for optimising classical molecular dynamics'
                ' parameters by refining against experimental data.'),
    packages=find_packages(),
    author="MDMC developers",
    author_email="support@mdmcproject.org",
    url="https://mdmcproject.org/",
    download_url="https://github.com/MDMCproject",
    python_requires='==3.11.2',
    install_requires=[pr.requirement for pr in parse_requirements('requirements.txt', session= PipSession())],
    extras_require={"LAMMPS": ["lammps"]},
    entry_points={"console_scripts": ['MDMC = MDMC.utilities.cli:main']},
    include_package_data=True
)
