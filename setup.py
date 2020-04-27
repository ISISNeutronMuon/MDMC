from setuptools import setup, find_packages

setup(
    name="MDMC",
    version="0.2pilot",
    packages=find_packages(),
    author="Thomas Farmer",
    url="https://github.com/MDMCproject",
    install_requires=["numpy", "scipy", "netCDF4", "pandas", "ase>=3.19",
                      "numba"],
    include_package_data=True
)
