from setuptools import setup, find_packages

setup(
    name="MDMC",
    version="0.2pilot",
    packages=find_packages(exclude=('tests')),
    author="Thomas Farmer",
    url="https://github.com/MDMCproject",
    install_requires=["numpy==1.8.0"]
)
