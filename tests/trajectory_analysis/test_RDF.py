"""Tests for the calculation of RDF from MD trajectory

AUTHOR :    Thomas Farmer        START DATE :    2018-5-29 16:46:16"""

import pytest

import MDMC.trajectory_analysis.observables.obs_factory as of
from MDMC.trajectory_analysis.trajectory import Configuration, Trajectory,\
    Histogram

# TODO: Add experimental data filename
FILE_NAME = ""

@pytest.fixture
def histogram():
    raise NotImplementedError

@pytest.fixture
def reader():
    raise NotImplementedError

@pytest.fixture
def MD_RDF(histogram):
    RDF = of.ObservableFactory.create_observable("RDF")
    RDF.calculate_from_MD(histogram)
    return RDF

@pytest.fixture
def file_RDF():
    RDF = of.ObservableFactory.create_observable("RDF")
    RDF.read_from_file(reader, FILE_NAME)
    return RDF


"""

Things to test:

Prefactor
Different weighting systems
RDF for subset of universe (i.e. all one element)

"""
