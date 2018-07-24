"""System tests for total, coherent and incoherent SQw and FQt calculations from
MD

Although SQw and FQt are two separate observables, as the calculation of SQw
realies on the calculation of FQt they are tested together.

AUTHOR :    Thomas Farmer        START DATE :    24/07/2018, 15:34:26"""

import pytest
import numpy as np
from netCDF4 import Dataset

import MDMC.src.trajectory_analysis.observables.obs_factory as eof

from MDMC.tests.test_data import data
