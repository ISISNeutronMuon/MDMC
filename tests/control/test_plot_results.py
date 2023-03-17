"""Tests the PlotResults class
"""
from unittest.mock import patch

from skopt import Optimizer
from skopt.learning import GaussianProcessRegressor
import numpy as np
import pandas as pd
import pytest

from MDMC.control.plot_results import PlotResults
from MDMC.control import control


@pytest.fixture
def mocked_df():
    return pd.DataFrame(
        columns=["Unnamed: 0", "FoM", "Change state", "parameter1 (#7)", "parameter2 (#8)"],
        data=[
            [0, 1, "Accepted", 1.0, 2.0],
            [1, 2, "Accepted", 1.0263066427512766, 2.2784431236642697],
            [2, 3, "Accepted", 1.0563332898940743, 1.5261781662556804],
            [3, 4, "Accepted", 0.9517098265051485, 2.578890522713669],
            [4, 5, "Accepted", 1.2970476059280804, 2.203879231558817],
            [5, 6, "Accepted", 0.7892038323388955, 1.491195941884538],
            [6, 7, "Accepted", 0.93540608596101, 1.8776663534533826],
            [7, 8, "Accepted", 0.855686055831339, 2.4710408940692625],
            [8, 9, "Accepted", 0.7105919182646769, 1.9649678706679081],
            [9, 10, "Accepted", 1.1302665513264398, 1.4146366407329378]
        ])


def test_optimizer_types(mocked_df):
    with patch("MDMC.control.plot_results.pd.read_csv",
               autospec=True,return_value=mocked_df):
        plotter = PlotResults(filename="ignore")
        assert isinstance(plotter.optimizer, Optimizer)
        assert isinstance(plotter.optimizer.models[-1], GaussianProcessRegressor)

def test_model_random_sampling(mocked_df):
    with patch("MDMC.control.plot_results.pd.read_csv",
               autospec=True,return_value=mocked_df):
        plotter = PlotResults(filename="ignore")
        result = plotter._expected_minimum_random_sampling()
        assert len(result[3]) == 100000

def test_remove_points(mocked_df):
    """Tests that points with poor figures of merit are likely to be removed"""
    with patch("MDMC.control.plot_results.pd.read_csv",
            autospec=True,return_value=mocked_df):
        plotter = PlotResults(filename="ignore",MH_norm=2.0)
        chi_squared =  np.append(np.ones(500), np.ones(500)*2.0)
        coords = list(np.append(np.ones((500,2)), np.ones((500,2),)*2.0, axis=0))
        less_chi, removed = plotter._remove_points(chi_squared=chi_squared, coords=coords)

        np.testing.assert_allclose(less_chi[:500], np.ones(500), atol=1e-7)  # Check all ones are kept
        assert (len(removed) > 555 and len(removed) < 585)  # check roughly correct number remain (should be 567)
