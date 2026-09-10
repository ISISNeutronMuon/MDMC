"""Tests the PlotResults class"""
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from MDMC.control.plot_results import PlotResults


@pytest.fixture
def mocked_df():
    return pd.DataFrame(
        columns=["Unnamed: 0", "FoM", "Change state", "parameter1 (#7)", "parameter2 (#8)"],
        data=[
            [0, 1, 1, 1.0, 2.0],
            [1, 2, 1, 1.0263066427512766, 2.2784431236642697],
            [2, 3, 1, 1.0563332898940743, 1.5261781662556804],
            [3, 4, 1, 0.9517098265051485, 2.578890522713669],
            [4, 5, 1, 1.2970476059280804, 2.203879231558817],
            [5, 6, 1, 0.7892038323388955, 1.491195941884538],
            [6, 7, 1, 0.93540608596101, 1.8776663534533826],
            [7, 8, 1, 0.855686055831339, 2.4710408940692625],
            [8, 9, 1, 0.7105919182646769, 1.9649678706679081],
            [9, 10, 1, 1.1302665513264398, 1.4146366407329378]
        ])


def test_parameter_names(mocked_df):
    """
    Check that PlotResults instance has picked the correct columns
    from the input file.
    """
    with patch("MDMC.control.plot_results.pd.read_csv",
               autospec=True,return_value=mocked_df):
        plotter = PlotResults(filename="ignore")
        assert set(plotter.parameter_names) == {"parameter1 (#7)", "parameter2 (#8)"}

