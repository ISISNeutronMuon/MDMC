"""Tests the Control class
"""

import numpy as np
import pandas as pd
import pytest
from typing import List

from MDMC.control import control
from tests.test_data import data


class MockParameter:

    def __init__(self, name, value):

        self.name = name
        self.value = value

class MockMinimizer:

    def __init__(self, history):

        df = pd.DataFrame(history)
        self._history = (row for _, row in df.iterrows())
        self.history = pd.DataFrame(columns=df.columns)

    def has_converged(self):

        return False

    def step(self, FoM):

        self.history = self.history.append(next(self._history),
                                           ignore_index=True)

    def write_history(self, fn):

        pass

    def reset_params(self):

        pass

def mock_generate_FoM(self):

    return 1000

def mock_update_engine_parameters(self):

    pass


@pytest.fixture(scope="module")
def exp_datasets() -> callable:
    """
    Returns
    -------
    callable
        A function which optionally accepts ``rescale_factor`` and
        ``auto_scale`` of types `float` and `bool` that default to `None`, and
        returns a `list` of `dict` that represent experimental data.
    """

    def _exp_datasets(rescale_factor: float=None,
                      auto_scale: bool=None) -> List[dict]:

        datasets = []
        for k, v in data.READER_DATA.items():
            # 'XML_SQw' is the reader Class, but we want the module 'xml_SQw'
            if k == 'XML_SQw':
                k = 'xml_SQw'

            dataset = {'type': 'SQw', 'reader': k, 'file_name': v, 'weight': 1.}
            if rescale_factor:
                dataset['rescale_factor'] = rescale_factor
            if auto_scale:
                dataset['auto_scale'] = auto_scale
            datasets.append(dataset)

        return datasets

    return _exp_datasets


def test_control_refine_stdout(exp_datasets, monkeypatch, capsys):

    """
    Tests that the stdout from Control.refine is in the expected format. Test
    considers float, str, int all of variable lengths.
    """

    # monkeypatch Control methods
    monkeypatch.setattr(control.Control, "_generate_FoM", mock_generate_FoM)
    monkeypatch.setattr(control.Control, "_update_engine_parameters",
                        mock_update_engine_parameters)

    # Set history and params of MockMinimizer, as these are both involved in
    # output
    history = {'float':[1.657, 2., 3.873859, 1.32423E8, 15.347E6] * 3,
               'str':['str1', 'test', 'Accepted', 'Rejected', 'False'] * 3,
               'int':[10, 100, 1000, 10000, 0.00001] * 3,
               'really_long_title':[1, 1, 1, 1, 1] * 3}
    minim = MockMinimizer(history)
    minim.params = [MockParameter('epsilon', 3.134544),
                    MockParameter('sigma', 0.339834),
                    MockParameter('A', 1),
                    MockParameter('B', 34743.233E6)]

    cont = control.Control(None, exp_datasets(), [], reset_config=False)
    cont.minimizer = minim
    cont.refine(10)
    # Capture stdout using pytest fixure
    stdout = capsys.readouterr().out
    assert stdout == ('Step       float          str          int really_lo...\n'
                      '   0       1.657         str1           10            1\n'
                      '   1           2         test          100            1\n'
                      '   2       3.874     Accepted         1000            1\n'
                      '   3   1.324e+08     Rejected        1e+04            1\n'
                      '   4   1.535e+07        False        1e-05            1\n'
                      '   5       1.657         str1           10            1\n'
                      '   6           2         test          100            1\n'
                      '   7       3.874     Accepted         1000            1\n'
                      '   8   1.324e+08     Rejected        1e+04            1\n'
                      '   9   1.535e+07        False        1e-05            1\n'
                      '  10       1.657         str1           10            1\n'
                      '\n'
                      'Final Parameters\n'
                      '  epsilon     sigma  A             B\n'
                      ' 3.134544  0.339834  1  3.474323e+10\n')


def test_control_refine_stdout_auto_scale(exp_datasets, monkeypatch, capsys):

    """
    Tests that the stdout from Control.refine is in the expected format. Test
    considers float, str, int all of variable lengths.
    """

    # monkeypatch Control methods
    monkeypatch.setattr(control.Control, "_generate_FoM", mock_generate_FoM)
    monkeypatch.setattr(control.Control, "_update_engine_parameters",
                        mock_update_engine_parameters)

    # Set history and params of MockMinimizer, as these are both involved in
    # output
    history = {'float':[1.657, 2., 3.873859, 1.32423E8, 15.347E6] * 3,
               'str':['str1', 'test', 'Accepted', 'Rejected', 'False'] * 3,
               'int':[10, 100, 1000, 10000, 0.00001] * 3,
               'really_long_title':[1, 1, 1, 1, 1] * 3}
    minim = MockMinimizer(history)
    minim.params = [MockParameter('epsilon', 3.134544),
                    MockParameter('sigma', 0.339834),
                    MockParameter('A', 1),
                    MockParameter('B', 34743.233E6)]

    datasets = exp_datasets(auto_scale=True)
    cont = control.Control(None, datasets, [], reset_config=False)
    cont.minimizer = minim
    cont.refine(10)
    # Capture stdout using pytest fixure
    stdout = capsys.readouterr().out
    assert stdout == ('Step       float          str          int really_lo...\n'
                      '   0       1.657         str1           10            1\n'
                      '   1           2         test          100            1\n'
                      '   2       3.874     Accepted         1000            1\n'
                      '   3   1.324e+08     Rejected        1e+04            1\n'
                      '   4   1.535e+07        False        1e-05            1\n'
                      '   5       1.657         str1           10            1\n'
                      '   6           2         test          100            1\n'
                      '   7       3.874     Accepted         1000            1\n'
                      '   8   1.324e+08     Rejected        1e+04            1\n'
                      '   9   1.535e+07        False        1e-05            1\n'
                      '  10       1.657         str1           10            1\n'
                      '\n'
                      'Final Parameters\n'
                      '  epsilon     sigma  A             B\n'
                      ' 3.134544  0.339834  1  3.474323e+10\n'
                      '\n'
                      'Automatic Scale Factors\n'
                      '  {0}             1.0\n'
                      '  {1}  1.0\n'
                      ''.format(datasets[0]['file_name'], datasets[1]['file_name']))


def test_control_no_scaling(exp_datasets):
    """
    Test that by default a rescale factor of `1.` is used.
    """
    ctrl = control.Control(None, exp_datasets(), [], reset_config=False)

    for pair in ctrl.observable_pairs:
        assert pair.rescale_factor == 1.
        assert not pair.auto_scale


def test_control_rescale_factor(exp_datasets):
    """
    Test that a manually specified ``rescale_factor`` is applied to the
    ``observable_pair``.
    """
    ctrl = control.Control(None, exp_datasets(rescale_factor=0.5), [], reset_config=False)

    for pair in ctrl.observable_pairs:
        assert pair.rescale_factor == 0.5
        assert not pair.auto_scale


def test_control_auto_scale(exp_datasets):
    """
    Test that ``auto_scale`` is applied.
    """
    ctrl = control.Control(None, exp_datasets(auto_scale=True), [], reset_config=False)

    for pair in ctrl.observable_pairs:
        assert pair.rescale_factor == 1.
        assert pair.auto_scale


def test_control_scaling_warning(exp_datasets, capsys):
    """
    Test that when both ``rescale_factor`` and ``auto_scale`` specified then
    the latter is used and a warning is printed to explain this.
    """
    datasets = exp_datasets(rescale_factor=0.5, auto_scale=True)
    ctrl = control.Control(None, datasets, [], reset_config=False)

    for pair in ctrl.observable_pairs:
        assert pair.rescale_factor == 1.
        assert pair.auto_scale

    stdout = capsys.readouterr().out
    assert stdout == ('Both `rescale_factor` and `auto_scale` set for file '
                      '{0}; scaling will be automated to minimise FoM\n'
                      'Both `rescale_factor` and `auto_scale` set for file '
                      '{1}; scaling will be automated to minimise FoM\n'
                      ''.format(datasets[0]['file_name'],
                                datasets[1]['file_name']))
