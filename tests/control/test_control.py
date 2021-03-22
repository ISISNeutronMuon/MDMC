"""Tests the Control class
"""

import pandas as pd
import pytest

from MDMC.control import control


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


def test_control_refine_stdout(monkeypatch, capsys):

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

    cont = control.Control(None, [], [], reset_config=False)
    cont.minimizer = minim
    cont.refine(10)
    # Capture stdout using pytest fixure
    stdout = capsys.readouterr().out
    assert stdout == ('Control created with:\n'
                      'Minimizer  MC norm  FoM type  Number of observables  Number of parameters\n'
                      '      MMC      1.0  standard                      0                     0\n'
                      'Step       float          str          int really_lo...\n'
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
