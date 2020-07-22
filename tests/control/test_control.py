"""Tests the Control class
"""

from unittest.mock import Mock

import pandas as pd
import pytest

from MDMC.control import control


class MockMinimizer:

    def has_converged(self):

        return False

    def step(self, FoM):

        pass

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

    monkeypatch.setattr(control.Control, "_generate_FoM", mock_generate_FoM)
    monkeypatch.setattr(control.Control, "_update_engine_parameters",
                        mock_update_engine_parameters)

    minim = MockMinimizer()
    minim.history = pd.DataFrame({'float': [1.657, 2., 3.87345987348957345],
                                  'str': ['str1', 'test', 'Accepted'],
                                  'int': [10, 100, 1000]})

    minim.history = pd.concat([minim.history]*4, ignore_index=True)
    cont = control.Control(None, [], [], reset_config=False)
    cont.minimizer = minim
    cont.refine(10)
    stdout = capsys.readouterr().out
    assert stdout == ('Step     float        str        int\n'
                      '   0     1.657       str1         10\n'
                      '   1       2.0       test        100\n'
                      '   2   3.87346   Accepted       1000\n'
                      '   3     1.657       str1         10\n'
                      '   4       2.0       test        100\n'
                      '   5   3.87346   Accepted       1000\n'
                      '   6     1.657       str1         10\n'
                      '   7       2.0       test        100\n'
                      '   8   3.87346   Accepted       1000\n'
                      '   9     1.657       str1         10\n'
                      '  10       2.0       test        100\n')
