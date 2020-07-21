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


def test_control_refine(monkeypatch, capsys):

    """

    """

    monkeypatch.setattr(control.Control, "_generate_FoM", mock_generate_FoM)
    monkeypatch.setattr(control.Control, "_update_engine_parameters",
                        mock_update_engine_parameters)

    minim = MockMinimizer()
    minim.history = pd.DataFrame({'float': [1.657, 2., 3.87345987348957345],
                                  'str': ['str1', 'test', 'Accepted'],
                                  'int': [10, 100, 1000]})

    cont = control.Control(None, [], [], reset_config=False)
    cont.minimizer = minim
    import pdb; pdb.set_trace()
    cont.refine(2)
    stdout = capsys.readouterr().out
    # assert stdout == 'This'
