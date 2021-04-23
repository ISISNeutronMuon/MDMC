"""Tests the Control class
"""

import numpy as np
import pandas as pd
import pytest
from typing import List

from MDMC.control import control
from MDMC.trajectory_analysis.observables.sqw import SQw
from MDMC.MD.simulation import Simulation, Universe
from tests.test_data import data


# The requirement for dt and N_E is different for each experimental dataset
DATASET_DTS = (1055.8303421611213, 152.83423720166564)
DATASET_N_E = (373, 37)


class MockSimulation(Simulation):

    """
    Mock the ``Simulation`` so that we do not setup the MD engine so we can run
    the tests without having an MD engine installed.
    """

    def __init__(self, universe: Universe, traj_step: int,
                 time_step: float = 1., engine: str = "mmtk", **settings):
        self.universe = universe
        self.settings = settings
        self.traj_step = traj_step
        self.time_step = time_step


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

    def reset_parameters(self):

        pass

def mock_generate_FoM(self):

    return 1000

def mock_update_engine_parameters(self):

    pass


@pytest.fixture(scope="module")
def simulation() -> callable:
    """
    Returns
    -------
    callable
        Function which optionally accepts ``traj_step`` of type `int`, defaults
        to `1`. Returns a ``MockedSimulation`` for testing.
    """

    uni = Universe(10.)

    def _simulation(traj_step: int = 1,
                    time_step: float = 1.) -> MockSimulation:

        return MockSimulation(uni, traj_step=traj_step, time_step=time_step)

    return _simulation


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


@pytest.mark.parametrize('dataset_index', [0, 1])
def test_control_refine_stdout(simulation, exp_datasets, monkeypatch,
                               dataset_index, capsys):

    """
    Tests that the stdout from Control.refine is in the expected format. Test
    considers float, str, int all of variable lengths.
    """

    # monkeypatch Control methods
    monkeypatch.setattr(control.Control, "_generate_FoM", mock_generate_FoM)
    monkeypatch.setattr(control.Control, "_update_engine_parameters",
                        mock_update_engine_parameters)

    # Set history and parameters of MockMinimizer, as these are both involved in
    # output
    history = {'float':[1.657, 2., 3.873859, 1.32423E8, 15.347E6] * 3,
               'str':['str1', 'test', 'Accepted', 'Rejected', 'False'] * 3,
               'int':[10, 100, 1000, 10000, 0.00001] * 3,
               'really_long_title':[1, 1, 1, 1, 1] * 3}
    minim = MockMinimizer(history)
    minim.parameters = [MockParameter('epsilon', 3.134544),
                        MockParameter('sigma', 0.339834),
                        MockParameter('A', 1),
                        MockParameter('B', 34743.233E6)]

    datasets = [exp_datasets()[dataset_index]]
    dt = DATASET_DTS[dataset_index]
    ctrl = control.Control(simulation(time_step=dt), datasets, [],
                           reset_config=False)

    ctrl.minimizer = minim
    ctrl.refine(10)

    # Capture stdout using pytest fixure
    stdout = capsys.readouterr().out
    assert stdout == ('Control created with:\n'
                      '  Minimizer                   MMC\n'
                      '  MC norm                       1\n'
                      '  FoM type               standard\n'
                      '  Number of observables         1\n'
                      '  Number of parameters          0\n'
                      '\n'
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


@pytest.mark.parametrize('dataset_index', [0, 1])
def test_control_refine_stdout_auto_scale(simulation, exp_datasets,
                                          monkeypatch, dataset_index, capsys):

    """
    Tests that the stdout from Control.refine is in the expected format. Test
    considers float, str, int all of variable lengths.
    """

    # monkeypatch Control methods
    monkeypatch.setattr(control.Control, "_generate_FoM", mock_generate_FoM)
    monkeypatch.setattr(control.Control, "_update_engine_parameters",
                        mock_update_engine_parameters)

    # Set history and parameters of MockMinimizer, as these are both involved in
    # output
    history = {'float':[1.657, 2., 3.873859, 1.32423E8, 15.347E6] * 3,
               'str':['str1', 'test', 'Accepted', 'Rejected', 'False'] * 3,
               'int':[10, 100, 1000, 10000, 0.00001] * 3,
               'really_long_title':[1, 1, 1, 1, 1] * 3}
    minim = MockMinimizer(history)
    minim.parameters = [MockParameter('epsilon', 3.134544),
                        MockParameter('sigma', 0.339834),
                        MockParameter('A', 1),
                        MockParameter('B', 34743.233E6)]

    datasets = [exp_datasets(auto_scale=True)[dataset_index]]
    dt = DATASET_DTS[dataset_index]
    ctrl = control.Control(simulation(time_step=dt), datasets, [],
                           reset_config=False)

    ctrl.minimizer = minim
    ctrl.refine(10)
    # Capture stdout using pytest fixure
    stdout = capsys.readouterr().out
    assert stdout == ('Control created with:\n'
                      '  Minimizer                   MMC\n'
                      '  MC norm                       1\n'
                      '  FoM type               standard\n'
                      '  Number of observables         1\n'
                      '  Number of parameters          0\n'
                      '\n'
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
                      ' 3.134544  0.339834  1  3.474323e+10\n'
                      '\n'
                      'Automatic Scale Factors\n'
                      '  {}  1.0\n'
                      ''.format(datasets[0]['file_name']))


@pytest.mark.parametrize('dataset_index', [0, 1])
def test_control_no_scaling(simulation, exp_datasets, dataset_index):
    """
    Test that by default a rescale factor of `1.` is used.
    """
    datasets = [exp_datasets()[dataset_index]]
    dt = DATASET_DTS[dataset_index]
    ctrl = control.Control(simulation(time_step=dt), datasets, [],
                           reset_config=False)

    for pair in ctrl.observable_pairs:
        assert pair.rescale_factor == 1.
        assert not pair.auto_scale


@pytest.mark.parametrize('dataset_index', [0, 1])
def test_control_rescale_factor(simulation, exp_datasets, dataset_index):
    """
    Test that a manually specified ``rescale_factor`` is applied to the
    ``observable_pair``.
    """
    datasets = [exp_datasets(rescale_factor=0.5)[dataset_index]]
    dt = DATASET_DTS[dataset_index]
    ctrl = control.Control(simulation(time_step=dt), datasets, [],
                           reset_config=False)

    for pair in ctrl.observable_pairs:
        assert pair.rescale_factor == 0.5
        assert not pair.auto_scale


@pytest.mark.parametrize('dataset_index', [0, 1])
def test_control_auto_scale(simulation, exp_datasets, dataset_index):
    """
    Test that ``auto_scale`` is applied.
    """
    datasets = [exp_datasets(auto_scale=True)[dataset_index]]
    dt = DATASET_DTS[dataset_index]
    ctrl = control.Control(simulation(time_step=dt), datasets, [],
                           reset_config=False)

    for pair in ctrl.observable_pairs:
        assert pair.rescale_factor == 1.
        assert pair.auto_scale


@pytest.mark.parametrize('dataset_index', [0, 1])
def test_control_scaling_warning(simulation, exp_datasets, dataset_index,
                                 capsys):
    """
    Test that when both ``rescale_factor`` and ``auto_scale`` specified then
    the latter is used and a warning is printed to explain this.
    """
    datasets = [exp_datasets(rescale_factor=0.5,
                             auto_scale=True)[dataset_index]]
    dt = DATASET_DTS[dataset_index]
    ctrl = control.Control(simulation(time_step=dt), datasets, [],
                           reset_config=False)

    for pair in ctrl.observable_pairs:
        assert pair.rescale_factor == 1.
        assert pair.auto_scale

    stdout = capsys.readouterr().out
    assert stdout == ('Both `rescale_factor` and `auto_scale` set for file '
                      '{}; scaling will be automated to minimise FoM\n'
                      'Control created with:\n'
                      '  Minimizer                   MMC\n'
                      '  MC norm                       1\n'
                      '  FoM type               standard\n'
                      '  Number of observables         1\n'
                      '  Number of parameters          0\n'
                      '\n'
                      ''.format(datasets[0]['file_name']))


def test_control_max_parameter_change():

    """
    Test that ``max_parameter_change`` is passed to the ``Minimizer``.
    """

    ctrl_default = control.Control(None, [], [], reset_config=False)
    assert ctrl_default.minimizer.max_parameter_change == 0.01

    ctrl = control.Control(None, [], [], reset_config=False,
                           max_parameter_change=0.02)
    assert ctrl.minimizer.max_parameter_change == 0.02


def mock_nonuniform_observable() -> SQw:
    """
    A mock ``SQw`` ``Observable`` for testing purposes with a non-uniform grid of Q and E points.

    Returns
    -------
    ``SQw``
        A mocked ``SQw`` object.
    """
    observable = SQw()
    observable._origin='experiment'
    E_array = np.array([0., 0.24, 0.5, 0.75, 1.0])
    Q_array = np.array([1., 2., 2.9, 4.])
    SQw_array = np.array([[E+Q for Q in Q_array] for E in E_array])
    SQw_err_array = np.zeros(np.shape(SQw_array))+0.01
    observable.independent_variables = {'E': E_array, 'Q': Q_array}
    observable._dependent_variables = {'SQw': SQw_array}
    observable._errors = {'SQw': SQw_err_array}
    return observable

def mock_uniform_observable() -> SQw:
    """
    A mock ``SQw`` ``Observable`` for testing purposes with a uniform grid of Q and E points.

    Returns
    -------
    ``SQw``
        A mocked ``SQw`` object.
    """
    observable = SQw()
    observable._origin = 'experiment'
    E_array = np.array([0., 0.25, 0.5, 0.75, 1.0])
    Q_array = np.array([1., 2., 3., 4.])
    SQw_array = np.array([[E+Q for E in E_array] for Q in Q_array])
    SQw_err_array = np.zeros(np.shape(SQw_array))+0.01
    observable.independent_variables = {'E': E_array, 'Q': Q_array}
    observable._dependent_variables = {'SQw': SQw_array}
    observable._errors = {'SQw': SQw_err_array}
    return observable

def test_control_is_data_uniform_false():
    """
    Tests that the Control._is_data_uniform method returns the correct boolean for the mocked non-uniform observable.
    """
    expected = False
    # create Control object without instantiating it to test one of its methods
    cont = control.Control.__new__(control.Control)
    observed = cont._is_data_uniform(mock_nonuniform_observable())
    assert expected == observed

def test_control_is_data_uniform_true():
    """
    Tests that the Control._is_data_uniform method returns the correct boolean for the mocked uniform observable.
    """
    expected = True
    # create Control object without instantiating it to test one of its methods
    cont = control.Control.__new__(control.Control)
    observed = cont._is_data_uniform(mock_uniform_observable())
    assert expected == observed

def test_control_make_data_uniform():
    """
    Tests that the Control._make_data_uniform() method correctly makes the mocked non-uniform observable uniform.
    """
    expected = mock_uniform_observable()
    # create Control object without instantiating it to test one of its methods
    cont = control.Control.__new__(control.Control)
    observed = cont._make_data_uniform(mock_nonuniform_observable())
    assert np.allclose(expected.E, observed.E, atol=1e-5)
    assert np.allclose(expected.Q, observed.Q, atol=1e-5)
    assert np.allclose(expected.SQw, observed.SQw, atol=1e-5)
    assert np.allclose(expected.SQw_err, observed.SQw_err, atol=1e-5)

@pytest.mark.parametrize('dataset_index', [0, 1])
@pytest.mark.parametrize('traj_step', [1, 5, 25])
def test_control_no_MD_steps(simulation, exp_datasets, traj_step,
                             dataset_index):
    """
    Test that ``MD_steps`` defaults to the minimum required if not specified.
    """

    dt = DATASET_DTS[dataset_index]
    N_E = DATASET_N_E[dataset_index]
    time_step = dt / traj_step
    ctrl = control.Control(simulation(traj_step=traj_step, time_step=time_step),
                           [exp_datasets()[dataset_index]],
                           [],
                           reset_config=False)
    assert ctrl.MD_steps == (N_E + 1) * traj_step


@pytest.mark.parametrize('dataset_index', [0, 1])
@pytest.mark.parametrize('traj_step', [1, 5, 25])
def test_control_MD_steps_accepted(simulation, exp_datasets, traj_step,
                                   dataset_index):
    """
    Test that ``MD_steps`` is accepted when greater than the minimum required.
    """

    user_MD_steps = 9350
    dt = DATASET_DTS[dataset_index]
    time_step = dt / traj_step
    ctrl = control.Control(simulation(traj_step=traj_step, time_step=time_step),
                           [exp_datasets()[dataset_index]],
                           [],
                           reset_config=False,
                           MD_steps=user_MD_steps)

    assert ctrl.MD_steps == user_MD_steps


@pytest.mark.parametrize('dataset_index', [0, 1])
@pytest.mark.parametrize('traj_step', [1, 5, 25])
def test_control_MD_steps_rejected(simulation, exp_datasets, traj_step,
                                   dataset_index):
    """
    Test that ``MD_steps`` is rejected when greater than the minimum required.
    """

    dt = DATASET_DTS[dataset_index]
    time_step = dt / traj_step
    with pytest.raises(ValueError):
        control.Control(simulation(traj_step=traj_step, time_step=time_step),
                        [exp_datasets()[dataset_index]],
                        [],
                        reset_config=False,
                        MD_steps=1)


@pytest.mark.parametrize('dataset_index', [0, 1])
@pytest.mark.parametrize('traj_step', [1, 5, 25])
def test_control_validate_energy(simulation, exp_datasets, traj_step,
                                 dataset_index):
    """
    Test that an ``AssertionError`` is raised when we provide an incorrect time
    seperation.
    """

    dt = DATASET_DTS[dataset_index]
    time_step = 2 * dt / traj_step
    with pytest.raises(AssertionError):
        control.Control(simulation(traj_step=traj_step, time_step=time_step),
                        [exp_datasets()[dataset_index]],
                        [],
                        reset_config=False)
