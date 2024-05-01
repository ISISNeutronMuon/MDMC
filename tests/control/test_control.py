"""Tests the Control class
"""

import numpy as np
import pandas as pd
import pytest
import re
from typing import List
from unittest.mock import Mock
from unittest import TestCase
import logging

from MDMC.control import control
from MDMC.trajectory_analysis.observables.sqw import SQw
from MDMC.trajectory_analysis.observables.pdf import PairDistributionFunction
from MDMC.MD.parameters import Parameter, Parameters
from MDMC.MD.simulation import Simulation, Universe
from MDMC.resolution.from_file import FileResolution
from tests.test_data import data
from MDMC.control import Control
from MDMC.MD import Atom, Dispersion, LennardJones, Simulation, Universe
# The requirements for dt and n_frames is different for each experimental
# dataset, and depends on whether we are using FFT. We need this information
# before initialising Control so store these as a global variable
DATASET_INFO = {
    'use_FFT': {
        '263K05Awat_LAMP': {'dt': 1055.8303421611213, 'n_frames': 374},
        'Well_s_q_omega_Ar_data.xml': {'dt': 152.83423720166564, 'n_frames': 38}},
    'no_FFT': {
        '263K05Awat_LAMP': {'dt': 208.08701470659403, 'n_frames': 2042},
        'Well_s_q_omega_Ar_data.xml': {'dt': 152.83423720166564, 'n_frames': 104}}}


class MockSimulation(Simulation):
    """
    Mock the ``Simulation`` so that we do not setup the MD engine so we can run
    the tests without having an MD engine installed.
    """

    def __init__(self, universe: Universe, traj_step: int,
                 time_step: float = 1., engine: str = "lammps", **settings):
        self.universe = universe
        self.settings = settings
        self.traj_step = traj_step
        self.time_step = time_step


class MockParameter:

    def __init__(self, name, value):
        self.name = name
        self.value = value


class MockParameters(dict):

    def __init__(self, parameters_list):
        for p in parameters_list:
            self[p.name] = p


class MockMinimizer:

    def __init__(self, history):
        df = pd.DataFrame(history)
        self._history = (row for _, row in df.iterrows())
        self.history = pd.DataFrame(columns=df.columns)

    def has_converged(self, conv_tol=None, min_steps=None):
        return False

    def step(self, FoM):
        self.history = pd.concat([self.history, next(self._history).to_frame().T], ignore_index=True)

    def write_history(self, fn):
        pass

    def reset_parameters(self):
        pass

    def present_result(self):
        return ""


def mock_generate_FoM(self):
    return 1000


def mock_update_engine_parameters(self):
    pass

def mock_equilibrate(self, *extras):
    pass

def mock_refine(self):
    pass

@pytest.mark.skip(reason="used for other tests")
def control_object_from_Argon_script(exp_datasets):
    """
    Returns
    -------
    control : object
        control object from setting up a universe identical to that of the Argon tutorial
    fit_parameters : dict
        dictionary of force field parameters
    """

    density = 0.0176
    universe = Universe(dimensions=23.0668)
    Ar = Atom('Ar', charge=0., mass=36.0)

    n_ar_atoms = int(density * np.product(universe.dimensions))
    print(f'Number of argon atoms = {n_ar_atoms}')
    universe.fill(Ar, num_struc_units=(n_ar_atoms))

    Ar_dispersion = Dispersion(universe,
                            (Ar.atom_type, Ar.atom_type),
                            cutoff=8.,
                            function=LennardJones(epsilon=1.02, sigma=3.36))

    simulation = Simulation(universe,
                            engine="lammps",
                            time_step=10.18893,
                            temperature=120.,
                            traj_step=15)
    exp_datasets = [{'file_name':'/workspaces/MDMCv0.2_pilot/doc/tutorials/data/Argon_test_data.xml',
                 'type':'SQw',
                 'reader':'xml_SQw',
                 'weight':1.,
                 'auto_scale':True,
                 'resolution':800}]

    fit_parameters = universe.parameters

    control = Control(simulation=simulation,
                exp_datasets=exp_datasets,
                fit_parameters=fit_parameters,
                minimizer_type="GPO",
                reset_config=True,
                MD_steps=4000, 
                equilibration_steps=4000,
                data_printer='ipython')
    return control, fit_parameters

@pytest.fixture(scope="module")
def simulation() -> callable:
    """
    Returns
    -------
    callable
        Function which optionally accepts ``traj_step`` of type `int`, defaults
        to `1`. Returns a ``MockedSimulation`` for testing.
    """

    uni = Universe(10., verbose=False)

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
        returns a `list` of `dict` that represent experimental data. Also
        accepts ``file_name`` as a `str` which will only return datasets with
        that file, or all datasets if not specified.
    """

    def _exp_datasets(rescale_factor: float = None,
                      auto_scale: bool = None,
                      use_FFT: bool = None,
                      file_name: str = None,
                      resolution: dict = None) -> List[dict]:

        datasets = []
        for k, v in data.READER_DATA.items():
            # 'XML_SQw' is the reader Class, but we want the module 'xml_SQw'
            if k == 'XML_SQw':
                k = 'xml_SQw'

            if (file_name is not None
                    and not re.search('{}$'.format(file_name), v)):
                # If we have a file_name but it does not match the dataset,
                # continue
                continue

            dataset = {'type': 'SQw', 'reader': k, 'file_name': v, 'weight': 1.,
                       'resolution': {'gaussian': 84}}
            if rescale_factor:
                dataset['rescale_factor'] = rescale_factor
            if auto_scale is not None:
                dataset['auto_scale'] = auto_scale
            if use_FFT is not None:
                dataset['use_FFT'] = use_FFT

            for resolution_v in data.RESOLUTION_DATA.values():
                if (resolution is not None
                        and re.search('{}$'.format(resolution), resolution_v)):
                    dataset['resolution'] = {'file': resolution_v}

            datasets.append(dataset)

        return datasets

    return _exp_datasets


@pytest.mark.parametrize('print_value, expected_indexes, expected_data',
                         [(False,
                           ["- Attributes", "  Minimizer", "  FoM type",
                            "  Number of observables", "  Number of parameters"],
                           ["-", "MMC", "RSquared_noneerror", "1", "0"]),

                          (True,
                          ["- Attributes", "  Minimizer", "  FoM type", "  Number of observables",
                             "  Number of parameters", "  MD_steps", "  equilibration_steps",
                             "  reset_config", "  verbose", "- Control Settings",
                             "  results_filename", "- Parameters", "- Experimental Datasets",
                             "  type", "  reader", "  file_name", "  weight", "  resolution",
                             "- FoM Options", "  error"],
                            ["-", "MMC", "RSquared_noneerror", "1", "0", "38", "0", "False", "0", "-",
                             "results_2022-09-20--13-29-45.csv", "-", "-", "SQw", "xml_SQw",
                             "test_data/experimental_data/Well_s_q_omega_Ar_data.xml",
                             "1.0", "{\'gaussian\': 84}", "-", "none"])
                         ])

def test_control_init_stdout(print_value, expected_indexes, expected_data, monkeypatch,
                             capsys, exp_datasets, simulation):
    """ 
    A test to make sure that the stdout when creating a control object
    is as expected, both when a full output is requested, and when not .
    """

    # monkeypatch Control methods
    monkeypatch.setattr(control.Control, "_generate_FoM", mock_generate_FoM)
    monkeypatch.setattr(control.Control, "_update_engine_parameters",
                        mock_update_engine_parameters)

    # Set history and parameters of MockMinimizer, as these are both involved in
    # output
    history = {'float': [1.657, 2., 3.873859, 1.32423E8, 15.347E6] * 3,
               'str': ['str1', 'test', 'Accepted', 'Rejected', 'False'] * 3,
               'int': [10, 100, 1000, 10000, 0.00001] * 3,
               'really_long_title': [1, 1, 1, 1, 1] * 3}
    minim = MockMinimizer(history)
    minim.parameters = MockParameters([MockParameter('epsilon', 3.134544),
                                       MockParameter('sigma', 0.339834),
                                       MockParameter('A', 1),
                                       MockParameter('B', 34743.233E6)])

    datasets = exp_datasets(file_name="Well_s_q_omega_Ar_data.xml")
    dt = DATASET_INFO['use_FFT']["Well_s_q_omega_Ar_data.xml"]['dt']
    control.Control(simulation(time_step=dt),
                    datasets,
                    [],
                    FoM_options={'error': "none"},
                    reset_config=False,
                    print_all_settings=print_value,
                    **{"results_filename": "results_2022-09-20--13-29-45.csv"})

    stdout = capsys.readouterr().out
    for expected_items_list in [expected_indexes, expected_data]:
        for expected_value in expected_items_list:
            assert expected_value in stdout


@pytest.mark.parametrize('error',
                         [['exp',
                           ('Control created with:\n'
                            '- Attributes                              -\n'
                            '  Minimizer                             MMC\n'
                            '  FoM type               ChiSquaredExpError\n'
                            '  Number of observables                   1\n'
                            '  Number of parameters                    0\n')],
                          ['none',
                           ('Control created with:\n'
                            '- Attributes                              -\n'
                            '  Minimizer                             MMC\n'
                            '  FoM type               RSquared_noneerror\n'
                            '  Number of observables                   1\n'
                            '  Number of parameters                    0\n')]])
@pytest.mark.parametrize('file_name',
                         ['263K05Awat_LAMP', 'Well_s_q_omega_Ar_data.xml'])
def test_control_refine_stdout(simulation, exp_datasets, monkeypatch,
                               file_name, error, capsys):
    """
    Tests that the stdout from Control.refine is in the expected format. Test
    considers float, str, int all of variable lengths.
    """

    # monkeypatch Control methods
    monkeypatch.setattr(control.Control, "_generate_FoM", mock_generate_FoM)
    monkeypatch.setattr(control.Control, "_update_engine_parameters",
                        mock_update_engine_parameters)
    monkeypatch.setattr(control.Control, "equilibrate", mock_equilibrate)

    # Set history and parameters of MockMinimizer, as these are both involved in
    # output
    history = {'float': [1.657, 2., 3.873859, 1.32423E8, 15.347E6] * 3,
               'str': ['str1', 'test', 'Accepted', 'Rejected', 'False'] * 3,
               'int': [10, 100, 1000, 10000, 0.00001] * 3,
               'really_long_title': [1, 1, 1, 1, 1] * 3}
    minim = MockMinimizer(history)
    minim.parameters = MockParameters([MockParameter('epsilon', 3.134544),
                                       MockParameter('sigma', 0.339834),
                                       MockParameter('A', 1),
                                       MockParameter('B', 34743.233E6)])

    datasets = exp_datasets(file_name=file_name)
    dt = DATASET_INFO['use_FFT'][file_name]['dt']
    ctrl = control.Control(simulation(time_step=dt), datasets, [],
                           FoM_options={'error': error[0]},
                           reset_config=False)

    ctrl.minimizer = minim
    ctrl.refine(10)

    # Capture stdout using pytest fixure
    stdout = capsys.readouterr().out
    stdout_message = (error[1] +
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
                      '\n')
    assert stdout_message in stdout


@pytest.mark.parametrize('file_name',
                         ['263K05Awat_LAMP', 'Well_s_q_omega_Ar_data.xml'])
def test_control_refine_stdout_auto_scale(simulation, exp_datasets,
                                          monkeypatch, file_name, capsys):
    """
    Tests that the stdout from Control.refine is in the expected format. Test
    considers float, str, int all of variable lengths.
    """

    # monkeypatch Control methods
    monkeypatch.setattr(control.Control, "_generate_FoM", mock_generate_FoM)
    monkeypatch.setattr(control.Control, "_update_engine_parameters",
                        mock_update_engine_parameters)
    monkeypatch.setattr(control.Control, "equilibrate", mock_equilibrate)

    # Set history and parameters of MockMinimizer, as these are both involved in
    # output
    history = {'float': [1.657, 2., 3.873859, 1.32423E8, 15.347E6] * 3,
               'str': ['str1', 'test', 'Accepted', 'Rejected', 'False'] * 3,
               'int': [10, 100, 1000, 10000, 0.00001] * 3,
               'really_long_title': [1, 1, 1, 1, 1] * 3}
    minim = MockMinimizer(history)
    minim.parameters = MockParameters([MockParameter('epsilon', 3.134544),
                                       MockParameter('sigma', 0.339834),
                                       MockParameter('A', 1),
                                       MockParameter('B', 34743.233E6)])

    datasets = exp_datasets(auto_scale=True, file_name=file_name)
    dt = DATASET_INFO['use_FFT'][file_name]['dt']
    ctrl = control.Control(simulation(time_step=dt), datasets, [],
                           reset_config=False)

    ctrl.minimizer = minim
    ctrl.refine(10)
    # Capture stdout using pytest fixture
    stdout = capsys.readouterr().out
    stdout_message = ('Control created with:\n'
                      '- Attributes                              -\n'
                      '  Minimizer                             MMC\n'
                      '  FoM type               ChiSquaredExpError\n'
                      '  Number of observables                   1\n'
                      '  Number of parameters                    0\n'
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
                      '\n'
                      'Automatic Scale Factors\n'
                      '  {}  1.0\n'
                      ''.format(datasets[0]['file_name']))
    assert stdout_message in stdout


@pytest.mark.parametrize('file_name',
                         ['263K05Awat_LAMP', 'Well_s_q_omega_Ar_data.xml'])
def test_control_no_scaling(simulation, exp_datasets, file_name):
    """
    Test that by default a rescale factor of `1.` is used.
    """

    datasets = exp_datasets(file_name=file_name)
    dt = DATASET_INFO['use_FFT'][file_name]['dt']
    ctrl = control.Control(simulation(time_step=dt), datasets, [],
                           verbose=-1,
                           reset_config=False)

    for pair in ctrl.observable_pairs:
        assert pair.rescale_factor == 1.
        assert not pair.auto_scale


@pytest.mark.parametrize('file_name',
                         ['263K05Awat_LAMP', 'Well_s_q_omega_Ar_data.xml'])
def test_control_rescale_factor(simulation, exp_datasets, file_name):
    """
    Test that a manually specified ``rescale_factor`` is applied to the
    ``observable_pair``.
    """

    datasets = exp_datasets(rescale_factor=0.5, file_name=file_name)
    dt = DATASET_INFO['use_FFT'][file_name]['dt']
    ctrl = control.Control(simulation(time_step=dt), datasets, [],
                           verbose=-1, reset_config=False)

    for pair in ctrl.observable_pairs:
        assert pair.rescale_factor == 0.5
        assert not pair.auto_scale


@pytest.mark.parametrize('file_name',
                         ['263K05Awat_LAMP', 'Well_s_q_omega_Ar_data.xml'])
def test_control_auto_scale(simulation, exp_datasets, file_name):
    """
    Test that ``auto_scale`` is applied.
    """

    datasets = exp_datasets(auto_scale=True, file_name=file_name)
    dt = DATASET_INFO['use_FFT'][file_name]['dt']
    ctrl = control.Control(simulation(time_step=dt), datasets, [],
                           verbose=-1, reset_config=False)

    for pair in ctrl.observable_pairs:
        assert pair.rescale_factor == 1.
        assert pair.auto_scale


@pytest.mark.parametrize('file_name',
                         ['263K05Awat_LAMP', 'Well_s_q_omega_Ar_data.xml'])
def test_control_scaling_warning(simulation, exp_datasets, file_name,
                                 capsys):
    """
    Test that when both ``rescale_factor`` and ``auto_scale`` specified then
    the latter is used and a warning is printed to explain this.
    """

    datasets = exp_datasets(rescale_factor=0.5,
                            auto_scale=True,
                            file_name=file_name)
    dt = DATASET_INFO['use_FFT'][file_name]['dt']
    ctrl = control.Control(simulation(time_step=dt), datasets, [],
                           reset_config=False)

    for pair in ctrl.observable_pairs:
        assert pair.rescale_factor == 1.
        assert pair.auto_scale

    stdout = capsys.readouterr().out
    stdout_message = ('Both `rescale_factor` and `auto_scale` set for file '
                      '{}; scaling will be automated to minimise FoM\n'
                      'Control created with:\n'
                      '- Attributes                              -\n'
                      '  Minimizer                             MMC\n'
                      '  FoM type               ChiSquaredExpError\n'
                      '  Number of observables                   1\n'
                      '  Number of parameters                    0\n'
                      '\n'
                      ''.format(datasets[0]['file_name']))
    assert stdout_message in stdout


@pytest.mark.parametrize('file_name',
                         ['263K05Awat_LAMP', 'Well_s_q_omega_Ar_data.xml'])
def test_control_use_FFT_default(simulation, exp_datasets, file_name):
    """
    Test that ``use_FFT`` defaults to True.
    """

    datasets = exp_datasets(file_name=file_name)
    dt = DATASET_INFO['use_FFT'][file_name]['dt']
    ctrl = control.Control(simulation(time_step=dt), datasets, [],
                           verbose=-1, reset_config=False)

    for pair in ctrl.observable_pairs:
        assert pair.exp_obs.use_FFT
        assert pair.MD_obs.use_FFT


@pytest.mark.parametrize('file_name',
                         ['263K05Awat_LAMP', 'Well_s_q_omega_Ar_data.xml'])
def test_control_use_FFT(simulation, exp_datasets, file_name):
    """
    Test that ``use_FFT`` is applied when specified.
    """

    datasets = exp_datasets(use_FFT=False, file_name=file_name)
    dt = DATASET_INFO['no_FFT'][file_name]['dt']
    ctrl = control.Control(simulation(time_step=dt), datasets, [],
                           verbose=-1, reset_config=False)

    for pair in ctrl.observable_pairs:
        assert not pair.exp_obs.use_FFT
        assert not pair.MD_obs.use_FFT


def test_control_max_parameter_change():
    """
    Test that ``max_parameter_change`` is passed to the ``Minimizer``.
    """

    ctrl_default = control.Control(None, [], [], minimizer_type="MMC",verbose=-1, reset_config=False)
    assert ctrl_default.minimizer.max_parameter_change == 0.01

    ctrl = control.Control(None, [], [], reset_config=False, verbose=-1, 
                           minimizer_type="MMC", max_parameter_change=0.02)
    assert ctrl.minimizer.max_parameter_change == 0.02


def mock_nonuniform_SQw() -> SQw:
    """
    A mock ``SQw`` ``Observable`` for testing purposes with a non-uniform grid of Q and E points.

    Returns
    -------
    ``SQw``
        A mocked ``SQw`` object.
    """
    observable = SQw()
    observable._origin = 'experiment'
    E_array = np.array([0., 0.24, 0.5, 0.75, 1.0])
    Q_array = np.array([1., 2., 2.9, 4.])
    SQw_array = np.array([[E + Q for E in E_array] for Q in Q_array])
    SQw_err_array = np.zeros(np.shape(SQw_array)) + 0.01
    observable.independent_variables = {'E': E_array, 'Q': Q_array}
    observable._dependent_variables = {'SQw': [SQw_array]}
    observable._errors = {'SQw': [SQw_err_array]}
    return observable


def mock_uniform_SQw() -> SQw:
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
    SQw_array = np.array([[E + Q for E in E_array] for Q in Q_array])
    SQw_err_array = np.zeros(np.shape(SQw_array)) + 0.01
    observable.independent_variables = {'E': E_array, 'Q': Q_array}
    observable._dependent_variables = {'SQw': [SQw_array]}
    observable._errors = {'SQw': [SQw_err_array]}
    return observable


def mock_nonuniform_PDF() -> PairDistributionFunction:
    """
    A mock ``PairDistributionFunction`` ``Observable`` for testing purposes with a non-uniform grid of r points.

    Returns
    -------
    ``PairDistributionFunction``
        A mocked ``PairDistributionFunction`` object.
    """
    observable = PairDistributionFunction()
    r_array = np.array([1., 1.9, 3.1, 4.])
    observable.independent_variables = {'r': r_array}
    observable._dependent_variables = {'PDF': [r_array * 2]}
    observable._errors = {'PDF': [r_array / 10]}
    return observable


def mock_uniform_PDF() -> PairDistributionFunction:
    """
    A mock ``PairDistributionFunction`` ``Observable`` for testing purposes with a uniform grid of r points.

    Returns
    -------
    ``PairDistributionFunction``
        A mocked ``PairDistributionFunction`` object.
    """
    observable = PairDistributionFunction()
    r_array = np.array([1., 2., 3., 4.])
    observable.independent_variables = {'r': r_array}
    observable._dependent_variables = {'PDF': [r_array * 2]}
    observable._errors = {'PDF': [r_array / 10]}
    return observable


@pytest.mark.parametrize('mock_observable',
                         [{'obs': mock_nonuniform_SQw(),
                           'exp': {'E': {'uniform': False, 'zeroed': True},
                                   'Q': {'uniform': False, 'zeroed': False}}},
                          {'obs': mock_uniform_SQw(),
                           'exp': {'E': {'uniform': True, 'zeroed': True},
                                   'Q': {'uniform': True, 'zeroed': False}}},
                          {'obs': mock_nonuniform_PDF(),
                           'exp': {'r': {'uniform': False, 'zeroed': False}}},
                          {'obs': mock_uniform_PDF(),
                           'exp': {'r': {'uniform': True, 'zeroed': False}}}])
def test_control_is_data_uniform(mock_observable):
    """
    Tests that the Control._is_data_uniform method returns the correct boolean for the mocked observables.
    """
    expected = mock_observable['exp']
    # create Control object without instantiating it to test one of its methods
    cont = control.Control
    observed = cont._is_data_uniform(mock_observable['obs'])
    assert expected == observed


@pytest.mark.parametrize('mock_observable',
                         [{'obs': mock_nonuniform_SQw(), 'exp': mock_uniform_SQw()},
                          {'obs': mock_nonuniform_PDF(), 'exp': mock_uniform_PDF()}])
def test_control_make_data_uniform(mock_observable):
    """
    Tests that the Control._make_data_uniform() method correctly makes the mocked non-uniform observables uniform.
    """
    expected = mock_observable['exp']
    # create Control object without instantiating it to test one of its methods
    cont = control.Control.__new__(control.Control)
    observed = cont._make_data_uniform(mock_observable['obs'])
    for var_key in observed.independent_variables:
        assert np.allclose(expected.independent_variables[var_key],
                           observed.independent_variables[var_key], atol=1e-5)
    for var_key in observed.dependent_variables:
        assert np.allclose(expected.dependent_variables[var_key],
                           observed.dependent_variables[var_key], atol=1e-5)
    for var_key in observed.errors:
        assert np.allclose(expected.errors[var_key], observed.errors[var_key], atol=1e-5)


@pytest.mark.parametrize('file_name',
                         ['263K05Awat_LAMP', 'Well_s_q_omega_Ar_data.xml'])
@pytest.mark.parametrize('traj_step', [1, 5, 25])
@pytest.mark.parametrize('use_FFT', [True, False])
def test_control_no_MD_steps(simulation, exp_datasets, use_FFT, traj_step,
                             file_name):
    """
    Test that ``MD_steps`` defaults to the minimum required if not specified.
    """

    if use_FFT:
        key = 'use_FFT'
    else:
        key = 'no_FFT'
    dt = DATASET_INFO[key][file_name]['dt']
    n_frames = DATASET_INFO[key][file_name]['n_frames']
    time_step = dt / traj_step
    ctrl = control.Control(simulation(traj_step=traj_step, time_step=time_step),
                           exp_datasets(use_FFT=use_FFT, file_name=file_name),
                           [], verbose=-1,
                           reset_config=False)
    assert ctrl.MD_steps == n_frames * traj_step


@pytest.mark.parametrize('file_name',
                         ['263K05Awat_LAMP', 'Well_s_q_omega_Ar_data.xml'])
@pytest.mark.parametrize('traj_step', [1, 5, 25])
@pytest.mark.parametrize('use_FFT', [True, False])
def test_control_MD_steps_accepted(simulation, exp_datasets, use_FFT,
                                   traj_step, file_name):
    """
    Test that ``MD_steps`` is accepted when greater than the minimum required,
    and rounded down to an integer number of ``nE * traj_steps`` if there is a
    maximum number of frames (i.e. when ``use_FFT == True`).
    """

    user_MD_steps = 51050
    if use_FFT:
        key = 'use_FFT'
        max_steps = traj_step * DATASET_INFO[key][file_name]['n_frames']
        expected_steps = max_steps * (user_MD_steps // max_steps)
    else:
        key = 'no_FFT'
        expected_steps = user_MD_steps

    dt = DATASET_INFO[key][file_name]['dt']
    time_step = dt / traj_step
    ctrl = control.Control(simulation(traj_step=traj_step, time_step=time_step),
                           exp_datasets(use_FFT=use_FFT, file_name=file_name),
                           [],
                           verbose=-1,
                           reset_config=False,
                           MD_steps=user_MD_steps)

    assert ctrl.MD_steps == expected_steps


@pytest.mark.parametrize('file_name',
                         ['263K05Awat_LAMP', 'Well_s_q_omega_Ar_data.xml'])
@pytest.mark.parametrize('traj_step', [1, 5, 25])
@pytest.mark.parametrize('use_FFT', [True, False])
def test_control_MD_steps_rejected(simulation, exp_datasets, use_FFT,
                                   traj_step, file_name):
    """
    Test that ``MD_steps`` is rejected when greater than the minimum required.
    """

    if use_FFT:
        key = 'use_FFT'
    else:
        key = 'no_FFT'
    dt = DATASET_INFO[key][file_name]['dt']
    time_step = dt / traj_step
    with pytest.raises(ValueError):
        control.Control(simulation(traj_step=traj_step, time_step=time_step),
                        exp_datasets(use_FFT=use_FFT, file_name=file_name),
                        [],
                        verbose=-1,
                        reset_config=False,
                        MD_steps=1)


@pytest.mark.parametrize('file_name',
                         ['263K05Awat_LAMP', 'Well_s_q_omega_Ar_data.xml'])
@pytest.mark.parametrize('traj_step', [1, 5, 25])
@pytest.mark.parametrize('use_FFT', [True, False])
def test_control_validate_energy(simulation, exp_datasets, use_FFT, traj_step,
                                 file_name):
    """
    Test that an ``AssertionError`` is raised when we provide an incorrect time
    separation.
    """

    if use_FFT:
        key = 'use_FFT'
    else:
        key = 'no_FFT'
    dt = DATASET_INFO[key][file_name]['dt']
    time_step = 2 * dt / traj_step
    with pytest.raises(AssertionError):
        control.Control(simulation(traj_step=traj_step, time_step=time_step),
                        exp_datasets(use_FFT=use_FFT, file_name=file_name),
                        [],
                        verbose=-1,
                        reset_config=False)


def test_control_fit_parameters(simulation):
    """
    Test that unsuitable fit_parameters are removed from the Control object:
      - Parameters with a value of 0
      - Parameters that are fixed
      - Parameters that are tied
    As these cannot be refined
    """

    tie_target = Parameter(-1., 'tie_target')
    tied_param = Parameter(2., 'tied')
    tied_param.set_tie(tie_target, '')
    fit_parameters = Parameters([Parameter(0., 'zero'),
                                 Parameter(1., 'fixed', fixed=True),
                                 tied_param,
                                 Parameter(3., 'constraints', constraints=(2.9, 3.1))])

    ctrl = control.Control(simulation(), [], fit_parameters=fit_parameters,
                           verbose=-1, reset_config=False)

    assert len(ctrl.fit_parameters) == 1
    assert 'constraints' in list(ctrl.fit_parameters.keys())[0]


def test_control_resolution_function(simulation, exp_datasets):
    """
    Test that when a resolution file is provided, a resolution function is added to both the
    experimental and MD observables.
    """

    file_name = '263K05Awat_LAMP'
    resolution_file = '262p7K0A5van_LAMP'

    dt = DATASET_INFO['use_FFT'][file_name]['dt']
    traj_step = 1
    time_step = dt / traj_step

    ctrl = control.Control(simulation(time_step=time_step, traj_step=traj_step),
                           exp_datasets(file_name=file_name, resolution=resolution_file),
                           [],
                           verbose=-1,
                           reset_config=False)

    assert type(ctrl.observable_pairs[0].exp_obs.resolution) == FileResolution
    assert type(ctrl.observable_pairs[0].MD_obs.resolution) == FileResolution

@pytest.mark.parametrize('steps', [0,None])
def test_control_equilibrate_auto_check(simulation, exp_datasets, steps, monkeypatch):
    """
    Tests that when the equilibration method is called with no steps specified
    (either 0 or None), then the auto_equilibrate method is called.
    """
    mock_auto_equilibrate = Mock()
    monkeypatch.setattr(control.Simulation, "auto_equilibrate", mock_auto_equilibrate)
    
    ctrl = control.Control(simulation(traj_step=1, time_step=1),
                        exp_datasets(use_FFT=False, file_name='263K05Awat_LAMP'),
                        [],
                        reset_config=False,
                        equilibration_steps=steps)
    
    ctrl.equilibrate(steps)
    mock_auto_equilibrate.assert_called()
    

@pytest.mark.parametrize('steps', [1,50])
def test_control_equilibrate_run_check(simulation,exp_datasets, steps, monkeypatch):
    """
    Tests that when the equilibration method is called with equilibration steps specified
    (an integer > 0), then the simulation.run method is called accordingly. 
    """
    mock_simulation_run = Mock()
    monkeypatch.setattr(control.Simulation, "run", mock_simulation_run)
    
    ctrl = control.Control(simulation(traj_step=1, time_step=1),
                        exp_datasets(use_FFT=False, file_name='263K05Awat_LAMP'),
                        [],
                        reset_config=False,
                        equilibration_steps=steps)
    
    ctrl.equilibrate(steps)
    mock_simulation_run.assert_called()

def test_control_q_value_trimming(exp_datasets):
    """
    Tests that the q_value trimming is done, by using a script which 
    """
    ctrl,fit_parameters = control_object_from_Argon_script(exp_datasets)
    
    fit_parameters['epsilon'].value = 1.02
    fit_parameters['sigma'].value = 3.36
    
    fit_parameters['sigma'].constraints = [2.0,3.8]
    fit_parameters['epsilon'].constraints = [0.5, 1.5]
    
    ctrl.equilibrate(n_steps=1000)

    recreated_q_values_pos = [6,9]
    manually_trimmed_arrays = [ctrl.observable_pairs[0].exp_obs.errors['SQw'][0][pos] 
                               for pos in recreated_q_values_pos]
    ctrl.refine(n_steps=1)
    auto_trimmed_arrays = ctrl.observable_pairs[0].exp_obs.errors['SQw'][0]
    
    assert np.array_equal(manually_trimmed_arrays,auto_trimmed_arrays)
    
def test_control_q_value_trimming_warning(exp_datasets, caplog):
    """
    
    """
    ctrl,fit_parameters = control_object_from_Argon_script(exp_datasets)
    
    fit_parameters['epsilon'].value = 1.02
    fit_parameters['sigma'].value = 3.36
    
    fit_parameters['sigma'].constraints = [2.0,3.8]
    fit_parameters['epsilon'].constraints = [0.5, 1.5]
    
    ctrl.equilibrate(n_steps=1000)

    caplog.set_level(logging.WARNING)
    ctrl.refine(n_steps=1)
    assert " The specified box size was not able to recreate the lowest q " 
    " values of the experimental data and so this data has been " 
    " trimmed accordingly." in caplog.text