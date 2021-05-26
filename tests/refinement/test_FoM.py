"""Tests for classes that derive from FigureOfMeritCalculator class and test for
ObservablePair class"""

import copy

import numpy as np
import pytest

import MDMC.refinement.FoM as fom
import MDMC.trajectory_analysis.observables.obs_factory as of

from tests.test_data import data


"""
ObservablePair
Test the following:

Takes exactly one exp_obs which has attributes independent, dependent and errors
Takes exactly one MD_obs which has attributes independent, dependent and errors
Observables have the same type
Observables have the same independent variables
Observables have the same shape for dependent variables
Observables have the same shape for errors
Checks that the difference is correctly calculated
Checks that the errors are returned in quadrature
Checks that the weight is a non-negative float


FoM calculators:
Test the following:

Returns a non-negative float
Can deal with multiple obs_pair inputs
"""

@pytest.fixture
def SQw_from_exp():

    SQw = of.ObservableFactory.create_observable('SQw')
    SQw.read_from_file(reader='LAMPSQw', file_name=data.READER_DATA['LAMPSQw'])
    return SQw

@pytest.fixture
def SQw_from_exp_diff():

    # SQw objects from files cannot be deepcopied so add a new fixture
    SQw = of.ObservableFactory.create_observable('SQw')
    SQw.read_from_file(reader='LAMPSQw', file_name=data.READER_DATA['LAMPSQw'])
    return SQw

@pytest.fixture
def SQw_from_MD(SQw_from_exp):

    SQw = of.ObservableFactory.create_observable('SQw')
    SQw._origin = 'MD'
    SQw.independent_variables = SQw_from_exp.independent_variables
    SQw._dependent_variables = SQw_from_exp.dependent_variables
    SQw._errors = SQw_from_exp.errors
    return SQw

@pytest.fixture
def observable_pair(SQw_from_exp, SQw_from_MD):

    return fom.ObservablePair(SQw_from_exp, SQw_from_MD, weight=1.)

@pytest.fixture
def SQw_dict():

    INDEP = np.arange(0., 100., 1.)
    DEP = np.arange(0., 200., 2.)
    ERR = DEP / 10.

    from_exp = of.ObservableFactory.create_observable('SQw')
    from_exp._origin = 'experiment'
    from_exp.independent_variables = {'indep':INDEP}
    from_exp._dependent_variables = {'dep':DEP}
    from_exp._errors = {'err':ERR}

    from_MD = copy.deepcopy(from_exp)
    from_MD.origin = 'MD'

    return {'dep':DEP, 'err':ERR, 'from_exp':from_exp,
            'from_MD':from_MD}


PAIRS_INFO = [(('experiment',
                np.arange(-5., 5.5, 0.5),
                np.random.random_sample(21) * np.random.random_integers(1, 1e9),
                np.random.random_sample(21) * np.random.random_integers(1, 1e9)
               ),
               ('MD',
                np.arange(-5, 5.5, 0.5),
                np.random.random_sample(21) * np.random.random_integers(1, 1e9),
                np.random.random_sample(21) * np.random.random_integers(1, 1e9)
               ),
              ),
             ]

@pytest.fixture
def pairs():

    obs_pairs = []
    for pair_info in PAIRS_INFO:

        obs_duo = []
        for obs_info in pair_info:
            obs = of.ObservableFactory.create_observable('SQw')
            obs._origin = obs_info[0]
            obs.independent_variables = {'indep':obs_info[1]}
            obs._dependent_variables = {'dep':obs_info[2]}
            obs._errors = {'err':obs_info[3]}

            obs_duo.append(obs)

        obs_pairs.append(fom.ObservablePair(obs_duo[0], obs_duo[1], weight=1.))

    return obs_pairs


def test_OP_identical_independent_variables(SQw_from_exp, SQw_from_exp_diff,
                                            SQw_from_MD, observable_pair):

    """
    Tests that ObservablePair observables have the same independent variables
    """

    # Test for no exception when init and when set with it
    pair = fom.ObservablePair(SQw_from_exp, SQw_from_MD, weight=1.)
    assert (pair.exp_obs.independent_variables ==
            pair.MD_obs.independent_variables)
    pair.MD_obs = SQw_from_MD
    pair.exp_obs = SQw_from_exp

    # Test for exceptions with different independent variables
    SQw_from_MD_diff = copy.deepcopy(SQw_from_MD)
    for k in SQw_from_exp_diff.independent_variables:
        SQw_from_exp_diff._independent_variables[k] *= 2.
        SQw_from_MD_diff.independent_variables[k] *= 3.

    init_exception_check(AssertionError, SQw_from_exp_diff, SQw_from_MD_diff)
    set_exception_check(AssertionError, SQw_from_exp_diff, SQw_from_MD_diff,
                        observable_pair)


def test_OP_shape_dependent_variables(SQw_from_exp, SQw_from_exp_diff,
                                      SQw_from_MD, observable_pair):

    """
    Tests that ObservablePair observables have dependent variables with the same
    shape
    """

    # Test for no exception when init with SQw
    pair = fom.ObservablePair(SQw_from_exp, SQw_from_MD, weight=1.)
    assert (np.shape(pair.exp_obs.dependent_variables) ==
            np.shape(pair.MD_obs.dependent_variables))
    pair.MD_obs = SQw_from_MD
    pair.exp_obs = SQw_from_exp

    # Tests for exception with different shape dependent variables
    SQw_from_MD_diff = copy.deepcopy(SQw_from_MD)
    for k in SQw_from_exp_diff.dependent_variables:
        SQw_from_exp_diff._dependent_variables[k] = [np.transpose(
            SQw_from_exp_diff.dependent_variables[k][0])]
        SQw_from_MD_diff._dependent_variables[k] = \
            [SQw_from_MD_diff.dependent_variables[k][0].flatten()]

    init_exception_check(AssertionError, SQw_from_exp_diff, SQw_from_MD_diff)
    set_exception_check(AssertionError, SQw_from_exp_diff, SQw_from_MD_diff,
                        observable_pair)


def test_OP_shape_errors(SQw_from_exp, SQw_from_exp_diff, SQw_from_MD,
                         observable_pair):

    """
    Tests that ObservablePair observables have errors with the same shape
    """

    # Test for no exception when init with SQw
    pair = fom.ObservablePair(SQw_from_exp, SQw_from_MD, weight=1.)
    assert (np.shape(pair.exp_obs.errors) == np.shape(pair.MD_obs.errors))
    pair.MD_obs = SQw_from_MD
    pair.exp_obs = SQw_from_exp

    # Tests foir exception with different shape dependent variables
    SQw_from_MD_diff = copy.deepcopy(SQw_from_MD)
    for k in SQw_from_exp_diff.errors:
        SQw_from_exp_diff._errors[k] = np.transpose(SQw_from_exp_diff.errors[k][0])
        SQw_from_MD_diff._errors[k] = SQw_from_MD_diff.errors[k][0].flatten()

    init_exception_check(AssertionError, SQw_from_exp_diff, SQw_from_MD_diff)
    set_exception_check(AssertionError, SQw_from_exp_diff, SQw_from_MD_diff,
                        observable_pair)


def test_OP_origins(SQw_from_exp, SQw_from_MD, observable_pair):

    """
    Tests that ObservablePairs can only be initialiazed with exactly one
    observable of each origin ('experiment' and 'MD')
    """

    init_exception_check(AssertionError, SQw_from_exp, SQw_from_exp)
    init_exception_check(AssertionError, SQw_from_MD, SQw_from_MD)

    set_exception_check(AssertionError, SQw_from_MD, SQw_from_exp,
                        observable_pair)


def test_OP_types(SQw_from_exp, observable_pair):

    """
    Tests that ObservablePairs can only have observables of the same type
    """

    # Create an incoherent SQw
    SQw_incoh_from_MD = of.ObservableFactory.create_observable('SQw_incoh')
    SQw_incoh_from_MD._origin = 'MD'
    SQw_incoh_from_MD.independent_variables = SQw_from_exp.independent_variables
    SQw_incoh_from_MD._dependent_variables = SQw_from_exp.dependent_variables
    SQw_incoh_from_MD._errors = SQw_from_exp.errors

    init_exception_check(AssertionError, SQw_from_exp, SQw_incoh_from_MD)

    # Create a coherent SQw
    SQw_coh_from_exp = of.ObservableFactory.create_observable('SQw_coh')
    SQw_coh_from_exp._origin = 'experiment'
    SQw_coh_from_exp.independent_variables = SQw_from_exp.independent_variables
    SQw_coh_from_exp._dependent_variables = SQw_from_exp.dependent_variables
    SQw_coh_from_exp._errors = SQw_from_exp.errors

    set_exception_check(AssertionError, SQw_coh_from_exp, SQw_incoh_from_MD,
                        observable_pair)


def test_OP_weight_validation(SQw_from_exp, SQw_from_MD, observable_pair):

    """
    Tests that the ObservablePair weight must be a non-negative float
    """

    valid_values = [45.454545,
                    int(1),
                    '+1e1']

    for weight in valid_values:
        fom.ObservablePair(SQw_from_exp, SQw_from_MD, weight=weight)
        observable_pair.weight = weight

    invalid_values = [np.float('inf'),
                      np.float('nan'),
                      np.float(-1.)]

    invalid_types = ['one',
                     '1,234',
                     '']

    for weight in invalid_values:
        init_exception_check(AssertionError, SQw_from_exp, SQw_from_MD,
                             weight=weight)
        with pytest.raises(AssertionError):
            observable_pair.weight = weight

    for weight in invalid_types:
        init_exception_check(TypeError, SQw_from_exp, SQw_from_MD,
                             weight=weight)
        with pytest.raises(TypeError):
            observable_pair.weight = weight


def test_difference_calculation(SQw_dict):

    """
    Tests the ObservablePair difference calculation is correct
    """

    from_exp = SQw_dict['from_exp']
    from_MD = SQw_dict['from_MD']
    from_MD._dependent_variables['dep'] = 2 * SQw_dict['dep']

    pair = fom.ObservablePair(from_exp, from_MD, weight=1.)
    assert np.all(pair.calculate_difference() == -SQw_dict['dep'])

    pair.exp_obs._dependent_variables['dep'] = 4 * SQw_dict['dep']
    assert np.all(pair.calculate_difference() == 2 * SQw_dict['dep'])

    rescaled_pair = fom.ObservablePair(from_exp, from_MD, weight=1.,
                                       rescale_factor=0.75)
    assert np.all(rescaled_pair.calculate_difference() == SQw_dict['dep'])


def test_error_calculation(SQw_dict):

    """
    Tests the errors are calculated in quadrature
    """

    ERR = SQw_dict['err']
    from_exp = SQw_dict['from_exp']
    from_MD = SQw_dict['from_MD']
    from_MD._errors['err'] = 2 * ERR

    pair = fom.ObservablePair(from_exp, from_MD, weight=1.)
    assert np.all(pair.calculate_errors() == (ERR ** 2 + (2 * ERR) ** 2) ** 0.5)

    ERR_NEG = np.arange(0., -19., -0.19)
    pair.MD_obs.errors['err'] = ERR_NEG
    assert np.all(pair.calculate_errors() == (ERR ** 2 + ERR_NEG ** 2) ** 0.5)

    pair.exp_obs.errors['err'] = ERR_NEG
    assert np.all(pair.calculate_errors() ==
                  (ERR_NEG ** 2 + ERR_NEG ** 2) ** 0.5)

    rescaled_pair = fom.ObservablePair(from_exp, from_MD, weight=1.,
                                       rescale_factor=0.75)
    assert (np.all(rescaled_pair.calculate_errors()
            == ((ERR_NEG * 0.75) ** 2 + ERR_NEG ** 2) ** 0.5))


def test_FoM_calculation(pairs):

    """
    Tests that the FoM calculation is valid for a pair of observables i.e. it is
    a non-negative float
    """

    calculator = fom.StandardFoMCalculator(pairs)
    assert calculator.calculate() >= 0


def test_multiple_FoM_calculation(pairs):

    """
    Tests that the FoM calculation is valid for for multiple pairs of
    observables
    """

    pairs += pairs
    calculator = fom.StandardFoMCalculator(pairs)
    assert calculator.calculate() >= 0


def test_FoM_calculation_dataset_size(pairs):

    """
    Tests that the FoM calculation normalises for the size of the dataset
    """

    # Create datasets with twice the number of entries
    pairs_large = pairs
    for pair in pairs_large:
        dependent_array = pair.exp_obs._dependent_variables['dep']
        pair.exp_obs._dependent_variables = {'dep': np.append(dependent_array, dependent_array)}
        pair.MD_obs._dependent_variables = {'dep': np.append(dependent_array, dependent_array)}

        error_array = pair.exp_obs._errors['err']
        pair.exp_obs._errors = {'err': np.append(error_array, error_array)}
        pair.MD_obs._errors = {'err': np.append(error_array, error_array)}

    calculator = fom.StandardFoMCalculator(pairs)
    calculator_large = fom.StandardFoMCalculator(pairs_large)
    assert calculator.calculate() == calculator_large.calculate()


@pytest.mark.parametrize('weight', [0.1, 1., 10.])
def test_weighted_FoM_calculation(pairs, weight):

    """
    Tests that the FoM calculation takes weighting into account
    """

    # Create datasets with infinite errors (which will give FoM of zero) and a
    # varying weight
    pairs_inf_error = pairs
    for pair in pairs_inf_error:
        pair.weight = weight
        error_shape = np.shape(pair.exp_obs._errors['err'])
        error_array = np.full(error_shape, np.float('inf'))
        pair.exp_obs._errors = {'err': error_array}

    calculator = fom.StandardFoMCalculator(pairs)
    calculator_weighted = fom.StandardFoMCalculator(pairs + pairs_inf_error)

    normal_FoM = calculator.calculate()
    weighted_FoM = calculator_weighted.calculate()

    assert weighted_FoM == normal_FoM / (pairs[0].weight + weight)


def init_exception_check(error, obs_from_exp, obs_from_MD, weight=1.):

    """
    Tests for error when ObservablePair is init with obs_from_exp and
    obs_from_MD

    Arguments:
    error - type of error which should be raised
    obs_from_exp - an experimental observable
    obs_from_MD - an MD observable
    weight - a non-negative float
    """

    with pytest.raises(error,
                       message="Expecting {0} upon init".format(str(error))):
        fom.ObservablePair(obs_from_exp, obs_from_MD, weight)


def set_exception_check(error, obs_from_exp, obs_from_MD, pair):

    """
    Tests for error when ObservablePair is set with obs_from_exp or obs_from_MD

    Arguments:
    error - type of error which should be raised
    obs_from_exp - an experimental observable
    obs_from_MD - an MD observable
    pair - an initial pair which is modified to test obs_from_exp and
    obs_from_MD
    """

    with pytest.raises(error,
                       message="Expecting {0} upon setting exp_obs".format(
                           str(error))):
        pair.exp_obs = obs_from_exp

    with pytest.raises(error,
                       message="Expecting {0} upon setting MD_obs".format(
                           str(error))):
        pair.MD_obs = obs_from_MD
