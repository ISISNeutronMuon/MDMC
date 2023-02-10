"""Tests for classes that derive from FigureOfMeritCalculator class and test for
ObservablePair class"""

import copy

import numpy as np
import pytest

from MDMC.refinement.FoM.FoM_factory import FoMFactory
from MDMC.refinement.FoM.FoM_abs import ObservablePair
#from MDMC.refinement.FoM.ChiSquared_nonerror import RSquared_noneerror
#from MDMC.refinement.FoM.ChiSquared_experror import ChiSquaredExpError
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

    return ObservablePair(SQw_from_exp, SQw_from_MD, weight=1.)

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
                np.random.random_sample(21) * np.arange(21),
                np.random.random_sample(21) * np.arange(21)
               ),
               ('MD',
                np.arange(-5, 5.5, 0.5),
                np.random.random_sample(21) * np.arange(21),
                np.random.random_sample(21) * np.arange(21)
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

        obs_pairs.append(ObservablePair(obs_duo[0], obs_duo[1], weight=1.))

    return obs_pairs


def test_OP_identical_independent_variables(SQw_from_exp, SQw_from_exp_diff,
                                            SQw_from_MD, observable_pair):

    """
    Tests that ObservablePair observables have the same independent variables
    """

    # Test for no exception when init and when set with it
    pair = ObservablePair(SQw_from_exp, SQw_from_MD, weight=1.)
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
    pair = ObservablePair(SQw_from_exp, SQw_from_MD, weight=1.)
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
    pair = ObservablePair(SQw_from_exp, SQw_from_MD, weight=1.)
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
        ObservablePair(SQw_from_exp, SQw_from_MD, weight=weight)
        observable_pair.weight = weight

    invalid_values = [float('inf'),
                      float('nan'),
                      float(-1.)]

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

    pair = ObservablePair(from_exp, from_MD, weight=1.)
    assert np.all(pair.calculate_difference() == -SQw_dict['dep'])

    pair.exp_obs._dependent_variables['dep'] = 4 * SQw_dict['dep']
    assert np.all(pair.calculate_difference() == 2 * SQw_dict['dep'])

    rescaled_pair = ObservablePair(from_exp, from_MD, weight=1.,
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

    pair = ObservablePair(from_exp, from_MD, weight=1.)
    assert np.all(pair.calculate_errors() == (ERR ** 2 + (2 * ERR) ** 2) ** 0.5)

    ERR_NEG = np.arange(0., -19., -0.19)
    pair.MD_obs.errors['err'] = ERR_NEG
    assert np.all(pair.calculate_errors() == (ERR ** 2 + ERR_NEG ** 2) ** 0.5)

    pair.exp_obs.errors['err'] = ERR_NEG
    assert np.all(pair.calculate_errors() ==
                  (ERR_NEG ** 2 + ERR_NEG ** 2) ** 0.5)

    rescaled_pair = ObservablePair(from_exp, from_MD, weight=1.,
                                       rescale_factor=0.75)
    assert (np.all(rescaled_pair.calculate_errors()
            == ((ERR_NEG * 0.75) ** 2 + ERR_NEG ** 2) ** 0.5))

@pytest.mark.parametrize('FoM_calculator', FoMFactory.get_FoM_names())
def test_FoM_calculation(FoM_calculator, pairs):

    """
    Tests that the FoM calculation is valid for a pair of observables i.e. it is
    a non-negative float
    """

    calculator = FoMFactory.create_FoM(FoM_calculator,pairs)
    assert calculator.calculate() >= 0


@pytest.mark.parametrize('FoM_calculator', FoMFactory.get_FoM_names() )
def test_multiple_FoM_calculation(FoM_calculator, pairs):

    """
    Tests that the FoM calculation is valid for for multiple pairs of
    observables
    """

    pairs += pairs
    calculator = FoMFactory.create_FoM(FoM_calculator,pairs)
    assert calculator.calculate() >= 0


@pytest.mark.parametrize('FoM_calculator', FoMFactory.get_FoM_names())
def test_FoM_calculation_data_point_norm(FoM_calculator, pairs):

    """
    Tests that the FoM calculation normalises for the size of the dataset
    """

    # Create datasets with twice the number of entries
    pairs_large = copy.deepcopy(pairs)
    for pair in pairs_large:
        dep_array_exp = pair.exp_obs._dependent_variables['dep']
        pair.exp_obs._dependent_variables = {'dep': np.append(dep_array_exp, dep_array_exp)}

        dep_array_MD = pair.MD_obs._dependent_variables['dep']
        pair.MD_obs._dependent_variables = {'dep': np.append(dep_array_MD, dep_array_MD)}

        error_array_exp = pair.exp_obs._errors['err']
        pair.exp_obs._errors = {'err': np.append(error_array_exp, error_array_exp)}

        error_array_MD = pair.MD_obs._errors['err']
        pair.MD_obs._errors = {'err': np.append(error_array_MD, error_array_MD)}

    calculator = FoMFactory.create_FoM(FoM_calculator,pairs, norm='data_points')
    calculator_large = FoMFactory.create_FoM(FoM_calculator,pairs_large, norm='data_points')
    assert np.isclose(calculator.calculate(), calculator_large.calculate())


@pytest.mark.parametrize('FoM_calculator', FoMFactory.get_FoM_names())
def test_FoM_calculation_dof_norm(FoM_calculator, pairs):

    """
    Tests that the FoM calculation normalises for the size of the dataset and degrees of freedom
    """

    dataset_size = pairs[0].exp_obs._dependent_variables['dep'].size

    # Set n_parameters to be half the dataset size, so in the second case we normalise by
    # `dataset_size - dataset_size / 2 = dataset_size / 2` giving twice the FoM
    calculator =FoMFactory.create_FoM(FoM_calculator,pairs, norm='dof', n_parameters=0)
    calculator_dof = FoMFactory.create_FoM(FoM_calculator,pairs, norm='dof', n_parameters=dataset_size / 2)

    assert calculator.calculate() == calculator_dof.calculate() / 2


@pytest.mark.parametrize('FoM_calculator', FoMFactory.get_FoM_names())
def test_FoM_calculation_no_norm(FoM_calculator, pairs):

    """
    Tests that the FoM calculation is increased for larger datasets when no norm is used
    """

    # Create datasets with twice the number of entries
    pairs_large = copy.deepcopy(pairs)
    for pair in pairs_large:
        dep_array_exp = pair.exp_obs._dependent_variables['dep']
        pair.exp_obs._dependent_variables = {'dep': np.append(dep_array_exp, dep_array_exp)}

        dep_array_MD = pair.MD_obs._dependent_variables['dep']
        pair.MD_obs._dependent_variables = {'dep': np.append(dep_array_MD, dep_array_MD)}

        error_array_exp = pair.exp_obs._errors['err']
        pair.exp_obs._errors = {'err': np.append(error_array_exp, error_array_exp)}

        error_array_MD = pair.MD_obs._errors['err']
        pair.MD_obs._errors = {'err': np.append(error_array_MD, error_array_MD)}

    calculator = FoMFactory.create_FoM(FoM_calculator,pairs, norm='none')
    calculator_large = FoMFactory.create_FoM(FoM_calculator,pairs_large, norm='none')
    assert np.isclose(calculator.calculate(), calculator_large.calculate() / 2)


@pytest.mark.parametrize('FoM_calculator',FoMFactory.get_FoM_names())
def test_FoM_calculation_raises_errors(FoM_calculator, pairs):

    """
    Tests that the FoM calculators raise a ValueError when given unrecognised or incompatible
    arguments.
    """

    # Unrecognised option for the norm
    with pytest.raises(ValueError):
        FoMFactory.create_FoM(FoM_calculator,pairs, norm='unrecognised')

    # DoF normalisation without providing number of parameters
    with pytest.raises(ValueError):
        FoMFactory.create_FoM(FoM_calculator,pairs, norm='dof')


@pytest.mark.parametrize('FoM_calculator', FoMFactory.get_FoM_names())
@pytest.mark.parametrize('weight', [0.1, 1., 10.])
def test_weighted_FoM_calculation(FoM_calculator, weight, pairs):

    """
    Tests that the FoM calculation takes weighting into account
    """

    normal_weight = pairs[0].weight
    calculator = FoMFactory.create_FoM(FoM_calculator,pairs)
    normal_FoM = calculator.calculate()

    # Create datasets with the same dependent variable for MD and exp (which
    # will give FoM of zero) and a varying weight
    pairs_weighted = copy.deepcopy(pairs)
    for pair in pairs_weighted:
        pair.weight = weight
        SQw_array = pair.MD_obs._dependent_variables['dep']
        pair.exp_obs._dependent_variables = {'dep':SQw_array}

    calculator_weighted =FoMFactory.create_FoM(FoM_calculator,pairs + pairs_weighted)
    weighted_FoM = calculator_weighted.calculate()

    assert weighted_FoM == normal_FoM / (normal_weight + weight)


@pytest.mark.parametrize('exp_error', [0.1, 1., 10.])
@pytest.mark.parametrize('MD_error', [0.1, 1., 10.])
def test_errors_ChiSquaredExpError(exp_error, MD_error, pairs):

    """
    Test that changing the value of the experimental errors affects the ChiSquaredExpError, but
    that changing the MD errors does not.
    """

    # Set errors to be constant for ease of testing
    for pair in pairs:
        error_shape = pair.exp_obs._errors['err'].shape
        pair.exp_obs._errors = {'err':np.ones(error_shape)}
        pair.MD_obs._errors = {'err':np.ones(error_shape)}

    calculator = FoMFactory.create_FoM('exp',pairs)
    normal_FoM = calculator.calculate()

    # Create datasets with scaled errors
    pairs_scaled_error = copy.deepcopy(pairs)
    for pair in pairs_scaled_error:
        error_shape = pair.exp_obs._errors['err'].shape
        pair.exp_obs._errors = {'err':np.full(error_shape, exp_error)}
        pair.MD_obs._errors = {'err':np.full(error_shape, MD_error)}

    calculator_weighted = FoMFactory.create_FoM('exp',pairs_scaled_error)
    scaled_FoM = calculator_weighted.calculate()

    assert np.isclose(scaled_FoM, normal_FoM / exp_error ** 2)


@pytest.mark.parametrize('exp_error', [0.1, 1., 10.])
@pytest.mark.parametrize('MD_error', [0.1, 1., 10.])
def test_errors_RSquaredNoError(exp_error, MD_error, pairs):

    """
    Test that changing the value of neither the experimental errors nor the MD errors affects the
    RSquared_noneerror.
    """

    # Set errors to be constant for ease of testing
    for pair in pairs:
        error_shape = pair.exp_obs._errors['err'].shape
        pair.exp_obs._errors = {'err':np.ones(error_shape)}
        pair.MD_obs._errors = {'err':np.ones(error_shape)}

    calculator = FoMFactory.create_FoM('none',pairs)
    normal_FoM = calculator.calculate()

    # Create datasets with scaled errors
    pairs_scaled_error = copy.deepcopy(pairs)
    for pair in pairs_scaled_error:
        error_shape = pair.exp_obs._errors['err'].shape
        pair.exp_obs._errors = {'err':np.full(error_shape, exp_error)}
        pair.MD_obs._errors = {'err':np.full(error_shape, MD_error)}

    calculator_weighted = FoMFactory.create_FoM('none',pairs_scaled_error)
    scaled_FoM = calculator_weighted.calculate()  # I don't think this is a good test, comparing values up to 5e8 different...

    assert np.isclose(scaled_FoM, normal_FoM)


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

    with pytest.raises(error):
        ObservablePair(obs_from_exp, obs_from_MD, weight)
        pytest.fail("Expecting {0} upon init")


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

    with pytest.raises(error):
        pair.exp_obs = obs_from_exp
        pytest_fail("Expecting {0} upon setting exp_obs")

    with pytest.raises(error):
        pair.MD_obs = obs_from_MD
        pytest_fail("Expecting {0} upon setting MD_obs")
