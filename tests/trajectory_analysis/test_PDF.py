"""Unit tests for the pair distribution function
"""

from itertools import combinations, combinations_with_replacement, product

import numpy as np
import pytest

from MDMC.MD.simulation import Universe
from MDMC.trajectory_analysis.compact_trajectory import configurations_as_compact_trajectory, CompactTrajectory
from MDMC.trajectory_analysis.trajectory import TemporalConfiguration
from MDMC.trajectory_analysis.observables.pdf import PairDistributionFunction


ELEMENTS = {'H', 'C', 'La'}

@pytest.fixture
def PDF():

    """
    Returns
    -------
    PairDistributionFunction
        An initiliazed but unmodified PairDistributionFunction
    """

    return PairDistributionFunction()

@pytest.fixture
def PDF_setup(PDF):

    """
    Returns
    -------
    PairDistributionFunction
        An initiliazed but with r values so that it can be used for testing
        settings
    """

    PDF.r = np.arange(0., 10., 0.5)
    return PDF

@pytest.fixture
def universe():

    """
    Returns
    -------
    Universe
        An empty Universe
    """

    return Universe(1.)

@pytest.fixture
def trajectory(universe):

    """
    Returns
    -------
    Trajectory
        A Trajectory with 1000 empty TemporalConfigurations
    """

    trj = configurations_as_compact_trajectory(*[TemporalConfiguration(time, universe=universe)
                       for time in range(0, 1000, 1)])
    trj.element_set = ELEMENTS
    return trj


@pytest.mark.parametrize('n_frames',
                         [None, 1, 10, 30, 100])
def test_set_n_frames(PDF_setup, trajectory, n_frames):

    """
    Tests that setting the number of frames results in the correct number of
    configurations being used for calculating the PDF, and that these
    configurations are evenly spaced.

    Also includes if n_frames is not specified but is set automatically to the
    total number of frames // 100
    """

    assert len(trajectory) == 1000
    if n_frames:
        settings = {'n_frames':n_frames}
    else:
        settings = {}
        n_frames = len(trajectory) // 100
    PDF_setup._parse_apply_MD_settings(trajectory, settings)
    assert len(PDF_setup.trajectory) == n_frames
    times = PDF_setup.trajectory.times
    # try/except accounts for n_frames == 1
    try:
        time_step = times[1] - times[0]
    except IndexError:
        time_step = 1000.
    assert np.all(np.arange(0., times[-1] + time_step, time_step) == times)


@pytest.mark.parametrize('n_frames', [0, -10, 1001])
def test_set_n_frames_error(PDF_setup, trajectory, n_frames):

    with pytest.raises(ValueError):
        PDF_setup._parse_apply_MD_settings(trajectory, {'n_frames':n_frames})


@pytest.mark.parametrize('partial_strings, expected',
                         [(None,
                           [('C', 'C'), ('C', 'H'), ('C', 'La'), ('H', 'H'),
                            ('H', 'La'), ('La', 'La')]),
                          ([('C', 'H')],
                           [('C', 'H')]),
                          ([('C', 'H'), ('H', 'La')],
                           [('C', 'H'), ('H', 'La')])
                         ])
def test_partial_strings_set(PDF_setup, trajectory, partial_strings, expected):

    """
    Tests that the PDF elements and partial strings are correctly set based on
    the partial strings that are passed to _parse_calc_MD_settings
    """

    settings = {'subset':partial_strings} if partial_strings else {}
    PDF_setup._parse_apply_MD_settings(trajectory, settings)
    assert sorted(PDF_setup.partial_strings) == sorted(expected)


def test_r_set(PDF_setup, trajectory):

    """
    Tests setting r values without r_min, r_max, r_step
    """

    r_values = np.arange(0., 10., 0.5)
    PDF_setup._parse_apply_MD_settings(trajectory, {'r':r_values})
    assert np.all(PDF_setup.r == r_values)
    assert PDF_setup.r_step == 0.5


@pytest.mark.parametrize('r_parameter', ['r_min', 'r_max', 'r_step'])
def test_r_set_error(PDF_setup, trajectory, r_parameter):

    """
    Tests that setting r values with either r_min, r_max, or r_step raises a
    TypeError
    """

    with pytest.raises(TypeError):
        # Set the r_parameter to an arbitrary value
        PDF_setup._parse_apply_MD_settings(trajectory, {'r':range(0, 10, 1),
                                                        r_parameter:0.})


@pytest.mark.parametrize('r_min, r_max, r_step',
                         [(0., 10., 0.5),
                          (0., 15., 2.),
                          (0., 1., 0.01)])
def test_r_set_range(PDF_setup, trajectory, r_min, r_max, r_step):

    """
    Tests setting r values using r_min, r_max and r_step, instead of passing r.
    """

    PDF_setup._parse_apply_MD_settings(trajectory, {'r_min':r_min,
                                                   'r_max':r_max,
                                                   'r_step':r_step})
    assert np.all(PDF_setup.r == np.arange(r_min, r_max + r_step, r_step))
    assert PDF_setup.r_step == r_step


def generate_positions(dimensions, number):

    """
    Generates positions for testing partitioning

    Parameters
    ----------
    dimensions : array
        The dimensions of the universe
    number : array
        The number of atoms in each direction

    Returns
    -------
    list
        A list of positions
    """

    components = np.array([np.linspace(0., dimensions[i], number[i]) for i in
                           range(3)])
    positions = list(product(*components))
    return np.array(list(map(np.array, positions)))


@pytest.mark.parametrize('positions, element_list, part_comps',
                         [(generate_positions(np.array([1.5, 2., 3.]),
                                              (3, 5, 5)),
                           ['H'] * 40 + ['C'] * 35,
                           (0.5, 0.5, 0.5),
                          )])
def test_partition(PDF, positions, element_list, part_comps):

    """
    Tests that all elements are included in the partitions, all positions are
    included in the partitions, and that each position is in the correct
    partition
    """

    PDF.elements = set(element_list)
    PDF.universe_dimensions = np.array([1.5, 2., 3.])
    # it is necessary to construct a CompactTrajectory here,
    # since now PDF._partition takes a trajectory as input.
    trajectory = CompactTrajectory(n_steps=1, n_atoms=len(element_list),
                                   universe = Universe(PDF.universe_dimensions))
    trajectory.writeOneStep(0,0.0,positions)
    trajectory.validateTypes(element_list)
    trajectory.setCharge(len(element_list)*[0.0])
    trajectory.labelAtoms({1:'C', 2:'H'}, {1:12.0, 2: 1.0})
    trajectory.postProcess()
    # now a CompactTrajectory has been constructed out of the input positions
    # and the testing can continue
    partitions = PDF._partition(trajectory, part_comps)
    assert sorted(list(PDF.elements)) == sorted(list(partitions.keys()))

    partition_positions = [pos for elem_partitions in partitions.values()
                           for pos in elem_partitions.values()]
    assert np.all(pos in partition_positions for pos in positions)

    for element, elem_partitions in partitions.items():
        for partition, elem_positions in elem_partitions.items():
            for position in elem_positions:
                # Determine list index of position in positions, then check that
                # the element is correct
                position_index = positions.tolist().index(list(position))
                expected_element = element_list[position_index]
                assert element == expected_element
                # Determine lower and upper bounds of partition (convert from
                # indexes) and check position is within these
                lower_bounds = np.multiply(partition, part_comps)
                upper_bounds = np.add(partition, 1.) * part_comps
                assert np.all(lower_bounds <= position)
                assert np.all(position < upper_bounds)


@pytest.mark.parametrize('number_partitions',
                         [(3, 3, 3),
                          (3, 4, 5),
                          (7, 3, 5)])
def test_partition_pairs(PDF, number_partitions):

    """
    Tests that _get_partition_pairs returns a complete list of all pairs of
    partitions, for different partition components.

    Tests the total number of partition pairs (which must be 13 * x * y * z,
    where x, y, z are the number of partitions in each dimension). Also tests
    that each expected pair occurs.

    Parameters
    ----------
    number_partitions : tuple
        A tuple of ints where each int is the number of partitions in that
        dimension
    """

    # _get_partition_pairs uses the lengths of a partition in each dimension and
    # the dimensions of the universe to calculate number_partitions, so this
    # is inverted in order to pass it the correct parameter
    PDF.universe_dimensions = np.multiply((1, 1, 1), number_partitions)
    actual = PDF._get_partition_pairs((1, 1, 1))
    assert np.shape(actual) == (np.product(number_partitions) * 13, 2, 3)

    expected = get_expected_partition_pairs(*number_partitions)
    # Pairs in actual may not be in the same order as expected, so sort them
    # both
    expected = [sorted(pair) for pair in expected]
    actual = [sorted(pair) for pair in actual]
    assert np.all(pair in actual for pair in expected)


def generate_position_pairs(start, stop, step):

    """
    Generates position pairs for testing histogramming

    Positions are of the form (n * step, n * step, n * step) for
    start <= n <= stop e.g. (0., 0., 0.), (0.5, 0.5, 0.5) (1.0, 1.0, 1.0) for
    start = 0., stop=1.5, step=0.5.

    All pair combinations of these positions are then returned.

    Parameters
    ----------
    start : int
    stop : int
    step : int

    Returns
    -------
    list of tuples
        (pos1, pos2) where pos1 and pos2 are 3 element arrays with all 3
        elements are the same float (which must be a multiple of step which is
        greater than or equal to start and less than stop)
    """

    return combinations(map(lambda x: np.array([x]*3),
                            np.arange(start, stop, step)), 2)


@pytest.mark.parametrize('unique_elements, b_cohs, expected',
                         [(['H', 'Na', 'C'],
                           {'Na':3.1, 'C':1.9},
                           {'Na':3.1, 'C':1.9, 'H':-3.7390}),
                          (['O', 'K'],
                           {'K':9.5},
                           {'K':9.5, 'O':5.803}),
                          (['H', 'O'],
                           {},
                           {'H':-3.7390, 'O':5.803})])
def test_set_weights(PDF, unique_elements, b_cohs, expected):

    """
    Tests that the correct weights of elements are determined
    """

    assert PDF._set_weights(unique_elements, b_cohs) == expected


@pytest.mark.parametrize('unique_element_dict, element_list',
                         [({'H':3, 'O':2, 'C':1},
                           ['H', 'H', 'H', 'O', 'O', 'C']),
                          ({'Na':4, 'K':1},
                           ['Na', 'K', 'Na', 'Na', 'Na', 'C']),
                          ({'B':0, 'Ar':3},
                           ['Ar', 'Ar', 'Ar'])])
def test_set_numbers(PDF, unique_element_dict, element_list):

    """
    Tests that the correct number of elements are determined
    """

    assert (PDF._set_numbers(unique_element_dict.keys(), element_list)
            == unique_element_dict)


def get_expected_partition_pairs(x, y, z):

    """
    This method is simpler to understand (and therefore more likely to be
    correct!) than the method employed in _get_partition_pairs, but less
    efficient (as redundant pairs are generated and then removed), particularly
    for large numbers of partitions

    Parameters
    ----------
    x, y, z : int
        The number of partitions in each dimension

    Returns
    -------
    list of tuples
        (partition1, partition2) where partition1 and partition 2 are 3 element
        tuples, where each element is a tuple.
    """

    def are_neighbours(pair, max_parts):

        """
        Checks whether two partitions are neighbours i.e. no dimension is
        separated by more than 1

        Parameters
        ----------
        pair : tuple
            (partition1, partition2) where partition1 and partition2 are 3
            element tuples
        max : tuple
            (x, y, z) where x, y and z are ints giving the maximum partition
            number ineach dimension

        Returns
        -------
        bool
            True if the partitions in pair are neighbours (including considering
            periodic boundary conditions)
        """

        # Iterate over components for both pairs
        for i in range(3):
            c1, c2 = pair[0][i], pair[1][i]
            # First condition checks if they are simple neighbours, second
            # checks if they are neighbours due to PBC
            if np.abs(c1 - c2) > 1 and sorted([c1, c2]) != [0, max_parts[i]]:
                return False
        return True

    x_rng, y_rng, z_rng = range(0, x, 1), range(0, y, 1), range(0, z, 1)
    all_partition_combinations = combinations(product(x_rng, y_rng, z_rng), 2)
    expected = [pair for pair in all_partition_combinations
                if are_neighbours(pair, [x-1, y-1, z-1])]

    return expected
