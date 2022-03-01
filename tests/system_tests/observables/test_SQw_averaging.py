"""System test for averaging ``SQw`` from multiple ``Trajectory`` objects """

import numpy as np
from tests.system_tests.observables.data_manager import trajectory, Q_vectors
from MDMC.trajectory_analysis.observables.obs_factory import ObservableFactory

def test_SQw_averaging(trajectory, Q_vectors):
    """
    test whether averaging SQw over multiple subtrajectories is consistent with SQw calculated
    from the full trajectory and whether errors are calculated
    """
    # Values are equivalent to those used by nMOLDYN to generate the test data
    DIMENSIONS = (39.4221067, 39.4221067, 39.4221067)
    E_RESOLUTION = {'gaussian': 49.99998257}
    subtrj_list = [trajectory[0:-2], trajectory[1:-1]]
    SQw_full_trj = ObservableFactory.create_observable('SQw')
    SQw_full_trj.use_FFT = True
    # the trajectories need to have the same length to give same number of energy points and
    # hence np.shape(SQw) so that they can be compared
    SQw_full_trj.calculate_from_MD(trajectory[1:-1],
                                Q_vectors=Q_vectors,
                                dimensions=DIMENSIONS,
                                energy_resolution=E_RESOLUTION)
    SQw_mean = ObservableFactory.create_observable('SQw')
    SQw_mean.use_FFT = True
    SQw_mean.calculate_from_MD(subtrj_list, Q_vectors=Q_vectors, dimensions=DIMENSIONS,
                               energy_resolution=E_RESOLUTION)
    np.testing.assert_allclose(SQw_full_trj.SQw[0], SQw_mean.SQw[0], rtol=3e-2)
    assert np.any(np.not_equal(SQw_full_trj.errors, SQw_mean.errors))
