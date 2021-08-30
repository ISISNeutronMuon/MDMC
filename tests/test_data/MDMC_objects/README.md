To load `trajectory.zip` refer the test fixture `trajectory()` in `tests\system_tests\observables\data_manager.py`:

```
try:
    import cPickle as pickle
except ImportError:
    import pickle
import zlib

import pytest

from tests.test_data import data


@pytest.fixture(scope="session")
def trajectory():

    """
    Returns
    -------
    Trajectory
        A 50 configuration trajectory (from a 50000 step simulation) of 2048
        water molecules
    """

    # Unzip and unpickle the trajectory
    compressed_trajectory = open(data.OBJECT_DATA['trajectory'], 'rb').read()
    pickled_trajectory = zlib.decompress(compressed_trajectory)
    return pickle.loads(pickled_trajectory, encoding='latin-1')
```

Corresponding methods should be used to save new `Trajectory` objects if needed, i.e. replacing `decompress` with `compress` and `loads` with `dumps`.
