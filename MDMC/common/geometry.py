"""A module with geometric objects and operations

AUTHOR :    Thomas Farmer        START DATE :    2018-5-29 18:32:39"""

import numpy as np

# TODO: Determine if it is better to set up all positions/velocities as Vectors or leave them as np arrays
# (better performance with np arrays even when both use linalg function)

class Vector(np.ndarray):

    """
    A numpy array with additional vector operations
    """

    def __new__(cls, x, dtype=None):
        return np.array(x).view(cls)

    def distance(self, vector):
        return np.linalg.norm(self-vector)

    def displacement(self, vector):
        return self - vector
