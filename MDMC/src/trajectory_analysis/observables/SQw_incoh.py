"""Module for incoherent SQw class

AUTHOR :    Thomas Farmer        START DATE :    20/07/2018, 16:28:02"""

import numpy as np

from MDMC.src.common.mathematics import correlation
from MDMC.src.trajectory_analysis.observables.SQw import AbstractSQw


class SQwIncoherent(AbstractSQw):

    """
    A class for containing, calculating and reading the incoherent dynamic
    structure factor
    """

    def _calculate_FQt_single_Q(self, Q_vector):

        n_atoms = len(self.trajectory.atoms)
        # Iterate over all atoms and over all trajectories to get atom positions
        for i in np.arange(n_atoms):
            for j in np.arange(len(self.trajectory.times)):
                atom_positions = self.trajectory.configurations[j] \
                                     .atom_positions[i]
                rho = self._calculate_rho(atom_positions, Q_vector)
                FQt_single_Q_atom = correlation(rho, normalise=True)

    def _calculate_rho(self, r, Q_vector):

        """
        Calculates time dependent number density in reciprocal space for all Q
        vectors

        As rho is the sum of the contributions for all of the specified Q
        vectors, these Q vectors should have the same Q value. rho is calculated
        for only a single atom.

        Arguments:
        Q_vector: Either a single Q vector or three orthogonal Q vectors
        """

        rho = []
        for time in self.trajectory.times:
            rho.append(np.exp(-1j * np.dot(Q_vector, r)))
