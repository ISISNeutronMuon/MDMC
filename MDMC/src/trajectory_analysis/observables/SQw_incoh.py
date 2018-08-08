"""Module for incoherent SQw class

AUTHOR :    Thomas Farmer        START DATE :    20/07/2018, 16:28:02"""

import numpy as np

from MDMC.src.common.atom_properties import B_INCOH
from MDMC.src.common.mathematics import correlation
from MDMC.src.trajectory_analysis.observables.SQw import AbstractSQw


class SQwIncoherent(AbstractSQw):

    """
    A class for containing, calculating and reading the incoherent dynamic
    structure factor
    """

    def _set_weights(self):

        element_weights = {element:B_INCOH[element]**2 for element
                           in self.trajectory.element_set}
        self.weights = [element_weights[atom.element] for atom
                        in self.trajectory.atoms]

    def _calculate_FQt_single_Q(self, Q_vector):

        n_atoms = len(self.trajectory.atoms)

        # Iterate over all atoms and over all trajectories to get atom positions
        FQt_single_Q = np.zeros(len(self.trajectory.times))
        for i in np.arange(n_atoms):
            atom_positions = [conf.atom_positions[i] for conf
                              in self.trajectory.configurations]


            rho = self._calculate_rho(atom_positions, Q_vector)
            FQt_single_Q_atom = correlation(rho, normalise=True)
            FQt_single_Q += FQt_single_Q_atom * self.weights[i]

        # Normalise to the number of orthogonal vectors
        try:
            norm = np.shape(Q_vector)[0]
        except IndexError:
            norm = 1.

        return FQt_single_Q / (n_atoms * norm)

    def _calculate_rho(self, positions, Q_vector):

        """
        Calculates time dependent number density in reciprocal space for all Q
        vectors

        As rho is the sum of the contributions for all of the specified Q
        vectors, these Q vectors should have the same Q value. rho is calculated
        for only a single atom.

        Arguments:
        Q_vector: Either a single Q vector or three orthogonal Q vectors
        """

        rho = [self._rho(r, Q_vector) for r in positions]

        return np.array(rho)
