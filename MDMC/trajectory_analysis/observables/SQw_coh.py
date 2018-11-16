"""Module for coherent SQw class

AUTHOR :    Thomas Farmer        START DATE :    20/07/2018, 16:28:02"""

import numpy as np

from MDMC.common.atom_properties import B_COH
from MDMC.common.mathematics import correlation
from MDMC.trajectory_analysis.observables.SQw import AbstractSQw


class SQwCoherent(AbstractSQw):

    """
    A class for containing, calculating and reading the coherent dynamic
    structure factor
    """
    def _set_weights(self):

        self.weights = {element:B_COH[element] for element
                           in self.trajectory.element_set}

    def _calculate_FQt_single_Q(self, Q_vector):

        rho = self._calculate_rho(Q_vector)

        elements = self.trajectory.element_set
        rho_element = {}
        n_atoms = 0
        for element in elements:
            indexes = np.where(np.array(self.trajectory.element_list)
                               == element)
            rho_element[element] = np.array([np.sum(rho_t[indexes], axis=0)
                                             for rho_t in rho])
            n_atoms += np.shape(indexes)[1]

        FQt_single_Q = np.zeros(len(self.t))
        for element1 in elements:
            for element2 in elements:
                FQt_single_Q += self.weights[element1] \
                                * self.weights[element2] \
                                * correlation(rho_element[element1],
                                              rho_element[element2],
                                              normalise=True)[:len(self.t)]

        # Normalise to the number of orthogonal vectors
        try:
            norm = np.shape(Q_vector)[0]
        except IndexError:
            norm = 1.

        return FQt_single_Q / (n_atoms * norm)
