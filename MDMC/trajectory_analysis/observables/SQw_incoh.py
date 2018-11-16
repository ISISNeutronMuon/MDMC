"""Module for incoherent SQw class

AUTHOR :    Thomas Farmer        START DATE :    20/07/2018, 16:28:02"""

import numpy as np

from MDMC.common.atom_properties import B_INCOH
from MDMC.common.mathematics import correlation
from MDMC.trajectory_analysis.observables.SQw import AbstractSQw


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

        """
        The length of the correlations is bounded by the length of the energies
        rather the times, as this allows energies to be calculated from
        trajectories with longer timescales than is required by the energy
        resolution.
        """

        n_atoms = len(self.trajectory.atoms)
        rho = self._calculate_rho(Q_vector)

        # Iterate over rho and autocorrelate for each atom
        FQt_single_Q = np.zeros(len(self.E))
        for i in np.arange(n_atoms):
            rho_atom = [rho_t[i] for rho_t in rho]
            FQt_single_Q_atom = correlation(rho_atom,
                                            normalise=True)[:len(self.E)]
            FQt_single_Q += FQt_single_Q_atom * self.weights[i]

        # Normalise to the number of orthogonal vectors
        try:
            norm = np.shape(Q_vector)[0]
        except IndexError:
            norm = 1.

        return FQt_single_Q / (n_atoms * norm)
