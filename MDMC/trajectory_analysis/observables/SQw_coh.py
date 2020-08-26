"""Module for coherent SQw class"""

import numpy as np

from MDMC.common.atom_properties import B_COH
from MDMC.common.mathematics import correlation
from MDMC.trajectory_analysis.observables.obs_factory import ObservableFactory
from MDMC.trajectory_analysis.observables.SQw import AbstractSQw, calculate_rho


@ObservableFactory.register(('CoherentDynamicStructureFactor',
                             'SQwCoherent',
                             'SQwCoh'))
class SQwCoherent(AbstractSQw):

    """
    A class for containing, calculating and reading the coherent dynamic
    structure factor
    """
    def _set_weights(self):

        """
        Calculate the neutron weighting for coherent scattering
        """

        self.weights = {element:B_COH[element] for element
                        in self.trajectory.element_set}

    def _calculate_FQt_single_Q(self, Q_vector):

        """
        Calculates the F(Q, t) for a single Q value

        The length of the correlations is bounded by the length of the energies
        rather the times, as this allows energies to be calculated from
        trajectories with longer timescales than is required by the energy
        resolution.

        Parameters
        ----------
        Q_vector : numpy.ndarray
            An ``array`` of one or more Q vectors with the same Q value

        Returns
        -------
        numpy.ndarray
            An ``array`` with dimensions of ``self.t``
        """

        elements = self.trajectory.element_set
        rho_element = {}
        n_atoms = 0
        for element in elements:
            indexes = np.where(np.array(self.trajectory.element_list)
                               == element)
            element_configs = [config.positions[indexes] for config
                               in self.trajectory]
            rho_config = np.zeros((len(element_configs), len(Q_vector)),
                                  dtype=complex)
            for i, positions in enumerate(element_configs):
                rho_config[i, :] = np.sum(calculate_rho(positions,
                                                        np.array(Q_vector)),
                                          axis=0)
            rho_element[element] = rho_config
            n_atoms += np.shape(indexes)[1]

        FQt_single_Q = np.zeros(len(self.E))
        for element1 in elements:
            for element2 in elements:
                FQt_single_Q += self.weights[element1] \
                                * self.weights[element2] \
                                * correlation(rho_element[element1],
                                              rho_element[element2],
                                              normalise=True)[:len(self.E)]

        # Normalise to the number of orthogonal vectors
        try:
            norm = np.shape(Q_vector)[0]
        except IndexError:
            norm = 1.

        return FQt_single_Q / (n_atoms * norm)
