"""Module for coherent SQw class"""

import numpy as np

from MDMC.common.atom_properties import B_COH
from MDMC.common.mathematics import correlation
from MDMC.trajectory_analysis.observables.obs_factory import ObservableFactory
from MDMC.trajectory_analysis.observables.sqw import AbstractSQw, calculate_rho


@ObservableFactory.register(('CoherentDynamicStructureFactor',
                             'SQwCoherent',
                             'SQwCoh',
                             'SQw_coh'))
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

    def _calculate_FQt_single_Q(self, single_Q_vectors):
        # Inherit docstring of abstract method

        n_t = len(self.t)
        elements = self.trajectory.element_set
        FQt_single_Q = np.zeros(n_t)
        rho_element = {}
        n_atoms = 0

        for element in elements:
            # Get the positions of all atoms (the configuration) of each
            # element over time such that ``element_configs`` has time as its
            # first dimension and each atom of ``element`` as its second
            indexes = np.where(np.array(self.trajectory.element_list)
                               == element)
            element_configs = [config.positions[indexes] for config
                               in self.trajectory]

            rho_config = np.zeros((len(element_configs),
                                   len(single_Q_vectors)),
                                  dtype=complex)
            for i, positions in enumerate(element_configs):
                # For each time frame ``i`` calculate the Fourier transformed
                # number density and sum over all positions but preserve the
                # second dimension, our array of Q vectors
                rho_unsummed = calculate_rho(positions,
                                             np.array(single_Q_vectors))
                rho_config[i, :] = np.sum(rho_unsummed, axis=0)

            rho_element[element] = rho_config
            n_atoms += np.shape(indexes)[1]

        for element1 in elements:
            for element2 in elements:
                # A sum over the Q vectors is performed within ``correlation``.
                FQt_single_Q += self.weights[element1] \
                                * self.weights[element2] \
                                * correlation(rho_element[element1],
                                              rho_element[element2],
                                              normalise=True)[:n_t]

        # Normalise to the number of orthogonal vectors
        try:
            norm = np.shape(single_Q_vectors)[0]
        except IndexError:
            norm = 1.

        return FQt_single_Q / (n_atoms * norm)
