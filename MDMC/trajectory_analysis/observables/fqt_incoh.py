"""Module for incoherent FQt class"""

import numpy as np

from MDMC.common.atom_properties import B_INCOH
from MDMC.common.mathematics import correlation
from MDMC.trajectory_analysis.observables.fqt import AbstractFQt, calculate_rho
from MDMC.trajectory_analysis.observables.obs_factory import ObservableFactory


@ObservableFactory.register(('IncoherentIntermediateScatteringFunction',
                             'FQtIncoherent'
                             'FQtIncoh',
                             'FQt_incoh'))
class FQtIncoherent(AbstractFQt):
    """
    A class for containing, calculating and reading the intermediate scattering
    function for the incoherent dynamic structure factor
    """

    def _set_weights(self):
        """
        Calculate the neutron weighting for incoherent scattering
        """

        element_weights = {element: B_INCOH[element]**2 for element
                           in self._trajectory.element_set}
        self.weights = [element_weights[atom.element] for atom
                        in self._trajectory.atoms]

    def _calculate_FQt_single_Q(self, single_Q_vectors):
        # Inherit docstring of abstract method

        n_t = len(self.t)
        n_atoms = len(self._trajectory.atoms)
        FQt_single_Q = np.zeros(n_t)

        # Arrange configs so that axes are [atoms, times, positions] i.e.
        # iterating over the first axis is iterating over each atom
        configs = np.swapaxes([config.positions for config in self._trajectory],
                              0,
                              1)
        for atom_positions, weight in zip(configs, self.weights):
            rho_atom = calculate_rho(atom_positions,
                                     np.array(single_Q_vectors))

            # A sum over the Q vectors is performed within ``correlation``.
            FQt_single_Q_atom = correlation(rho_atom,
                                            normalise=True)[:n_t]
            FQt_single_Q += FQt_single_Q_atom * weight

        # Normalise to the number of orthogonal vectors
        try:
            norm = np.shape(single_Q_vectors)[0]
        except IndexError:
            norm = 1.

        return FQt_single_Q / (n_atoms * norm)
