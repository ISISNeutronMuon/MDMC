"""Module for incoherent FQt class"""

import numpy as np

from MDMC.common.atom_properties import B_INCOH
from MDMC.common.mathematics import faster_autocorrelation
from MDMC.trajectory_analysis.observables.fqt import AbstractFQt, calculate_rho
from MDMC.trajectory_analysis.observables.obs import executor
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

    def _set_weights(self) -> None:
        """Calculate the neutron weighting for incoherent scattering"""

        element_weights = {element: B_INCOH[element]**2 for element
                           in self._trajectory.element_set}
        self.weights = [element_weights[atom.element] for atom
                        in [self._trajectory.exportAtom(atom_number=x) for x
                            in range(self._trajectory.n_atoms)]]

    def _calculate_FQt_single_Q(self, single_Q_vectors: list) -> 'np.ndarray':
        # Inherit docstring of abstract method

        n_t = len(self.t)
        n_atoms = self._trajectory.n_atoms
        FQt_single_Q = np.zeros(n_t)
        weight = self.weights

        configs = np.swapaxes(self._trajectory.position,
                              1,
                              2)
        configs = np.swapaxes(configs,
                              0,
                              2)
        rho_all = calculate_rho(configs, np.array(single_Q_vectors))
        futures = (executor.submit(faster_autocorrelation,
                                    rho.T,
                                    weights = np.array(weight))
                                    for rho in rho_all)
        results = [future.result()[:n_t] for future in futures]
        for q_num in range(len(single_Q_vectors)):
            FQt_single_Q += results[q_num]

        # Normalise to the number of orthogonal vectors
        try:
            norm = np.shape(single_Q_vectors)[0]
        except IndexError:
            norm = 1

        return FQt_single_Q / (n_atoms * norm)
