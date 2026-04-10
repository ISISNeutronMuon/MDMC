"""
Module for incoherent FQt class.
"""

import numpy as np

from MDMC.common.mathematics import faster_autocorrelation
from MDMC.common.periodictable_objects import create_list_of_element_objects
from MDMC.trajectory_analysis.observables.concurrency_tools import core_batch, create_executor
from MDMC.trajectory_analysis.observables.fqt import (
    AbstractFQt,
    calc_incoherent_scatt_length,
    calculate_rho,
)
from MDMC.trajectory_analysis.observables.obs_factory import ObservableFactory


@ObservableFactory.register(('IncoherentIntermediateScatteringFunction',
                             'FQtIncoherent'
                             'FQtIncoh',
                             'FQt_incoh'))
class FQtIncoherent(AbstractFQt):
    """
    Class for processing intermediate scattering function for incoherent dynamic structure factor.
    """

    def _set_weights(self) -> None:
        """Calculate the neutron weighting for incoherent scattering."""
        elements_list = create_list_of_element_objects(self._trajectory.element_set)

        element_weights = {str(element):  element.neutron.b_c_i**2\
                           if element.neutron.b_c_i is not None \
                           else calc_incoherent_scatt_length(str(element))**2 for element
                           in elements_list}
        self.weights = [element_weights[str(atom.element)] for atom
                        in [self._trajectory.exportAtom(atom_number=x) for x
                        in range(self._trajectory.n_atoms)]]

    def _calculate_FQt_single_Q(self, single_Q_vectors: list) -> np.ndarray:
        # Inherit docstring of abstract method

        n_t = len(self.t)
        n_atoms = self._trajectory.n_atoms
        FQt_single_Q = np.zeros(n_t)
        weight = self.weights
        executor = create_executor()

        configs = np.swapaxes(self._trajectory.position, 1, 2)
        configs = np.swapaxes(configs, 0, 2)
        rho_all = calculate_rho(configs, np.array(single_Q_vectors))
        futures = core_batch(executor.submit(faster_autocorrelation,
                                             rho.T,
                                             weights=np.array(weight))
                             for rho in rho_all)

        for future_batch in futures:
            results = [future.result() for future in future_batch]
            for result in results:
                FQt_single_Q += result

        # Normalise to the number of orthogonal vectors
        try:
            norm = np.shape(single_Q_vectors)[0]
        except IndexError:
            norm = 1

        return FQt_single_Q / (n_atoms * norm)
