"""Module for coherent FQt class"""

import numpy as np

from MDMC.common.atom_properties import B_COH
from MDMC.common.mathematics import faster_correlation
from MDMC.trajectory_analysis.observables.fqt import AbstractFQt, calculate_rho
from MDMC.trajectory_analysis.observables.obs import executor
from MDMC.trajectory_analysis.observables.obs_factory import ObservableFactory


@ObservableFactory.register(('CoherentIntermediateScatteringFunction',
                             'FQtCoherent',
                             'FQtCoh',
                             'FQt_coh'))
class FQtCoherent(AbstractFQt):
    """
    A class for containing, calculating and reading the intermediate scattering
    function for the coherent dynamic structure factor
    """

    def _set_weights(self) -> None:
        """Calculate the neutron weighting for coherent scattering"""

        self.weights = {element: B_COH[element] for element
                        in self._trajectory.element_set}

    def _calculate_FQt_single_Q(self, single_Q_vectors: 'np.ndarray') -> 'np.ndarray':
        # Inherit docstring of abstract method

        n_t = len(self.t)
        elements = self._trajectory.element_set
        FQt_single_Q = np.zeros(n_t)
        rho_element = {}
        n_atoms = 0

        def helper_coherent(configs: np.ndarray, q_vector: np.ndarray):
            """A wrapper for the calculate_rho function and the summation
            of the resulting array. This part of the calculation is handled
            by numpy, and so it is easy to run in parallel.

            Arguments
            ---------
            configs: numpy.ndarray
                array of atom positions, size (N_timesteps, 3, N_atoms)
            q_vector: numpy.ndarray
                q vector in array form, size (3)

            Returns:
                array of rho values summed over the atoms in the system,
                size (N_timesteps)
            """
            return calculate_rho(configs, q_vector).sum(axis = 1)

        for element in elements:
            # Get the positions of all atoms (the configuration) of each
            # element over time such that ``element_configs`` has time as its
            # first dimension and each atom of ``element`` as its second
            indexes = np.where(np.array(self._trajectory.element_list)
                               == element)
            element_configs = self._trajectory.position[:, indexes[0], :]
            rho_config = np.zeros((len(element_configs),
                                   len(single_Q_vectors)),
                                  dtype=complex)

            # For the np.dot product to be broadcast correctly,
            # the [x, y, z] atom positions have to be on axis 1.
            # For this reason we swap the axes, moving the
            # axis of atom numbers to axis 2.
            # Time axis is still axis 0.
            configs = np.swapaxes(element_configs, 1, 2)
            # The single_Q_vectors array contains many q vectors
            # with similar values of |Q|.
            # The following lines split the calculation by multiplying
            # the trajectory by each q vector separately.
            futures = [executor.submit(helper_coherent,
                                       configs, single_Q_vectors[q_num])
                                       for q_num in range(len(single_Q_vectors))]
            # The following line makes the code wait for all the calculations to finish.
            results = [future.result() for future in futures]
            # At this stage, the results list is fully populated,
            # and the following loop writes the results into the rho_config array.
            for q_num in range(len(single_Q_vectors)):
                rho_config[:, q_num] = results[q_num]

            rho_element[element] = rho_config
            n_atoms += np.shape(indexes)[1]

        for element1 in elements:
            for element2 in elements:
                # A sum over the Q vectors is performed within ``correlation``.
                FQt_single_Q += self.weights[element1] \
                    * self.weights[element2] \
                    * faster_correlation(rho_element[element1],
                                  rho_element[element2])[:n_t]

        # Normalise to the number of orthogonal vectors
        try:
            norm = np.shape(single_Q_vectors)[0]
        except IndexError:
            norm = 1

        return FQt_single_Q / (n_atoms * norm)
