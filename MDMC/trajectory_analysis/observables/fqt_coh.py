# MDMC is a package for the optimisation of classical potentials with experimental data
# Copyright (C) 2026 MDMC Developers
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
Module for coherent FQt class.
"""

import numpy as np

from MDMC.common.mathematics import faster_correlation
from MDMC.common.periodictable_objects import create_list_of_element_objects
from MDMC.trajectory_analysis.observables.fqt import AbstractFQt
from MDMC.trajectory_analysis.observables.obs_factory import ObservableFactory


@ObservableFactory.register(
    ("CoherentIntermediateScatteringFunction", "FQtCoherent", "FQtCoh", "FQt_coh"),
)
class FQtCoherent(AbstractFQt):
    """
    Class for processing the intermediate scattering function for coherent dynamic structure factor.
    """

    def _set_weights(self) -> None:
        """Calculate the neutron weighting for coherent scattering."""
        elements_list = create_list_of_element_objects(self._trajectory.element_set)

        self.weights = {
            str(element): element.neutron.b_c if element.neutron.b_c is not None else 0
            for element in elements_list
        }

    def _calculate_FQt_single_Q(self, single_Q_vectors: list) -> np.ndarray:
        # Inherit docstring of abstract method

        n_t = len(self.t)
        elements = self._trajectory.element_set
        FQt_single_Q = np.zeros(n_t)
        rho_element = {}
        n_atoms = 0

        for element in elements:
            # Get the positions of all atoms (the configuration) of each
            # element over time such that ``element_configs`` has time as its
            # first dimension and each atom of ``element`` as its second
            indexes = np.where(np.array(self._trajectory.element_list) == element)
            element_configs = self._trajectory.position[:, indexes[0], :]

            rho_element[element] = self.calculate_rho_config(element_configs, single_Q_vectors)
            n_atoms += np.shape(indexes)[1]

        for element1 in elements:
            for element2 in elements:
                # A sum over the Q vectors is performed within ``correlation``.
                FQt_single_Q += (
                    self.weights[element1]
                    * self.weights[element2]
                    * faster_correlation(rho_element[element1], rho_element[element2])[:n_t]
                )

        # Normalise to the number of orthogonal vectors
        try:
            norm = np.shape(single_Q_vectors)[0]
        except IndexError:
            norm = 1

        return FQt_single_Q / (n_atoms * norm)
