"""Module for incoherent SQw class"""

import numpy as np

from MDMC.common.atom_properties import B_INCOH
from MDMC.common.mathematics import correlation
from MDMC.trajectory_analysis.observables.obs_factory import ObservableFactory
from MDMC.trajectory_analysis.observables.sqw import AbstractSQw, calculate_rho


@ObservableFactory.register(('IncoherentDynamicStructureFactor',
                             'SQwIncoherent'
                             'SQwIncoh',
                             'SQw_incoh'))
class SQwIncoherent(AbstractSQw):

    """
    A class for containing, calculating and reading the incoherent dynamic
    structure factor
    """

    def _set_weights(self):

        """
        Calculate the neutron weighting for incoherent scattering
        """

        element_weights = {element:B_INCOH[element]**2 for element
                           in self.trajectory.element_set}
        self.weights = [element_weights[atom.element] for atom
                        in self.trajectory.atoms]

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

        n_atoms = len(self.trajectory.atoms)
        FQt_single_Q = np.zeros(len(self.E))

        # Arrange configs so that axes are [atoms, times, positions] i.e.
        # iterating over the first axis is iterating over each atom
        configs = np.swapaxes([config.positions for config in self.trajectory],
                              0,
                              1)
        for atom_positions, weight in zip(configs, self.weights):
            rho_atom = calculate_rho(atom_positions, np.array(Q_vector))
            FQt_single_Q_atom = correlation(rho_atom,
                                            normalise=True)[:len(self.E)]
            FQt_single_Q += FQt_single_Q_atom * weight

        # Normalise to the number of orthogonal vectors
        try:
            norm = np.shape(Q_vector)[0]
        except IndexError:
            norm = 1.

        return FQt_single_Q / (n_atoms * norm)
