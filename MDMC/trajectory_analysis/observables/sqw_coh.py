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

        r"""
        Calculates the F(Q, t) from an array of vectors corresponding to a
        single value of Q.

        The length of the correlations is bounded by the length of the
        ``self.E + 1`` rather than ``self.t``, as this allows energies to be
        calculated from trajectories with longer timescales than is required by
        the energy resolution.

        We start by calculating the Fourier transformed number densities for
        the atoms :math:`j` of element :math:`\alpha`:

        .. math::

            \rho_{\alpha, m_Q}(n_t, p) = \sum_{j \in \alpha} e^{-i \vec{q_{p}} \cdot \vec{r_j}}

        Where :math:`n_t` indexes time and :math:`p` indexes momentum vector.
        F(Q,t) is calculated from the correlation :math:`C` between these
        number densities, where we make use of the correlation theorem of
        discrete periodic functions to speed up calculation using the FFT
        [see E.O. Brigham, The Fast Fourier Transform, 1974]:

        .. math::

            C_{\alpha,\beta,m_Q}(n_t, p) = \Re \Big[\frac{1}{N_E - |n_t|} \mathcal{F}_t^{-1} \big[ \tilde{\rho'}^*_{\alpha, m_Q}(n_E, p) \tilde{\rho'}_{\beta, m_Q}(n_E, p) \big] \Big]

        Where we denote the Fourier transform of :math:`\rho` as:

        .. math::

            \tilde{\rho'}_{\alpha, m_Q}(n_E, p) = \mathcal{F}_t \big[ \rho'_{\alpha, m_Q}(n_t, p) \big]

        Where :math:`\rho'` denotes that :math:`\rho` has been padded with
        zeros in the time domain to give it twice its orginal length.

        For the coherent contribution, we calculate:

        .. math::

            F_{m_Q}^{coh}(n_t) = \sum_{\alpha} \sum_{\beta} B^{coh}_\alpha B^{coh}_\beta \sum_p C_{\alpha,\beta, m_Q}(n_t, p)

        The ideal (not accounting for instrument resolution) scattering
        function is normalised by both the number of atoms that contributed and
        the number of Q vectors used for the single value of Q:

        .. math::

            F_{m_Q}^{ideal}(n_t) = \frac{ F_{m_Q}^{coh}(n_t) }{N_{atoms} N_p}

        Parameters
        ----------
        single_Q_vectors : numpy.ndarray
            An array of Q vectors with approximately the same magnitude

        Returns
        -------
        numpy.ndarray
            An ``array`` with length ``len(self.E) + 1``
        """

        n_t = self.maximum_frames
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
