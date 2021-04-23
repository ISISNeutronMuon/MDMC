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

    def _calculate_FQt_single_Q(self, single_Q_vectors):

        r"""
        Calculates the incoherent F(Q, t) from an array of vectors
        corresponding to a single value of Q.

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

        For the incoherent contribution, we calculate:

        .. math::

            F_{m_Q}^{inc}(n_t) = \sum_{\alpha} \big( B^{inc}_\alpha \big) ^2 \sum_p C_{\alpha,\alpha, m_Q}(n_t, p)

        The ideal (not accounting for instrument resolution) scattering
        function is normalised by both the number of atoms that contributed and
        the number of Q vectors used for the single value of Q:

        .. math::

            F_{m_Q}^{ideal}(n_t) = \frac{ F_{m_Q}^{inc}(n_t) }{ N_{atoms} N_p }

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
        n_atoms = len(self.trajectory.atoms)
        FQt_single_Q = np.zeros(n_t)

        # Arrange configs so that axes are [atoms, times, positions] i.e.
        # iterating over the first axis is iterating over each atom
        configs = np.swapaxes([config.positions for config in self.trajectory],
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
