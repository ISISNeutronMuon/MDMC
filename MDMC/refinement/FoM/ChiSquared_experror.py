"""The class for Chi Squared figure of merit calculation with errors"""
import numpy as np

from MDMC.refinement.FoM.FoM_abs import FigureOfMerit, ObservablePair


class ChiSquaredExpError(FigureOfMerit):
    # ignore line too long linting as it is necessary for LaTeX formatting
    # pylint: disable=line-too-long
    r"""
    Calculates the weighted sum of the Figure of Merits for a number of datasets:

    .. math::

        FoM_{total} = \frac{\sum_{i} FoM_{i}}{\sum_{i} w_{i}}

    Here the weighted Figure of Merit for the :math:`i`-th dataset, :math:`FoM_{i}`, is given by
    a sum of the square difference between data points for a single ``ObservablePair``, normalised
    by the errors and the number of data points, i.e. the reduced chi-squared:

    .. math::

        FoM_{i} = \frac{w_{i}}{\nu_{i}} \sum_{j} \frac{(D_{j}^{exp} - D_{j}^{sim})^2}{(\sigma_{j}^{exp})^2}

    where the sum is over the :math:`N_{i}` data points in the ``ObservablePair`` corresponding to
    the :math:`i`-th dataset, and the normalisation factor :math:`\nu_{i}` is either :math:`N_{i}`,
    :math:`N_{i} - M` where :math:`M` is the number of fitting parameters, or :math:`1`.
    :math:`w_{i}` is an importance weighting assigned to the :math:`i`-th dataset. :math:`D_{j}`
    are the individual data points in the 1-D or 2-D array of the experimental ``Observable``
    (:math:`exp`) or simulated ``Observable`` (:math:`sim`), and :math:`\sigma_{j}^{exp}` are the
    elements in a 1-D or 2-D array corresponding to the error of the :math:`j`-th data point. Note
    that the subtraction and division over the arrays are element-wise. Note also that if the
    experimental ``Observable`` is not on an absolute scale, an additional ``rescale_factor`` can be
    specified (or automatically determined) by the ``ObservablePair`` to scale the experimental
    data points and errors by a simple linear scaling.
    """

    def calculate_single_FoM(self, obs_pair: ObservablePair):
        # ignore line too long linting as it is necessary for LaTeX formatting
        # pylint: disable=line-too-long
        r"""
        Performs the error normalised square difference for an ``ObservablePair``. If
        ``obs_pair.auto_scale`` is `True`, then this will also set ``obs_pair.rescale`` to the
        value which minimizes the FoM. If we label ``rescale_factor``:math:`=\lambda` then the
        minimum of the FoM is obtained as:

        .. math::


            FoM_{i}(\lambda) &=& w_{i} \sum_{j} \left(\frac{\lambda*D_{j}^{exp} - \\\\
            D_{j}^{sim}}{\lambda*\sigma_{j}^{exp}}\right)^2 \\\\
            \left. \frac{dFoM_{i}}{d\lambda}\right|_{\lambda=\lambda_{min}} &=& 0 \\\\
            \lambda_{min} &=& \frac{A}{B} \\\\

        where we have:

        .. math::

            A &=& \sum_{j}\left(\frac{D_{j}^{sim}}{\sigma_{j}^{exp}}\right)^2 \\\\
            B &=& \sum_{j} \frac{D_{j}^{exp}*D_{j}^{sim}}{(\sigma_{j}^{exp})^2}

        Parameters
        ----------
        obs_pair : ObservablePair
            An ``ObservablePair`` for which the FoM is calculated

        Returns
        -------
        float
            The FoM for the obs_pair
        """

        if obs_pair.auto_scale:
            exp_errors = np.array(*obs_pair.exp_obs.errors.values())
            exp_values = np.array(
                *obs_pair.exp_obs.dependent_variables.values())
            MD_values = np.array(*obs_pair.MD_obs.dependent_variables.values())
            obs_pair.rescale_factor = (np.sum((MD_values / exp_errors) ** 2) / np.sum(
                MD_values * exp_values / exp_errors ** 2))
            print(obs_pair.rescale_factor)

        
        norm_factor = self.data_norm_factor(obs_pair=obs_pair)
        value_unreduced = np.sum((obs_pair.calculate_difference()
                                  / obs_pair.calculate_exp_errors()) ** 2)
        print(value_unreduced, norm_factor, obs_pair.weight)
        return obs_pair.weight * value_unreduced / norm_factor
