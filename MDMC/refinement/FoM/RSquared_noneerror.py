"""
The class for R-Squared figure of merit calculation with no errors.
"""
import numpy as np

from MDMC.refinement.FoM.FoM_abs import FigureOfMerit, ObservablePair


class RSquared_noneerror(FigureOfMerit):
    r"""
    Calculate the weighted sum of the Figure of Merits for a number of datasets.

    This is done by:

    .. math::

        FoM_{total} = \frac{\sum_{i} FoM_{i}}{\sum_{i} w_{i}}

    Here the weighted Figure of Merit for the :math:`i`-th dataset, :math:`FoM_{i}`, is given by
    the sum of the square difference between data points for a single
    ``ObservablePair``, i.e. the reduced chi-squared:

    .. math::

        FoM_{i} = \frac{w_{i}}{\nu_{i}} \sum_{j} (D_{j}^{exp} - D_{j}^{sim})^2

    where the sum is over the :math:`N_{i}` data points in the ``ObservablePair`` corresponding to
    the :math:`i`-th dataset, and the normalisation factor :math:`\nu_{i}` is either :math:`N_{i}`,
    :math:`N_{i} - M` where :math:`M` is the number of fitting parameters, or :math:`1`.
    :math:`w_{i}` is an importance weighting assigned to the :math:`i`-th dataset. :math:`D_{j}`
    are the individual data points in the 1-D or 2-D array of the experimental ``Observable``
    (:math:`exp`) or simulated ``Observable`` (:math:`sim`). Note that the subtraction and division
    over the arrays are element-wise. Note also that if the experimental ``Observable`` is not on
    an absolute scale, an additional ``rescale_factor`` can be specified (or automatically
    determined) by the ``ObservablePair`` to scale the experimental data points and errors by a
    simple linear scaling.
    """

    def calculate_single_FoM(self, obs_pair: ObservablePair):
        r"""
        Perform the square difference for an ``ObservablePair``.

        If ``obs_pair.auto_scale`` is `True`, then this will also set ``obs_pair.rescale``
        to the value which minimizes the FoM. If we label ``rescale_factor``:math:`=\lambda`
        then the minimum of the FoM is obtained as:

        .. math::


            FoM_{i}(\lambda) &=& w_{i} \sum_{j}
                \left(\lambda*D_{j}^{exp} - D_{j}^{sim}\right)^2 \\\\
            \left. \frac{dFoM_{i}}{d\lambda}\right|_{\lambda=\lambda_{min}} &=& 0 \\\\
            \lambda_{min} &=& \frac{A}{B} \\\\

        where we have:

        .. math::

            A &=& \sum_{j} D_{j}^{exp}*D_{j}^{sim} \\\\
            B &=& \sum_{j}\left(D_{j}^{exp}\right)^2

        Parameters
        ----------
        obs_pair : ObservablePair
            An ``ObservablePair`` for which the FoM is calculated.

        Returns
        -------
        float
            The FoM for the obs_pair.
        """

        # ignore line too long linting as it is necessary for LaTeX formatting
        # pylint: disable=line-too-long
        if obs_pair.auto_scale:
            exp_values = np.array(
                *obs_pair.exp_obs.dependent_variables.values())
            MD_values = np.array(*obs_pair.MD_obs.dependent_variables.values())
            A = np.sum(MD_values * exp_values)
            B = np.sum(exp_values ** 2)
            obs_pair.rescale_factor = A / B

        norm_factor = self.data_norm_factor(obs_pair=obs_pair)
        value_unreduced = np.sum(obs_pair.calculate_difference() ** 2)
        return obs_pair.weight * value_unreduced / norm_factor
