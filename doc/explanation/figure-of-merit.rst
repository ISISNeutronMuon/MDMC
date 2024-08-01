.. _fom-explanation-label :

Figure of Merit (FoM)
=====================

What is the figure of merit?
----------------------------

When we fit a model to some data, the figure of merit is the value of a `merit function <https://mathworld.wolfram.com/MeritFunction.html>`_, 
which calculates how different the model is from the data. This means we can use it as 
a quantitative assessment of how 'good' a model is; the smaller the figure of merit,
the closer our model is to the data.

MDMC uses the `chi-squared test <https://en.wikipedia.org/wiki/Chi-squared_test>`_ as a merit
function; this is popular for non-linear optimisation.

Indeed, from the use of chi-squared one can note overlap between hypothesis testing and
assessment of merit - in hypothesis testing, our 'model' is the result we'd expect
under the null hypothesis, and our data is the results of our experiment. The
hypothesis test is a quantitative indicator of how different these are.

How does MDMC use the figure of merit?
--------------------------------------

MDMC uses figure of merit as the basis of refinement. We `create a simulation with some
parameters <../how-to/use-MDMC/notebooks/running-a-simulation.ipynb>`_, calculate the
figure of merit of a `dynamical property <../how-to/use-MDMC/notebooks/creating-an-observable.ipynb>`_
between our simulation and experimental data, and use :ref:`minimizer-explanation-label`
to adjust the simulation parameters in order to minimize the figure of merit.

The chi-squared figure of merit test
------------------------------------
In MDMC, the dynamical property is called an `Observable`. A number of datasets can be
given to an optimisation with weights of how important each one is to the refinement.
MDMC calculates the weighted average of the Figure of Merits:

.. math::

    FoM_{total} = \sum_{i}\frac{FoM_{i}}{w_{i}}

Here the weighted Figure of Merit for the :math:`i`-th dataset, :math:`FoM_{i}`, is given by
a sum of the square difference between data points for a single pair of observables, normalised
by the errors and the number of data points, i.e. the reduced chi-squared:

.. math::

    FoM_{i} = \frac{w_{i}}{\nu_{i}} \sum\limits_{j} \frac{(D_{j}^{\textrm{exp}} - D_{j}^{\textrm{sim}})^2}{(\sigma_{j}^{\textrm{exp}})^2}

where the sum is over the :math:`N_{i}` data points in the pair of observables corresponding to
the :math:`i`-th dataset, and the normalisation factor :math:`\nu_{i}` is either :math:`N_{i}`,
:math:`N_{i} - M` where :math:`M` is the number of fitting parameters, or :math:`1`.
:math:`w_{i}` is an importance weighting assigned to the :math:`i`-th dataset. 

The sets :math:`D_{j}^{\textrm{sim}}` and :math:`D_{j}^{\textrm{exp}}` are the individual data points of the 
experimental and simulated ``Observable`` respectively, and :math:`\sigma_{j}^{\textrm{exp}}` corresponds
to the error of the :math:`j`-th data point. Note that the subtraction and division over 
the arrays are element-wise. 

Rescaling
^^^^^^^^^

Note also that if the experimental ``Observable`` is not on an absolute scale,
an additional ``rescale_factor`` can be specified (or automatically determined) in the code
to scale the experimental data points and errors by a simple linear scaling.

If automatically determined, this rescaling is done to the value which minimizes the FoM. If we label
the rescale factor :math:`\lambda` then the minimum of the FoM is obtained as:

.. math::


    FoM_{i}(\lambda) &=& w_{i} \sum\limits_{j} \left(\frac{\lambda*D_{j}^{\textrm{exp}} - \\\\
    D_{j}^{\textrm{sim}}}{\lambda*\sigma_{j}^{\textrm{exp}}\right)^2 \\\\
    \left. \frac{dFoM_{i}}{d\lambda}\right|_{\lambda=\lambda_{\textrm{min}}} &=& 0 \\\\
    \lambda_{\textrm{min}} &=& \frac{A}{B} \\\\

where we have:

.. math::

    A &=& \sum\limits_{j}\left(\frac{D_{j}^{\textrm{sim}}}{\sigma_{j}^{\textrm{exp}}}\right)^2 \\\\
    B &=& \sum\limits_{j} \frac{D_{j}^{\textrm{exp}}*D_{j}^{\textrm{sim}}}{(\sigma_{j}^{\textrm{exp}})^2}
