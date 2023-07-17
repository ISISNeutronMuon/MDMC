.. _fom-explanation-label :

Figure of Merit (FoM)
=====================

What is the figure of merit?
----------------------------

When we fit a model to some data, the figure of merit is the value of a [merit function](https://mathworld.wolfram.com/MeritFunction.html), 
which calculates how different the model is to the data. This means we can use it as 
a quantitative assessment of how 'good' a model is; the smaller the figure of merit,
the closer our model is to the data.

Popular merit functions are the [Akaike information criterion](https://en.wikipedia.org/wiki/Akaike_information_criterion)
or the [chi-squared test](https://en.wikipedia.org/wiki/Chi-squared_test), which MDMC uses.

As seen with the chi-squared test, one can note overlap between hypothesis testing and
assessment of merit - in hypothesis testing, our 'model' is the result we'd expect
under the null hypothesis, and our data is the results of our experiment. The
hypothesis test is a quantitative indicator of how different these are.

How does MDMC use the figure of merit?
--------------------------------------

MDMC uses figure of merit as the basis of refinement. We [create a simulation with some
parameters](../how-to/use-MDMC/notebooks/running-a-simulation.ipynb), calculate the
figure of merit of a [dynamical property](../how-to/use-MDMC/notebooks/creating-an-observable.ipynb)
between our simulation and experimental data, and use a [minimization algorithm](minimizers.rst)
to adjust the simulation parameters in order to minimize the figure of merit.

The chi-squared figure of merit test
------------------------------------