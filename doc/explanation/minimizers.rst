.. _minimizer-explanation-label :

Minimizers
==========

What is a minimizer?
--------------------

A minimizer is an `optimisation algorithm or heuristic <https://en.wikipedia.org/wiki/Mathematical_optimization>`_
which takes a set of input parameters and an output function of the inputs, and aims to find which
combination of inputs makes the output function smallest.

Example: Mr. Minimizer
----------------------

As a very simple example, imagine an explorer, 'Mr. Minimizer', has parachuted down into uncharted land.
His job is to find the deepest valley in a large area of the land, and he has been given a set
of instructions on how to find it:

1. Take a step in a random direction.
2. If that step is uphill, go back to where you were before you took the step, and
   go to step 1.
3. If the step is downhill, stay there and go to step 1.

Through this, Mr. Minimizer is guaranteed to end up at the bottom of *some* valley.
However, if he ends up at the bottom of a valley, he has no way of knowing whether
it's the deepest in all the land, and he is stuck down there. More complicated algorithms
have ways of dealing with this (in fact, this is what the Metropolis-Hastings algorithm -
explained below - does!) Nonetheless, this is an example of a simple minimization heuristic; our
inputs are the x and y coordinates of Mr. Minimizer's position, and the output is
his altitude at that location. We call the space of all possible combinations of inputs
the "parameter space", and the output function the "objective function" (objective
as in 'goal' or 'target').

Minimization is a huge field of mathematics, and many more sophisticated algorithms exist. 

How does MDMC use minimization?
-------------------------------

MDMC's parameter space for minimization are the parameters governing the forces between
molecules in a given simulation, and it then aims to minimize the :ref:`fom-explanation-label`
between a simulation using those parameters and experimental data. Through this, it finds
the parameters which create a simulation that most closely resembles the experimental data.

Derivative-free optimisation
----------------------------

There are a variety of popular, ubiquitous minimization algorithms, such as the
`Levenberg-Marquardt algorithm <https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm>`_
or the `BFGS algorithm <https://en.wikipedia.org/wiki/Broyden%E2%80%93Fletcher%E2%80%93Goldfarb%E2%80%93Shanno_algorithm>`_. 
However, many of these do not solve the minimization problem that
MDMC aims to solve. Many fast algorithms rely on being able to calculate the slope or
gradient of the objective function - in MDMC's case, the figure of merit,
which is based on the match-up between experimental and simulated data. Experimental data
is noisy, which simply makes the figure of merit function 'not 
`smooth enough <https://en.wikipedia.org/wiki/Smoothness>`_' to use many of these algorithms.

We thus turn to `'derivative-free optimisation' <https://en.wikipedia.org/wiki/Derivative-free_optimization>`_,
which is a subfield of optimisation that avoids needing gradient information. We will now detail
the minimizers available in MDMC.

Metropolis-Hastings algorithm
-----------------------------
The Metropolis-Hastings algorithm (in MDMC, this is called `MMC`, for 'Metropolis Monte Carlo') 
is a 'random walk' Monte Carlo algorithm; in essence, a far more sophisticated version 
of the instructions given to Mr. Minimizer in the first section. It uses a more complicated
method for deciding whether or not to backtrack that avoids it getting stuck in small,
'local' valleys.

MMC starts at an initial point in parameter space and proposes a random direction to take a step in;
this proposed point to step to is called the "candidate". It then accepts or rejects the candidate,
i.e. decides whether or not to *take* that step, at random. The probability of acceptance
is based on the 'acceptance ratio', whether the objective function at the candidate inputs is
higher or lower than the current point. This means that unlike our earlier Mr. Minimizer, this
algorithm is willing to 'walk uphill', particularly up shallow hills.

MMC is a robust algorithm - it is *guaranteed* to eventually find the minimum point in the
entire space. However, it is quite slow as it can reject steps; when a step is rejected, the
time used to simulate the function and calculate the figure of merit is essentially wasted.
Furthermore, if it starts very far away from the minimum, it can only take finite-sized steps.
This means it might take a long time to randomly wander over to the vague region of the
minimum if the initial 'guess' of the parameters is not very good.

Gaussian Process Regression
---------------------------
The Gaussian Process Regression (GPR) algorithm aims to 'map out' parameter space. It first
creates a grid of values in parameter space and calculates the objective function at each
of these points. It uses these values to 'fit' an approximate topography to the space,
and then predicts the values inbetween by interpolation to predict where the lowest
point is.

The MDMC GPR algorithm creates the grid of values via `'Latin hypercube sampling' <https://en.wikipedia.org/wiki/Latin_hypercube>`_. 
If we wanted to take a sample size of 4 from a 2D space, a 'Latin square sample' would
divide the space into a grid of 4 rows and 4 columns, and then take 4 samples
such that none of the samples are on the same row or column; imagine a 4x4 chess grid
where we have placed 4 rooks in such a way that none of them can capture each other.
This ensures our samples are random, but still more-or-less evenly distributed. A *hypercube* is
the term for the equivalent of a cube in any number of dimensions (e.g. 2D hypercube is a square,
3D hypercube is a cube, so on), so a *Latin* hypercube is the same concept in any number of dimensions 
(for MDMC, as many dimensions as there are parameters).

This method can be extremely effective, as it quickly produces an accurate prediction
without needing an initial 'guess'. However, it can be more expensive computationally
than Metropolis-Hastings, as it has to take samples over a region around the initial
'guess' which may or may not be useful in finding the minimum, especially if the region is very large.
It is also not mathematically guaranteed to find the exact minimum (but will usually be very close!)

Gaussian Process Optimisation
-----------------------------
Gaussian Process Optimisation (GPO) combines 'exploration' of the space as in
the Gaussian Process Regression algorithm, as well as 'exploitation' of
minima, 'valleys' in the parameter space.