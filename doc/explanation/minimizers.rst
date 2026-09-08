.. _minimizer-explanation-label :

Minimizers
==========

What is a minimizer?
--------------------

A minimizer is an `optimisation algorithm or heuristic <https://en.wikipedia.org/wiki/Mathematical_optimization>`_
which takes a set of input parameters and a function of the inputs, and aims to find which
combination of inputs makes the result of the function the smallest (minimal).

Example: Gradient Descent
-------------------------

To visualise a minimizer, picture a hilly terrain. 
The minimizer is trying to find the lowest valley in this terrain (the global minimum). 
For this example, our initial parameters can be pictured as the position of a ball on this terrain. 
Gradient descent does this by rolling the ball down the steepest path it can find, stopping and reassessing (i.e. without momentum) at regular intervals. 
(note here, that the valley we start in may not be the deepest valley in the landscape and we may only find the bottom of the valley we started in, the local minimum, initial parameters are key)
Other minimizers might be pictured as an (or many) automata moving around trying to find the lowest valley; 
many minimizers are based on a concept of a 'walk'. 

The `gradient descent <https://en.wikipedia.org/wiki/Gradient_descent>`_ algorithm
proceeds as follows:

1. Start at some position.
2. Calculate the gradient, or slope, of the current location.
3. Take a step (of some fixed size) in the direction where the slope goes
   downhill the steepest.
4. Repeat from to step 2 until reaching a equilibrium point (a point where the gradient
   is zero in all directions).

Through this, the minimizer is guaranteed to end up at the bottom of *some* valley.
However, if it ends up at the bottom of a valley, it has no way of knowing whether
it's the deepest in all the land, and it is stuck down there. More complicated algorithms
have ways of dealing with this.

Nonetheless, this is an example of minimization; our inputs are the x and y coordinates of the minimizer's
current position, and the output is their altitude at that location.
We call the space of all possible combinations of inputs the "parameter space",
and the output function the "objective function" (objective as in 'goal' or 'target').

Minimization is a huge field of mathematics, and many more sophisticated algorithms exist.
Popular, ubiquitous minimization algorithms include the
`Levenberg-Marquardt algorithm <https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm>`_
or the `BFGS algorithm <https://en.wikipedia.org/wiki/Broyden%E2%80%93Fletcher%E2%80%93Goldfarb%E2%80%93Shanno_algorithm>`_.

How does MDMC use minimization?
-------------------------------

MDMC's parameter space for minimization are the parameters governing the interactions between
molecules in a given simulation. It aims to minimize the :ref:`fom-explanation-label`
between a simulation using those parameters and experimental data. Through this, it finds
the parameters which create a simulation that most closely reproduces the experimental data.

Derivative-free optimisation
----------------------------

Many fast minimization algorithms, such as the ones above, rely on being able to calculate the slope or gradient
of the objective function. However, MDMC's objective function (the figure of merit) is based on
the difference between experimental and simulated data. Both experimental data and molecular dynamics
simulations are noisy, which makes the figure of merit 'not `smooth enough <https://en.wikipedia.org/wiki/Smoothness>`_'
to use many of these algorithms; the idea of a 'slope' does not make sense!

We thus turn to `'derivative-free optimisation' <https://en.wikipedia.org/wiki/Derivative-free_optimization>`_,
which is a subfield of optimisation that avoids needing gradient information. Derivative-free optimisation
is also known as 'black-box optimisation'; the objective function is a 'black-box', where we do not
have some mathematical formula for it.

Covariance Matrix Adaptation Evolution Strategy
-----------------------------------------------

The `CMA-ES package <https://cma-es.github.io/apidocs-pycma/index.html>`_ provides an optimisation
algorithm which generates the parameters within a given standard deviation around the initial values,
and determines their covariance matrix based on the calculated values at each point. This approach
requires a larger number of function evaluations before convergence is reached. At the same time,
it can handle noisy data and functions with local minima.
