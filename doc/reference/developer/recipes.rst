.. _dev_doc_recipes-label:

Recipes for 'simple' additions
==============================

This page will list 'recipes' for how to add simple expansions to MDMC.

New approximation functions
---------------------------

If you would like to add a new approximation function for Control objects (e.g. Gaussian, Lorentzian), MDMC's handling of these functions is built to be expandable. Do the following:

Let `myfunction` be the name of your function.

1. Change the docstrings for the `Control` class in `control.py` and the `AbstractSQw` class to list your function as a resolution function.
2. go to `MDMC/resolution/` and add a file called myfunction.py. This file should define a subclass of Resolution named MyfunctionResolution (note the title case). This has a mandatory method, `apply(self, fqt, t)`, which takes your function and applies it to an FQt array.

If your function is numeric, you should create another method, `window_in_t(self, t)` which calculates the Fourier transform of your function on an array t. `apply()` can then simply be:

.. code-block:: python

    def apply(self, fqt, t):
        N_Q, N_T = np.shape(fqt)
        window = self.window_in_t(t[:N_T])

        return np.broadcast_to(window, (N_Q, N_T)) * fqt

Here, you are done implementation-wise! Factory patterns will handle the actual implementation of your function. However, you should add some tests:

6. Create a new file in the directory `MDMC/tests/system_tests/observables/` and name it `test_SQw_myfunction`. In here, please add tests which validate your new function's calculations against a benchmark (either a similar calculation made in a third-party software, or done by hand).

If your function is not simply numeric (i.e. requires __init__ arguments other than a single numeric value), some tweaking of `MDMC/trajectory_analysis/observables/sqw.py` and `tests/resolution/test_resolution_factory.py` will also be necessary, and your function may take a lot more work.

Now you should be done; if you create a `Control` object with a dataset that has resolution `{'MY_FUNCTION': x}`, it should apply your resolution function to this data.
