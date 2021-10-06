.. _dev_doc_recipes-label:

Recipes for 'simple' additions
==============================

This page will list 'recipes' for how to add simple expansions to MDMC.

New approximation functions
---------------------------

If you would like to add a new approximation function for Control objects (e.g. Gaussian, Lorentzian), MDMC's handling of these functions is built to be expandable. Do the following:

Let MY_FUNCTION be the name of your function.

1. Change the docstrings for the `Control` class in `control.py` and the `AbstractSQw` class to list your function as a resolution function.
2. open `MDMC/common/resolution_functions.py` and add your function.
3. open `MDMC/trajectory_analysis/sqw_resolution_windows/resolution_windows.py` and add a function named MY_FUNCTION_window, which is the mathematics used to create the Fourier transform of your function on a Numpy array. Note you can use MY_FUNCTION() from resolution_functions.py if needed.

Here, you are done implementation-wise! Factory patterns will handle the actual implementation of your function. However, you should add some tests:

5. Open `tests/trajectory_analysis/test_resolution_window_factory.py` and add your function to the @pytest.mark.parametrize decorator of `test_resolution_window_factory`; do this by adding a tuple to the list, where the tuple is ("MY_FUNCTION", MY_FUNCTION_window).
6. Create a new file in the directory `MDMC/tests/system_tests/observables/` and name it `test_SQw_MY_FUNCTION`. In here, please add tests which validate your new function's calculations against a benchmark (either a similar calculation made in a third-party software, or done by hand).

Now you should be done; if you create a `Control` object with a dataset that has resolution `{'MY_FUNCTION': x}`, it should apply your resolution function to this data.
