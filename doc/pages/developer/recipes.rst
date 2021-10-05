.. _dev_doc_recipes-label:

Recipes for 'simple' additions
==============================

This page will list 'recipes' for how to add simple expansions to MDMC.

New approximation functions
---------------------------

If you would like to add a new approximation function for Control objects (e.g. Gaussian, Lorentzian), MDMC's handling of these functions is built to be expandable. Do the following:

1. Change the docstrings for the `Control` class in `control.py` and the `AbstractSQw` class to list your function.
2. open `MDMC/common/resolution_functions.py` and add your function.
3. Open sqw.py, add your function to the list imported from `MDMC.common.resolution_functions`, then find the following block of code:

.. code-block:: python

  if 'energy_resolution' in settings:
  	if type(settings['energy_resolution']) in [float, int]:  # if a number, assume Gaussian and convert to dict
  		warnings.warn("Assuming energy resolution is Gaussian. To change this,"
  		" input energy resolution as {'function': 'value'}, where"
  		" 'function' is your desired resolution approximation function.", SyntaxWarning)
  		settings['energy_resolution'] = {'gaussian': settings['energy_resolution']}
  	# process energy resolution and function type
  	if type(settings['energy_resolution']) == dict:
  		if 'gaussian' in settings['energy_resolution']:
  			# Convert the user friendly ueV into preferred system unit of meV
  			self.e_res = settings['energy_resolution']['gaussian'] / 1000
  			self.approximation_function = gaussian
                    
etc. See the bottom of this code block (starting with Gaussian) is an if loop listing the current accepted approximation functions. In this loop (after the last `elif`, before the `else`), add the following, where MY_FUNCTION is the function you added to `resolution_functions.py`:

.. code-block:: python

  if 'MY_FUNCTION' in settings['energy_resolution']:
  	self.e_res = settings['energy_resolution']['MY_FUNCTION'] / 1000
  	self.approximation_function = MY_FUNCTION
  	
4. After this, there should be an `else` subroutine which raises a `NameError`, with a list of accepted functions. Add `'MY_FUNCTION'` to this list.
5. Scroll down and find the following block of code:

 .. code-block:: python
 
  resolution_function = self.resolution_functions.get('SQw')
  if resolution_function is not None:
  	window = self._calculate_resolution_window(resolution_function)
  elif self.e_res is not None:
  	if function == gaussian:
  
etc. After this block (starting with Gaussian) is an if loop going through the list of functions. To the bottom of this list (before the `else`) add the following, again where MY_FUNCTION is your function:

.. code-block:: python

  if function == MY_FUNCTION
  [MATH GOES HERE]
  
where [MATH GOES HERE] is the mathematics used to apply the Fourier transform of MY_FUNCTION to an array.

5. Open `test_control.py`, and find the test `test_control_refine_other_functions`. The line above it should be a `@pytest.mark.parametrize' decorator; its second parameter should be a list of function names starting with `'lorentzian'`. Add `'MY_FUNCTION'` to this list. 

6. Create a new file in the directory `MDMC/tests/system_tests/observables/` and name it `test_SQw_MY_FUNCTION`. In here, please add tests which validate your new function's calculations against a benchmark (either a similar calculation made in a third-party software, or done by hand). 

Now you should be done; if you create a `Control` object with a dataset that has resolution `{'MY_FUNCTION': x}`, it should apply your resolution function to this data.
