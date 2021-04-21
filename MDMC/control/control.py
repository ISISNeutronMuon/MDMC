"""A module for performing the refinement"""

from copy import deepcopy

import numpy as np
import pandas as pd
from typing import List
from scipy.interpolate import interp2d

from MDMC.common.decorators import repr_decorator
from MDMC.MD.parameters import Parameters
from MDMC.MD.simulation import Simulation
from MDMC.refinement import minimizer, FoM
from MDMC.trajectory_analysis.observables.obs_factory \
    import ObservableFactory
from MDMC.trajectory_analysis.observables.sqw import SQw
from MDMC.trajectory_analysis.observables.obs import Observable

# Define the current limitations on data structure for each ``type`` of ``Observable``. The dictionary keys are
# ``Observable`` types for which some of the independent variables have restrictions on them being uniform
# (the first boolean in the list for each variable) or starting from zero (second boolean in the list)
# The booleans are 'True' if there is a restrictions, e.g. for ``SQw`` ``Observables`` the energy 'E' needs to
# be uniform and start at zero, whereas reciprocal points 'Q' only need to be uniform.
Uniformity_requirements = {'SQw': {'E': [True, True], 'Q': [True, False]}, 'PDF': {'r': [True, False]}}

@repr_decorator('simulation', 'exp_datasets', 'FoM_calculator', 'minimizer',
                'reset_config', 'fit_parameters', 'MD_steps', 'settings')
class Control:

    """
    Controls the MDMC refinement

    Parameters
    ----------
    simulation : Simulation
        Performs a simulation for a given set of potential ``Parameter``
        objects.
    exp_datasets : list of dicts
        Each `dict` is an experimental dataset, containing the file name
        (``file_name``), the type of observable (``type``), the reader required
        for the file (``reader``), and the weighting of the dataset in the
        Figure of Merit calculation(``weighting``). Optionally, can also
        contain ``rescale_factor`` which will be applied to the experimental
        data when comparing it to the calculated observable. Default is `1.`.
        Alternatively, can optionally specify ``auto_scale`` which will set the
        ``rescale_factor`` automatically to minimise the FoM. Default is
        `False`. If both ``rescale_factor`` and ``auto_scale`` are provided
        then a warning is printed and ``auto_scale`` takes precedence.
    fit_parameters : Parameters, list of Parameter
        All parameters which will be refined.
    MC_norm : float, optional
        Determines the accept/reject ratio of the MC. Default is 1.
    minimizier_type : str, optional
        The ``Minimizer`` type. Default is 'MMC'.
    FoM_type : str, optional
        The type of ``FigureOfMeritCalculator``. Default is ``standard``.
    reset_config : bool, optional
        Determines if the configuration is reset to the end of the last accepted
        state. Default is `True`.
    MD_steps : int, optional
        Number of molecular dynamics steps for each step of the refinement.
        When not provided, the minimum number of steps needed for successful
        calculation of the observables is used. Default is `None`.
    **settings
        ``energy_resolution`` : float
            Instrument energy resolution as the FWHM in ``ueV``.

    Example
    -------
    An example of an exp_dataset list is::

        [{'file_name':data.LAMP_SQW_FILE,
          'type':'SQw',
          'reader':'LAMPSQw',
          'weight':1.,
          'rescale_factor':0.5},
         {'file_name:data.ANOTHER_FILE',
          'type':'FQt',
          'reader':'GENERIC_READER',
          'weight':0.5,
          'auto_scale':True}]

    Attributes
    ----------
    simulation : Simulation
        The ``Simulation`` on which is used to perform the refinement
    exp_datasets : `list of dicts`
        One `dict` per experimental dataset used for the refinement
    fit_parameters : Parameters
        All ``Parameter`` objects which will be refined
    minimizer : Minimizer
        Refines the potential parameters.
    settings : `dict`
        Settings for the MD and minimization.
    observable_pairs : `list` of ``ObservablePairs``
        Experimental observable/MD observable pairs which are used to calculate
        the Figure of Merit
    FoM_calculator : FigureOfMeritCalculator
        Calculates the FoM `float` from the ``observable_pairs``.
    MD_steps : `int`
        Number of molecular dynamics steps for each step of the refinement
    """

    MINIMIZER_DICT = {"MMC":minimizer.MMC}
    FOM_DICT = {"standard":FoM.StandardFoMCalculator}

    def __init__(self, simulation: Simulation, exp_datasets: List[dict],
                 fit_parameters: Parameters, MC_norm: float=1.,
                 minimizer_type: str='MMC', FoM_type: str='standard',
                 reset_config: bool=True, MD_steps: int=None, **settings):

        self.simulation = simulation
        self.exp_datasets = exp_datasets
        self.fit_parameters = Parameters(fit_parameters)
        # Minimizer FoM_old is always initialised to infinity, so that first MC
        # step (i.e. the setup) is always accepted.
        self.minimizer = self.MINIMIZER_DICT[minimizer_type](MC_norm,
                                                             self.fit_parameters)
        self.reset_config = reset_config
        self.settings = settings

        # Create experimental observables from datasets and placeholders for
        # experimental observables calculated from MD
        self.observable_pairs = []
        minimum_MD_steps = 0
        for dset in exp_datasets:
            exp_observable = self._read_observable_from_file(dset['type'],
                                                             dset['reader'],
                                                             dset['file_name'])
            with self._is_data_uniform(exp_observable) as uniformity_check:
                if dset['type'] in Uniformity_requirements \
                        and uniformity_check != Uniformity_requirements[dset['type']]:
                    exp_observable = self._make_data_uniform(exp_observable, uniformity_check)

            MD_observable = self._create_empty_observable(exp_observable)

            auto_scale = dset.get('auto_scale', False)
            rescale_factor = dset.get('rescale_factor')
            if auto_scale and rescale_factor:
                print('Both `rescale_factor` and `auto_scale` set for file {};'
                      ' scaling will be automated to minimise FoM'
                      ''.format(dset['file_name']))
                rescale_factor = 1.
            elif not rescale_factor:
                rescale_factor = 1.

            observable_pair = FoM.ObservablePair(exp_observable,
                                                 MD_observable,
                                                 dset['weight'],
                                                 rescale_factor=rescale_factor,
                                                 auto_scale=auto_scale)
            self.observable_pairs.append(observable_pair)
            minimum_MD_steps = max(minimum_MD_steps,
                                   self._calculate_MD_steps(observable_pair))

        self.FoM_calculator = self.FOM_DICT[FoM_type](self.observable_pairs)

        # Use specified MD_steps if supplied, else calculate
        if MD_steps:
            try:
                assert MD_steps >= minimum_MD_steps
                self.MD_steps = MD_steps
            except AssertionError as error:
                raise ValueError('Experimental datasets provided require a '
                                 'minimum MD_steps value of {} in order to '
                                 'calculate observables'.format(minimum_MD_steps)
                                 ) from error
        else:
            self.MD_steps = minimum_MD_steps

        setup_frame = pd.DataFrame([[minimizer_type],
                                    [MC_norm],
                                    [FoM_type],
                                    [len(self.observable_pairs)],
                                    [len(self.fit_parameters)]],
                                    index=['  Minimizer',
                                           '  MC norm',
                                           '  FoM type',
                                           '  Number of observables',
                                           '  Number of parameters'])

        print('Control created with:\n{}\n'
              ''.format(setup_frame.to_string(index=True, header=False)))

    def __str__(self):

        exp_dataset_types = [dataset['type'] for dataset in self.exp_datasets]
        return "{0} refining {1} {2} using {3} data types".format(
            self.__class__.__name__,
            len(self.fit_parameters),
            'parameter' if len(self.fit_parameters) == 1 else 'parameters',
            exp_dataset_types)

    def refine(self, n_steps):

        """
        Refines the specified potential parameters

        Parameters
        ----------
        n_steps : int
            maximum number of steps for the refinement

        Examples
        --------
        Perform a refinement with a maximum of 100 steps:

            .. highlight:: python
            .. code-block:: python

            control.refine(100)
        """

        count = -1

        self._print_header()
        while count < n_steps and not self.minimizer.has_converged():
            self.step()
            count += 1

        # Try/except accounts for n_steps <= -1
        try:
            # Reset the minimizer parameters to those from the final FoM:
            # to account for a current side effect of step()
            self.minimizer.reset_parameters()
            self._update_engine_parameters()
        except TypeError:
            pass

        # print values of final parameters
        parameter_df = pd.DataFrame({p.name:p.value for p in self.minimizer.parameters},
                                    index=[0])
        print('\nFinal Parameters\n{}'
              ''.format(parameter_df.to_string(index=False)))

        # If automatically scaling data print the scale factor for each dataset
        scaling_keys = []
        scaling_values = []
        for i, observable_pair in enumerate(self.observable_pairs):
            if observable_pair.auto_scale:
                dset = self.exp_datasets[i]
                scaling_keys.append('  {}'.format(dset['file_name']))
                scaling_values.append([observable_pair.rescale_factor])

        if len(scaling_keys) > 0 and len(scaling_values) > 0:
            scaling_df = pd.DataFrame(scaling_values, index=scaling_keys)
            print('\nAutomatic Scale Factors\n{}'
                  ''.format(scaling_df.to_string(index=True, header=False)))

    def step(self):

        """
        Do a full step: generate and run MD to calulate FoM for existing
        parameters, iterate parameters a step forward and reset MD (phasespace)
        if previous step was rejected and reset_config = true
        """

        # Generate FoM by running MD for this step and then calculate FoM
        fom = self._generate_FoM()
        # Select new parameters to consider
        self.minimizer.step(fom)
        # Update the MD engine with new parameters
        self._update_engine_parameters()
        self._print_data()

        # When reset_config=true reset the MD (phasespace) back if the
        # previous step was rejected
        if self.reset_config:
            if self.minimizer.state_changed:
                # Set MD engine to remember new config
                self.simulation.engine.save_config()
            else:
                # Set MD engine to reset to old config
                self.simulation.engine.reset_config()

        self.minimizer.write_history('results.csv')

    def _print_data(self):

        with pd.option_context('display.max_colwidth', 12,
                               'display.precision', 5,
                               'display.float_format', '{:.4g}'.format):
            n_step = self.minimizer.history.iloc[-1].name
            output = self.minimizer.history.loc[[n_step]].to_string(
                col_space=12, index=False, header=False).split('\n')
            data = '{:4d}'.format(n_step) + ''.join(output)
            print(data)

    def _print_header(self):

        def format_column(column):
            column = column if len(column) < 13 else column[:9] + '...'
            return ' ' * (12 - len(column)) + column

        columns = ' '.join([format_column(col) for col
                            in self.minimizer.history.columns])
        header = 'Step' + columns
        print(header)

    def _generate_FoM(self):

        """
        Run the MD for an iteration/step, calculate observable, compare with
        observed observed and return the FoM

        Returns
        -------
        `float`
            Non-negative `float` FoM
        """

        self._run_MD()
        self._calculate_observables(self.simulation, self.observable_pairs)

        return self.FoM_calculator.calculate()

    def _run_MD(self):

        """
        Run a molecular dynamics simulation
        """

        self.simulation.run(self.MD_steps, verbose=False)

    def _update_engine_parameters(self):

        """
        Update the force field parameters of the MD engine
        """

        self.simulation.engine.update_parameters()

    def _read_observable_from_file(self, type, reader, file_name):

        """
        Creates an Observable of the specified type and reads in data from file

        Parameters
        ----------
        type : str
            The ``type`` of the ``Observable``.
        reader : str
            The ``type`` of the ``Reader``.
        file_name : str
            The absolute or relative path and the file name.

        Returns
        -------
        ``Observable``
            An ``Observable`` of specified ``type``
        """

        observable = ObservableFactory.create_observable(type)
        observable.read_from_file(reader=reader, file_name=file_name)
        return observable

    def _create_empty_observable(self, exp_observable):

        """
        Creates a ``Observable`` without data but with independent variables
        specified from another ``Observable``.  This is a placeholder in which
        the ``Observable`` can be calculated from an MD trajectory.

        Parameters
        ----------
        exp_observable : Observable
            An ``Observable`` with defined independent variables.

        Returns
        -------
        ``Observable``
            An ``Observable`` with only independent variables and
            ``origin == 'MD'``
        """

        observable = ObservableFactory.create_observable(exp_observable.name)
        observable.origin = 'MD'
        observable.independent_variables = deepcopy(
            exp_observable.independent_variables)
        return observable

    def _calculate_observables(self, simulation, observable_pairs):

        """
        Calculates all of the ``Observable`` objects from the MD
        trajectory/configurations

        Parameters
        ----------
        simulation : Simulation
            ``MDEngine`` with defined trajectory attribute
        observable_pairs : list of ObservablePairs
            ``ObservablesPairs`` for which the MD ``Observable`` will be
            calculated
        """

        trj = simulation.engine.convert_trajectory()
        for pair in observable_pairs:
            maximum_frames = pair.MD_obs.maximum_frames
            if maximum_frames:
                pair.MD_obs.calculate_from_MD(trj[:maximum_frames],
                                              **self.settings)
            else:
                pair.MD_obs.calculate_from_MD(trj, **self.settings)

    def _calculate_MD_steps(self, observable_pair: FoM.ObservablePair):

        """
        Calculates the minimum number of steps required for the MD engine in
        order to calculate MD ``Observables`` objects with the same independent
        variables as the experimental ``Observable`` objects.

        Parameters
        ----------
        observable_pair : ObservablePair
            ``ObservablesPair`` for which the required number of ``MD_steps``
            is calculated

        Returns
        -------
        `int`
            Number of ``MD_steps``
        """
        traj_step = self.simulation.settings.get('traj_step')
        minimum_frames = observable_pair.exp_obs.minimum_frames

        return traj_step * minimum_frames

    def _is_data_uniform(self, observable: Observable) -> dict:
        """
        Checks if the values of the independent variables of an ``Observable`` are uniformly spaced and checks if
        they start at zero.

        Parameters
        ----------
        observable : Observable
            The ``Observable`` for which to check the independent variables.

        Returns
        -------
        `dict`
            A dictionary with the keys the same as the independent variables of the ``Observable`` and the values
            are a list of two booleans: the first if the points are uniformly spaced and the second if they start at
            zero.
        """
        uniformity_dict = {}
        for var_key, var_data in observable.independent_variables.items():
            uniform_data = np.linspace(min(var_data), max(var_data), num=len(var_data))
            is_uniform = np.allclose(var_data, uniform_data, rtol=1e-5)
            uniformity_dict[var_key] = [is_uniform, min(var_data) == 0]
        return uniformity_dict

    def _make_data_uniform(self, observable: Observable, uniformity_check: dict = None) -> Observable:
        """
        Takes an ``Observable`` and returns an Observable with its values of the variables interpolated onto a
        uniform grid. Currently limited to ``Observables`` with 2D ``dependent_variables`` (i.e. SQW).
        Optionally takes a dictionary ('uniformity_check`) as input that specifies if the
        ``independent_variables`` of the ``Observable`` are currently uniform or start at zero; this dictionary gets
        compared to the current limitations of MDMC to determine which variables need to be made uniform and/or made to
        start at zero.

        Parameters
        ----------
        observable : Observable
            An ``Observable`` for which the independent variables need to be made uniform / to start at zero. Currently
            limited to ``SQw`` ``Observables``.
        uniformity_check : dict
            A dictionary of the format {'variable': [bool, bool]}, where the two booleans specify if the
            independent 'variable' is uniform and starts at zero, respectively.

        Returns
        -------
        ``Observable``
            Returns a copy of the passed ``Observable`` with the independent variables put onto uniform grid (for the
            variables where that is necessary) and the dependent variables interpolated onto the same grid
        """
        observable = observable
        # determine which independent variables are uniform/start at zero if not know
        if uniformity_check is None:
            uniformity_check = self._is_data_uniform(observable)
        observable_name = observable.name
        # initialise a helper dictionary to hold the data points for the uniform independent variables
        indep_var_uniform = {}
        # loop through all independent variables and make them uniform if needed
        for var_key, var_uniformity in uniformity_check.items():
            # check if the independent variable has requirements on uniformity or starting at zero and if these are
            # satisfied already
            if var_key in Uniformity_requirements[observable_name] and \
                    var_uniformity != Uniformity_requirements[observable_name]:
                data = observable.independent_variables[var_key]
                if Uniformity_requirements[observable_name][var_key][1]:
                    minimum = 0
                else:
                    minimum = min(data)
                uniform_data = np.linspace(minimum, max(data), num=len(data))
                indep_var_uniform[var_key] = uniform_data
            # if uniformity requirements are satisfied already, add the data points to the helper dictionary
            else:
                indep_var_uniform[var_key] = observable.independent_variables[var_key]

        # loop through the dependent variables and interpolate them
        for var_key, data in observable.dependent_variables.items():
            # determine the correct order of independent variables used in the interpolation
            var_shape = observable.dependent_variables_shape
            # note: if x = [0,1,2];  y = [0,3]; z = [[1,2,3], [4,5,6]]; then np.shape(z)=(np.size(y), np.size(x)),
            # so because observable.dependent_variables_shape gives the order of np.shape(depedent_variable) the
            # order of the x (Q) and y (E) data is the following:
            x_data = observable.independent_variables[var_shape[var_key][1]]
            y_data = observable.independent_variables[var_shape[var_key][0]]
            data_interpol = interp2d(x_data, y_data, data)
            x_uniform = indep_var_uniform[var_shape[var_key][1]]
            y_uniform = indep_var_uniform[var_shape[var_key][0]]
            uniform_data = data_interpol(x_uniform, y_uniform)
            observable._dependent_variables[var_key] = uniform_data
            # repeat the interpolation for the errors
            err_data = observable.errors[var_key]
            err_data[err_data == np.float('inf')] = 0
            err_interpol = interp2d(x_data, y_data, err_data)
            err_uniform = err_interpol(x_uniform, y_uniform)
            observable._errors[var_key] = err_uniform
        # finally, set the independent variables of the ``Observable`` to the uniform ones
        observable.independent_variables = indep_var_uniform
        return observable

