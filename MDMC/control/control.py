"""A module for performing the refinement"""

from copy import deepcopy

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
from typing import List
from scipy.interpolate import interp1d, interp2d
from typing import Dict

from MDMC.common.constants import h, h_bar
from MDMC.common.decorators import repr_decorator
from MDMC.MD.parameters import Parameters
from MDMC.MD.simulation import Simulation
from MDMC.refinement import minimizer, FoM
from MDMC.trajectory_analysis.observables.obs_factory \
    import ObservableFactory
from MDMC.trajectory_analysis.observables.obs import Observable


@repr_decorator('simulation', 'exp_datasets', 'FoM_calculator', 'minimizer',
                'reset_config', 'fit_parameters', 'MD_steps',
                'max_parameter_change', 'settings')
class Control:

    """
    Controls the MDMC refinement

    Parameters
    ----------
    simulation : Simulation
        Performs a simulation for a given set of potential ``Parameter``
        objects.
    exp_datasets : list of dicts
        Each `dict` represents an experimental dataset, containing the
        following keys:
          - ``file_name`` (`str`) the file name
          - ``type`` (`str`) the type of observable
          - ``reader`` (`str`) the reader required for the file
          - ``weighting`` (`float`) the weighting of the dataset to be used in
            the Figure of Merit calculation
          - ``resolution_file`` (`str`, optional, defaults to `None`) a file in
            the same format as ``file_name`` containing results of a vanadium
            sample which is used to determine instrument energy resolution for
            this dataset (overriding the ``energy_resolution`` setting)
          - ``rescale_factor`` (`float`, optional, defaults to `1.`) applied to
            the experimental data when calculating the FoM to ensure it is on
            the same scale as the calculated observable
          - ``auto_scale`` (`bool`, optional, defaults to `False`) set the
            ``rescale_factor`` automatically to minimise the FoM, if both
            ``rescale_factor`` and ``auto_scale`` are provided then a warning
            is printed and ``auto_scale`` takes precedence
          - ``use_FFT`` (`bool`, optional, defaults to `True`) whether to use
            Fast Fourier Transforms in the calculation of dependent variables.
            FFT speeds up calculation but places restrictions on spacing in the
            independent variable domain(s). This option may not be supported
            for all ``Observable``s
        Note that the default (and preferred) behaviour of the scaling settings requires that the
        dataset provided has been properly scaled and normalised for the refinement process.
        Arbitrary or automatic rescaling should be undertaken with care, as it does not take into
        account any physical aspects of scaling the data, such as the presence or absence of
        background events from peaks outside its range.
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
    max_parameter_change : float, optional
        Maximum factor by which a Parameter can change each step of the
        refinement. Defaults to `0.01`
    MD_steps : int, optional
        Number of molecular dynamics steps for each step of the refinement.
        When not provided, the minimum number of steps needed for successful
        calculation of the observables is used. If provided, the actual number
        of steps may be reduced to prevent running MD that won't be used when
        calculting dependent variables. Default is `None`.
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
          'resolution_file':data.LAMP_SQW_VAN_FILE
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
                 reset_config: bool=True, MD_steps: int=None,
                 max_parameter_change: float=0.01, **settings):

        self.simulation = simulation
        self.exp_datasets = exp_datasets
        self.fit_parameters = Parameters(fit_parameters)
        # Minimizer FoM_old is always initialised to infinity, so that first MC
        # step (i.e. the setup) is always accepted.
        self.minimizer = self.MINIMIZER_DICT[minimizer_type](MC_norm,
                                                             self.fit_parameters,
                                                             max_parameter_change=max_parameter_change)
        self.reset_config = reset_config
        self.settings = settings

        # Create experimental observables from datasets and placeholders for
        # experimental observables calculated from MD
        self.observable_pairs = []
        minimum_MD_steps = 0
        for dset in exp_datasets:
            use_FFT = dset.get('use_FFT', True)
            exp_observable = self._read_observable_from_file(dset['type'],
                                                             dset['reader'],
                                                             dset['file_name'])
            exp_observable.use_FFT = use_FFT

            if exp_observable.uniformity_requirements:
                exp_observable = self._make_data_uniform(exp_observable)

            MD_observable = self._create_empty_observable(exp_observable)
            MD_observable.use_FFT = use_FFT

            self._validate_energy(MD_observable)

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

            # Take the largest minimum number of MD_steps needed by any dataset
            min_MD_steps_dset = self._calculate_minimum_MD_steps(observable_pair)
            minimum_MD_steps = max(minimum_MD_steps, min_MD_steps_dset)

        self.FoM_calculator = self.FOM_DICT[FoM_type](self.observable_pairs)

        # Use specified MD_steps if supplied, else calculate
        if MD_steps:
            try:
                assert MD_steps >= minimum_MD_steps
                # Set self.MD_steps to be the largest number required by any of
                # our observable pairs
                maximum_MD_steps = minimum_MD_steps
                for pair in self.observable_pairs:
                    max_MD_steps_pair = self._calculate_maximum_MD_steps(MD_steps, pair)
                    maximum_MD_steps = max(maximum_MD_steps, max_MD_steps_pair)
                self.MD_steps = maximum_MD_steps
            except AssertionError as error:
                raise ValueError('Experimental datasets provided require a '
                                 'minimum MD_steps value of {} in order to '
                                 'calculate observables'.format(minimum_MD_steps)
                                 ) from error
        else:
            self.MD_steps = minimum_MD_steps

        for i, dset in enumerate(exp_datasets):
            if dset.get('resolution_file'):
                resolution_functions = self._read_resolution_from_file(dset['type'],
                                                                       dset['reader'],
                                                                       dset['resolution_file'])
                self.observable_pairs[i].exp_obs.resolution_functions = resolution_functions
                self.observable_pairs[i].MD_obs.resolution_functions = resolution_functions

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

    def _read_observable_from_file(self, type: str, reader: str, file_name: str,
                                   resolution_file_name: str = None):

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

    def _read_resolution_from_file(self, data_type: str, reader: str, file_name: str):

        """
        Reads resolution data for the specified ``data_type`` from file and interpolates it
        to give a dictionary of general resolution functions in the time domain for each dependent
        variable.

        Note that if this resolution function is used on data outside its original range, then it
        will use nearest neighbour extrapolation. Additionally, the input will be reflected in the
        time/energy domain as symmetry about 0 is assumed. If for whatever reason this is not
        appropriate for the data in question, this function should not be used.

        This may not be supported for all ``Observable`` types.

        Parameters
        ----------
        data_type : str
            The ``type`` of the ``Observable``.
        reader : str
            The ``type`` of the ``Reader``.
        file_name : str
            The absolute or relative path of the resolution file name.

        Returns
        -------
        dict
            A dictionary with keys for each dependent variable, where the
            values are resolution functions for that variable.
        """

        resolution_obs = self._read_observable_from_file(data_type,
                                                         reader,
                                                         file_name)

        dt = self.simulation.time_step * self.simulation.traj_step
        return resolution_obs.calculate_resolution_functions(dt)

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
        observable.resolution_functions = exp_observable.resolution_functions
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
            maximum_frames = pair.MD_obs.maximum_frames()
            if maximum_frames:
                # If there is a limit on the number of frames the observable
                # can use in calculations, split the trajectory into as many
                # subsets of this length as we can
                sub_trj = []
                n_averages = len(trj) // maximum_frames
                for i in range(n_averages):
                    sub_trj.append(trj[i * maximum_frames : (i + 1) * maximum_frames])
                pair.MD_obs.calculate_from_MD(sub_trj, **self.settings)
            else:
                # Otherwise, provide the whole trajectory
                pair.MD_obs.calculate_from_MD([trj], **self.settings)

    def _calculate_minimum_MD_steps(self, observable_pair: FoM.ObservablePair):

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
        time_step = self.simulation.time_step
        traj_step = self.simulation.traj_step
        # Calculate the time separation between trajectory frames, dt
        dt = time_step * traj_step
        # Calculate the minimum number of trajectory frames needed for the
        # calculation of the dependent_variables of observable_pair
        minimum_frames = observable_pair.exp_obs.minimum_frames(dt)

        return traj_step * minimum_frames

    def _calculate_maximum_MD_steps(self, MD_steps: int,
                                    observable_pair: FoM.ObservablePair):

        """
        Calculates the maximum number of steps that ``observable_pair`` will be
        able to use when calculating dependent variables whilst still being
        below the ``MD_steps`` specified by the user. Any additional steps
        beyond this would not contribute to the calculation.

        If ``observable_pair.exp_obs.maximum_frames()`` is None, then all frames can be used so we
        just return the largest multiple of ``traj_steps``.

        Otherwise, we calculate the largest multiple of
        ``traj_step * observable_pair.exp_obs.maximum_frames()``, as we can then calculate the
        dependent variable multiple times by taking subsets of the total trajectory.

        Parameters
        ----------
        MD_steps : int
            The hard upper limit on the number of steps to run, as specified by
            the user
        observable_pair : ObservablePair
            ``ObservablesPair`` for which the required number of ``MD_steps``
            is calculated

        Returns
        -------
        int
            Maximum number of ``MD_steps`` needed
        """
        traj_step = self.simulation.traj_step
        maximum_frames = observable_pair.exp_obs.maximum_frames()

        if maximum_frames is None:
            maximum_steps = traj_step
        else:
            maximum_steps = traj_step * maximum_frames

        return maximum_steps * (MD_steps // (maximum_steps))

    def _is_data_uniform(self, observable: Observable) -> Dict[str, Dict[str, bool]]:
        """
        Checks if the values for each independent variable of an ``Observable`` are uniformly
        spaced and if they start at zero. This information is returned in a single dictionary.

        Parameters
        ----------
        observable : Observable
            The ``Observable`` for which to check the independent variables.

        Returns
        -------
        `Dict[str, Dict[str, bool]]`
            An outer dictionary where the independent variables of the ``Observable`` are the keys,
            and the values are another dictionary corresponding to that variable. This inner
            dictionary has the same format for all variables, with the two keys 'uniform' and
            'zeroed'. The values for these keys are booleans that state whether the data fulfils
            the corresponding requirement.

        Examples
        --------
        >>> control._is_data_uniform(self, observable)
        {'E': {'uniform': True, 'zeroed': True}, 'Q': {'uniform': True, 'zeroed': False}}
        """
        uniformity_dict = {}
        for var_key, var_data in observable.independent_variables.items():
            uniform_data = np.linspace(min(var_data), max(var_data), num=len(var_data))
            is_uniform = np.allclose(var_data, uniform_data, rtol=1e-5)
            uniformity_dict[var_key] = {'uniform': is_uniform, 'zeroed': var_data[0] == 0}
        return uniformity_dict

    def _make_data_uniform(self, observable: Observable) -> Observable:
        """
        Takes an ``Observable``, checks the requirements for its ``independent_variables`` to be uniform or start at
        zero, creates uniform grids for the variables that do not satisfy their requirement, interpolates the
        ``dependent_variables`` as needed, and returns an ``Observable`` with the uniform/interpolated variables.
        Currently limited to ``Observables`` with two-dimensional ``dependent_variables`` (e.g. SQw).

        Parameters
        ----------
        observable : Observable
            An ``Observable`` for which the independent variables need to be made uniform / to start at zero. Currently
            limited to ``Observables`` for which the ``dependent_variables`` are two-dimensional.

        Returns
        -------
        ``Observable``
            Returns a copy of the passed ``Observable`` with the independent variables put onto uniform grid (for the
            variables where that is necessary) and the dependent variables interpolated onto the same grid
        """
        # get the uniformity requirements from the Observable
        uniformity_required = observable.uniformity_requirements
        if uniformity_required is None:
            return observable

        # determine for all independent_variables if they are currently uniform or start at zero
        uniformity_state = self._is_data_uniform(observable)
        # initialise helper list for the independent_variables that need to be made uniform
        indep_vars_to_be_changed = []
        # loop through requirements
        for var_key, var_required in uniformity_required.items():
            var_state = uniformity_state[var_key]
            # if the variable has a requirement AND it is not satisfied (for either uniformity OR zero-start)
            # then add it to the list of variables that need to be changed
            if (var_required['uniform'] and not var_state['uniform']) or \
                    (var_required['zeroed'] and not var_state['zeroed']):
                indep_vars_to_be_changed.append(var_key)

        # if all uniformity requirements are already satisfied simply return the original observable
        if not indep_vars_to_be_changed:
            return observable

        # initialise a helper dictionary to hold the new independent variables
        indep_var_uniform = {}
        # loop through all independent variables
        for var_key in uniformity_state:
            # check if the independent variable needs to be made uniform
            if var_key in indep_vars_to_be_changed:
                data = observable.independent_variables[var_key]
                if uniformity_required[var_key]['zeroed']:
                    minimum = 0
                else:
                    minimum = min(data)
                uniform_data = np.linspace(minimum, max(data), num=len(data))
                indep_var_uniform[var_key] = uniform_data
            # if uniformity requirements are satisfied already, add the data points to the helper dictionary
            else:
                indep_var_uniform[var_key] = observable.independent_variables[var_key]

        # get the indexing order of independent variables within the dependent variables
        var_indexing = observable.dependent_variables_structure

        # loop through the dependent variables and interpolate them
        for var_key, data_list in observable.dependent_variables.items():
            # Experimental Observables should only have 1 element in data_list
            try:
                assert len(data_list) == 1
                data = data_list[0]
            except AssertionError as error:
                msg = ('Expected experimental dataset to only have one dependent '
                       'variable entry for {0}, but found {1} instead'
                       ''.format(var_key, len(data_list)))
                raise AssertionError(msg) from error

            # determine the dimension of the dependent variable
            var_dimension = data.ndim
            # interpolation for 1D
            if var_dimension == 1:
                x_data = observable.independent_variables[var_indexing[var_key][0]]
                data_interpol = interp1d(x_data, data)
                x_uniform = indep_var_uniform[var_indexing[var_key][0]]
                uniform_data = data_interpol(x_uniform)
                # repeat the interpolation for the errors
                err_data = observable.errors[var_key][0]
                err_data[err_data == np.float('inf')] = 0
                err_interpol = interp1d(x_data, err_data)
                err_uniform = err_interpol(x_uniform)
                err_uniform[err_uniform == 0.] = np.float('inf')
            # interpolation for 2D
            elif var_dimension == 2:
                # note: the interp2d interpolation function requires input of the form
                # interp2d(x, y, z)
                # where if np.size(x)=m and np.size(y)=n then np.shape(z)=(n,m)
                # E.g. if x = [0,1,2]; y = [0,3]; z = [[1,2,3], [4,5,6]]
                # Because Observable.dependent_variables_structure gives the order in which the independent variables
                # are represented in the np.shape of the data, we have to reverse the order of the x and y arrays
                # for interp2d:
                x_data = observable.independent_variables[var_indexing[var_key][1]]
                y_data = observable.independent_variables[var_indexing[var_key][0]]
                data_interpol = interp2d(x_data, y_data, data)
                # get the independent_variables that satisfy the uniformity requirements as created earlier
                x_uniform = indep_var_uniform[var_indexing[var_key][1]]
                y_uniform = indep_var_uniform[var_indexing[var_key][0]]
                uniform_data = data_interpol(x_uniform, y_uniform)
                # repeat the interpolation for the errors
                err_data = observable.errors[var_key][0]
                err_data[err_data == np.float('inf')] = 0
                err_interpol = interp2d(x_data, y_data, err_data)
                err_uniform = err_interpol(x_uniform, y_uniform)
                err_uniform[err_uniform == 0.] = np.float('inf')
            else:
                raise NotImplementedError('Only 1D and 2D data can currently be made uniform')
            # save the uniform data and errors back into the Observable
            observable._dependent_variables[var_key] = [uniform_data]
            observable._errors[var_key] = [err_uniform]
        # finally, set the independent variables of the ``Observable`` to the uniform ones
        observable.independent_variables = indep_var_uniform
        return observable

    def _validate_energy(self, obs: Observable):

        """
        Try and validate the energy of the ``Observable`` provided, and pass if
        it does not have a ``validate_energy`` function itself

        Parameters
        ----------
        obs : Observable
            ``Observable`` to validate

        Returns
        -------
        None

        Raises
        ------
        AssertionError
        """

        # Calculate the time separation between trajectory frames, dt, imposed
        # by the simulation
        dt = self.simulation.traj_step * self.simulation.time_step
        try:
            obs.validate_energy(dt)
        except AttributeError:
            pass
