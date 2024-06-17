"""A module for performing the refinement"""
import statistics
from copy import deepcopy
from typing import List, Dict, Union
from contextlib import suppress
from datetime import datetime
from pathlib import Path

import logging
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d, interp2d
from verbosemanager import VerboseManager

from MDMC.control.plot_results import PlotResults, data_printers
from MDMC.common.decorators import repr_decorator
from MDMC.MD.parameters import Parameters
from MDMC.MD.simulation import Simulation
from MDMC.refinement.minimizers.minimizer_factory import MinimizerFactory
from MDMC.refinement.FoM.FoM_factory import FoMFactory
from MDMC.refinement.FoM.FoM_abs import ObservablePair
from MDMC.resolution.resolution_factory import ResolutionFactory
from MDMC.trajectory_analysis.observables.obs_factory \
    import ObservableFactory
from MDMC.trajectory_analysis.observables.obs import Observable


@repr_decorator('simulation', 'exp_datasets', 'FoM_calculator', 'minimizer',
                'reset_config', 'fit_parameters', 'MD_steps',
                'verbose')
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
          - ``resolution`` : (`dict`, but can be `str`, `float`, or None)
            Should be a one-line dict of format {'file' : str} or {key : float}
            if {'file' : str}, the str should be a file in
            the same format as ``file_name`` containing results of a vanadium
            sample which is used to determine instrument energy resolution for
            this dataset.
            If {key : float}, the key should be the type of resolution function.
            allowed types are 'gaussian' and 'lorentzian'
            Can also be 'lazily' given as just a `str` or `float`.
            If just a string is given, it is assumed to be a filename.
            If just a float is given, it is assumed to be Gaussian.
            The float should be the instrument energy resolution as the FWHM in ``ueV`` (micro eV).
            If you have already accounted for instrument resolution in your dataset,
            this field can be set to None, and resolution application will be skipped.
            This must be done explicitly.
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
        All parameters which will be refined. Note that any ``Parameter`` that is ``fixed``,
        ``tied`` or equal to 0 will not be passed to the minimizer as these cannot be refined.
        Those with ``constraints`` set are still passed.
    minimizer_type : str, optional
        The ``Minimizer`` type. Default is 'MMC'.
    FoM_options : dict of {str : str}, optional
        Defines the details of the ``FigureOfMeritCalculator`` to use. Default is `None`, in which
        case the first option for each of the following keys is used:
          - ``errors`` {'exp', 'none'} Whether to weight the differences between MD and
            experimental data with the experimental error or not (errors from MD are not currently
            supported).
          - ``norm`` {'data_points', 'dof', 'none'} How to normalise the FoM, either by the total
            number of data_points (n), the degrees of freedom (n - m, where m is the number of
            parameters being varied), or not at all.
    reset_config : bool, optional
        Determines if the configuration is reset to the end of the last accepted
        state. Default is `True`.
    MD_steps : int, optional
            Number of molecular dynamics steps for each step of the refinement.
            When not provided, the minimum number of steps needed for successful
            calculation of the observables is used. If provided, the actual number
            of steps may be reduced to prevent running MD that won't be used when
            calculating dependent variables. Default is `None`.
    equilibration_steps : int, optional
            Number of molecular dynamics steps used to equilibrate the ``Universe``
            in between each refinement step. When changes to the ``Parameters`` are small,
            this equilibration can be much shorter than the equilibration needed before
            starting the refinement process, but in general will vary depending on
            the details of the ``Universe`` and ``Parameters``. Default is 0.
    verbose: int, optional
            The level of verbosity:
            Verbose level -1 hides all outputs for tests.
            Verbose level 0 gives no information.
            Verbose level 1 gives final time for the whole method.
            Verbose level 2 gives final time and also a progress bar.
            Verbose level 3 gives final time, a progress bar, and time per step.
    print_full_settings: bool, optional
        Whether to print all settings/attributes/parameters passed to this object,
        defaults to False.
    **settings: dict, optional
        n_steps: int
            The maximum number of refinement steps to be. An optional parameter that,
            if specified, will be used in the ``refine`` method unless a different value
            of ``n_steps`` is explicitly passed when running ``refine``.
        cont_slicing : bool, optional
            Flag to decide between two possible behaviours when the number of ``MD_steps`` is
            larger than the minimum required to calculate the observables. If ``False`` (default)
            then the ``CompactTrajectory`` is sliced into non-overlapping sub-``CompactTrajectory``
            blocks for each of which the observable is calculated. If ``True``, then the
            ``CompactTrajectory`` is sliced into as many non-identical sub-``CompactTrajectory``
            blocks as possible (with overlap allowed).
        results_filename: str
            The name of the file in which the results of the MDMC run will be stored
        MC_norm: int
            1 if the MMC minimiser is to be used for parameter optimisation.
        use_average : bool, optional
            Optional parameter relevant in case ``MD_steps`` is larger than the minimum required
            and the MD ``CompactTrajectory`` is sliced into sub-``CompactTrajectory`` blocks.
            If ``True`` (default) the observables are averaged over the sub-``CompactTrajectory``
            blocks. If ``False`` they are not averaged.
        data_printer: str, default 'plaintext'
            How to display the data during minimisation. Current options are 'plaintext' (default,
            plaintext printing) or 'ipython' (prettier HTML printing via iPython)

    Example
    -------
    An example of an exp_dataset list is::

        [{'file_name':data.LAMP_SQW_FILE,
          'type':'SQw',
          'reader':'LAMPSQw',
          'weight':1.,
          'resolution':{'file':data.LAMP_SQW_VAN_FILE}
          'rescale_factor':0.5},
         {'file_name:data.ANOTHER_FILE',
          'type':'FQt',
          'reader':'GENERIC_READER',
          'weight':0.5,
          'resolution':{'gaussian':2.35}
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
    observable_pairs : `list` of ``ObservablePairs``
        Experimental observable/MD observable pairs which are used to calculate
        the Figure of Merit
    FoM_calculator : FigureOfMeritCalculator
        Calculates the FoM `float` from the ``observable_pairs``.
    MD_steps : `int`
        Number of molecular dynamics steps for each step of the refinement
    """

    def __init__(self, simulation: Simulation, exp_datasets: List[dict],
                 fit_parameters: Parameters,
                 minimizer_type: str = 'MMC', FoM_options: dict = None,
                 reset_config: bool = True, MD_steps: int = None,
                 equilibration_steps: int = 0,
                 previous_history: Union[str,Path] = None,
                 verbose: int = 0,
                 print_all_settings: bool = False,
                 **settings: dict):

        self.previous_history = previous_history
        self.step_timings: list = []
        self.simulation = simulation
        self.exp_datasets = exp_datasets
        self.verbose = verbose
        self.timings: dict = {'equilibrate': [],
                        '_run_MD': [],
                        'convert_trajectory': [],
                        'FoM_calculator': [],
                        'minimizer': [],
                        'TOTAL STEP': []}

        # Remove any fixed, tied or parameters equal to 0 as these cannot be refined
        # if a Parameters object, convert to list first for comprehension
        if isinstance(fit_parameters, Parameters):
            fit_parameters = list(fit_parameters.values())
        fit_parameters = [p for p in fit_parameters
                          if (not (p.fixed or p.tied) and p.value != 0)]
        self.fit_parameters = Parameters(fit_parameters)
        self.reset_config = reset_config
        self.equilibration_steps = equilibration_steps
        self.settings = settings
        self.n_steps = settings.get('n_steps')
        self.results_filename = settings.get('results_filename',
                                f'results_{datetime.now().strftime("%Y-%m-%d--%H-%M-%S")}.csv')
        settings['results_filename'] = self.results_filename

        # Minimizer FoM_old is always initialised to infinity, so that first MC
        # step (i.e. the setup) is always accepted.
        # pylint: disable=line-too-long
        # disable this pylint warning as this can't be fixed in a way that looks good
        self.minimizer = MinimizerFactory.create_minimizer(minimizer_type, self,self.fit_parameters,
                                                           previous_history, **settings)
        self.first_loaded_step = bool(previous_history)

        # Create experimental observables from datasets and placeholders for
        # experimental observables calculated from MD
        self.observable_pairs = []
        minimum_MD_steps = 0
        for dset in exp_datasets:
            try:
                use_FFT = dset['use_FFT']
            except KeyError:
                use_FFT = True

            # keep the keys in the _dset_input_check function consistent with the ones retrieved from dset
            self._input_check(dset, inputs = ['type','reader', 'file_name'])
            exp_observable = self._read_observable_from_file(dset['type'],
                                                        dset['reader'],
                                                        dset['file_name'],
                                                        use_FFT)

            if exp_observable.uniformity_requirements:
                exp_observable = self._make_data_uniform(exp_observable)

            MD_observable = self._create_empty_observable(exp_observable, exp_observable.use_FFT)

            self._validate_energy(MD_observable)

            auto_scale = dset.get('auto_scale', False)
            rescale_factor = dset.get('rescale_factor')
            if auto_scale and rescale_factor and self.verbose != -1:
                print('Both `rescale_factor` and `auto_scale` set for file {};'
                      ' scaling will be automated to minimise FoM'
                      ''.format(dset['file_name']))
                rescale_factor = 1.
            elif not rescale_factor:
                rescale_factor = 1.

            observable_pair = ObservablePair(exp_observable,
                                             MD_observable,
                                             dset['weight'],
                                             rescale_factor=rescale_factor,
                                             auto_scale=auto_scale)
            self.observable_pairs.append(observable_pair)

            # Take the largest minimum number of MD_steps needed by any dataset
            min_MD_steps_dset = self._calculate_minimum_MD_steps(
                observable_pair)
            minimum_MD_steps = max(minimum_MD_steps, min_MD_steps_dset)

        if FoM_options is None or FoM_options.get('error') is None:
            FoM_error = 'exp'
        else:
            FoM_error = FoM_options.get('error')

        if FoM_options is None or FoM_options.get('norm') is None:
            FoM_norm = 'data_points'
        else:
            FoM_norm = FoM_options.get('norm')

        self.FoM_calculator = FoMFactory.create_FoM(FoM_error, self.observable_pairs,
                                                    norm=FoM_norm,
                                                    n_parameters=len(self.fit_parameters))

        # Use specified MD_steps if supplied, else calculate
        # cont_slicing produces small sub-trajectories, so calculation is unnecessary
        if MD_steps:
            try:
                assert MD_steps >= minimum_MD_steps
                if self.settings.get('cont_slicing', False):
                    self.MD_steps = MD_steps
                else:
                    # Set self.MD_steps to be the largest number required by any of
                    # our observable pairs
                    maximum_MD_steps = minimum_MD_steps
                    for pair in self.observable_pairs:
                        max_MD_steps_pair = self._calculate_maximum_MD_steps(
                            MD_steps, pair)
                        maximum_MD_steps = max(maximum_MD_steps, max_MD_steps_pair)
                    self.MD_steps = maximum_MD_steps
            except AssertionError as error:
                raise ValueError('Experimental datasets provided require a '
                                 f'minimum MD_steps value of {minimum_MD_steps} in order to '
                                 'calculate observables') from error
        else:
            self.MD_steps = minimum_MD_steps

        # read in resolutions for experimental datasets
        for i, dset in enumerate(exp_datasets):
            resolution_factory = ResolutionFactory()
            dt = self.simulation.time_step * self.simulation.traj_step

            if 'resolution' not in dset.keys():
                # create list of user keys for resolutions to add to the error
                userkeys = []
                for setting in resolution_factory.resolutions:
                    userkeys.append(setting.lower().replace('resolution', ''))
                raise KeyError("A resolution function must be added. Recognised functions are " +
                               str(userkeys) +
                               ". If you meant to apply no resolution,"
                               " then specify resolution as None for the exp_dataset parameters.")

            resolution = resolution_factory.create_instance(dset['resolution'], dset['type'],
                                                            dset['reader'], dt)

            self.observable_pairs[i].exp_obs.resolution = resolution
            self.observable_pairs[i].MD_obs.resolution = resolution

        # setup the dataframe for stdout
        data_array = [
            ["-"],
            [minimizer_type],
            [self.FoM_calculator.__class__.__name__],
            [len(self.observable_pairs)],
            [len(self.fit_parameters)],
        ]

        index_array = [
            '- Attributes',
            '  Minimizer',
            '  FoM type',
            '  Number of observables',
            '  Number of parameters'
        ]

        # Printing settings
        if print_all_settings:
            for item in ["MD_steps", "equilibration_steps", "reset_config", "verbose"]:
                index_array.append(f'  {item}')
                data_array.append([self.__dict__[item]])

            # pylint: disable=consider-using-dict-items
            index_array.append("- Control Settings")
            data_array.append(["-"])
            for setting in settings:
                index_array.append(f'  {setting}')
                data_array.append([settings[setting]])

            index_array.append("- Parameters")
            data_array.append(["-"])
            for parameter in fit_parameters:
                index_array.append(f'  {parameter.name}')
                data_array.append([parameter.value])

            index_array.append("- Experimental Datasets")
            data_array.append(["-"])
            if exp_datasets is not None:
                for dataset in exp_datasets:
                    for key in dataset.keys():
                        index_array.append(f'  {key}')
                        data_array.append([dataset[key]])

            index_array.append("- FoM Options")
            data_array.append(["-"])
            if FoM_options is not None:
                for key in FoM_options.keys():
                    index_array.append(f'  {key}')
                    data_array.append([FoM_options[key]])

        setup_frame = pd.DataFrame(data=data_array,
                                   index=index_array)
        if self.verbose != -1:
            print(f'Control created with:\n{setup_frame.to_string(index=True, header=False)}\n')

        # set up data printer and attach minimiser history to it
        self.data_printer = data_printers[settings.get('data_printer', 'plaintext')]()

    def __str__(self) -> str:
        exp_dataset_types = [dataset['type'] for dataset in self.exp_datasets]

        # plural adds "s" to end of "parameter" if there is more than one parameter
        plural = ("" if len(self.fit_parameters) == 1 else "s")
        return (f"{self.__class__.__name__} refining {len(self.fit_parameters)} parameter{plural} "
                f"using {exp_dataset_types} data types")

    def refine(self, n_steps: int = None) -> None:
        """
        Refines the specified potential parameters

        Parameters
        ----------
        n_steps : int
            Maximum number of steps for the refinement. Must be specified either when creating
            the ``Control`` object or when calling this method. The value specified to this
            method supersedes the value passed (if any) when the ``Control`` object was created.
            If nothing is passed, the method will check if a number was specified when the
            ``Control`` object was created and use that value.

        Examples
        --------
        Perform a refinement with a maximum of 100 steps:

            .. highlight:: python
            .. code-block:: python

            control.refine(100)
        """

        if not n_steps:
            if self.n_steps is None:
                raise ValueError('The number of maximum refinement steps needs to be'
                                 ' specified and greater than 0.')
        else:
            self.n_steps = n_steps

        # calculate verbose steps
        # 4 steps per refinement step, and n + 1 steps total
        verbose_steps = (self.n_steps + 1) * 4
        # initialise step timings list for average step timings at end
        self.step_timings = []

        count = -1

        if self.verbose != -1:
            self.data_printer.print_header(self.minimizer.history)  # creates stdout header
            verbose_manager = VerboseManager.instance()
            verbose_manager.start(verbose_steps, verbose=self.verbose)
            while count < n_steps and not self.minimizer.has_converged():
                if count >= 0:
                    self.equilibrate(self.equilibration_steps)

                verbose_manager.header(f"Step {count + 1}")
                self.step()  # advance the refinement by one step
                count += 1
                if self.verbose == 3:  # if progress bar is there, ensure data is on new line
                    print("")
                self.data_printer.print_data(self.minimizer.history)

        else:
            while count < n_steps and not self.minimizer.has_converged():
                if count >= 0:
                    self.equilibrate(self.equilibration_steps)
                self.step()  # advance the refinement by one step
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
        result_string = self.minimizer.present_result()
        if self.verbose != -1:
            print(result_string)

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
            if self.verbose != -1:
                print(
                f'\nAutomatic Scale Factors\n{scaling_df.to_string(index=True, header=False)}')

        if self.verbose >= 1:
            average_timing = statistics.mean(self.step_timings)
            if self.verbose != -1:
                print(
                f'\nAverage time per step was {np.round_(average_timing, 2)} seconds.')

        verbose_manager.finish("Refinement")

    def minimize(self, n_steps: int,
                 minimize_every: int = 10,
                 verbose: bool = False, output_log: str = None,
                 work_dir: str = None, **settings: dict) -> None:
        """
        Performs an MD run intertwined with periodic structure relaxation.
        This way after a local minimum is found, the system is taken
        out of the minimum to explore a larger volume of the parameter
        space.

        Parameters
        ----------
        n_steps : int
            Total number of the MD run steps
        minimize_every: int, optional
            Number of MD steps between two consecutive minimizations
        verbose: bool, optional
            Whether to print statements when the minimization has been started and completed
            (including the number of minimization steps and time taken). Default is `False`.
        output_log: str, optional
            Log file for the MD engine to write to. Default is `None`.
        work_dir: str, optional
            Working directory for the MD engine to write to. Default is `None`.
        **settings
            ``etol`` (`float`)
                If the energy change between iterations is less than ``etol``,
                minimization is stopped. Default depends on engine used.
            ``ftol`` (`float`)
                If the magnitude of the global force is less than ``ftol``,
                minimization is stopped. Default depends on engine used.
            ``maxiter`` (`int`)
                Maximum number of iterations of a single structure
                relaxation procedure. Default depends on engine used.
            ``maxeval`` (`int`)
                Maximum number of force evaluations to perform. Default depends
                on engine used.
        """

        self.simulation.minimize(n_steps,minimize_every,verbose,
                                 output_log,work_dir, **settings)

    def equilibrate(self, n_steps: int = None, verbose: bool = False,
            output_log: str = None, work_dir: str = None, **settings: dict) -> None:

        """
        Run molecular dynamics to equilibrate the ``Universe``.

        Parameters
        ----------
        n_steps : int
            Number of simulation steps to run.
        verbose: bool, optional
            Whether to print statements upon starting and completing the run.
            Default is `False`.
        output_log: str, optional
            Log file for the MD engine to write to. Default is `None`.
        work_dir: str, optional
            Working directory for the MD engine to write to. Default is `None`.
        """
        if not n_steps:
            self.simulation.auto_equilibrate()
        else:
            self.simulation.run(n_steps, True, verbose, output_log, work_dir,**settings)

    def step(self) -> None:
        """
        Do a full step: generate and run MD to calculate FoM for existing
        parameters, iterate parameters a step forward and reset MD (phasespace)
        if previous step was rejected and reset_config = true
        """
        verbose_manager = VerboseManager.instance()
        verbose_manager.start(4, verbose=self.verbose)
        # Generate FoM by running MD for this step and then calculate FoM
        if self.first_loaded_step:
            fom = self.minimizer.FoM_old
            self.first_loaded_step = False
        else:
            fom = self._generate_FoM()

        verbose_manager.step("Selecting new parameters and updating engine")
        # Select new parameters to consider
        self.minimizer.step(fom)
        # Update the MD engine with new parameters
        self._update_engine_parameters()
        # When reset_config=true reset the MD (phasespace) back if the
        # previous step was rejected
        if self.reset_config:
            if self.minimizer.state_changed:
                # Set MD engine to remember new config
                self.simulation.engine.save_config()
            else:
                # Set MD engine to reset to old config
                self.simulation.engine.reset_config()

        self.minimizer.write_history(self.results_filename)

        step_timings = verbose_manager.finish("Refinement step")
        self.step_timings.append(step_timings)


    def plot_results(self, filename: str=None, points: int=100000, MH_norm: float=20.0) -> None:
        """
        Instantiates an insstance of the PlotResults class and generates a cornerplot
        from the data in self.results_filename

        Parameters
        ----------
        filename : str, optional
            The filename and path to the history file. Defaults to None which
            then uses self.results_filename
        points : int, optional
            The number of samples to initially generate, defaults to 100,000
        MH_norm : float, optional
            The denominator of the exponent, controlling how likley points are to be kept,
            defaults to 20.0

        Returns
        -------
        corner plot : Matplotlib.figure.Figure
            A plot displaying every parameter combination with their variances and covariances
        """
        if filename is None:
            filename = self.results_filename
        plotter = PlotResults(filename, MH_norm=MH_norm, points=points,
                              quantiles=[0.34, 0.5, 0.68])
        cornerplot, means, stds = plotter.create_cornerplot()

        if self.verbose != -1:
            print(f'Parameter means = {means}, Parameter errors = {stds}')

        return cornerplot


    def _generate_FoM(self) -> float:
        """
        Run the MD for an iteration/step, calculate observable, compare with
        observed and return the FoM

        Returns
        -------
        `float`
            Non-negative `float` FoM
        """
        self._run_MD()
        self._calculate_observables(self.simulation, self.observable_pairs)

        FoM_value = self.FoM_calculator.calculate()

        return FoM_value

    def _run_MD(self) -> None:
        """
        Run a molecular dynamics simulation
        """

        self.simulation.run(self.MD_steps, verbose=False)

    def _update_engine_parameters(self) -> None:
        """
        Update the force field parameters of the MD engine
        """

        self.simulation.engine.update_parameters()

    @staticmethod
    def _read_observable_from_file(obstype: str, reader: str, file_name: str,
                                   use_FFT: bool = True) -> Observable:
        """
        Creates an Observable of the specified type and reads in data from file

        Parameters
        ----------
        obstype : str
            The ``type`` of the ``Observable``.
        reader : str
            The ``type`` of the ``Reader``.
        file_name : str
            The absolute or relative path and the file name.
        use_FFT: bool, optional
            boolean determining if the FFT should be used, default is True

        Returns
        -------
        ``Observable``
            An ``Observable`` of specified ``type``
        """

        observable = ObservableFactory.create_observable(obstype)
        observable.read_from_file(reader=reader, file_name=file_name)
        observable.use_FFT = use_FFT
        return observable

    @staticmethod
    def _create_empty_observable(exp_observable: Observable, use_FFT: bool = True) -> Observable:
        """
        Creates a ``Observable`` without data but with independent variables
        specified from another ``Observable``.  This is a placeholder in which
        the ``Observable`` can be calculated from an MD trajectory.

        Parameters
        ----------
        exp_observable : Observable
            An ``Observable`` with defined independent variables.
        use_FFT: bool, optional
            boolean determining if the FFT should be used, default is True

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
        observable.use_FFT =  use_FFT
        return observable

    def _calculate_observables(self,
                               simulation: Simulation,
                               observable_pairs: 'list[ObservablePair]') -> None:
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
        verbose_manager = VerboseManager.instance()
        # this is a subprocess, so we don't bother giving maximum steps since it won't be used
        verbose_manager.start(0, verbose=self.verbose)

        verbose_manager.step("Converting trajectory")
        trj = simulation.engine.convert_trajectory()

        verbose_manager.step("Calculating observables from the MD trajectory")
        for pair in observable_pairs:
            obs_timings = pair.MD_obs.calculate_from_MD(trj, verbose=self.verbose, **self.settings)
            if self.verbose == 1 and obs_timings is not None:
                for key, value in obs_timings.items():
                    if key not in self.timings:
                        self.timings[key] = []
                    self.timings[key] += value
        verbose_manager.finish("Calculating observables")

    def _calculate_minimum_MD_steps(self, observable_pair: ObservablePair) -> int:
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
                                    observable_pair: ObservablePair) -> int:
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

    @staticmethod
    def _is_data_uniform(observable: Observable) -> Dict[str, Dict[str, bool]]:
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
        >>> _is_data_uniform(observable)
        {'E': {'uniform': True, 'zeroed': True}, 'Q': {'uniform': True, 'zeroed': False}}
        """
        uniformity_dict = {}
        for var_key, var_data in observable.independent_variables.items():
            uniform_data = np.linspace(
                min(var_data), max(var_data), num=len(var_data))
            is_uniform = np.allclose(var_data, uniform_data, rtol=1e-5)
            uniformity_dict[var_key] = {
                'uniform': is_uniform, 'zeroed': var_data[0] == 0}
        return uniformity_dict

    def _make_data_uniform(self, observable: Observable) -> Observable:
        """
        Takes an ``Observable``, checks the requirements for its ``independent_variables``
        to be uniform or start at zero, creates uniform grids for the variables that do not
        satisfy their requirement, interpolates the ``dependent_variables`` as needed,
        and returns an ``Observable`` with the uniform/interpolated variables.
        Limited to ``Observables`` with two-dimensional ``dependent_variables`` (e.g. SQw).
        This may change in future.

        Parameters
        ----------
        observable : Observable
            An ``Observable`` for which the independent variables
            need to be made uniform / to start at zero. Currently
            limited to ``Observables`` for which the ``dependent_variables`` are two-dimensional.

        Returns
        -------
        ``Observable``
            Returns a copy of the passed ``Observable`` with the independent variables put onto
            uniform grid (for the variables where that is necessary)
            and the dependent variables interpolated onto the same grid
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
            # if the variable has a requirement AND it is not satisfied
            # (for either uniformity OR zero-start)
            # then add it to the list of variables that need to be changed
            if (var_required['uniform'] and not var_state['uniform']) or \
                    (var_required['zeroed'] and not var_state['zeroed']):
                indep_vars_to_be_changed.append(var_key)

        # if all uniformity requirements are already satisfied
        # simply return the original observable
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
            # if uniformity requirements are satisfied already,
            # add the data points to the helper dictionary
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
                raise AssertionError('Expected experimental dataset to only have one dependent '
                                     f'variable entry for {var_key}, '
                                     f'but found {len(data_list)} instead') from error

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
                err_data[err_data == float('inf')] = 0
                err_interpol = interp1d(x_data, err_data)
                err_uniform = err_interpol(x_uniform)
                err_uniform[err_uniform == 0.] = float('inf')
            # interpolation for 2D
            elif var_dimension == 2:
                # note: the interp2d interpolation function requires input of the form
                # interp2d(x, y, z)
                # where if np.size(x)=m and np.size(y)=n then np.shape(z)=(n,m)
                # E.g. if x = [0,1,2]; y = [0,3]; z = [[1,2,3], [4,5,6]]
                # Because Observable.dependent_variables_structure gives the order in which
                # the independent variables are represented in the np.shape of the data,
                # we have to reverse the order of the x and y arrays for interp2d:
                x_data = observable.independent_variables[var_indexing[var_key][1]]
                y_data = observable.independent_variables[var_indexing[var_key][0]]
                data_interpol = interp2d(x_data, y_data, data)
                # get the independent_variables that satisfy the uniformity requirements
                # as created earlier
                x_uniform = indep_var_uniform[var_indexing[var_key][1]]
                y_uniform = indep_var_uniform[var_indexing[var_key][0]]
                uniform_data = data_interpol(x_uniform, y_uniform)
                # repeat the interpolation for the errors
                err_data = observable.errors[var_key][0]
                err_data[err_data == float('inf')] = 0
                err_interpol = interp2d(x_data, y_data, err_data)
                err_uniform = err_interpol(x_uniform, y_uniform)
                err_uniform[err_uniform == 0.] = float('inf')
            else:
                raise NotImplementedError(
                    'Only 1D and 2D data can currently be made uniform')
            # save the uniform data and errors back into the Observable
            observable.dependent_variables[var_key] = [uniform_data]
            observable.errors[var_key] = [err_uniform]
        # finally, set the independent variables of the ``Observable`` to the uniform ones
        observable.independent_variables = indep_var_uniform
        return observable

    def _validate_energy(self, obs: Observable) -> None:
        """
        Try and validate the energy of the ``Observable`` provided, and pass if
        it does not have a ``validate_energy`` function itself. The time step and trajectory step
        may be changed in the validate_energy method of the specific observable if necessary, to
        achieve an energy separation that matches that of the experimental data.

        Parameters
        ----------
        obs : Observable
            ``Observable`` to validate

        Returns
        -------
        None
        """

        with suppress(AttributeError):

            changed, traj_step, time_step, self.dt_required \
                  = obs.validate_energy(self.simulation.time_step)
            if changed:
                self.simulation.traj_step = traj_step
                self.simulation.time_step = time_step
                logging.warning(" The given traj_step and time_step values were not"
                    " compatibile with the dataset specified.\nThe values "
                    "(whilst prioritising time_step) have been changed to"
                    " traj_step: %d, and time_step: %f. \n"
                    "Context: for this dataset, traj_step multiplied by time_step"
                    " must be ~= %f (6 d.p). \n" , traj_step,time_step,self.dt_required)

    def _input_check(self, general_set, inputs) -> None:
        """

        Handles error for retrieving data from a set where the input is not found.
        This was made in a general way to be used for any dataset or set of inputs.

        Parameters
        ----------
        general_set : A general dataset
            A variable that contains a set of information for input checking against,
            for example 'dset' contains the observable type, file_name and reader type,
            for checking inputs.

        Returns
        -------
        None
        """

        for input_check in inputs:
            try:
                general_set[input_check]
            except KeyError as error:
                raise KeyError("There was an issue retrieving the input: "
                               f" {input_check} "
                                "from the dataset, please check your inputs again.") from error
