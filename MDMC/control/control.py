"""A module for performing the refinement"""

from copy import deepcopy

import numpy as np

from MDMC.common.decorators import repr_decorator
from MDMC.MD.parameters import Parameters
from MDMC.refinement import minimizer, FoM
from MDMC.trajectory_analysis.observables.obs_factory \
    import ObservableFactory


@repr_decorator('simulation', 'exp_datasets', 'FoM_calculator', 'minimizer',
                'reset_config', 'fit_params', 'settings')
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
        Figure of Merit calculation(``weighting``).
    fit_params : Parameters, list of Parameter
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
    **settings
        Keyword arguments.

    Example
    -------
    An example of an exp_dataset list is::

        [{'file_name':data.LAMP_SQW_FILE,
          'type':'SQw',
          'reader':'LAMPSQw',
          'weight':1.},
         {'file_name:data.ANOTHER_FILE',
          'type':'FQt',
          'reader':'GENERIC_READER',
          'weight':0.5}]

    Attributes
    ----------
    simulation : Simulation
        The ``Simulation`` on which is used to perform the refinement
    exp_datasets : `list of dicts`
        One `dict` per experimental dataset used for the refinement
    fit_params : Parameters
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

    def __init__(self, simulation, exp_datasets, fit_params, MC_norm=1.,
                 minimizer_type='MMC', FoM_type='standard',
                 reset_config=True, **settings):

        self.simulation = simulation
        self.exp_datasets = exp_datasets
        self.fit_params = Parameters(fit_params)
        # Minimizer FoM_old is always initialised to infinity, so that first MC
        # step (i.e. the setup) is always accepted.
        self.minimizer = self.MINIMIZER_DICT[minimizer_type](MC_norm,
                                                             self.fit_params)
        self.reset_config = reset_config
        self.settings = settings

        # Create experimental observables from datasets and placeholders for
        # experimental observables calculated from MD
        self.observable_pairs = []
        for dset in exp_datasets:
            exp_observable = self._read_observable_from_file(dset['type'],
                                                             dset['reader'],
                                                             dset['file_name'])
            MD_observable = self._create_empty_observable(exp_observable)
            observable_pair = FoM.ObservablePair(exp_observable,
                                                 MD_observable,
                                                 dset['weight'])
            self.observable_pairs.append(observable_pair)

        self.FoM_calculator = self.FOM_DICT[FoM_type](self.observable_pairs)

        # Use specified MD_steps if supplied, else calculate
        self.MD_steps = settings.get('MD_steps')

    def __str__(self):

        exp_dataset_types = [dataset['type'] for dataset in self.exp_datasets]
        return "{0} refining {1} {2} using {3} data types".format(
            self.__class__.__name__,
            len(self.fit_params),
            'parameter' if len(self.fit_params) == 1 else 'parameters',
            exp_dataset_types)

    def refine(self, n_steps):

        """
        Refines the specified potential parameters

        Parameters
        ----------
        n_steps : int
            maximum number of steps for the refinement
        """

        count = -1

        while count < n_steps and not self.minimizer.has_converged():

            fom = self._generate_FoM()
            self.minimizer.step(fom)
            print(self.minimizer.history.loc[[count + 1]].to_string(
                header=False))
            self.simulation.engine.update_parameters()
            count += 1

            if self.reset_config:
                if self.minimizer.state_changed:
                    # Set MD engine to remember new config
                    self.simulation.engine.save_config()
                else:
                    # Set MD engine to reset to old config
                    self.simulation.engine.reset_config()

        # Try/except accounts for n_steps <= -1
        try:
            # Reset the minimizer params to those from the final FoM
            self.minimizer.reset_params()
            self.simulation.engine.update_parameters()
        except TypeError:
            pass

        self.minimizer.write_history('results.dat')

    def _generate_FoM(self):

        """
        The methods required to generate a FoM

        Returns
        -------
        `float`
            Non-negative `float` FoM
        """

        self._run_MD()
        self._calculate_observables(self.simulation, self.observable_pairs)

        # TODO: Remove arbitrary normalization
        for pair in self.observable_pairs:
            exp_norm = np.max(pair.exp_obs._dependent_variables['SQw'])
            md_norm = np.max(pair.MD_obs._dependent_variables['SQw'])
            pair.exp_obs._dependent_variables['SQw'] /= exp_norm
            pair.exp_obs._errors['SQw'] /= exp_norm
            pair.MD_obs._dependent_variables['SQw'] /= md_norm
            pair.MD_obs._errors['SQw'] /= md_norm

        return self._calculate_FoM()

    def _run_MD(self):

        """
        Run a molecular dynamics simulation
        """

        self.simulation.run(self.MD_steps)

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

        # slc = self._calculate_trajectory_slice(self.observable_pairs[0].exp_obs,
        # )
        trj = simulation.engine.convert_trajectory()
        for pair in observable_pairs:
            pair.MD_obs.calculate_from_MD(trj, **self.settings)

    def _calculate_FoM(self):

        """
        Calculates the total FoM for all ``ObservablePair`` objects

        Returns
        -------
        `float`
            Non-negative `float` FoM
        """

        return self.FoM_calculator.calculate()

    def _calculate_MD_steps(self):

        """
        Calculates the minimum number of steps required for the MD engine in
        order to calculate MD ``Observables`` objects with the same independent
        variables as the experimental ``Observable`` objects.

        THIS METHOD IS NOT IMPLEMENTED

        Returns
        -------
        `int`
            Number of molecular dynamics steps

        Raises
        ------
        NotImplementedError
            THIS METHOD IS NOT IMPLEMENTED
        """

        raise NotImplementedError
    #
    # def _calculate_trajectory_slice(self, exp_obs, traj_step):
    #
    #     """
    #     Calculates the slice of the trajectory that is required for calculating
    #     the MD observables for the same independent variables as the
    #     experimental observables
    #
    #     Arugments:
    #     exp_obs - an observable with origin='experiment'
    #     traj_step - the integer step size of the MD trajectory captures
    #
    #     Returns:
    #     A slice
    #     """
    #
    #     n_steps = len(exp_obs.E)
    #
    #     start = 0
    #     stop = self.MD_steps / traj_step
    #     step = int(round((self.MD_steps - start) / n_steps))
    #
    #     return slice(start, stop + 1, step)
