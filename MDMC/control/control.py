"""A module for performing the refinement

AUTHOR :    Thomas Farmer        START DATE :    2018-6-18 15:47:47"""

import numpy as np

from MDMC.refinement import minimizer, FoM
from MDMC.trajectory_analysis.observables.obs_factory \
    import ObservableFactory


class MDMCControl(object):

    """
    Controls the MDMC refinement
    """

    # TODO: Better implementation of minimizer instantiation - maybe factory pattern
    MINIMIZER_DICT = {"MMC":minimizer.MMC}
    FOM_DICT = {"standard":FoM.StandardFoMCalculator}

    def __init__(self, MD_engine, exp_datasets, fit_params, MC_norm=1.,
        minimizer_type='MMC', FoM_type='standard', **settings):

        """
        Creates experimental observables from datasets and placeholders for
        experimental observables calculated from MD

        Minimizer FoM_old is initialized to infinity so that the first MC step (i.e. the
        setup) is always accepted. Settings for calculating observables can be
        specified.

        Arguments:
        MD_engine - MDEngine
        exp_datasets - a list of dictionaries with one dictionary fof each
        dataset. Each dictionary contains the file name, the type of observable,
        the reader required for the file, and the weighting of the dataset in
        the Figure of Merit calculation. For example:

        exp_datasets = [{'file_name':data.LAMP_SQW_FILE,
                         'type':'SQw',
                         'reader':'LAMPSQw',
                         'weight':1.},
                        {'file_name:data.ANOTHER_FILE',
                         'type':'FQt',
                         'reader':'GENERIC_READER',
                         'weight':0.5}]

        MC_norm - a float which determines the accept/reject ratio of the MC
        fit_params - a list of all parameters which will be refined
        minimizier_type - a string with the minimizer type. 'MMC' is the default
        FoM_type - a string with the type of Figure of Merit calculation
        """

        self.MD_engine = MD_engine
        self.exp_datasets = exp_datasets
        self.fit_params = fit_params
        self.minimizer = self.MINIMIZER_DICT[minimizer_type](MC_norm,
                                                             self.fit_params)
        self.settings = settings

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

    def refine(self, n_steps):

        """
        Refines the specified potential parameters

        Arguments:
        n_steps - integer maximum number of steps for the refinement
        """

        count = -1

        while count < n_steps and not self.minimizer.has_converged():

            FoM = self.generate_FoM()
            self.minimizer.step(FoM)
            count += 1

        # Try/except accounts for n_steps = -1
        try:
            # Reset the minimizer params to those from the final FoM
            self.minimizer.reset_params()
        except TypeError:
            pass

        print self.fit_params

    def generate_FoM(self):

        """
        The methods required to generate a FoM

        Returns:
        Non-negative float FoM
        """

        self.run_MD()
        self._calculate_observables(self.MD_engine, self.observable_pairs)

        # TODO: Remove arbitrary normalization
        for pair in observable_pairs:
            pair.exp_obs._dependent_variables['SQw'] /= \
                np.max(pair.exp_obs._dependent_variables['SQw'])
            pair.MD_obs._dependent_variables['SQw'] /= \
                np.max(pair.MD_obs._dependent_variables['SQw'])

        return self._calculate_FoM()

    def run_MD(self):

        self.MD_engine.run(self.MD_steps)

    def _read_observable_from_file(self, type, reader, file_name):

        """
        Creates an observable of the specified type and reads in data from file

        Arguments:
        type - string specifying the type of the observable
        reader - string specifying the type of the reader
        file_name - string with the absolute or relative path and the file name

        Returns:
        Observable of type 'type'
        """

        observable = ObservableFactory.create_observable(type)
        observable.read_from_file(reader=reader, file_name=file_name)
        return observable

    def _create_empty_observable(self, exp_observable):

        """
        Creates a observable without data but with independent variables
        specified from another observable.  This is a placeholder in which
        the observable can be calculated from an MD trajectory.

        Arguments:
        exp_observable - an observable with defined independent variables

        Returns:
        An observable with only independent variables and origin = 'MD'
        """

        observable = ObservableFactory.create_observable(exp_observable.name)
        observable.origin = 'MD'
        observable.independent_variables = exp_observable.independent_variables
        return observable

    def _calculate_observables(self, MD_engine, observable_pairs):

        """
        Calculates all of the observables from the MD trajectory/configurations

        Arguments:
        MD_engine - an MDEngine object
        observable_pairs - a list of observable pairs
        """

        trj = MD_engine.engine.convert_trajectory()
        for pair in observable_pairs:
            pair.MD_obs.calculate_from_MD(trj, **self.settings)

    def _calculate_FoM(self):

        """
        Calculates the total FoM for all observable pairs

        Returns:
        Non-negative float FoM
        """

        return self.FoM_calculator.calculate()

    def _calculate_MD_steps(self):

        """
        Calculates the minimum number of steps required for the MD engine in
        order to calculate MD observables with the same independent variables as
        the experimental observables.

        Returns:
        integer number of steps
        """

        raise NotImplementedError
