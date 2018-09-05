"""A module for performing the refinement

AUTHOR :    Thomas Farmer        START DATE :    2018-6-18 15:47:47"""


from MDMC.trajectory_analysis.exp_obs_factory \
    import ExperimentalObservableFactory
import MDMC.refinement.minimizer as minim
import MDMC.refinement.FoM


class MDMCControl(object):

    """
    Controls the MDMC refinement
    """

    # TODO: Better implementation of minimizer instantiation - maybe factory pattern
    MINIMIZER_DICT = {"MMC":minim.MMC}

    # TODO: Change __init__ so that a minimizer instance is passed
    # TODO: Find a better solution for passing settings for refinement - or can it be avoided altogether?
    def __init__(self, n_steps, MD_engine, exp_datasets, fit_params,
        minimizer_type, FoM_calculator = FoM.StandardFoMCalculator(),
        **settings):

        """
        Creates experimental observables from datasets and placeholders for
        experimental observables calculated from MD

        Minimizer FoM_old is initialized to infinity so that the first MC step (i.e. the
        setup) is always accepted. Settings for calculating observables can be
        specified.
        """

        self.MD_engine = MD_engine
        self.exp_datasets = exp_datasets
        self.fit_params = fit_params
        self.minimizer = self.MINIMIZER_DICT[minimizer_type]()
        self.FoM_calculator = FoM_calculator
        self.settings = settings
        self.n_steps = n_steps
        self.count = 0

        self.exp_observables = []
        for dataset in exp_datasets:
            exp_observables.append(self._read_observable_from_file(dataset))

        self.MD_observables = []
        for exp_observable in exp_observables:
            MD_observables.append(self._create_empty_observable(exp_observable))

        self.refine(fit_params)


    def test_convergence(self):

        """
        Tests whether the refinement has converged
        """

        raise NotImplementedError

    def refine(self, fit_params):

        """
        Refines the specified potential parameters
        """

        self.generate_FoM()

        for _ in range(n_steps):
            self.minimizer.step()
            if self.minimizer.has_converged():
                break
            self.generate_FoM()


        while True:

            if self.minimizer.test_convergence() or count >= n_steps:

                break

            if self.minimizer.change_state():
                self.minimizer.FoM_old = self.minimizer.FoM
                self.minimizer.fit_params_old = fit_params
                self.count += 1
                self.minimizer.change_parameters(self.fit_params)
            else:
                self.count += 1

        print self.fit_params

    def generate_FoM(self):

        """
        The methods requires to generate a FoM
        """

        self.run_MD()
        self.calculate_observables(self.MD_engine, self.MD_observables)
        self.minimizer.FoM = self.calculate_FoM(self.exp_observables,
            self.MD_observables)


    def run_MD(self):

        self.MD_engine.run()

    def _read_observable_from_file(self, dataset):

        """
        Creates an observable of the specified type and reads in data from file
        """

        observable = ExperimentalObservableFactory.create_observable(
            dataset['type'])
        return observable.read_from_file(reader=dataset['reader'],
            file_name=dataset['file_name'])

    def _create_empty_observable(self, exp_observable):

        """
        Creates a observable without data but with independent variables
        specified from another observable

        This is a placeholder in which the the observable can be calculated from
        an MD trajectory
        """

        observable = ExperimentalObservableFactory.create_observable(
            exp_observable.name)
        observable.independent_variables = exp_observable.independent_variables
        return observable

    def _calculate_observables(self, MD_engine, observables):

        """
        Calculates all of the observables from the MD trajectory/configurations
        """

        for observable in observables:
            observable.calculate_from_MD(MD_engine, self.settings)

    def _calculate_FoM(self, exp_observables, MD_observables):

        data_pairs = [dict(chain(tup[0].items(), tup[1].items))
            for tup in zip(exp_observables, MD_observables)]
        return self.FoM_calculator().calculate_FoM(data_pairs)
