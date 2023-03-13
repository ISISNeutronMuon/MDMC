import numpy as np
import pandas as pd
import corner
from scipy.optimize import OptimizeResult

from skopt import Optimizer



class PlotResults():
    """
    A class to read in any refinement, create a Gaussain Process Optimizer and then do sampling 
    on the result to create a corner plot.
    """

    def __init__(self, filename):
        self.filename = filename
        self.optimizer = Optimizer(self.parameter_bounds,"GP", acq_func="gp_hedge",
                acq_optimizer="sampling", initial_point_generator="lhs", n_initial_points=20,
                model_queue_size=1)
        self.parameter_names, self.parameter_coordinates, self.FoMs = self.get_measured_points()

    def get_measured_points(self) -> tuple:
        """Opens the dataframe in `filename` and extracts the measured parameters names, values
        and associated figures of merit.
        Returns:
        --------
        tuple of (parameter names, parameter coordinates, FoM's) 
        """
        records = pd.read_csv(self.filename, delimiter=',')
        records = records.astype(dtype=float, errors='ignore')
        # Convert to float where possible (i.e. not a string)

        FoMs = records['FoM'].to_list()

        records = records.drop(columns=['Unnamed: 0', 'FoM', 'Change state'])
        # TODO this is hard coded to creation of history, may want to change

        coordinates = records.values.tolist()
        names = records.tolist()

        return names, coordinates, FoMs

    @staticmethod
    def _expected_minimum_random_sampling(optimized_result: OptimizeResult,
                    n_random_starts: int=100000) -> 'tuple[list, float, list, list[list]]':
        """
        This is almost verbatim a copy of code from scikit-optimize but with the samples as
        an additional output:
        https://github.com/scikit-optimize/scikit-optimize/blob/de32b5fd2205a1e58526f3cacd0422a26d315d0f/skopt/utils.py#L259

        Parameters
        ----------
        optimized_result : `OptimizeResult`, scipy object
            The optimization result returned by a `skopt` minimizer.
        n_random_starts : int, default=100000
            The number of random points for the minimization of the surrogate
            model.

        Returns
        -------
        min_x : list
            location of the minimum.
        y_random[index_best_objective] : float
            the surrogate function value at the minimum.
        y_random : np.array
            An array of length "n_random_starts" containing surrogate function values at each point
        random_samples : list[list]
            A list of length "n_random_starts" containing the coordinates of each prediction
        """

        # sample points from search space, set a random seed for reproducibility = 7 w.l.o.g.
        random_samples = optimized_result.space.rvs(n_random_starts, random_state=7)

        # make estimations with surrogate
        model = optimized_result.models[-1]
        y_random = model.predict(optimized_result.space.transform(random_samples))
        index_best_objective = np.argmin(y_random)
        min_x = random_samples[index_best_objective]

        return min_x, y_random[index_best_objective], y_random, random_samples


    @staticmethod
    def _remove_points(chi_squared: 'list[float]', coords: 'list[list]',
                       MC_norm: float=20.0) -> 'tuple[list, list]':
        """
        Removes points with poor figure of merit based on a Metropolis-Hastings type rule,
        where the likelihood of keeping a point is dependent on the exponent of the difference
        between its figure of merit, and that of the best one found, divided by MC_norm.

        Parameters
        ----------
        chi_squared : list[float]
            A list of the predicted chi-squared value at each coordinate
        coords : list[list]
            A list of the coordinates at which all of the chi-squared predictions are made
        MC_norm : float, optional
            The denominator of the exponent used to control the liklihood of keeping a point,
            defaults to 20.0

        Returns
        -------
        reduced_chi : list
            A list of the remaining chi-squared points
        reduced_coords : list[list]
            A list of the remaining coordinates
        """
        np.random.seed(16)  # Set for reproducible output - will always retain same points
        lowest_chi = min(chi_squared)

        points_to_keep = np.random.random(size=chi_squared.shape) < \
                         np.exp((lowest_chi - chi_squared)/(lowest_chi/MC_norm))
        reduced_chi=chi_squared[points_to_keep]
        reduced_coords = np.array(coords)[points_to_keep]

        return reduced_chi, reduced_coords

    def plot_results(self, points: int=100000, MC_norm: float=20.0) -> None:
        """
        Performs a random sample across the coordinate space giving a predicted figure of merit at
        every point. Then removes points with poor figures of merit, according to a
        Metropolis-Hastings type rule, where the likelihood of keeping a point is dependant on the
        exponent of the difference between its figure of merit, and that of the best one found,
        divided by MC_norm. A corner plot is then returned (a matplotlib figure object), which can
        be displayed or exported.

        Parameters
        ----------
        points : int, optional
            The number of samples to initially generate, defaults to 100,000
        MC_norm : float, optional
            The denominator of the exponent, controlling how likley points are to be kept,
            defaults to 20.0

        Returns
        -------
        corner plot : Matplotlib.figure.Figure
            A plot displaying every parameter combination with their variances and covariances
        """
        try:
            _, _, y_random, coords = \
            self._expected_minimum_random_sampling(self.minimizer.optimizer, n_random_starts=points)
        except IndexError:
            msg = ("\n \n Your model has not been run for enough iterations to make a reasonable "
                   "guess at the best figure of merit. Please run for at least 20 steps. \n")
            print(msg)

            return None

        _, reduced_coordinate_list = self._remove_points(y_random, coords, MC_norm)

        data = np.empty(shape=np.array(reduced_coordinate_list).shape)
        for i in range(np.array(reduced_coordinate_list).shape[1]):
            data[:,i] = np.array(reduced_coordinate_list)[:,i]

        labels = [str(name) for name in self.fit_parameters]
        cornerplot = corner.corner(data, labels = labels, quantiles = [0.34, 0.5, 0.68])

        return cornerplot