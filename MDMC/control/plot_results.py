"""A module for plotting data and results of a minimization."""
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import corner
import IPython.display

from skopt import Optimizer


class PlotResults():
    """
    A class to read in any completed refinement history file, create a Gaussain Process Optimizer
    and then do sampling on the result to create a corner plot.

    Parameters:
    -----------
    filename : str
        path to the file to load in the refinement history
    quantiles : list, optional
        optional, list of the quantiles to be plotted on the corner plot, defaults
        to [0.34, 0.5, 0.68], e.g. 1-sigma
    MH_norm : float, optional
        The Metropolis-Hastings normalising factor to determine if points should be
        kept or not, defaults to 20
    points : int, optional
        Number of points to plot on the corner plot, defaults to 100,000
    """

    def __init__(self, filename: str, quantiles: 'list[float]'=None,
                 MH_norm: float=20, points: int =100000):
        self.filename = filename
        self.quantiles = [0.34, 0.5, 0.68] if quantiles is None else quantiles
        self.MH_norm = MH_norm
        self.points = points

        self.parameter_names, self.parameter_coords,\
        self.minmax_coords, self.FoMs = self.get_measured_points()

        # Create the optimizer
        self.optimizer = Optimizer(self.minmax_coords,"GP", acq_func="gp_hedge",
                                   acq_optimizer="sampling", model_queue_size=1)
        # Train the optimizer
        self.optimizer.tell(self.parameter_coords, self.FoMs)

    def get_measured_points(self) -> tuple:
        """Opens the dataframe in `filename` and extracts the measured parameters names, values
        and associated figures of merit.
        Returns:
        --------
        tuple of (parameter names, parameter coordinates, min and max parameters, FoM's)
        """
        records = pd.read_csv(self.filename, delimiter=',')
        records = records.astype(dtype=float, errors='ignore')
        # Convert to float where possible (i.e. not a string)

        FoMs = records['FoM'].to_list()
        records = records.drop(columns=['Unnamed: 0', 'FoM', 'Change state',
                                        'Pred coords', 'Pred FoM'], errors='ignore')
        # TODO this is hard coded to creation of history, may want to change

        coordinates = records.values.tolist()
        names = records.columns.tolist()
        minmax_coordinates = [(min(np.array(coord)),
                               max(np.array(coord))) for coord in np.array(coordinates).T]
        return names, coordinates, minmax_coordinates, FoMs


    def _expected_minimum_random_sampling(self) -> 'tuple[list, float, list, list[list]]':
        """
        This is almost verbatim a copy of code from scikit-optimize but with the samples as
        an additional output:
        https://github.com/scikit-optimize/scikit-optimize/blob/de32b5fd2205a1e58526f3cacd0422a26d315d0f/skopt/utils.py#L259

        Returns
        -------
        min_x : list
            location of the minimum.
        y_random[index_best_objective] : float
            the surrogate function value at the minimum.
        y_random : np.array
            An array of length "self.points" containing surrogate function values at each point
        random_samples : list[list]
            A list of length "self.points" containing the coordinates of each prediction
        """

        # sample points from search space, set a random seed for reproducibility = 7 w.l.o.g.
        random_samples = self.optimizer.space.rvs(self.points, random_state=7)

        # make estimations with surrogate
        model = self.optimizer.models[-1]
        y_random = model.predict(self.optimizer.space.transform(random_samples))
        index_best_objective = np.argmin(y_random)
        min_x = random_samples[index_best_objective]

        return min_x, y_random[index_best_objective], y_random, random_samples


    def _remove_points(self, chi_squared: 'list[float]',
                       coords: 'list[list]') -> 'tuple[list, list]':
        """
        Removes points with poor figure of merit based on a Metropolis-Hastings type rule,
        where the likelihood of keeping a point is dependent on the exponent of the difference
        between its figure of merit, and that of the best one found, divided by MH_norm.

        Parameters
        ----------
        chi_squared : list[float]
            A list of the predicted chi-squared value at each coordinate
        coords : list[list]
            A list of the coordinates at which all of the chi-squared predictions are made

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
                         np.exp((lowest_chi - chi_squared)/(lowest_chi/self.MH_norm))
        reduced_chi=chi_squared[points_to_keep]
        reduced_coords = np.array(coords)[points_to_keep]

        return reduced_chi, reduced_coords

    def create_cornerplot(self) -> None:
        """
        Performs a random sample across the coordinate space giving a predicted figure of merit at
        every point. Then removes points with poor figures of merit, according to a
        Metropolis-Hastings type rule, where the likelihood of keeping a point is dependant on the
        exponent of the difference between its figure of merit, and that of the best one found,
        divided by MC_norm. A corner plot is then returned (a matplotlib figure object), which can
        be displayed or exported.

        Returns
        -------
        corner plot : Matplotlib.figure.Figure
            A plot displaying every parameter combination with their variances and covariances
        """
        try:
            _, _, y_random, coords = \
            self._expected_minimum_random_sampling()
        except IndexError:
            msg = ("\n \n Your data file apears not to have any points in, please check you have "
                   "run the refinement and it saved correctly. \n")
            print(msg)

            return None

        _, reduced_coordinate_list = self._remove_points(y_random, coords)

        data = np.empty(shape=np.array(reduced_coordinate_list).shape)
        for i in range(np.array(reduced_coordinate_list).shape[1]):
            data[:,i] = np.array(reduced_coordinate_list)[:,i]

        labels = [str(name) for name in self.parameter_names]
        cornerplot = corner.corner(data, labels = labels, quantiles = [0.34, 0.5, 0.68])

        mean, std = np.mean(data, axis=0), np.std(data, axis=0)

        return cornerplot, mean, std


class DataPrinter(ABC):
    """
    A class for printing data during a minimisation.
    """

    @abstractmethod
    def print_data(self, history):
        """
        Update table at the end of a refinement step.

        Parameters:
        history
            The history of the minimizer data is printed from.
        """
        raise NotImplementedError

    @abstractmethod
    def print_header(self, history):
        """
        Create table headers at the start of refinement.
        
        Parameters:
        history
            The history of the minimizer data is printed from.
        """
        raise NotImplementedError


class PlaintextDataPrinter(DataPrinter):
    """Plaintext data printer."""

    def print_data(self, history) -> None:
        with pd.option_context('display.max_colwidth', 12,
                               'display.precision', 5,
                               'display.float_format', '{:.4g}'.format):
            n_step = history.iloc[-1].name
            output = history.loc[[n_step]].to_string(
                col_space=12, index=False, header=False).split('\n')
            data = '{:4d}'.format(n_step) + ''.join(output)
            print(data)

    def print_header(self, history) -> None:
        def format_column(column):
            column = column if len(column) < 13 else column[:9] + '...'
            return ' ' * (12 - len(column)) + column

        columns = ' '.join([format_column(col) for col
                            in history.columns])
        header = 'Step' + columns
        print(header)


class IPythonDataPrinter(DataPrinter):
    """Prettier IPython data printer, for Jupyter Notebooks, etc."""
    def __init__(self):
        self.display = IPython.display.DisplayHandle()

    def print_data(self, history) -> None:
        history_table = pd.DataFrame(history, 
                                     columns=history.columns)
        history_table.index.name = "Step"
        self.display.update(history_table)

    def print_header(self, history) -> None:
        history_table = pd.DataFrame(columns=history.columns)
        self.display.display(history_table)


data_printers = {
    'plaintext': PlaintextDataPrinter,
    'ipython': IPythonDataPrinter
}
