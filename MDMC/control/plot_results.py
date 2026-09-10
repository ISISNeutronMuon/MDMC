# MDMC is a package for the optimisation of classical potentials with experimental data
# Copyright (C) 2026 MDMC Developers
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""A module for plotting data and results of a minimization."""

from abc import ABC, abstractmethod

import corner
import IPython.display
import matplotlib.pyplot as mpl
import numpy as np
import pandas as pd
from matplotlib.figure import Figure


class PlotResults:
    """
    A class to read in any completed refinement history file, create a Gaussain Process Optimizer
    and then do sampling on the result to create a corner plot.

    Parameters:
    -----------
    filename : str
        Path to the file to load in the refinement history.
    quantiles : list, optional
        Optional, list of the quantiles to be plotted on the corner plot, defaults
        to [0.34, 0.5, 0.68], e.g. 1-sigma.
    output_filename : str | None
        Name stem to be included in output plot file names. Set to None if figures
        are not saved to file, e.g. when running in Jupyter Notebooks.
    """

    def __init__(
        self,
        filename,
        quantiles: list[float] = None,
        output_filename: str | None = None,
    ):
        self.filename = filename
        self.quantiles = [0.34, 0.5, 0.68] if quantiles is None else quantiles
        self.output_filename = output_filename

        self.parameter_names, self.parameter_coords, self.minmax_coords, self.FoMs = (
            self.get_measured_points()
        )

    def get_measured_points(self) -> tuple:
        """Opens the dataframe in `filename` and extracts the measured parameters names, values
        and associated figures of merit.
        Returns:
        --------
        tuple of (parameter names, parameter coordinates, min and max parameters, FoM's)
        """
        records = pd.read_csv(self.filename, delimiter=",")
        records = records.astype(dtype=float, errors="ignore")
        # Convert to float where possible (i.e. not a string)

        FoMs = records["FoM"].to_list()
        records = records.drop(
            columns=["Unnamed: 0", "FoM", "Change state", "CMA iteration"], errors="ignore"
        )
        # TODO this is hard coded to creation of history, may want to change

        coordinates = records.values.tolist()
        names = records.columns.tolist()
        minmax_coordinates = [
            (min(np.array(coord)), max(np.array(coord))) for coord in np.array(coordinates).T
        ]
        return names, coordinates, minmax_coordinates, FoMs

    def _create_figure(self, nrows: int = 1) -> Figure | None:
        """Create a temporary figure which can be used for plotting results.

        If no output filename has been given to PlotResults, the plotters
        will create their own figures, which should be the standard approach
        in Jupyter Notebooks, allowing the figure to be displayed by the
        notebook plugins. Otherwise, the plots will be saved to file, which
        should be the approach in Python scripts.

        Parameters
        ----------
        nrows : int, optional
            Planned number of rows in the figure, by default 1

        Returns
        -------
        Figure | None
            Empty matplotlib figure, or None if not saving to a file.
        """
        if self.output_filename is not None:
            return mpl.figure(figsize=(12.0, 4.0 * (nrows + 1)), dpi=192)
        else:
            return None

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

        index_best_objective = np.argmin(self.FoMs)
        min_x = self.parameter_coords[index_best_objective]

        param_array = np.array(self.parameter_coords)

        labels = [str(name) for name in self.parameter_names]

        target_figure = self._create_figure()

        cornerplot = corner.corner(
            param_array,
            labels=labels,
            quantiles=[0.34, 0.5, 0.68],
            range=self.minmax_coords,
            fig=target_figure,
        )

        if target_figure is not None:
            target_figure.savefig(f"{self.output_filename}_cornerplot.png")

        mean, std = np.mean(param_array, axis=0), np.std(param_array, axis=0)

        return cornerplot, mean, std

    def create_parameter_plots(self, cutoff_fraction: float = 0.3):
        """Show the force field parameters plotted against the FoM.

        The results are sorted by FoM. The cutoff_fraction argument
        determines how many data points out of the total amount will
        be included in the plot.

        Parameters
        ----------
        cutoff_fraction : float, optional
            The fraction of best points to be included, 1.0 for all, by default 0.3
        """
        if self.output_filename is None:
            return
        sorting = np.argsort(self.FoMs)
        sorted_FoM = np.array(self.FoMs)[sorting]
        param_vals = np.array(self.parameter_coords).T
        nrows = len(param_vals)
        temp_fig = self._create_figure(nrows=nrows)
        cutoff_index = np.ceil(len(self.FoMs) * cutoff_fraction).astype(int)
        for pindex, name, param in zip(range(nrows), self.parameter_names, param_vals, strict=True):
            splot = temp_fig.add_subplot(nrows, 1, pindex + 1)
            # sharex="all")
            par_data = np.array(param)[sorting]
            splot.plot(sorted_FoM[:cutoff_index], par_data[:cutoff_index], "o")
            splot.set_title(name)
            splot.set_xlabel("FoM")
            splot.set_ylabel(name)
        temp_fig.savefig(f"{self.output_filename}_parameters.png")


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
        with pd.option_context(
            "display.max_colwidth",
            12,
            "display.precision",
            5,
            "display.float_format",
            "{:.4g}".format,
        ):
            n_step = history.iloc[-1].name
            output = (
                history.loc[[n_step]].to_string(col_space=12, index=False, header=False).split("\n")
            )
            data = f"{n_step:4d}{''.join(output)}"
            print(data)

    def print_header(self, history) -> None:
        def format_column(column):
            column = column if len(column) < 13 else column[:9] + "..."
            return " " * (12 - len(column)) + column

        columns = " ".join([format_column(col) for col in history.columns])
        header = "Step" + columns
        print(header)


class IPythonDataPrinter(DataPrinter):
    """Prettier IPython data printer, for Jupyter Notebooks, etc."""

    def __init__(self):
        self.display = IPython.display.DisplayHandle()

    def print_data(self, history) -> None:
        history_table = pd.DataFrame(history, columns=history.columns)
        history_table.index.name = "Step"
        self.display.update(history_table)

    def print_header(self, history) -> None:
        history_table = pd.DataFrame(columns=history.columns)
        self.display.display(history_table)


data_printers = {
    "plaintext": PlaintextDataPrinter,
    "ipython": IPythonDataPrinter,
}
