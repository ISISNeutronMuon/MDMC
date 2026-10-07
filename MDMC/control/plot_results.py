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
import numpy.typing as npt
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
    cutoff_fraction : float
        Fraction of the data points to keep, 1.0 means all, by default0.5
    """

    def __init__(
        self,
        filename,
        quantiles: list[float] = None,
        output_filename: str | None = None,
        cutoff_fraction: float = 0.5,
    ):
        self.filename = filename
        self.quantiles = [0.34, 0.5, 0.68] if quantiles is None else quantiles
        self.output_filename = output_filename
        self.cutoff_fraction = cutoff_fraction

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

    def _trim_results(
        self, cutoff_fraction: float | None = None
    ) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        """Return only a specified fraction of the results with the best FoM.

        Parameters and FoM values are sorted based on FoM values, and
        all the results above the specified fraction are not included in the output.

        Parameters
        ----------
        cutoff_fraction : float | None, optional
            Fraction of the data points to keep, 1.0 means all, by default None

        Returns
        -------
        tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]
            Parameter array and FoM array, both trimmed and sorted by FoM
        """
        cutoff = cutoff_fraction if cutoff_fraction else self.cutoff_fraction
        sorting = np.argsort(self.FoMs)
        sorted_FoM = np.array(self.FoMs)[sorting]
        param_vals = np.array(self.parameter_coords)[sorting]
        cutoff_index = np.ceil(len(self.FoMs) * cutoff).astype(int)
        return param_vals[:cutoff_index], sorted_FoM[:cutoff_index]

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
        """Plot parameters as a function of other parameters.

        This compares the parameters pairwise and provides a visual estimate
        of correlation between refinement parameters.

        Returns
        -------
        corner plot : Matplotlib.figure.Figure
            A plot displaying every parameter combination with their variances and covariances
        """
        param_array, _ = self._trim_results()

        labels = [str(name) for name in self.parameter_names]

        target_figure = self._create_figure()

        cornerplot = corner.corner(
            param_array,
            labels=labels,
            quantiles=self.quantiles,
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
        param_vals, sorted_FoM = self._trim_results()
        nrows = len(self.parameter_names)
        temp_fig = self._create_figure(nrows=nrows)
        x_axis = None
        for pindex, name, param in zip(
            range(nrows), self.parameter_names, param_vals.T, strict=True
        ):
            if x_axis is None:
                splot = temp_fig.add_subplot(nrows, 1, pindex + 1)
                x_axis = splot
            else:
                splot = temp_fig.add_subplot(nrows, 1, pindex + 1, sharex=x_axis)
            splot.plot(sorted_FoM, param, "o")
            splot.set_xlabel("FoM")
            splot.set_ylabel(name)
        temp_fig.savefig(f"{self.output_filename}_parameters.png")

    def create_FoM_plot(self, use_logscale: bool = True):
        """Show the FoM values over the entire refinement.

        By default the plot will use logarithmic scale for the FoM values.

        Parameters
        ----------
        use_logscale : bool, optional
            Apply logarithmic scale to the plotted results, by default True
        """
        if self.output_filename is None:
            return
        unsorted_FoM = np.array(self.FoMs)
        temp_fig = self._create_figure()
        splot = temp_fig.add_subplot(1, 1, 1)
        splot.plot(unsorted_FoM, "o")
        splot.set_title("Refinement FoM evolution")
        splot.set_xlabel("step #")
        splot.set_ylabel("FoM")
        if use_logscale:
            splot.set_yscale("log")
        temp_fig.savefig(f"{self.output_filename}_FoM.png")


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
