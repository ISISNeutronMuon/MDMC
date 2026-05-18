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

"""
Utility functions for plotting.

These are additional functions which can be used to plot specific MDMC data,
e.g. dynamic plotting during a refinement.  All plotting requires ``matplotlib`` to
be installed, and dynamic plotting requires execution to be performed within a
Jupyter notebook in order to display correctly.
"""

import warnings
from contextlib import suppress
from types import MethodType

from MDMC.common.df_operations import filter_dataframe
from MDMC.control import Control

try:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
except ModuleNotFoundError as error:
    raise ModuleNotFoundError(
        "MDMC plotting utilities require matplotlib to be installed.",
    ) from error


# Defaults for text and plot output sizes

#: Default VBox height.
VBOX_HEIGHT = "73%"
#: Default number of text lines.
N_TEXT_LINES = 5
#: Default plot height.
PLOT_HEIGHT = 360
#: Default CNVS width.
CNVS_WIDTH = 800


# pylint: disable=import-outside-toplevel, protected-access
# we are importing things out-of-order and copying variables on purpose here
def plot_progress(inst: Control, ynames: str) -> Control:
    """
    Plot current progress of a control.

    Modifies an instance of :any:`MDMC.control.Control` so that the progress of 1
    or more variables is plotted with each step when :any:`refine` is called.

    This takes an instance of ``Control`` as a parameter and
    returns a modified instance, which can be treated exactly as the original
    instance.  See the examples section for more details.

    Applying ``plot_progress`` will also change the text output so that only the
    more recent five steps are shown.

    Parameters
    ----------
    inst : Control
        An instance of the ``MDMC.control.Control`` class.
    ynames : str, list of str
        One or more str with the name of the variable to be displayed with each
        step of the refinement. These variables must correspond to the column
        names in ``inst.minimizer.history``, for example the names of the
        parameters that are being refined. It is recommended that a maximum of
        8 names is provided, as otherwise the graph sizes become too small.

    Returns
    -------
    Control
        An instance of the ``MDMC.control.Control`` class, which is
        modified so that a plot is displayed when ``inst.refine`` is called.

    Notes
    -----
    This plotting should only be used in a Jupyter notebook and requires
    ipywidgets and matplotlib to interactively display the progress. The
    matplotlib backend must be set to 'notebook' before calling `refine`. This
    can be done by executing the following magic call within a Jupyter notebook
    cell:

    .. highlight:: python
    .. code-block:: python

        %matplotlib notebook

    Examples
    --------
    Modifying a ``Control`` instance to plot the progress of the 'FoM' with each
    refinement step. This should be executed within a Jupyter notebook:

    .. highlight:: python
    .. code-block:: python

        %matplotlib notebook
        control = Control(...)  # ... represents some parameters
        control = plot_progress(control, 'FoM')
        control.refine(100)

    First the matplotlib backend is set to 'notebook', then the ``Control``
    instance is modified, and then a refinement is run. With each step of the
    refinement a graph of 'FoM' against 'Steps' will be plotted.

    Modifying a ``Control`` instance to plot the progress of the 'FoM', 'sigma',
    and 'epsilon' with each refinement step. This should be executed within a
    Jupyter notebook:

    .. highlight:: python
    .. code-block:: python

        %matplotlib notebook
        control = Control(...)  # ... represents some parameters
        control = plot_progress(control, ['FoM', 'sigma', 'epsilon'])
        control.refine(100)

    With each step of the refinement a graph of 'FoM' against 'Steps' will be
    plotted, a graph of 'sigma' against 'Steps' will be plotted, and a graph of
    'epsilon' against 'Steps' will be plotted.
    """

    # If required modules are not installed, warn user and return unmodified
    # instance
    try:
        from IPython import display
        from ipywidgets import Output, VBox
    except ModuleNotFoundError as err:
        warnings.warn(f"plot_progress requires {err.name}. No plots will be displayed.")
        return inst

    # This font size and linewidth were suitable for OSX dev environment
    plt.rcParams.update({"font.size": 22, "axes.linewidth": 5})

    # copies of the original instance methods are kept so that they can be
    # called within the replacement methods
    orig_refine = inst.refine
    orig_step = inst.step
    orig_print_data = inst._print_data
    orig_print_header = inst._print_header

    # Add protected ynames and vbox variables to instance
    # Force ynames to be list so that it can be iterated over
    inst._ynames = [ynames] if isinstance(ynames, str) else ynames

    # Create a VBox for the text consisting of N empty outputs. These are then
    # dynamically replaced by lines containing text output with each step.
    # height layout setting is used to reduce padding between lines of text.
    inst._vbox = VBox([Output()] * N_TEXT_LINES, layout={"height": VBOX_HEIGHT})

    # Basic validation of user input
    if not inst._ynames:
        raise ValueError("ynames must contain at least one str")
    for yname in inst._ynames:
        if yname not in inst.minimizer.history:
            raise ValueError(f"{yname} is not a variable in the minimizer history")

    # Redefined refine so that figure plotting is set, and the vbox containing
    # the text is displayed after the header
    def refine(self, *args):
        figure, axs = plt.subplots(len(self._ynames), 1, squeeze=False)
        inst.figure, inst.axes = figure, axs.flatten()
        for yname, ax in zip(self._ynames, inst.axes):
            ax.set_ylabel(yname)
            if ax is inst.axes[-1]:
                ax.set_xlabel("Steps")
                ax.xaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=1))
            else:
                ax.set_xticklabels([])
        # This fudge to change the dpi and resize the canvas is required because
        # of a bug in matplotlib when canvas.draw is called dynamically within
        # a loop (the bug reduces canvas._dpi_ratio to 1 which results in graphs
        # being plotted half size until the execution is completed)
        self.figure.canvas._dpi_ratio = 2
        height = min(len(self._ynames), 4) * PLOT_HEIGHT
        # This try/except allows IPython/Jupyter console to run without error
        # (although it does not live plot)
        with suppress(AttributeError):
            self.figure.canvas.handle_resize({"width": CNVS_WIDTH, "height": height})
        self.figure.canvas.draw()
        orig_print_header()
        display.display(self._vbox)
        orig_refine(*args)

    # Redefine step so that history is also plotted
    def step(self):
        orig_step()
        self.plot_history()

    # Redefine print data so that output is captured and added to vbox rather
    # than displayed. This stops Jupyter from autoscrolling after a certain
    # number of lines of text are displayed.
    def print_data(self):
        text_output = Output()
        with text_output:
            orig_print_data()
        self._vbox.children = self._vbox.children[1:] + (text_output,)

    # Used to change inst._print_header to pass through as header printing is
    # handled in substitute refine instead
    def print_header(self):
        pass

    def plot_history(self):
        history = self.minimizer.history
        for yname, ax in zip(self._ynames, self.axes):
            acp_rows = filter_dataframe(["Accepted"], history, column_names=["Change state"])
            rej_rows = filter_dataframe(["Rejected"], history, column_names=["Change state"])
            ax.plot(
                acp_rows.index.astype(int),
                acp_rows[yname],
                linestyle="",
                marker="o",
                color="tab:blue",
                markersize=12,
            )
            ax.plot(
                rej_rows.index.astype(int),
                rej_rows[yname],
                linestyle="",
                marker="x",
                color="tab:red",
                markersize=12,
                markeredgewidth=5,
            )
        self.figure.canvas.draw()

    # Set new methods for inst (MethodType required because they are bound)
    inst.refine = MethodType(refine, inst)
    inst.step = MethodType(step, inst)
    inst.plot_history = MethodType(plot_history, inst)
    inst._print_data = MethodType(print_data, inst)
    inst._print_header = MethodType(print_header, inst)

    return inst
