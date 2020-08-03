"""Plotting related utilities
"""

from collections import deque
from types import MethodType

try:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
except ModuleNotFoundError:
    raise ModuleNotFoundError('MDMC plotting utilities require matplotlib to be'
                              ' installed.')

from MDMC.common.df_operations import filter_dataframe


def plot_progress(inst, ynames):

    """
    Modifies an instance of ``MDMC.control.Control`` so that the progress of 1
    or more variables is plotted with each step when ``refine`` is called.

    This takes an instance of ``MDMC.control.Control`` as a parameter and
    returns a modified instance, which can be treated exactly as the original
    instance.  See the examples section for more details.

    **This plotting should only be used in a Jupyter notebook and requires
    matplotlib to interactively display the progress. The matplotlib backend
    must be set to 'notebook' before calling `refine`. This can be done by
    executing the following magic call within a Jupyter notebook cell:**

        .. highlight:: python
        .. code-block:: python

            %matplotlib notebook

    Parameters
    ----------
    inst : MDMC.control.Control
        An instance of the ``MDMC.control.Control`` class, which will be
        modified so that a plot is displayed when ``inst.refine`` is called.
    ynames : str, list of str
        One or more str with the name of the variable to be displayed with each
        step of the refinement. These variables must correspond to the column
        names in ``inst.minimizer.history``, for example the names of the
        parameters that are being refined. It is recommended that a maximum of
        8 names is provided, as otherwise the graph sizes become too small.

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

    from IPython import display
    from ipywidgets import Output, VBox

    plt.rcParams.update({'font.size': 22, 'axes.linewidth': 5})

    orig_refine = inst.refine
    orig_step = inst.step
    orig_print_data = inst._print_data
    # Force ynames to be list so that it can be iterated over
    inst.ynames = [ynames] if isinstance(ynames, str) else ynames
    # inst.text_output_deque = deque(maxlen=5)
    inst.vbox = VBox([Output()] * 5, layout={'height':'75%'})
    display.display(inst.vbox)

    # Basic validation of user input
    if len(ynames) < 1:
        raise ValueError('ynames must contain at least one str')
    for yname in ynames:
        if yname not in inst.minimizer.history:
            raise ValueError('{0} is not a variable in the minimizer'
                             ' history'.format(yname))

    def refine(self, *args):
        figure, axs = plt.subplots(len(self.ynames), 1, squeeze=False)
        # figure.tight_layout(pad=1.01)
        inst.figure, inst.axes = figure, axs.flatten()
        for yname, ax in zip(self.ynames, inst.axes):
            ax.set_ylabel(yname)
            if ax is inst.axes[-1]:
                ax.set_xlabel('Steps')
                ax.xaxis.set_major_locator(MaxNLocator(integer=True,
                                                       min_n_ticks=1))
            else:
                ax.set_xticklabels([])
        # This fudge to change the dpi and resize the canvas is required because
        # of a bug in matplotlib when canvas.draw is called dynamically within
        # a loop (the bug reduces canvas._dpi_ratio to 1 which results in graphs
        # being plotted half size until the execution is completed)
        self.figure.canvas._dpi_ratio = 2
        height = min(len(self.ynames), 4) * 400
        self.figure.canvas.handle_resize({'width':800, 'height':height})
        self.figure.canvas.draw()

        orig_refine(*args)

    def step(self):
        orig_step()
        self.plot_history()

    def print_data(self):
        text_output = Output()
        with text_output:
            orig_print_data()
        inst.vbox.children = inst.vbox.children[1:] + (text_output, )
        #
        # if len(self.text_output_deque) == self.text_output_deque.maxlen:
        #     outdated_text_output = self.text_output_deque.popleft()
        #     outdated_text_output.close()
        # text_output = Output()
        # with text_output:
        #     orig_print_data()
        # display.display(text_output)
        # self.text_output_deque.append(text_output)

    def plot_history(self):
        history = self.minimizer.history
        for yname, ax in zip(self.ynames, self.axes):
            acp_rows = filter_dataframe(['Accepted'], history,
                                        column_names=['Change state'])
            rej_rows = filter_dataframe(['Rejected'], history,
                                        column_names=['Change state'])
            ax.plot(acp_rows.index.astype(int), acp_rows[yname], linestyle='',
                    marker='o', color='tab:blue', markersize=12)
            ax.plot(rej_rows.index.astype(int), rej_rows[yname], linestyle='',
                    marker='x', color='tab:red', markersize=12,
                    markeredgewidth=5)
        self.figure.canvas.draw()

    # Set new methods for inst
    inst.refine = MethodType(refine, inst)
    inst.step = MethodType(step, inst)
    inst.plot_history = MethodType(plot_history, inst)
    inst._print_data = MethodType(print_data, inst)

    return inst
