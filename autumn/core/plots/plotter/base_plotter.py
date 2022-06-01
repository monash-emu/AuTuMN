import copy
from abc import ABC, abstractmethod

import numpy
from matplotlib import pyplot


class BasePlotter(ABC):
    """
    Used to plot things
    """

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def save_figure(self, fig, filename: str, subdir=None, title_text=None, dpi_request=300):
        pass

    def get_figure(
        self,
        n_panels=1,
        room_for_legend=False,
        requested_grid=None,
        share_xaxis=False,
        share_yaxis="none",
    ):
        """
        Initialise the subplots (or single plot) according to the number of panels required.

        Args:
            n_panels: The number of panels needed
            room_for_legend: Whether room is needed for a legend - applies to single axis plots only
            requested_grid: Shape of grid panels requested at call to method
            share_yaxis: String to pass to the sharey option
        Returns:
            fig: The figure object
            axes: A list containing each of the axes
            max_dims: The number of rows or columns of sub-plots, whichever is greater
        """
        pyplot.style.use("ggplot")
        n_rows, n_cols = requested_grid if requested_grid else find_subplot_grid(n_panels)
        indices = []
        horizontal_position_one_axis = 0.11 if room_for_legend else 0.15
        if n_panels == 1:
            fig = pyplot.figure()
            axes = fig.add_axes([horizontal_position_one_axis, 0.15, 0.69, 0.7])
        elif n_panels == 2:
            fig, axes = pyplot.subplots(1, 2, sharey=share_yaxis)
            fig.set_figheight(3.5)
            fig.subplots_adjust(bottom=0.15, top=0.85)
        else:
            fig, axes = pyplot.subplots(n_rows, n_cols, sharex=share_xaxis, sharey=share_yaxis)
            for panel in range(n_rows * n_cols):
                indices.append(new_find_panel_grid_indices(panel, n_rows, n_cols))
        return fig, axes, max([n_rows, n_cols]), n_rows, n_cols, indices

    def get_plot_title(self, title: str):
        """
        Get a more readable version of a string for figure title.
        """
        return self.translation_dict.get(title, title)

    def tidy_x_axis(self, axis, start, end, max_dims, labels_off=False, x_label=None):
        """
        Function to tidy x-axis of a plot panel - currently only used in the scale-up vars, but intended to be written
        in such a way as to be extendable to other types of plotting.

        Args:
            axis: The plotting axis
            start: Lowest x-value being plotted
            end: Highest x-value being plotted
            max_dim: Maximum number of rows or columns of subplots in figure
            labels_off: Whether to turn all tick labels off on this axis
            x_label: Text for the x-axis label if required
        """

        # range
        axis.set_xlim(left=start, right=end)

        # ticks and their labels
        if labels_off:
            axis.tick_params(axis="x", labelbottom=False)
        elif len(axis.get_xticks()) > 7:
            for label in axis.xaxis.get_ticklabels()[::2]:
                label.set_visible(False)
        axis.tick_params(axis="x", length=3, pad=6, labelsize=get_label_font_size(max_dims))

        # axis label
        if x_label is not None:
            axis.set_xlabel(self.get_plot_title(x_label), fontsize=get_label_font_size(max_dims))

    def tidy_y_axis(
        self,
        axis,
        quantity,
        max_dims,
        left_axis=True,
        max_value=1e6,
        space_at_top=0.1,
        y_label=None,
        y_lims=None,
        allow_negative=False,
    ):
        """
        General approach to tidying up the vertical axis of a plot, depends on whether it is the left-most panel.

        Args:
            axis: The axis itself
            quantity: The name of the quantity being plotted (which can be used to determine the sort of variable it is)
            max_dims: Maximum number of rows or columns of subplots on the figure
            left_axis: Boolean for whether the axis is the left-most panel
            max_value: The maximum value in the data being plotted
            space_at_top: Relative amount of space to leave at the top, above the maximum value of the plotted data
            y_label: A label for the y-axis, if required
            y_lims: 2-element tuple for the y-limit, if required
            allow_negative: Whether to set the bottom of the axis to zero
        """

        # axis range
        if y_lims:
            axis.set_ylim(y_lims)
        elif "prop_" in quantity and axis.get_ylim()[1] > 1.0:
            axis.set_ylim(top=1.004)
        elif "prop_" in quantity and max_value > 0.7:
            axis.set_ylim(bottom=0.0, top=1.0)
        elif "prop_" in quantity or "likelihood" in quantity or "cost" in quantity:
            pass
        elif axis.get_ylim()[1] < max_value * (1.0 + space_at_top):
            pass
            # axis.set_ylim(top=max_value * (1. + space_at_top))
        if not allow_negative:
            axis.set_ylim(bottom=0.0)

        # ticks
        axis.tick_params(axis="y", length=3.0, pad=6, labelsize=get_label_font_size(max_dims))

        # tick labels
        if not left_axis:
            pyplot.setp(axis.get_yticklabels(), visible=False)
        elif "prop_" in quantity:
            axis.yaxis.set_major_formatter(FuncFormatter("{0:.0%}".format))

        # axis label
        if y_label and left_axis:
            axis.set_ylabel(self.get_plot_title(y_label), fontsize=get_label_font_size(max_dims))


COLOR_THEME = 100 * [
    (0.0, 0.0, 0.0),
    (57.0 / 255.0, 106.0 / 255.0, 177.0 / 255.0),  # blue
    (218.0 / 255.0, 124.0 / 255.0, 48.0 / 255.0),  # orange
    (62.0 / 255.0, 150.0 / 255.0, 81.0 / 255.0),   # green
    (204.0 / 255.0, 37.0 / 255.0, 41.0 / 255.0),  # red
    (107.0 / 255.0, 76.0 / 255.0, 154.0 / 255.0),  # purple
    (146.0 / 255.0, 36.0 / 255.0, 40.0 / 255.0),
    (148.0 / 255.0, 139.0 / 255.0, 61.0 / 255.0),
    (0.0, 0.0, 125.0 / 255.0),
    (210.0 / 255.0, 70.0 / 255.0, 0.0),
    (100.0 / 255.0, 150.0 / 255.0, 1.0),
    (65.0 / 255.0, 65.0 / 255.0, 65.0 / 255.0),
    (220.0 / 255.0, 25.0 / 255.0, 25.0 / 255.0),
    (120.0 / 255.0, 55.0 / 255.0, 20.0 / 255.0),
    (120.0 / 255.0, 55.0 / 255.0, 110.0 / 255.0),
    (135.0 / 255.0, 135.0 / 255.0, 30.0 / 255.0),
    (120.0 / 255.0, 120.0 / 255.0, 120.0 / 255.0),
    (220.0 / 255.0, 20.0 / 255.0, 170.0 / 255.0),
    (20.0 / 255.0, 65.0 / 255.0, 20.0 / 255.0),
    (15.0 / 255.0, 145.0 / 255.0, 25.0 / 255.0),
    (15.0 / 255.0, 185.0 / 255.0, 240.0 / 255.0),
    (10.0 / 255.0, 0.0, 110.0 / 255.0),
]


def find_subplot_grid(n_plots):
    """
    Find a convenient number of rows and columns for a required number of subplots. First take the root of the number of
    subplots and round up to find the smallest square that could accommodate all of them. Next find out how many rows
    that many subplots would fill out by dividing the number of plots by the number of columns and rounding up. This
    will potentially leave a few panels blank at the end and number of rows will equal the number of columns or the
    number of rows will be on fewer.

    Args:
        n_plots: The number of subplots needed
    Returns:
        The number of rows of subplots
        n_cols: The number of columns of subplots
    """

    n_cols = int(numpy.ceil(numpy.sqrt(n_plots)))
    return int(numpy.ceil(n_plots / float(n_cols))), n_cols


def find_panel_grid_indices(axes, index, n_rows, n_columns):
    """
    Find the subplot index for a plot panel from the number of the panel and the number of columns of sub-plots.

    Args:
        axes: All the plot axes to be searched from
        index: The number of the panel counting up from zero
        n_rows: Number of rows of sub-plots in figure
        n_columns: Number of columns of sub-plots in figure
    """

    row, column = (
        numpy.floor_divide(index, n_columns),
        (index + 1) % n_columns - 1 if n_rows > 1 else None,
    )
    return axes[row, column] if n_rows > 1 else axes[index]


def new_find_panel_grid_indices(index, n_rows, n_columns):
    """
    Find the subplot index for a plot panel from the number of the panel and the number of columns of sub-plots.

    Args:
        index: The number of the panel counting up from zero
        n_rows: Number of rows of sub-plots in figure
        n_columns: Number of columns of sub-plots in figure
    """

    row, column = (
        numpy.floor_divide(index, n_columns),
        index % n_columns if n_rows > 1 else None,
    )
    return row, column


def get_label_font_size(max_dim):
    """
    Find standardised font size that can be applied across all figures.

    Args:
        max_dim: The number of rows or columns, whichever is the greater
    """

    label_font_sizes = {1: 8, 2: 7}
    return label_font_sizes[max_dim] if max_dim in label_font_sizes else 6


def scale_axes(vals, max_val, y_sig_figs):
    """
    General function to scale a set of axes and produce text that can be added to the axis label. Written here as a
    separate function from the tidy_axis method below because it can then be applied to both x- and y-axes.

    Args:
        vals: List of the current y-ticks
        max_val: The maximum value of this list
        y_sig_figs: The preferred number of significant figures for the ticks
    Returns:
        labels: List of the modified tick labels
        axis_modifier: The text to be added to the axis
    """

    y_number_format = "%." + str(y_sig_figs) + "f"
    y_number_format_around_one = "%." + str(max(2, y_sig_figs)) + "f"
    if max_val < 5e-9:
        labels = [y_number_format % (v * 1e12) for v in vals]
        axis_modifier = "Trillionth "
    elif max_val < 5e-6:
        labels = [y_number_format % (v * 1e9) for v in vals]
        axis_modifier = "Billionth "
    elif max_val < 5e-3:
        labels = [y_number_format % (v * 1e6) for v in vals]
        axis_modifier = "Millionth "
    elif max_val < 5e-2:
        labels = [y_number_format % (v * 1e3) for v in vals]
        axis_modifier = "Thousandth "
    elif max_val < 0.1:
        labels = [y_number_format % (v * 1e2) for v in vals]
        axis_modifier = "Hundredth "
    elif max_val < 5:
        labels = [y_number_format_around_one % v for v in vals]
        axis_modifier = ""
    elif max_val < 5e3:
        labels = [y_number_format % v for v in vals]
        axis_modifier = ""
    elif max_val < 5e6:
        labels = [y_number_format % (v / 1e3) for v in vals]
        axis_modifier = "Thousand "
    elif max_val < 5e9:
        labels = [y_number_format % (v / 1e6) for v in vals]
        axis_modifier = "Million "
    else:
        labels = [y_number_format % (v / 1e9) for v in vals]
        axis_modifier = "Billion "
    return labels, axis_modifier


def increment_list_for_patch(new_data, cumulative_data):
    """
    Takes a list of cumulative data totals, preserves the previous values and adds a new list to it. This is to allow
    patches to be plotted that have the previous data values as their base and the results of this stacking as their
    top.

    Args:
        new_data: The new data to be stacked up
        cumulative_data: The previous running totals
    Returns:
        previous_data: The previous running total (was cumulative_data)
        The new running total as the new values for cumulative_data
    """

    previous_data = copy.copy(cumulative_data)
    return (
        previous_data,
        [last + current for last, current in zip(cumulative_data, new_data)],
    )


def add_title_to_plot(fig, n_panels, content):
    """
    Function to add title to the top of a figure and handle multiple panels if necessary.

    Args:
        fig: The figure object to have a title added to it
        n_panels: Integer for the total number of panels on the figure
        content: Unprocessed string to determine text for the title
    """

    # if few panels, bigger and lower title
    greater_heights = {1: 0.92, 2: 0.98}
    greater_font_sizes = {1: 14, 2: 11}
    fig.suptitle(
        content,
        y=greater_heights[n_panels] if n_panels in greater_heights else 0.96,
        fontsize=greater_font_sizes[n_panels] if n_panels in greater_font_sizes else 10,
    )
