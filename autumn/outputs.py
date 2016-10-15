
import os
import glob
import datetime
import autumn.model
import autumn.economics
from autumn.spreadsheet import read_input_data_xls
import numpy as np
import openpyxl as xl
import tool_kit
from docx import Document
from matplotlib import pyplot, patches, style
import numpy
import pylab
import platform
import os
import warnings
import economics
import pandas
import copy


def relax_y_axis(ax):

    """
    Matplotlib's default values often place curves very close to the top
    of axes and sometimes extend down to small fractions for plots that are
    proportions. This over-rides some of these defaults, that I don't like.
    Args:
        ax: Axis with default y-limits to be revised

    Returns:
        ylims: New y-lims that look better

    """

    ylims = list(ax.get_ylim())
    if ylims[0] < ylims[1] * .75:
        ylims[0] = 0.
    else:
        ylims[0] = ylims[0] * .6
    ylims[1] = ylims[1] * 1.1

    return ylims


def find_smallest_factors_of_integer(n):

    """
    Quick method to iterate through integers to find the smallest whole number
    fractions. Written only to be called by find_subplot_numbers.

    Args:
        n: Integer to be factorised

    Returns:
        answer: The two smallest factors of the integer

    """

    answer = [1E3, 1E3]
    for i in range(1, n + 1):
        if n % i == 0 and i+(n/i) < sum(answer):
            answer = [i, n/i]
    return answer


def humanise_y_ticks(ax):

    """
    Coded by Bosco, does a few things, including rounding
    axis values to thousands, millions or billions and abbreviating
    these to single letters.

    Args:
        ax: The adapted axis

    """


    vals = list(ax.get_yticks())
    max_val = max([abs(v) for v in vals])
    if max_val < 1e3:
        return
    if max_val >= 1e3 and max_val < 1e6:
        labels = ["%.1fK" % (v/1e3) for v in vals]
    elif max_val >= 1e6 and max_val < 1e9:
        labels = ["%.1fM" % (v/1e6) for v in vals]
    elif max_val >= 1e9:
        labels = ["%.1fB" % (v/1e9) for v in vals]
    is_fraction = False
    for label in labels:
        if label[-3:-1] != ".0":
            is_fraction = True
    if not is_fraction:
        labels = [l[:-3] + l[-1] for l in labels]
    ax.set_yticklabels(labels)


def make_axes_with_room_for_legend():

    """
    Create axes for a figure with a single plot with a reasonable
    amount of space around.

    Returns:
        ax: The axes that can be plotted on

    """

    fig = pyplot.figure()
    ax = fig.add_axes([0.1, 0.1, 0.6, 0.75])
    return ax


def set_axes_props(
        ax, xlabel=None, ylabel=None, title=None, is_legend=True,
        axis_labels=None):

    frame_colour = "grey"

    # Hide top and right border of plot
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    if xlabel is not None:
        ax.set_xlabel(xlabel)

    if ylabel is not None:
        ax.set_ylabel(ylabel)

    if is_legend:
        if axis_labels:
            handles, labels = ax.get_legend_handles_labels()
            leg = ax.legend(
                handles,
                axis_labels,
                bbox_to_anchor=(1.05, 1),
                loc=2,
                borderaxespad=0.,
                frameon=False,
                prop={'size': 7})
        else:
            leg = ax.legend(
                bbox_to_anchor=(1.05, 1),
                loc=2,
                borderaxespad=0.,
                frameon=False,
                prop={'size':7})
        for text in leg.get_texts():
            text.set_color(frame_colour)

    if title is not None:
        t = ax.set_title(title)
        t.set_color(frame_colour)

    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontname('Arial')
        label.set_fontsize(8)

    ax.tick_params(color=frame_colour, labelcolor=frame_colour)
    for spine in ax.spines.values():
        spine.set_edgecolor(frame_colour)
    ax.xaxis.label.set_color(frame_colour)
    ax.yaxis.label.set_color(frame_colour)

    humanise_y_ticks(ax)


def get_nice_font_size(subplot_grid):

    # Simple function to return a reasonable font size
    # as appropriate to the number of rows of subplots in the figure
    return 2. + 8. / subplot_grid[0]


def find_reasonable_year_ticks(start_time, end_time):

    """
    Simple method to find some reasonably spaced x-ticks and making sure there
    aren't too many of them

    Args:
        start_time: Plotting start time
        end_time: Plotting end time

    Returns:
        xticks: List of where the x ticks should go
    """

    # If the range is divisible by 15
    if (start_time - end_time) % 15 == 0:
        xticks_any_length = numpy.arange(start_time, end_time + 15, 15)
    # Otherwise if it's divisible by 10
    elif (start_time - end_time) % 10 == 0:
        xticks_any_length = numpy.arange(start_time, end_time + 10, 10)
    # Otherwise just give up on having ticks along axis
    else:
        xticks_any_length = [start_time, end_time]

    xticks = []
    if len(xticks_any_length) > 10:
        for i in range(len(xticks_any_length)):
            if i % 2 == 0:
                xticks += [xticks_any_length[i]]
    else:
        xticks = xticks_any_length

    return xticks


def find_standard_output_styles(labels, lightening_factor=1.):

    """
    Function to find some standardised colours for the outputs we'll typically
    be reporting on - i.e. incidence, prevalence, mortality and notifications.
    Incidence is black/grey, prevalence green, mortality red and notifications blue.

    Args:
        labels: List containing strings for the outputs that colours are needed for.
        lightening_factor: Float between zero and one that specifies how much lighter to make
            the colours - with 0. being no additional lightening (black or dark green/red/blue)
            and 1. being completely lightened to reach white.

    Returns:
        colour: Colour for plotting
        indices: List of strings to be used to find the data in the data object
        yaxis_label: Unit of measurement for outcome
        title: Title for plot (so far usually a subplot)
        patch_colour: Colour half way between colour and white
    """

    colour = []
    indices = []
    yaxis_label = []
    title = []
    patch_colour = []

    if "incidence" in labels:
        colour += [(lightening_factor, lightening_factor, lightening_factor)]
        indices += ['e_inc_100k']
        yaxis_label += ['Per 100,000 per year']
        title += ["Incidence"]
    if "mortality" in labels:
        colour += [(1., lightening_factor, lightening_factor)]
        indices += ['e_mort_exc_tbhiv_100k']
        yaxis_label += ['Per 100,000 per year']
        title += ["Mortality"]
    if "prevalence" in labels:
        colour += [(lightening_factor, 0.5 + 0.5 * lightening_factor, lightening_factor)]
        indices += ['e_prev_100k']
        yaxis_label += ['Per 100,000']
        title += ["Prevalence"]
    if "notifications" in labels:
        colour += [(lightening_factor, lightening_factor, 0.5 + 0.5 * lightening_factor)]
        yaxis_label += ['']
        title += ["Notifications"]

    # Create a colour half-way between the line colour and white for patches
    for i in range(len(colour)):
        patch_colour += [[]]
        for j in range(len(colour[i])):
            patch_colour[i] += [1. - (1. - colour[i][j]) / 2.]

    return colour, indices, yaxis_label, title, patch_colour


def make_related_line_styles(labels, strain_or_organ):

    colours = {}
    patterns = {}
    compartment_full_names = {}
    markers = {}
    for label in labels:
        colour, pattern, compartment_full_name, marker =\
            get_line_style(label, strain_or_organ)
        colours[label] = colour
        patterns[label] = pattern
        compartment_full_names[label] = compartment_full_name
        markers[label] = marker
    return colours, patterns, compartment_full_names, markers


def get_line_style(label, strain_or_organ):

    # Unassigned groups remain black
    colour = (0, 0, 0)
    if "susceptible_vac" in label:  # susceptible_unvac remains black
        colour = (0.3, 0.3, 0.3)
    elif "susceptible_treated" in label:
        colour = (0.6, 0.6, 0.6)
    if "latent" in label:  # latent_early remains as for latent
        colour = (0, 0.4, 0.8)
    if "latent_late" in label:
        colour = (0, 0.2, 0.4)
    if "active" in label:
        colour = (0.9, 0, 0)
    elif "detect" in label:
        colour = (0, 0.5, 0)
    elif "missed" in label:
        colour = (0.5, 0, 0.5)
    if "treatment" in label:  # treatment_infect remains as for treatment
        colour = (1, 0.5, 0)
    if "treatment_noninfect" in label:
        colour = (1, 1, 0)

    pattern = get_line_pattern(label, strain_or_organ)

    category_full_name = label
    if "susceptible" in label:
        category_full_name = "Susceptible"
    if "susceptible_fully" in label:
        category_full_name = "Fully susceptible"
    elif "susceptible_vac" in label:
        category_full_name = "BCG vaccinated, susceptible"
    elif "susceptible_treated" in label:
        category_full_name = "Previously treated, susceptible"
    if "latent" in label:
        category_full_name = "Latent"
    if "latent_early" in label:
        category_full_name = "Early latent"
    elif "latent_late" in label:
        category_full_name = "Late latent"
    if "active" in label:
        category_full_name = "Active, yet to present"
    elif "detect" in label:
        category_full_name = "Detected"
    elif "missed" in label:
        category_full_name = "Missed"
    if "treatment" in label:
        category_full_name = "Under treatment"
    if "treatment_infect" in label:
        category_full_name = "Infectious under treatment"
    elif "treatment_noninfect" in label:
        category_full_name = "Non-infectious under treatment"

    if "smearpos" in label:
        category_full_name += ", \nsmear-positive"
    elif "smearneg" in label:
        category_full_name += ", \nsmear-negative"
    elif "extrapul" in label:
        category_full_name += ", \nextrapulmonary"

    if "_ds" in label:
        category_full_name += ", \nDS-TB"
    elif "_mdr" in label:
        category_full_name += ", \nMDR-TB"
    elif "_xdr" in label:
        category_full_name += ", \nXDR-TB"

    marker = ""

    return colour, pattern, category_full_name, marker


def get_line_pattern(label, strain_or_organ):

    pattern = "-"  # Default solid line
    if strain_or_organ == "organ":
        if "smearneg" in label:
            pattern = "-."
        elif "extrapul" in label:
            pattern = ":"
    elif strain_or_organ == "strain":
        if "_mdr" in label:
            pattern = '-.'
        elif "_xdr" in label:
            pattern = '.'

    return pattern


def make_plot_title(model, labels):

    try:
        if labels is model.labels:
            title = "by each individual compartment"
        elif labels is model.compartment_types \
                or labels is model.compartment_types_bystrain:
            title = "by types of compartments"
        elif labels is model.broad_compartment_types_byorgan:
            title = "by organ involvement"
        elif labels is model.broad_compartment_types \
                or labels is model.broad_compartment_types_bystrain:
            title = "by broad types of compartments"
        elif labels is model.groups["ever_infected"]:
            title = "within ever infected compartments"
        elif labels is model.groups["infected"]:
            title = "within infected compartments"
        elif labels is model.groups["active"]:
            title = "within active disease compartments"
        elif labels is model.groups["infectious"]:
            title = "within infectious compartments"
        elif labels is model.groups["identified"]:
            title = "within identified compartments"
        elif labels is model.groups["treatment"]:
            title = "within treatment compartments"
        else:
            title = "not sure"
        return title
    except:
        return ""


def create_patch_from_dictionary(dict):

    """
    Creates an array that can be used as a patch for plotting
    Args:
        dict: Dictionary with keys 'lower_limit', 'upper_limit' and 'year'
            (at least, although 'point_estimate' will also usually be there)

    Returns:
        patch_array: The patch array for plotting
    """

    patch_array = numpy.zeros(shape=(len(dict['lower_limit']) * 2, 2))
    j = 0
    for i in dict['lower_limit']:
        # Years going forwards
        patch_array[j][0] = i
        # Years going backwards
        patch_array[-(j + 1)][0] = i
        # Lower limit data going forwards
        patch_array[j][1] = dict['lower_limit'][i]
        # Upper limit data going backwards
        patch_array[-(j + 1)][1] = dict['upper_limit'][i]
        j += 1

    return patch_array


def save_png(png):

    # Should be redundant once Project module complete

    if png is not None:
        pylab.savefig(png, dpi=300)


def plot_flows(model, labels, png=None):

    colours, patterns, compartment_full_names\
        = make_related_line_styles(labels)
    ax = make_axes_with_room_for_legend()
    axis_labels = []
    for i_plot, plot_label in enumerate(labels):
        ax.plot(
            model.times,
            model.get_flow_soln(plot_label) / 1E3,
            label=plot_label, linewidth=1,
            color=colours[plot_label],
            linestyle=patterns[plot_label])
        axis_labels.append(compartment_full_names[plot_label])
    set_axes_props(ax, 'Year', 'Change per year, thousands',
                   'Aggregate flows in/out of compartment',
                   True, axis_labels)
    save_png(png)


def plot_comparative_age_parameters(data_strat_list,
                                    data_value_list,
                                    model_value_list,
                                    model_strat_list,
                                    parameter_name):

    # Get good tick labels from the stratum lists
    data_strat_labels = []
    for i in range(len(data_strat_list)):
        data_strat_labels += [tool_kit.turn_strat_into_label(data_strat_list[i])]
    model_strat_labels = []
    for i in range(len(model_strat_list)):
        model_strat_labels += [tool_kit.turn_strat_into_label(model_strat_list[i])]

    # Find a reasonable upper limit for the y-axis
    ymax = max(data_value_list + model_value_list) * 1.2

    # Plot original data bar charts
    subplot_grid = (1, 2)
    fig = pyplot.figure()
    ax = fig.add_axes([0.1, 0.2, 0.35, 0.6])
    x_positions = range(len(data_strat_list))
    width = .6
    ax.bar(x_positions, data_value_list, width)
    ax.set_ylabel('Parameter value',
                  fontsize=get_nice_font_size(subplot_grid))
    ax.set_title('Input data', fontsize=12)
    ax.set_xticklabels(data_strat_labels, rotation=45)
    ax.set_xticks(x_positions)
    ax.set_ylim(0., ymax)
    ax.set_xlim(-1. + width, x_positions[-1] + 1)

    # Plot adjusted parameters bar charts
    ax = fig.add_axes([0.55, 0.2, 0.35, 0.6])
    x_positions = range(len(model_strat_list))
    ax.bar(x_positions, model_value_list, width)
    ax.set_title('Model implementation', fontsize=12)
    ax.set_xticklabels(model_strat_labels, rotation=45)
    ax.set_xticks(x_positions)
    ax.set_ylim(0., ymax)
    ax.set_xlim(-1. + width, x_positions[-1] + 1)

    # Overall title
    fig.suptitle(tool_kit.capitalise_first_letter(tool_kit.replace_underscore_with_space(parameter_name))
                 + ' adjustment',
                 fontsize=15)

    # Find directory and save
    out_dir = 'fullmodel_graphs'
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    base = os.path.join(out_dir, parameter_name)
    save_png(base + '_param_adjustment.png')


def find_subplot_numbers(n):
    """
    Method to find a good number of rows and columns for subplots of figure.

    Args:
        n: Total number of subplots.

    Returns:
        answer: List of two elements, being the rows and columns of the subplots.

    """

    # Find a nice number of subplots for a panel plot
    answer = find_smallest_factors_of_integer(n)
    i = 0
    while i < 10:
        if abs(answer[0] - answer[1]) > 3:
            n = n + 1
            answer = find_smallest_factors_of_integer(n)
        i = i + 1

    return answer


class Project:

    def __init__(self, model_runner, gui_inputs):

        """
        Initialises an object of class Project, that will contain all the information (data + outputs) for writing a
        report for a country
        Args:
            models: dictionary such as: models = {'baseline': model, 'scenario_1': model_1,  ...}
        """

        self.model_runner = model_runner
        self.inputs = self.model_runner.inputs
        self.gui_inputs = gui_inputs
        self.country = self.gui_inputs['country'].lower()
        self.name = 'test_' + self.country
        self.scenarios = []
        self.out_dir_project = os.path.join('projects', self.name)
        if not os.path.isdir(self.out_dir_project):
            os.makedirs(self.out_dir_project)
        self.figure_number = 1
        self.classifications = ['demo_', 'econ_', 'epi_', 'program_prop_', 'program_timeperiod_']
        self.output_colours = {}
        self.program_colours = {}
        self.programs = []
        self.suptitle_size = 13
        self.classified_scaleups = {}

        # Extract some characteristics from the models within model runner
        self.scenarios = self.model_runner.model_dict.keys()
        self.scenarios.reverse()
        self.programs = self.model_runner.model_dict['baseline'].interventions_to_cost

    #################################
    # General methods for use below #
    #################################

    def find_years_to_write(self, scenario, output, minimum=0, maximum=3000, step=1):

        """
        Find years that need to be written into a spreadsheet or document.

        Args:
            scenario: Model scenario to be written.
            output: Epidemiological or economic output.
            minimum: First year (integer).
            maximum: Last year (integer).
            step: Time step (integer).

        """

        requested_years = range(minimum, maximum, step)
        years = []
        for y in self.model_runner.epi_outputs_integer_dict[scenario][output]:
            if y in requested_years:
                years += [y]
        return years

    def find_var_index(self, var):

        """
        Finds the index number for a var in the var arrays. (Arbitrarily uses the baseline model from the model runner.)

        Args:
            var: String for the var that we're looking for.

        Returns:
            The var's index (unnamed).

        """

        return self.model_runner.model_dict['baseline'].var_labels.index(var)

    def set_and_update_figure(self):

        """
        If called at the start of each plotting function, will create a figure that is numbered according to
        self.figure_number, which is then updated at each call. This stops figures plotting on the same axis
        and saves you having to worry about how many figures are being opened.

        """

        fig = pyplot.figure(self.figure_number)
        self.figure_number += 1
        return fig

    def make_single_axis(self, fig):

        """
        Create axes for a figure with a single plot with a reasonable
        amount of space around.

        Returns:
            ax: The axes that can be plotted on

        """

        ax = fig.add_axes([0.1, 0.1, 0.6, 0.75])
        return ax

    def make_legend_to_single_axis(self, ax, scenario_handles, scenario_labels):

        """
        Standardised format to legend at side of single axis plot
        Args:
            ax: The axis that needs a legend.
            scenario_handles: The elements for the legend.
            scenario_labels: List of strings to name the elements of the legend.

        """

        ax.legend(scenario_handles,
                  scenario_labels,
                  bbox_to_anchor=(1.05, 1),
                  loc=2,
                  borderaxespad=0.,
                  frameon=False,
                  prop={'size': 7})

    def make_default_line_styles(self, n, return_all=True):

        """
        Produces a standard set of line styles that isn't adapted to
        the data being plotted.

        Args:
            n: The number of line-styles
            return_all: Whether to return all of the styles up to n or just the last one

        Returns:
            line_styles: A list of standard line-styles, or if return_all is False,
                then the single item (for methods that are iterating through plots.

        """

        # Iterate through a standard set of line styles
        for i in range(n):
            line_styles = []
            for line in ["-", ":", "-.", "--"]:
                for colour in "krbgmcy":
                    line_styles.append(line + colour)

        if return_all:
            return line_styles
        else:
            return line_styles[n - 1]

    def scale_axes(self, max_value):

        """
        Method to find how much the axis in question should be scaled down by for plotting numbers with high values
        (especially costs) and provide a string to amend the axis appropriately.
        Args:
            max_value: The highest value in the plot (or a group of plots)

        Returns:
            multiplier: The value to scale the axis by.
            multiplier_label: The text to add to the y-axis after scaling

        """

        if max_value < 1e3:
            multiplier = 1.
            multiplier_label = ''
        elif max_value >= 1e3 and max_value < 1e6:
            multiplier = 1e-3
            multiplier_label = 'Thousand'
        elif max_value >= 1e6 and max_value < 1e9:
            multiplier = 1e-6
            multiplier_label = 'Million'
        elif max_value >= 1e9 and max_value < 1e12:
            multiplier = 1e-9
            multiplier_label = 'Billion'
        elif max_value >= 1e12:
            multiplier = 1e-12
            multiplier_label = 'Trillion'

        return multiplier, multiplier_label

    def save_figure(self, fig, last_part_of_name_for_figure):

        """
        Simple method to standardise names for output figure files.
        Args:
            last_part_of_name_for_figure: The part of the figure name that is variable and input from the
                plotting method.

        """

        png = os.path.join(self.out_dir_project, self.country + last_part_of_name_for_figure + '.png')
        fig.savefig(png, dpi=300)

    #########################################
    # Methods to collect data for later use #
    #########################################

    def create_full_output_dict(self, outputs):

        """
        Creates a dictionary for each requested output at every time point in that model's times attribute.

        """

        for scenario in self.scenarios:
            self.full_output_dict[scenario] = {}
            self.full_output_lists[scenario] = {}
            self.full_uncertainty_dicts[scenario] = {}

            for label in outputs:
                self.full_output_lists[scenario]['times'] = self.model_runner.model_dict[scenario].times
                self.full_output_lists[scenario]['cost_times'] = self.model_runner.model_dict[scenario].cost_times
                if 'costs_' not in label:
                    array_name = 'var_array'
                    times_name = 'times'
                    index_label = self.find_var_index(label)
                else:
                    array_name = 'costs'
                    times_name = 'cost_times'
                    index_label = self.model_runner.model_dict[scenario].interventions_to_cost.index(label[6:])
                self.full_output_lists[scenario][label] \
                    = list(self.model_runner.results['scenarios'][scenario][array_name][:, index_label])
                self.full_output_dict[scenario][label] = dict(zip(self.full_output_lists[scenario][times_name],
                                                                  self.full_output_lists[scenario][label]))

                # Add centiles of uncertainty data to full dictionary
                if self.gui_inputs['output_uncertainty']:
                    self.full_uncertainty_dicts[scenario][label] = {}
                    for working_centile in range(101) + [2.5, 97.5]:
                        self.full_uncertainty_dicts[scenario][label]['centile_' + str(working_centile)] \
                            = dict(zip(self.full_output_lists[scenario][times_name],
                                   np.percentile(
                                       self.model_runner.results['uncertainty'][scenario]['var_array'][
                                       :, index_label, self.model_runner.accepted_indices],
                                       working_centile,
                                       axis=1)
                                       )
                                   )

    #################################################
    # Methods for outputting to Office applications #
    #################################################

    def master_outputs_runner(self):

        """
        Method to work through all the fundamental output methods, which then call all the specific output
        methods for plotting and writing as required.

        """

        self.write_spreadsheets()
        self.write_documents()
        self.run_plotting()
        self.open_output_directory()

    def write_spreadsheets(self):

        """
        Determine whether to write to spreadsheets by scenario or by output

        """

        if self.gui_inputs['output_spreadsheets']:
            if self.gui_inputs['output_by_scenario']:
                print('Writing scenario spreadsheets')
                self.write_xls_by_scenario()
            else:
                print('Writing output indicator spreadsheets')
                self.write_xls_by_output()

    def write_xls_by_output(self):

        # Write a new file for each output
        for output in self.model_runner.epi_outputs_integer_dict['baseline']:

            # Make filename
            path = os.path.join(self.out_dir_project, output)
            path += ".xlsx"

            # Get active sheet
            wb = xl.Workbook()
            sheet = wb.active
            sheet.title = output

            # Write a new file for each epidemiological indicator
            for scenario in self.model_runner.epi_outputs_integer_dict:
                self.write_xls_column_or_row(sheet, scenario, output)

            # Save workbook
            wb.save(path)

    def write_xls_column_or_row(self, sheet, scenario, output):

        # Find years to write
        years = self.find_years_to_write(scenario,
                                         output,
                                         int(self.inputs.model_constants['report_start_time']),
                                         int(self.inputs.model_constants['report_end_time']),
                                         int(self.inputs.model_constants['report_step_time']))

        # Write data
        if self.gui_inputs['output_horizontally']:
            self.write_horizontally_by_scenario(sheet, output, years)
        else:
            self.write_vertically_by_scenario(sheet, output, years)

    def write_xls_by_scenario(self):

        # Write a new file for each scenario
        for scenario in self.model_runner.epi_outputs_integer_dict:

            # Make filename
            path = os.path.join(self.out_dir_project, scenario)
            path += '.xlsx'

            # Get active sheet
            wb = xl.Workbook()
            sheet = wb.active
            sheet.title = scenario

            for output in self.model_runner.epi_outputs_integer_dict['baseline']:
                self.write_xls_column_or_row(sheet, scenario, output)

            # Save workbook
            wb.save(path)

    def write_horizontally_by_scenario(self, sheet, output, years):

        sheet.cell(row=1, column=1).value = 'Year'
        for y, year in enumerate(years):
            sheet.cell(row=1, column=y+1).value = year
        for s, scenario in enumerate(self.scenarios):
            sheet.cell(row=s+1, column=1).value = \
                tool_kit.replace_underscore_with_space(
                    tool_kit.capitalise_first_letter(scenario))
            for y, year in enumerate(years):
                if year in self.model_runner.epi_outputs_integer_dict[scenario][output]:
                    sheet.cell(row=s+1, column=y+1).value \
                        = self.model_runner.epi_outputs_integer_dict[scenario][output][year]

    def write_horizontally_by_output(self, sheet, scenario, years):

        sheet.cell(row=1, column=1).value = 'Year'
        for y, year in enumerate(years):
            sheet.cell(row=1, column=y+1).value = year
        for o, output in enumerate(self.model_runner.epi_outputs_integer_dict['baseline']):
            sheet.cell(row=o+1, column=1).value = \
                tool_kit.replace_underscore_with_space(
                    tool_kit.capitalise_first_letter(output))
            for y, year in enumerate(years):
                if year in self.model_runner.epi_outputs_integer_dict[scenario][output]:
                    sheet.cell(row=r, column=y+1).value \
                        = self.model_runner.epi_outputs_integer_dict[scenario][output][year]

    def write_vertically_by_scenario(self, sheet, output, years):

        sheet.cell(row=1, column=1).value = 'Year'
        for y, year in enumerate(years):
            sheet.cell(row=y+1, column=1).value = year
        for s, scenario in enumerate(self.scenarios):
            sheet.cell(row=1, column=s+1).value = \
                tool_kit.replace_underscore_with_space(
                    tool_kit.capitalise_first_letter(scenario))
            for y, year in enumerate(years):
                if year in self.model_runner.epi_outputs_integer_dict[scenario][output]:
                    sheet.cell(row=y+1, column=s+1).value \
                        = self.model_runner.epi_outputs_integer_dict[scenario][output][year]

    def write_vertically_by_output(self, sheet, scenario, years):

        sheet.cell(row=1, column=1).value = 'Year'
        for y, year in enumerate(years):
            sheet.cell(row=y+1, column=1).value = year
        for o, output in enumerate(self.model_runner.epi_outputs_integer_dict['baseline']):
            sheet.cell(row=1, column=o+1).value = \
                tool_kit.replace_underscore_with_space(
                    tool_kit.capitalise_first_letter(output))
            for y, year in enumerate(years):
                if year in self.model_runner.epi_outputs_integer_dict[scenario][output]:
                    sheet.cell(row=y+1, column=o+1).value \
                        = self.model_runner.epi_outputs_integer_dict[scenario][output][year]

    def write_documents(self):

        """
        Determine whether to write to documents by scenario or by output

        """

        if self.gui_inputs['output_documents']:
            if self.gui_inputs['output_by_scenario']:
                print('Writing scenario documents')
                self.write_docs_by_scenario()
            else:
                print('Writing output indicator documents')
                self.write_docs_by_output()

    def write_docs_by_output(self):

        # Write a new file for each output
        for output in self.model_runner.epi_outputs_integer_dict['baseline']:

            # Initialise document
            path = os.path.join(self.out_dir_project, output)
            path += ".docx"
            document = Document()
            table = document.add_table(rows=1, cols=len(self.scenarios) + 1)

            # Write headers
            header_cells = table.rows[0].cells
            header_cells[0].text = 'Year'
            for s, scenario in enumerate(self.scenarios):
                header_cells[s + 1].text \
                    = tool_kit.capitalise_first_letter(tool_kit.replace_underscore_with_space(scenario))

            # Find years to write
            years = self.find_years_to_write('baseline',
                                             output,
                                             int(self.inputs.model_constants['report_start_time']),
                                             int(self.inputs.model_constants['report_end_time']),
                                             int(self.inputs.model_constants['report_step_time']))

            for year in years:

                # Add row to table
                row_cells = table.add_row().cells
                row_cells[0].text = str(year)

                for s, scenario in enumerate(self.scenarios):
                    if year in self.model_runner.epi_outputs_integer_dict[scenario][output]:
                        row_cells[s + 1].text = '%.2f' % self.model_runner.epi_outputs_integer_dict[
                            scenario][output][year]

            # Save document
            document.save(path)

    def write_docs_by_scenario(self):

        # Write a new file for each output
        outputs = self.model_runner.epi_outputs_integer_dict['baseline'].keys()

        for scenario in self.scenarios:

            # Initialise document
            path = os.path.join(self.out_dir_project, scenario)
            path += ".docx"
            document = Document()
            table = document.add_table(rows=1, cols=len(outputs) + 1)

            # Write headers
            header_cells = table.rows[0].cells
            header_cells[0].text = 'Year'
            for o, output in enumerate(outputs):
                header_cells[o + 1].text \
                    = tool_kit.capitalise_first_letter(tool_kit.replace_underscore_with_space(output))

            # Find years to write
            years = self.find_years_to_write(scenario,
                                             output,
                                             int(self.inputs.model_constants['report_start_time']),
                                             int(self.inputs.model_constants['report_end_time']),
                                             int(self.inputs.model_constants['report_step_time']))

            for year in years:

                # Add row to table
                row_cells = table.add_row().cells
                row_cells[0].text = str(year)

                for o, output in enumerate(outputs):
                    if year in self.model_runner.epi_outputs_integer_dict[scenario][output]:
                        row_cells[o + 1].text = '%.2f' % self.model_runner.epi_outputs_integer_dict[
                            scenario][output][year]

            # Save document
            document.save(path)

    def run_plotting(self):

        # Find some general output colours
        output_colours = self.make_default_line_styles(5, True)
        for s, scenario in enumerate(self.scenarios):
            self.output_colours[scenario] = output_colours[s]
        for p, program in enumerate(self.programs):
            # +1 is to avoid starting from black, which doesn't look as nice for programs as for baseline scenario
            self.program_colours[program] = output_colours[p + 1]

        # Plot main outputs
        if self.gui_inputs['output_gtb_plots']:
            self.plot_outputs_against_gtb(
                ['incidence', 'mortality', 'prevalence', 'notifications'])

        # Plot scale-up functions - currently only doing this for the baseline model run
        if self.gui_inputs['output_scaleups']:
            self.classify_scaleups()
            self.plot_scaleup_fns_against_data()
            self.plot_programmatic_scaleups()

        # Plot economic outputs
        if self.gui_inputs['output_plot_economics']:
            self.plot_cost_coverage_curves()
            self.plot_cost_over_time()

        # Plot compartment population sizes
        if self.gui_inputs['output_compartment_populations']:
            self.plot_populations()

        # Plot fractions
        if self.gui_inputs['output_fractions']:
            self.plot_fractions('strain')

        # Plot outputs by age group
        if self.gui_inputs['output_by_age']:
            if len(self.inputs.agegroups) > 1:
                self.plot_outputs_by_age()
            else:
                warnings.warn('Requested outputs by age, but model is not age stratified.')

        # Plot proportions of population
        if self.gui_inputs['output_age_fractions']:
            self.plot_stratified_populations(age_or_comorbidity='age')

        # Plot comorbidity proportions
        if self.gui_inputs['output_comorbidity_fractions']:
            self.plot_stratified_populations(age_or_comorbidity='comorbidity')

        # Make a flow-diagram
        if self.gui_inputs['output_flow_diagram']:
            png = os.path.join(self.out_dir_project, self.country + '_flow_diagram' + '.png')
            self.model_runner.model_dict['baseline'].make_flow_diagram(png)

        # Plot comorbidity proportions
        if self.gui_inputs['output_plot_comorbidity_checks'] \
                and len(self.model_runner.model_dict['baseline'].comorbidities) > 1:
            self.plot_comorb_checks()

    def plot_outputs_against_gtb(self, outputs):

        """
        Produces the plot for the main outputs, can handle multiple scenarios.
        Save as png at the end.
        Note that if running a series of scenarios, it is expected that the last scenario to
        be run will be baseline, which should have scenario set to None.

        Args:
            outputs: A list of the outputs to be plotted

        """

        # Standard preliminaries
        start_time = self.inputs.model_constants['plot_start_time']
        scenario_labels = []
        colour, indices, yaxis_label, title, patch_colour = \
            find_standard_output_styles(outputs, lightening_factor=0.3)
        subplot_grid = find_subplot_numbers(len(outputs))
        fig = self.set_and_update_figure()

        # Loop through outputs
        for out, output in enumerate(outputs):

            ax = fig.add_subplot(subplot_grid[0], subplot_grid[1], out + 1)

            # Plotting of GTB data
            plotting_data = {}

            # Notifications
            if output == 'notifications':
                plotting_data['point_estimate'] = self.inputs.original_data['notifications']['c_newinc']

            # Other reported data
            else:
                for indicator in self.inputs.original_data['tb']:
                    if indices[out] in indicator and '_lo' in indicator:
                        plotting_data['lower_limit'] = self.inputs.original_data['tb'][indicator]
                    elif indices[out] in indicator and '_hi' in indicator:
                        plotting_data['upper_limit'] = self.inputs.original_data['tb'][indicator]
                    elif indices[out] in indicator:
                        plotting_data['point_estimate'] = self.inputs.original_data['tb'][indicator]

                # Create and plot the patch array
                patch_array = create_patch_from_dictionary(plotting_data)
                patch = patches.Polygon(patch_array, color=patch_colour[out])
                ax.add_patch(patch)

            # Plot point estimates
            ax.plot(plotting_data['point_estimate'].keys(), plotting_data['point_estimate'].values(),
                    color=colour[out], linewidth=0.5)

            # Loop through scenarios that have been run and plot
            for scenario in reversed(self.scenarios):

                if not self.gui_inputs['output_uncertainty']:

                    # Plot the modelled data
                    ax.plot(
                        self.model_runner.model_dict[scenario].times,
                        self.model_runner.epi_outputs[scenario][output],
                        color=self.output_colours[scenario][1],
                        linestyle=self.output_colours[scenario][0],
                        linewidth=1.5)

                else:
                    modelled_time, modelled_values = \
                        tool_kit.get_truncated_lists_from_dict(self.full_uncertainty_dicts[scenario][output]['centile_50'],
                                                               start_time)
                    modelled_time_upper, modelled_values_upper = \
                        tool_kit.get_truncated_lists_from_dict(self.full_uncertainty_dicts[scenario][output]['centile_97.5'],
                                                               start_time)
                    modelled_time_lower, modelled_values_lower = \
                        tool_kit.get_truncated_lists_from_dict(self.full_uncertainty_dicts[scenario][output]['centile_2.5'],
                                                               start_time)

                    # Plot the modelled data
                    ax.plot(
                        modelled_time,
                        modelled_values,
                        color=self.output_colours[scenario][1],
                        linestyle=self.output_colours[scenario][0],
                        linewidth=1.5)
                    ax.plot(
                        modelled_time_upper,
                        modelled_values_upper,
                        color=self.output_colours[scenario][1],
                        linestyle='--',
                        linewidth=1)
                    ax.plot(
                        modelled_time_lower,
                        modelled_values_lower,
                        color=self.output_colours[scenario][1],
                        linestyle='--',
                        linewidth=1)

                # Add scenario label
                scenario_labels += [tool_kit.capitalise_first_letter(tool_kit.replace_underscore_with_space(scenario))]

            # Set vertical plot axis dimensions
            ax.set_xlim((start_time, self.inputs.model_constants['plot_end_time']))

            # Set x-ticks
            ax.set_xticks(find_reasonable_year_ticks(start_time, self.inputs.model_constants['plot_end_time']))

            # Adjust size of labels of x-ticks
            for axis_to_change in [ax.xaxis, ax.yaxis]:
                for tick in axis_to_change.get_major_ticks():
                    tick.label.set_fontsize(get_nice_font_size(subplot_grid))

            # Add the sub-plot title with slightly larger titles than the rest of the text on the panel
            ax.set_title(title[out], fontsize=get_nice_font_size(subplot_grid) + 2.)

            # Label the y axis with the smaller text size
            ax.set_ylabel(yaxis_label[out], fontsize=get_nice_font_size(subplot_grid))

            # Add the legend
            scenario_handles = ax.lines[1:]
            if out == len(outputs) - 1:
                ax.legend(scenario_handles,
                          scenario_labels,
                          fontsize=get_nice_font_size(subplot_grid),
                          frameon=False)

        # Add main title and save
        fig.suptitle(tool_kit.capitalise_first_letter(self.country) + ' model outputs', fontsize=self.suptitle_size)
        self.save_figure(fig, '_main_outputs')

    def classify_scaleups(self):

        """
        Classifies the time variant parameters according to their type (e.g. programmatic, economic, demographic, etc.)

        """

        for classification in self.classifications:
            self.classified_scaleups[classification] = []
            for fn in self.model_runner.model_dict['baseline'].scaleup_fns:
                if classification in fn:
                    self.classified_scaleups[classification] += [fn]

    def plot_scaleup_fns_against_data(self):

        """
        Plot each scale-up function as a separate panel against the data it is fitted to.

        """

        # Different figure for each type of function
        for classification in self.classified_scaleups:

            # Find the list of the scale-up functions to work with and some x-values
            functions = self.classified_scaleups[classification]

            # Standard prelims
            subplot_grid = find_subplot_numbers(len(functions))
            fig = self.set_and_update_figure()

            # Find some x-values
            start_time = self.inputs.model_constants['plot_start_time']
            end_time = self.inputs.model_constants['plot_end_time']
            x_vals = numpy.linspace(start_time, end_time, 1e3)

            # Main title for whole figure
            title = self.inputs.country + ' ' + \
                    tool_kit.find_title_from_dictionary(classification) + ' parameter'
            if len(functions) > 1:
                title += 's'
            fig.suptitle(title, fontsize=self.suptitle_size)

            # Iterate through functions
            for f, function in enumerate(functions):

                # Initialise subplot area
                ax = fig.add_subplot(subplot_grid[0], subplot_grid[1], f + 1)

                # Iterate through the scenarios
                scenario_labels = []
                for scenario in reversed(self.scenarios):

                    # Line plot of scaling parameter functions
                    ax.plot(x_vals,
                            map(self.model_runner.model_dict[scenario].scaleup_fns[function],
                                x_vals),
                            color=self.output_colours[scenario][1])

                    # Record the name of the scenario for the legend
                    scenario_labels \
                        += [tool_kit.capitalise_first_letter(tool_kit.replace_underscore_with_space(scenario))]

                # Plot the raw data from which the scale-up functions were produced
                data_to_plot = {}
                for year in self.inputs.scaleup_data[None][function]:
                    if year > start_time:
                        data_to_plot[year] = self.inputs.scaleup_data[None][function][year]

                # Scatter plot data from which they are derived
                ax.scatter(data_to_plot.keys(), data_to_plot.values(), color='k', s=6)

                # Adjust tick font size and add panel title
                ax.set_xticks([start_time, end_time])
                for axis_to_change in [ax.xaxis, ax.yaxis]:
                    for tick in axis_to_change.get_major_ticks():
                        tick.label.set_fontsize(get_nice_font_size(subplot_grid))
                title = tool_kit.find_title_from_dictionary(function)
                ax.set_title(title, fontsize=get_nice_font_size(subplot_grid))
                ylims = relax_y_axis(ax)
                ax.set_ylim(bottom=ylims[0], top=ylims[1])

                # Add legend to last plot
                scenario_handles = ax.lines
                if f == len(functions) - 1:
                    ax.legend(scenario_handles,
                              scenario_labels,
                              fontsize=get_nice_font_size(subplot_grid),
                              frameon=False)

            # Save
            self.save_figure(fig, '_' + classification + '_scale_ups')

    def plot_programmatic_scaleups(self):

        """
        Plots only the programmatic time-variant functions on a single set of axes

        """

        # Functions to plot are those in the program_prop_ category of the classified scaleups
        # (classify_scaleups must have been run)
        functions = self.classified_scaleups['program_prop_']

        # Standard prelims
        fig = self.set_and_update_figure()
        line_styles = self.make_default_line_styles(len(functions), True)

        # Get some x values for plotting
        x_vals = numpy.linspace(self.inputs.model_constants['plot_start_time'],
                                self.inputs.model_constants['plot_end_time'],
                                1e3)

        # Plot functions for baseline model run only
        scenario_labels = []
        ax = self.make_single_axis(fig)
        for figure_number, function in enumerate(functions):
            ax.plot(x_vals,
                    map(self.inputs.scaleup_fns[None][function],
                        x_vals), line_styles[figure_number],
                    label=function)
            scenario_labels += [tool_kit.find_title_from_dictionary(function)]

        # Make title, legend, generally tidy up and save
        title = tool_kit.capitalise_first_letter(self.country) + ' ' + \
                tool_kit.find_title_from_dictionary('program_prop_') + \
                ' parameters'
        set_axes_props(ax, 'Year', 'Parameter value',
                       title, True, functions)
        ylims = relax_y_axis(ax)
        ax.set_ylim(bottom=ylims[0], top=ylims[1])
        scenario_handles = ax.lines
        self.make_legend_to_single_axis(ax, scenario_handles, scenario_labels)
        self.save_figure(fig, '_programmatic_scale_ups')

    def plot_cost_coverage_curves(self):

        """
        Plots cost-coverage curves at times specified in the report times inputs in control panel.

        """

        # Plot figures by scenario
        for scenario in self.scenarios:

            fig = self.set_and_update_figure()

            # Subplots by program
            subplot_grid = find_subplot_numbers(len(self.programs))
            for p, program in enumerate(self.programs):

                ax = fig.add_subplot(subplot_grid[0], subplot_grid[1], p + 1)
                scenario_labels = []

                # Make times that each curve is produced for from control panel inputs
                times = range(int(self.inputs.model_constants['cost_curve_start_time']),
                              int(self.inputs.model_constants['cost_curve_end_time']),
                              int(self.inputs.model_constants['cost_curve_step_time']))

                for t, time in enumerate(times):
                    time_index = tool_kit.find_first_list_element_at_least_value(
                        self.model_runner.model_dict[scenario].times, time)
                    y_values = []
                    x_values = []
                    for i in numpy.linspace(0, 1, 101):

                        # Make cost coverage curve
                        if i < self.inputs.model_constants['econ_saturation_' + program]:
                            cost = economics.get_cost_from_coverage(i,
                                                                    self.inputs.model_constants['econ_inflectioncost_'
                                                                                                + program],
                                                                    self.inputs.model_constants['econ_saturation_'
                                                                                                + program],
                                                                    self.inputs.model_constants['econ_unitcost_'
                                                                                                + program],
                                                                    self.model_runner.model_dict[
                                                                        scenario].var_array[
                                                                        time_index,
                                                                        self.model_runner.model_dict[
                                                                            scenario].var_labels.index('popsize_'
                                                                                                       + program)])
                            x_values += [cost]
                            y_values += [i]

                    # Find darkness
                    darkness = .9 - (float(t) / float(len(times))) * .9

                    # Scale data
                    multiplier, multiplier_label = self.scale_axes(max(x_values))
                    x_values_to_plot = [x * multiplier for x in x_values]

                    # Plot
                    ax.plot(x_values_to_plot, y_values, color=(darkness, darkness, darkness))

                    # Find label for legend
                    scenario_labels += [str(int(time))]

                # Legend to last panel
                if p == len(self.programs) - 1:
                    scenario_handles = ax.lines
                    self.make_legend_to_single_axis(ax, scenario_handles, scenario_labels)
                ax.set_title(tool_kit.find_title_from_dictionary('program_prop_' + program)[:-9],
                             fontsize=get_nice_font_size(subplot_grid)+2)

                # X-axis label
                ax.set_xlabel(multiplier_label + ' $US', fontsize=get_nice_font_size(subplot_grid), labelpad=1)
                for axis_to_change in [ax.xaxis, ax.yaxis]:
                    for tick in axis_to_change.get_major_ticks():
                        tick.label.set_fontsize(get_nice_font_size(subplot_grid))

            # Finish off with title and save file for scenario
            fig.suptitle('Cost-coverage curves for ' + tool_kit.replace_underscore_with_space(scenario),
                         fontsize=self.suptitle_size)
            self.save_figure(fig, '_' + scenario + '_cost_coverage')

    def plot_cost_over_time(self):

        """
        Method that produces plots for individual and cumulative program costs for each scenario as separate figures.
        Panels of figures are the different sorts of costs (i.e. whether discounting and inflation have been applied).

        """

        cost_types = ['discounted_inflated', 'inflated', 'discounted', 'raw']

        # Separate figures for each scenario
        for scenario in self.scenarios:

            # Standard prelims, but separate for each type of plot - individual and stacked
            fig_individual = self.set_and_update_figure()
            fig_stacked = self.set_and_update_figure()
            fig_relative = self.set_and_update_figure()
            subplot_grid = find_subplot_numbers(len(cost_types))

            # Find the maximum of any type of cost across all of the programs
            max_cost = 0.
            max_stacked_cost = 0.

            for program in self.programs:
                for cost_type in cost_types:
                    if max(self.model_runner.cost_outputs[scenario][cost_type + '_cost_' + program]) > max_cost:
                        max_cost = max(self.model_runner.cost_outputs[scenario][cost_type + '_cost_' + program])
            # for cost_type in cost_types:
            #     if max(self.model_runner.outputs[scenario][cost_type + '_cost_all_programs']) > max_stacked_cost:
            #         max_stacked_cost = max(self.model_runner.outputs[scenario][cost_type + '_cost_all_programs'])

            # Scale vertical axis and amend axis label as appropriate
            multiplier_individual, multiplier_individual_label = self.scale_axes(max_cost)
            # multiplier_stacked, multiplier_stacked_label = self.scale_axes(max_stacked_cost)

            # Find the index for the first time after the current time
            reference_time_index \
                = tool_kit.find_first_list_element_above_value(self.model_runner.cost_outputs[scenario]['times'],
                                                               self.inputs.model_constants['reference_time'])
            for c, cost_type in enumerate(cost_types):

                # Plot each type of cost to its own subplot and ensure same y-axis scale
                if c == 0:
                    ax_individual = fig_individual.add_subplot(subplot_grid[0], subplot_grid[1], c + 1)
                    ax_stacked = fig_stacked.add_subplot(subplot_grid[0], subplot_grid[1], c + 1)
                    ax_relative = fig_relative.add_subplot(subplot_grid[0], subplot_grid[1], c + 1)
                    ax_individual_first = copy.copy(ax_individual)
                    ax_stacked_first = copy.copy(ax_stacked)
                    ax_reference_first = copy.copy(ax_relative)
                else:
                    ax_individual = fig_individual.add_subplot(subplot_grid[0], subplot_grid[1], c + 1,
                                                               sharey=ax_individual_first)
                    ax_stacked = fig_stacked.add_subplot(subplot_grid[0], subplot_grid[1], c + 1,
                                                         sharey=ax_stacked_first)
                    ax_relative = fig_relative.add_subplot(subplot_grid[0], subplot_grid[1], c + 1,
                                                           sharey=ax_reference_first)

                # Create empty list for legend
                program_labels = []
                cumulative_data = [0.] * len(self.model_runner.cost_outputs[scenario]['times'])

                for program in self.model_runner.model_dict[scenario].interventions_to_cost:

                    # Record the previous data for plotting as an independent object for the lower edge of the fill
                    previous_data = copy.copy(cumulative_data)

                    # Calculate the cumulative sum for the upper edge of the fill
                    for i in range(len(self.model_runner.cost_outputs[scenario]['times'])):
                        cumulative_data[i] += self.model_runner.cost_outputs[scenario][cost_type + '_cost_' + program][i]

                    # Scale all the data
                    data = self.model_runner.cost_outputs[scenario][cost_type + '_cost_' + program]

                    individual_data = [d * multiplier_individual for d in data]
                    # cumulative_data_to_plot = [d * multiplier_stacked for d in cumulative_data]
                    # previous_data_to_plot = [d * multiplier_stacked for d in previous_data]

                    reference_cost \
                        = self.model_runner.cost_outputs[scenario][cost_type + '_cost_' + program][reference_time_index]
                    relative_data = [(d - reference_cost) * multiplier_individual for d in data]

                    # Plot lines
                    ax_individual.plot(self.model_runner.cost_outputs[scenario]['times'],
                                       individual_data,
                                       color=self.program_colours[program][1])
                    ax_relative.plot(self.model_runner.cost_outputs[scenario]['times'],
                                     relative_data,
                                     color=self.program_colours[program][1])

                    # Plot stacked
                    # ax_stacked.fill_between(self.model_runner.model_dict[scenario].cost_times,
                    #                         previous_data_to_plot,
                    #                         cumulative_data_to_plot,
                    #                         color=self.program_colours[program][1],
                    #                         linewidth=0.)

                    # Record label for legend
                    program_labels += [tool_kit.find_title_from_dictionary(program)]

                # Axis title and y-axis label
                for ax in [ax_individual, ax_stacked, ax_relative]:
                    ax.set_title(tool_kit.capitalise_first_letter(tool_kit.replace_underscore_with_space(cost_type)),
                                 fontsize=8)
                    ax.set_ylabel(multiplier_individual_label + ' $US',
                                  fontsize=get_nice_font_size(subplot_grid))

                    # Tidy ticks
                    for tick in ax.xaxis.get_major_ticks():
                        tick.label.set_fontsize(get_nice_font_size(subplot_grid))
                    for tick in ax.yaxis.get_major_ticks():
                        tick.label.set_fontsize(get_nice_font_size(subplot_grid))

                    # Add the legend to last subplot panel
                    if c == len(cost_types) - 1:
                        ax.legend(ax_individual.lines,
                                  program_labels,
                                  fontsize=get_nice_font_size(subplot_grid),
                                  frameon=False)

                # Set x-limits
                for ax in [ax_individual, ax_stacked, ax_relative]:
                    ax.set_xlim(self.inputs.model_constants['plot_start_time'],
                                self.inputs.model_constants['plot_end_time'])

            # Finishing off with title and save
            fig_individual.suptitle('Individual program costs for ' + tool_kit.find_title_from_dictionary(scenario),
                                    fontsize=self.suptitle_size)
            self.save_figure(fig_individual, '_' + scenario + '_timecost_individual')
            fig_stacked.suptitle('Stacked program costs for ' + tool_kit.find_title_from_dictionary(scenario),
                                 fontsize=self.suptitle_size)
            self.save_figure(fig_stacked, '_' + scenario + '_timecost_stacked')
            fig_relative.suptitle('Relative program costs for ' + tool_kit.find_title_from_dictionary(scenario),
                                  fontsize=self.suptitle_size)
            self.save_figure(fig_relative, '_' + scenario + '_timecost_relative')

    def plot_populations(self, strain_or_organ='organ'):

        """
        Plot population by the compartment to which they belong.

        Args:
            strain_or_organ: Whether the plotting style should be done by strain or by organ.

        """

        # Standard prelims
        fig = self.set_and_update_figure()
        ax = self.make_single_axis(fig)

        # Get plotting styles
        colours, patterns, compartment_full_names, markers \
            = make_related_line_styles(self.model_runner.model_dict['baseline'].labels, strain_or_organ)

        # Initialise empty list for legend
        axis_labels = []

        # Plot total population
        ax.plot(
            self.model_runner.epi_outputs['baseline']['times'],
            self.model_runner.epi_outputs['baseline']['population'],
            'k',
            label='total', linewidth=2)
        axis_labels.append('Number of persons')

        # Plot sub-populations
        for plot_label in self.model_runner.model_dict['baseline'].labels:
            ax.plot(self.model_runner.epi_outputs['baseline']['times'],
                    self.model_runner.model_dict['baseline'].compartment_soln[plot_label],
                    label=plot_label, linewidth=1,
                    color=colours[plot_label],
                    marker=markers[plot_label],
                    linestyle=patterns[plot_label])
            axis_labels.append(compartment_full_names[plot_label])

        # Finishing touches
        ax.set_xlim(self.inputs.model_constants['plot_start_time'],
                    self.inputs.model_constants['plot_end_time'])
        title = make_plot_title('baseline', self.model_runner.model_dict['baseline'].labels)
        set_axes_props(ax, 'Year', 'Persons', 'Population, ' + title, True, axis_labels)

        # Saving
        self.save_figure(fig, '_population')

    def plot_fractions(self, strain_or_organ):

        """
        Plot population fractions by the compartment to which they belong.

        Args:
            strain_or_organ: Whether the plotting style should be done by strain or by organ.

        """

        # Get values to be plotted
        subgroup_solns, subgroup_fractions = autumn.tool_kit.find_fractions(self.model_runner.model_dict['baseline'])
        for i, category in enumerate(subgroup_fractions):
            values = subgroup_fractions[category]

            # Standard prelims
            fig = self.set_and_update_figure()
            ax = self.make_single_axis(fig)

            # Get plotting styles
            colours, patterns, compartment_full_names, markers \
                = make_related_line_styles(values.keys(), strain_or_organ)

            # Initialise empty list for legend
            axis_labels = []

            # Plot population fractions
            for plot_label in values.keys():
                ax.plot(
                    self.model_runner.model_dict['baseline'].times,
                    values[plot_label],
                    label=plot_label, linewidth=1,
                    color=colours[plot_label],
                    marker=markers[plot_label],
                    linestyle=patterns[plot_label])
                axis_labels.append(compartment_full_names[plot_label])

            # Finishing touches
            ax.set_xlim(self.inputs.model_constants['plot_start_time'],
                        self.inputs.model_constants['plot_end_time'])
            title = make_plot_title(self.model_runner.model_dict['baseline'], values.keys())
            set_axes_props(ax, 'Year', 'Proportion of population', 'Population, ' + title, True, axis_labels)

            # Saving
            self.save_figure(fig, '_fraction')

    def plot_outputs_by_age(self):

        """
        Produces the plot for the main outputs by age, can handle multiple scenarios (if required).
        Save as png at the end.
        Note that if running a series of scenarios, it is expected that the last scenario to
        be run will be baseline, which should have scenario set to None.
        This function is a bit less flexible than plot_outputs_against_gtb, in which you can select the
        outputs you want to plot. This one is constrained to incidence and mortality (which are the only
        ones currently calculated in the model object.

        """

        # Old code from when scenario used to be an input argument to this method
        scenario = None

        # Get the colours for the model outputs
        if scenario is None:
            # Last scenario to run should be baseline and should be run last
            # to lay a black line over the top for comparison
            output_colour = ['-k']
        else:
            # Otherwise cycling through colours
            output_colour = [self.make_default_line_styles(scenario, False)]

        subplot_grid = find_subplot_numbers(len(self.model_runner.model_dict['baseline'].agegroups) * 2 + 1)

        # Not sure whether we have to specify a figure number
        fig = self.set_and_update_figure()

        # Overall title
        fig.suptitle(self.country + ' burden by age group', fontsize=self.suptitle_size)

        for o, output in enumerate(['incidence', 'mortality']):

            # Find the highest incidence value in the time period considered across all age groups
            ymax = 0.
            for agegroup in self.model_runner.model_dict['baseline'].agegroups:
                new_ymax = max(self.model_runner.epi_outputs['baseline'][output + agegroup])
                if new_ymax > ymax:
                    ymax = new_ymax

            for i, agegroup in enumerate(self.model_runner.model_dict['baseline'].agegroups + ['']):

                ax = fig.add_subplot(subplot_grid[0],
                                     subplot_grid[1],
                                     i + o * (len(self.model_runner.model_dict['baseline'].agegroups) + 1))

                # Plot the modelled data
                ax.plot(
                    self.model_runner.epi_outputs['baseline']['times'],
                    self.model_runner.epi_outputs['baseline'][output + agegroup],
                    color=output_colour[0][1],
                    linestyle=output_colour[0][0],
                    linewidth=1.5)

                # Adjust size of labels of x-ticks
                for axis_to_change in [ax.xaxis, ax.yaxis]:
                    for tick in axis_to_change.get_major_ticks():
                        tick.label.set_fontsize(get_nice_font_size(subplot_grid))

                # Add the sub-plot title with slightly larger titles than the rest of the text on the panel
                ax.set_title(tool_kit.capitalise_first_letter(output) + ', '
                             + tool_kit.turn_strat_into_label(agegroup), fontsize=get_nice_font_size(subplot_grid))

                # Label the y axis with the smaller text size
                if i == 0:
                    ax.set_ylabel('Per 100,000 per year', fontsize=get_nice_font_size(subplot_grid))

                # Set upper y-limit to the maximum value for any age group during the period of interest
                ax.set_ylim(bottom=0., top=ymax)

                # Get the handles, except for the last one, which plots the data
                scenario_handles = ax.lines[:-1]

                # Make some string labels for these handles
                # (this code could probably be better)
                scenario_labels = []
                for s in range(len(scenario_handles)):
                    if s < len(scenario_handles) - 1:
                        scenario_labels += ['Scenario ' + str(s + 1)]
                    else:
                        scenario_labels += ['Baseline']

                # Draw the legend
                ax.legend(scenario_handles,
                          scenario_labels,
                          fontsize=get_nice_font_size(subplot_grid) - 2.,
                          frameon=False)

                # Finishing touches
                ax.set_xlim(self.inputs.model_constants['plot_start_time'],
                            self.inputs.model_constants['plot_end_time'])

        # Saving
        self.save_figure(fig, '_output_by_age')

    def plot_stratified_populations(self, age_or_comorbidity='age'):

        """
        Function to plot population by age group both as raw numbers and as proportions,
        both from the start of the model and using the input argument

        """

        if age_or_comorbidity == 'age':
            stratification = self.model_runner.model_dict['baseline'].agegroups
        elif age_or_comorbidity == 'comorbidity':
            stratification = self.model_runner.model_dict['baseline'].comorbidities
        else:
            stratification = None

        if stratification is None:
            warnings.warn('Plotting by stratification requested, but type of stratification requested unknown')
        elif len(stratification) < 2:
            warnings.warn('No stratification to plot')
        else:
            # Standard prelims
            fig = self.set_and_update_figure()
            colours = self.make_default_line_styles(len(stratification), return_all=True)

            # Loop over starting from the model start and the specified starting time
            for i_time, plot_left_time in enumerate(['plot_start_time', 'early_time']):

                # Find starting times
                title_time_text = tool_kit.find_title_from_dictionary(plot_left_time)

                # Initialise some variables
                times = self.model_runner.model_dict['baseline'].times
                lower_plot_margin_count = numpy.zeros(len(times))
                upper_plot_margin_count = numpy.zeros(len(times))
                lower_plot_margin_fraction = numpy.zeros(len(times))
                upper_plot_margin_fraction = numpy.zeros(len(times))
                legd_text = []

                for i, stratum in enumerate(stratification):

                    # Find numbers or fractions in that group
                    stratum_count = self.model_runner.epi_outputs['baseline']['population' + stratum]
                    stratum_fraction = self.model_runner.epi_outputs['baseline']['fraction' + stratum]

                    # Add group values to the upper plot range for area plot
                    for j in range(len(upper_plot_margin_count)):
                        upper_plot_margin_count[j] += stratum_count[j]
                        upper_plot_margin_fraction[j] += stratum_fraction[j]

                    # Plot
                    ax_upper = fig.add_subplot(2, 2, 1 + i_time)
                    ax_upper.fill_between(times,
                                          lower_plot_margin_count,
                                          upper_plot_margin_count,
                                          facecolors=colours[i][1])

                    # Create proxy for legend
                    ax_upper.plot([], [], color=colours[i][1], linewidth=6)
                    if age_or_comorbidity == 'age':
                        legd_text += [tool_kit.turn_strat_into_label(stratum)]
                    elif age_or_comorbidity == 'comorbidity':
                        legd_text += [tool_kit.find_title_from_dictionary(stratum)]

                    # Cosmetic changes at the end
                    if i == len(stratification) - 1:
                        ax_upper.set_title('Total numbers from ' + title_time_text, fontsize=8)
                        if i_time == 1:
                            ax_upper.legend(reversed(ax_upper.lines),
                                            reversed(legd_text), loc=2, frameon=False, fontsize=8)

                    # Plot population proportions
                    ax_lower = fig.add_subplot(2, 2, 3 + i_time)
                    ax_lower.fill_between(times, lower_plot_margin_fraction, upper_plot_margin_fraction,
                                          facecolors=colours[i][1])

                    # Cosmetic changes at the end
                    if i == len(stratification) - 1:
                        ax_lower.set_ylim((0., 1.))
                        ax_lower.set_title('Proportion of population from ' + title_time_text, fontsize=8)

                    # Add group values to the lower plot range for next iteration
                    for j in range(len(lower_plot_margin_count)):
                        lower_plot_margin_count[j] += stratum_count[j]
                        lower_plot_margin_fraction[j] += stratum_fraction[j]

                    for axis_to_change in [ax_upper.xaxis, ax_upper.yaxis]:
                        for tick in axis_to_change.get_major_ticks():
                            tick.label.set_fontsize(get_nice_font_size([2]))
                    for axis_to_change in [ax_lower.xaxis, ax_lower.yaxis]:
                        for tick in axis_to_change.get_major_ticks():
                            tick.label.set_fontsize(get_nice_font_size([2]))

                    ax_upper.set_xlim(self.inputs.model_constants[plot_left_time],
                                      self.inputs.model_constants['plot_end_time'])
                    ax_lower.set_xlim(self.inputs.model_constants[plot_left_time],
                                      self.inputs.model_constants['plot_end_time'])

            # Finish up
            fig.suptitle('Population by ' + tool_kit.find_title_from_dictionary(age_or_comorbidity),
                         fontsize=self.suptitle_size)
            self.save_figure(fig, '_comorbidity_proportions')

    def plot_intervention_costs_by_scenario(self, year_start, year_end, horizontal=False, plot_options=None):

        """

        Eike, 02/09/16

        Function for plotting total cost of interventions under different scenarios over a given range of years.

        Args:
            year_start: integer, start year of time frame over which to calculate total costs
            year_end:   integer, end year of time frame over which to calculate total costs (included)
            horizontal: boolean, plot stacked bar chart horizontally
            plot_options: dictionary, options for generating plot

        Will throw error if defined year range is not present in economic model outputs!

        """

        # set and check options / data ranges
        intervention_names_dict = {"vaccination": "Vaccination", "xpert": "GeneXpert", "xpertacf": "GeneXpert ACF", "smearacf": "Smear ACF",
                              "treatment_support": "Treatment Support", "ipt_age0to5": "IPT 0-5 y.o.", "ipt_age5to15": "IPT 5-15 y.o."}

        defaults = {
            "interventions": self.models['baseline'].interventions_to_cost,
            "x_label_rotation": 45,
            "y_label": "Total Cost ($)\n",
            "legend_size": 10,
            "legend_frame": False,
            "plot_style": "ggplot",
            "title": "Projected total costs {sy} - {ey}\n".format(sy=year_start, ey=year_end)
        }

        if plot_options is None:
            options = defaults
        else:
            for key, value in plot_options.items():
                defaults[key] = value
            options = defaults

        intervention_names = []
        for i in range(len(options['interventions'])):
            if options['interventions'][i] in intervention_names_dict.keys():
                intervention_names.append(intervention_names_dict[options['interventions'][i]])
            else:
                intervention_names.append(options['interventions'][i])

        if options["plot_style"] is not None:
            style.use(options["plot_style"])

        years = range(year_start, year_end + 1)

        # make data frame (columns: interventions, rows: scenarios)
        data_frame = pandas.DataFrame(index=self.scenarios, columns=intervention_names)

        for scenario in self.scenarios:
            data_frame.loc[scenario] = [sum([self.model_runner.epi_outputs_integer_dict[scenario]["cost_" + intervention][year]
                                             for year in years]) for intervention in options["interventions"]]

        data_frame.columns = intervention_names

        # make and style plot
        if horizontal:
            plot = data_frame.plot.barh(stacked=True, rot=options["x_label_rotation"], title=options["title"])
            plot.set_xlabel(options['y_label'])
        else:
            plot = data_frame.plot.bar(stacked=True, rot=options["x_label_rotation"], title=options["title"])
            plot.set_ylabel(options["y_label"])

        humanise_y_ticks(plot)

        handles, labels = plot.get_legend_handles_labels()
        lgd = plot.legend(handles, labels, bbox_to_anchor=(1, 0.5), loc='center left',
                          fontsize=options["legend_size"], frameon=options["legend_frame"])

        # save plot
        pyplot.savefig(os.path.join(self.out_dir_project, self.country + '_totalcost' + '.png'),
                       bbox_extra_artists=(lgd,), bbox_inches='tight')

    def plot_comorb_checks(self):

        """
        Plots actual comorbidity fractions against targets.

        """

        # Initial bits
        fig = self.set_and_update_figure()
        ax = self.make_single_axis(fig)

        # Plotting
        for comorb in self.model_runner.model_dict['baseline'].comorbidities:
            ax.plot(self.model_runner.model_dict['baseline'].times[2:],
                    self.model_runner.model_dict['baseline'].actual_comorb_props[comorb], 'g-')
            ax.plot(self.model_runner.model_dict['baseline'].times[1:],
                    self.model_runner.model_dict['baseline'].target_comorb_props[comorb], 'k--')
            ax.set_xlim([self.inputs.model_constants['recent_time'], self.inputs.model_constants['current_time']])

        # End bits
        fig.suptitle('Population by comorbidity', fontsize=self.suptitle_size)
        ax.set_xlabel('Year', fontsize=get_nice_font_size([1, 1]), labelpad=1)
        ax.set_ylabel('Proportion', fontsize=get_nice_font_size([1, 1]), labelpad=1)
        for axis_to_change in [ax.xaxis, ax.yaxis]:
            for tick in axis_to_change.get_major_ticks():
                tick.label.set_fontsize(get_nice_font_size([1, 1]))
        self.save_figure(fig, '_comorb_checks')

    def open_output_directory(self):

        """
        Opens the directory into which all the outputs have been placed

        """

        operating_system = platform.system()
        if 'Windows' in operating_system:
            os.system('start ' + ' ' + self.out_dir_project)
        elif 'Darwin' in operating_system:
            os.system('open ' + ' ' + self.out_dir_project)











