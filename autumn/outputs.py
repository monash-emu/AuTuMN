
import openpyxl as xl
import tool_kit as t_k
from docx import Document
from matplotlib import pyplot, patches, style, cm
import numpy
import pylab
import platform
import os
import warnings
import economics
import pandas
import copy
import model_runner
import scipy


def find_smallest_factors_of_integer(n):
    """
    Quick method to iterate through integers to find the smallest whole number fractions.
    Written only to be called by find_subplot_numbers.

    Args:
        n: Integer to be factorised
    Returns:
        answer: The two smallest factors of the integer
    """

    answer = [1e3, 1e3]
    for i in range(1, n + 1):
        if n % i == 0 and i+(n/i) < sum(answer):
            answer = [i, n/i]
    return answer


def humanise_y_ticks(ax):
    """
    Coded by Bosco, does a few things, including rounding axis values to thousands, millions or billions and
    abbreviating these to single letters.

    Args:
        ax: The adapted axis
    """

    vals = list(ax.get_yticks())
    max_val = max([abs(v) for v in vals])
    if max_val < 1e3:
        return
    if max_val >= 1e3 and max_val < 1e6:
        labels = ['%.1fK' % (v/1e3) for v in vals]
    elif max_val >= 1e6 and max_val < 1e9:
        labels = ['%.1fM' % (v/1e6) for v in vals]
    elif max_val >= 1e9:
        labels = ['%.1fB' % (v/1e9) for v in vals]
    is_fraction = False
    for label in labels:
        if label[-3:-1] != '.0':
            is_fraction = True
    if not is_fraction:
        labels = [l[:-3] + l[-1] for l in labels]
    ax.set_yticklabels(labels)


def make_single_axis(fig):
    """
    Create axes for a figure with a single plot with a reasonable amount of space around.

    Returns:
        ax: The axes that can be plotted on
    """

    return fig.add_axes([0.1, 0.1, 0.6, 0.75])


def make_axes_with_room_for_legend():
    """
    Create axes for a figure with a single plot with a reasonable amount of space around.

    Returns:
        The axes that can be plotted on
    """

    fig = pyplot.figure()
    return fig.add_axes([0.1, 0.1, 0.6, 0.75])


def set_axes_props(ax, xlabel=None, ylabel=None, title=None, is_legend=True, axis_labels=None, side='left'):

    frame_colour = "grey"

    # Hide top and right border of plot
    if side == 'left':
        ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position(side)
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
    """
    Simple function to return a reasonable font size as appropriate to the number of rows of subplots in the figure.
    """

    return 2. + 8. / subplot_grid[0]


def find_reasonable_year_ticks(start_time, end_time):
    """
    Function to find a reasonable spacing between years for x-ticks.

    Args:
        start_time: Float for left x-limit in years
        end_time: Float for right x-limit in years
    Returns:
        times: The times for the x ticks
    """

    duration = end_time - start_time
    if duration > 1e3:
        spacing = 1e2
    elif duration > 75.:
        spacing = 25.
    elif duration > 25.:
        spacing = 10.
    elif duration > 15.:
        spacing = 5.
    elif duration > 5.:
        spacing = 2.
    else:
        spacing = 1.

    times = []
    working_time = start_time
    while working_time <= end_time:
        times.append(working_time)
        working_time += spacing
    return times


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

    if 'incidence' in labels:
        colour += [(lightening_factor, lightening_factor, lightening_factor)]
        indices += ['e_inc_100k']
        yaxis_label += ['Per 100,000 per year']
        title += ['Incidence']
    if 'mortality' in labels:
        colour += [(1., lightening_factor, lightening_factor)]
        indices += ['e_mort_exc_tbhiv_100k']
        yaxis_label += ['Per 100,000 per year']
        title += ['Mortality']
    if 'prevalence' in labels:
        colour += [(lightening_factor, 0.5 + 0.5 * lightening_factor, lightening_factor)]
        indices += ['e_prev_100k']
        yaxis_label += ['Per 100,000']
        title += ['Prevalence']
    if 'notifications' in labels:
        colour += [(lightening_factor, lightening_factor, 0.5 + 0.5 * lightening_factor)]
        yaxis_label += ['']
        title += ['Notifications']
    if 'perc_incidence' in labels:
        colour += [(lightening_factor, lightening_factor, lightening_factor)]
        yaxis_label += ['Percentage']
        title += ['Proportion of incidence']

    # create a colour half-way between the line colour and white for patches
    for i in range(len(colour)):
        patch_colour += [[]]
        for j in range(len(colour[i])):
            patch_colour[i] += [1. - (1. - colour[i][j]) / 2.]

    return colour, indices, yaxis_label, title, patch_colour


def make_related_line_styles(labels, strain_or_organ):
    """
    Make line styles for compartments.
    Args:
        labels: List of compartment names
        strain_or_organ: Whether to make patterns refer to strain or organ status
    Returns:
        colours: Colours for plotting
        patterns: Line patterns for plotting
        compartment_full_names: Full names of compartments
        markers: Marker styles
    """

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


def make_default_line_styles(n, return_all=True):
        """
        Produces a standard set of line styles that isn't adapted to the data being plotted.

        Args:
            n: The number of line-styles
            return_all: Whether to return all of the styles up to n or just the last one
        Returns:
            line_styles: A list of standard line-styles, or if return_all is False, then the single style
        """

        # iterate through a standard set of line styles
        for i in range(n):
            line_styles = []
            for line in ['-', ':', '-.', '--']:
                for colour in 'krbgmcy':
                    line_styles.append(line + colour)

        styles_to_return = line_styles if return_all else line_styles[n - 1]
        return styles_to_return


def make_legend_to_single_axis(ax, scenario_handles, scenario_labels):
    """
    Standardised format to legend at side of single axis plot.

    Args:
        ax: The axis that needs a legend
        scenario_handles: The elements for the legend
        scenario_labels: List of strings to name the elements of the legend
    """

    ax.legend(scenario_handles,
              scenario_labels,
              bbox_to_anchor=(1.05, 1),
              loc=2,
              borderaxespad=0.,
              frameon=False,
              prop={'size': 7})


def get_line_style(label, strain_or_organ):

    """
    Get some colours and patterns for lines for compartments - called by make_related_line_styles only.

    Args:
        label: Compartment name
        strain_or_organ: Whether to change line pattern by strain or by organ
    """

    # Unassigned groups remain black
    colour = (0, 0, 0)
    if 'susceptible_immune' in label:  # susceptible_unvac remains black
        colour = (0.3, 0.3, 0.3)
    if 'latent' in label:  # latent_early remains as for latent
        colour = (0, 0.4, 0.8)
    if 'latent_late' in label:
        colour = (0, 0.2, 0.4)
    if 'active' in label:
        colour = (0.9, 0, 0)
    elif 'detect' in label:
        colour = (0, 0.5, 0)
    elif 'missed' in label:
        colour = (0.5, 0, 0.5)
    if 'treatment' in label:  # treatment_infect remains as for treatment
        colour = (1, 0.5, 0)
    if 'treatment_noninfect' in label:
        colour = (1, 1, 0)

    pattern = get_line_pattern(label, strain_or_organ)

    category_full_name = label
    if 'susceptible' in label:
        category_full_name = 'Susceptible'
    if 'susceptible_fully' in label:
        category_full_name = 'Fully susceptible'
    elif 'susceptible_immune' in label:
        category_full_name = 'BCG vaccinated, susceptible'
    if 'latent' in label:
        category_full_name = 'Latent'
    if 'latent_early' in label:
        category_full_name = 'Early latent'
    elif 'latent_late' in label:
        category_full_name = 'Late latent'
    if 'active' in label:
        category_full_name = 'Active, yet to present'
    elif 'detect' in label:
        category_full_name = 'Detected'
    elif 'missed' in label:
        category_full_name = 'Missed'
    if 'treatment' in label:
        category_full_name = 'Under treatment'
    if 'treatment_infect' in label:
        category_full_name = 'Infectious under treatment'
    elif 'treatment_noninfect' in label:
        category_full_name = 'Non-infectious under treatment'

    if 'smearpos' in label:
        category_full_name += ', \nsmear-positive'
    elif 'smearneg' in label:
        category_full_name += ', \nsmear-negative'
    elif 'extrapul' in label:
        category_full_name += ', \nextrapulmonary'

    if '_ds' in label:
        category_full_name += ', \nDS-TB'
    elif '_mdr' in label:
        category_full_name += ', \nMDR-TB'
    elif '_xdr' in label:
        category_full_name += ', \nXDR-TB'

    marker = ''

    return colour, pattern, category_full_name, marker


def get_line_pattern(label, strain_or_organ):

    """
    Get pattern for a compartment.

    Args:
        label: Compartment name
        strain_or_organ: Whether to change pattern by strain or by organ
    Returns:
        pattern: Line pattern for plotting
    """

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


def create_patch_from_list(x_list, lower_border, upper_border):
    """
    Creates an array that can be used to plot a patch using the add patch plotting function in matplotlib.

    Args:
        x_list: The x-values to go forward and backward with
        lower_border: The lower edge of the patch
        upper_border: The upper edge of the patch
    Returns:
        patch_array: An array for use in plotting patches
            (with length of double the length of the inputs lists and height of two)
    """

    assert len(x_list) == len(lower_border) == len(upper_border), \
        'Attempted to create patch out of lists of unequal length'
    patch_array = numpy.zeros(shape=(len(x_list) * 2, 2))
    for x_num, x_value in enumerate(x_list):
        patch_array[x_num][0] = x_value  # x_values going forwards
        patch_array[-(x_num + 1)][0] = x_value  # years going backwards
        patch_array[x_num][1] = lower_border[x_num]  # lower limit data going forwards
        patch_array[-(x_num + 1)][1] = upper_border[x_num]  # upper limit data going backwards
    return patch_array


def extract_dict_to_list_key_ordering(dictionary, key_string):
    """
    Create a dictionary with each element lists, one giving the "times" that the list element refers to and the others
    giving the data content that these times refer to - maintainting the order that the keys were originally in.

    Args:
        dictionary: The dictionary containing the data to be extracted (N.B. Assumption is that the keys refer to times)
        key_string: The key of interest within the dictionary
    Returns:
        extracted_lists: Dictionary containing the extracted lists with keys 'times' and "key_string"
    """

    extracted_lists = {}
    extracted_lists['times'] = sorted(dictionary.keys())
    extracted_lists[key_string] = []
    for time in extracted_lists['times']: extracted_lists[key_string].append(dictionary[time])
    return extracted_lists


def find_exponential_constants(times, y_values):
    """
    In order to find an exponential function that passes through the point (times[0], y_values[0])
    and (times[1], y_values[1]) and is of the form: y = exp(-a * (t - b)), where t is the independent variable.

    Args:
        times: List of the two time or x coordinates of the points to be fitted to
        y_values: List of the two outputs or y coordinates of the points to be fitted to
    Returns:
        a: Parameter for the horizontal transformation of the function
        b: Parameter for the horizontal translation of the function
    """

    b = (times[0] * numpy.log(y_values[1]) - times[1] * numpy.log(y_values[0])) \
        / (numpy.log(y_values[1]) - numpy.log(y_values[0]))
    a = - numpy.log(y_values[0]) / (times[0] - b)
    return a, b


def plot_endtb_targets(ax, output, base_value, plot_colour, annotate=True):
    """
    Plot the End TB Targets and the direction that we need to head to achieve them.

    Args:
        ax: The axis to be plotted on to
        o: Output number
        output: Output string
        gtb_data: GTB data values
        plot_colour: List of colours for plotting
    """

    # End TB Targets
    times = [2015., 2020., 2025., 2030., 2035.]
    target_text = ['', 'M', 'M', 'S', 'E']  # M for milestone, S for Sustainable Development Goal, E for End TB Target

    if output == 'mortality':

        # find targets
        target_props = [1., .65, .25, .1, .05]
        target_values = [base_value * t for t in target_props]

        # plot the individual targets themselves
        ax.plot(times[1:], target_values[1:],
                marker='o', markersize=4, color=plot_colour, markeredgewidth=0., linewidth=0.)

        # cycle through times and plot
        for t in range(len(times) - 1):
            times_to_plot, output_to_reach_target = find_times_from_exp_function(t, times, target_values)
            ax.plot(times_to_plot, output_to_reach_target, color=plot_colour, linewidth=.5)

    elif output == 'incidence':

        # find targets
        target_props = [1., .8, .5, .2, .1]
        target_values = [base_value * t for t in target_props]

        # plot the individual targets themselves
        ax.plot(times[1:], target_values[1:],
                marker='o', markersize=4, color=plot_colour, markeredgewidth=0., linewidth=0.)

        # cycle through times and plot
        for t in range(len(times) - 1):
            times_to_plot, output_to_reach_target = find_times_from_exp_function(t, times, target_values)
            ax.plot(times_to_plot, output_to_reach_target, color=plot_colour, linewidth=.5)

    # annotate points with letters if requested
    if annotate:
        for i, text in enumerate(target_props):
            ax.annotate(target_text[i], (times[i], target_values[i] + target_values[1] / 20.),
                        horizontalalignment='center', verticalalignment='bottom', fontsize=8, color=plot_colour)


def find_times_from_exp_function(t, times, target_values, number_x_values=1e2):
    """
    Find the times to plot and the outputs tracking towards the targets from the list of times and target values,
    using the function to fit exponential functions.

    Args:
        t: The sequence number for the time point
        times: The list of times being worked through
        target_values: The list fo target values corresponding to the times
    Returns:
        times_to_plot: List of the x-values or times for plotting
        outputs_to_reach_target: Corresponding list of values for the output needed to track towards the target
    """

    a, b = find_exponential_constants([times[t], times[t + 1]], [target_values[t], target_values[t + 1]])
    times_to_plot = numpy.linspace(times[t], times[t + 1], number_x_values)
    output_to_reach_target = [numpy.exp(-a * (x - b)) for x in times_to_plot]
    return times_to_plot, output_to_reach_target


def save_png(png):
    # should be redundant once Project module complete

    if png is not None: pylab.savefig(png, dpi=300)


def plot_comparative_age_parameters(data_strat_list, data_value_list, model_value_list, model_strat_list,
                                    parameter_name):

    # get good tick labels from the stratum lists
    data_strat_labels = []
    for i in range(len(data_strat_list)):
        data_strat_labels += [t_k.turn_strat_into_label(data_strat_list[i])]
    model_strat_labels = []
    for i in range(len(model_strat_list)):
        model_strat_labels += [t_k.turn_strat_into_label(model_strat_list[i])]

    # find a reasonable upper limit for the y-axis
    ymax = max(data_value_list + model_value_list) * 1.2

    # plot original data bar charts
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

    # plot adjusted parameters bar charts
    ax = fig.add_axes([0.55, 0.2, 0.35, 0.6])
    x_positions = range(len(model_strat_list))
    ax.bar(x_positions, model_value_list, width)
    ax.set_title('Model implementation', fontsize=12)
    ax.set_xticklabels(model_strat_labels, rotation=45)
    ax.set_xticks(x_positions)
    ax.set_ylim(0., ymax)
    ax.set_xlim(-1. + width, x_positions[-1] + 1)

    # overall title
    fig.suptitle(t_k.capitalise_and_remove_underscore(parameter_name)
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
            n += 1
            answer = find_smallest_factors_of_integer(n)
        i += 1
    return answer


def get_string_for_funding(funding):

    """
    Returns an easy to read string corresponding to a level of funding (ex 9,000,000 becomes $9M)
    """

    s = '$'
    if funding >= 1e9:
        letter = 'B'
        factor = 9
    elif funding >= 1e6:
        letter = 'M'
        factor = 6
    elif funding >=1e3:
        letter = 'K'
        factor = 3
    else:
        letter = ''
        factor = 0
    number = funding / (10**factor)
    s += str(number) + letter
    return s


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

    y_number_format = '%.' + str(y_sig_figs) + 'f'
    y_number_format_around_one = '%.' + str(max(2, y_sig_figs)) + 'f'
    if max_val < 5e-9:
        labels = [y_number_format % (v * 1e12) for v in vals]
        axis_modifier = 'Trillionth '
    elif max_val < 5e-6:
        labels = [y_number_format % (v * 1e9) for v in vals]
        axis_modifier = 'Billionth '
    elif max_val < 5e-3:
        labels = [y_number_format % (v * 1e6) for v in vals]
        axis_modifier = 'Millionth '
    elif max_val < 5e-2:
        labels = [y_number_format % (v * 1e3) for v in vals]
        axis_modifier = 'Thousandth '
    elif max_val < .1:
        labels = [y_number_format % (v * 1e2) for v in vals]
        axis_modifier = 'Hundredth '
    elif max_val < 5:
        labels = [y_number_format_around_one % v for v in vals]
        axis_modifier = ''
    elif max_val < 5e3:
        labels = [y_number_format % v for v in vals]
        axis_modifier = ''
    elif max_val < 5e6:
        labels = [y_number_format % (v / 1e3) for v in vals]
        axis_modifier = 'Thousand '
    elif max_val < 5e9:
        labels = [y_number_format % (v / 1e6) for v in vals]
        axis_modifier = 'Million '
    else:
        labels = [y_number_format % (v / 1e9) for v in vals]
        axis_modifier = 'Billion '
    return labels, axis_modifier


def write_param_to_sheet(country_sheet, working_list, median_run_index):
    """
    Function to write a single parameter value into a cell of the input spreadsheets - to be used as part of
    automatic calibration.

    Args:
        country_sheet: Spreadsheet object
        working_list: List to take value from
        median_run_index: Integer index of the median
    """

    for param in working_list:

        if working_list[param]:

            # find value to write from list and index
            value = working_list[param][median_run_index]

            # over-write existing parameter value if present
            param_found = False
            for row in country_sheet.rows:
                if row[0].value == param: row[1].value, param_found = value, True

            # if parameter not found in existing spreadsheet, write into new row at the bottom
            if not param_found:
                max_row = country_sheet.max_row
                country_sheet.cell(row=max_row + 1, column=1).value = param
                country_sheet.cell(row=max_row + 1, column=2).value = value


class Project:
    def __init__(self, runner, gui_inputs):
        """
        Initialises an object of class Project, that will contain all the information (data + outputs) for writing a
        report for a country.

        Args:
            runner: The main model runner object used to execute all the analyses
            gui_inputs: All inputs from the graphical user interface
        """

        self.model_runner = runner
        self.gui_inputs = gui_inputs

        (self.inputs, self.run_mode) \
            = [None for _ in range(2)]
        (self.output_colours, self.uncertainty_output_colours, self.program_colours, self.classified_scaleups,
         self.outputs) \
            = [{} for _ in range(5)]
        (self.grid, self.plot_rejected_runs, self.plot_true_outcomes) \
            = [False for _ in range(3)]
        (self.accepted_no_burn_in_indices, self.scenarios, self.interventions_to_cost, self.accepted_indices) \
            = [[] for _ in range(4)]
        self.uncertainty_centiles = {'epi': {}, 'cost': {}}
        for attribute in ['inputs', 'outputs']:
            setattr(self, attribute, getattr(self.model_runner, attribute))
        for attribute in ['scenarios', 'interventions_to_cost', 'run_mode']:
            setattr(self, attribute, getattr(self.model_runner.inputs, attribute))
        self.figure_number, self.title_size = 1, 13
        self.country = self.gui_inputs['country'].lower()
        self.out_dir_project = os.path.join('projects', 'test_' + self.country)
        self.figure_formats = ['png']   # allow for multiple formats. e.g. ['png', 'pdf']
        if not os.path.isdir(self.out_dir_project):
            os.makedirs(self.out_dir_project)
        self.years_to_write \
            = range(int(self.inputs.model_constants['report_start_time']),
                    int(self.inputs.model_constants['report_end_time']),
                    int(self.inputs.model_constants['report_step_time']))
        self.classifications \
            = ['demo_', 'econ_', 'epi_prop_smear', 'program_prop_', 'program_timeperiod_',
               'program_prop_novel', 'program_prop_treatment', 'program_prop_detect',
               'int_prop_vaccination', 'program_prop_treatment_success',
               'program_prop_treatment_death', 'transmission_modifier', 'algorithm']
        self.quantities_to_write_back = ['all_parameters', 'all_compartment_values', 'adjustments']
        self.gtb_available_outputs = ['incidence', 'mortality', 'prevalence', 'notifications']
        self.level_conversion_dict = {'lower_limit': '_lo', 'upper_limit': '_hi', 'point_estimate': ''}

        # to have a look at some individual vars scaling over time
        self.vars_to_view = ['riskgroup_prop_diabetes']

        # comes up so often that we need to find this index, that easiest to do in instantiation
        self.start_time_index \
            = t_k.find_first_list_element_at_least_value(self.outputs['manual']['epi'][0]['times'],
                                                         self.inputs.model_constants['plot_start_time'])

    ''' master method to call the others '''

    def master_outputs_runner(self):
        """
        Method to work through all the fundamental output methods, which then call all the specific output
        methods for plotting and writing as required.
        """

        self.model_runner.add_comment_to_gui_window('Creating outputs')

        # processing methods that are only required for outputs
        if self.run_mode == 'epi_uncertainty':
            self.find_uncertainty_indices()
            for output_type in ['epi', 'cost']:
                self.uncertainty_centiles[output_type] = self.find_uncertainty_centiles('epi_uncertainty', output_type)
        elif self.run_mode == 'int_uncertainty':
            for output_type in ['epi', 'cost']:
                self.uncertainty_centiles[output_type] = self.find_uncertainty_centiles('int_uncertainty', output_type)

        # write automatic calibration values back to sheets
        if self.run_mode == 'epi_uncertainty' and self.gui_inputs['write_uncertainty_outcome_params']:
            self.write_automatic_calibration_outputs()

        # write spreadsheets with sheet for each scenario or each output
        if self.gui_inputs['output_spreadsheets']:
            self.model_runner.add_comment_to_gui_window('Writing output spreadsheets')
            if self.gui_inputs['output_by_scenario']:
                self.write_xls_by_scenario()
            else:
                self.write_xls_by_output()

        # write documents - with document for each scenario or each output
        if self.gui_inputs['output_documents']:
            self.model_runner.add_comment_to_gui_window('Writing output documents')
            if self.gui_inputs['output_by_scenario']:
                self.write_docs_by_scenario()
                if self.run_mode == 'int_uncertainty':
                    self.print_int_uncertainty_relative_change(year=2035.)
            else:
                self.write_docs_by_output()

        # master plotting method
        self.model_runner.add_comment_to_gui_window('Creating plot figures')
        self.run_plotting()

        # open the directory to which everything has been written to save the user a click or two
        self.open_output_directory()

    ''' general methods for use by specific methods below '''

    def find_var_index(self, var):
        """
        Finds the index number for a var in the var arrays. (Arbitrarily uses the baseline model from the model runner.)

        Args:
            var: String for the var that we're looking for
        Returns:
            The var's index
        """

        return self.model_runner.models[0].var_labels.index(var)

    def find_start_index(self, scenario=0):
        """
        Very simple, but commonly used bit of code to determine whether to start from the start of the epidemiological
        outputs, or - if we're dealing with the baseline scenario - to start from the appropriate time index.

        Args:
            scenario: Scenario number (i.e. zero for baseline or single integer for scenario)
        Returns:
            Index that can be used to find starting point in epidemiological output lists
        """

        index = 0 if scenario else self.start_time_index
        return index

    def set_and_update_figure(self):
        """
        If called at the start of each plotting function, will create a figure that is numbered according to
        self.figure_number, which is then updated at each call. This stops figures plotting on the same axis
        and saves you having to worry about how many figures are being opened.
        """

        fig = pyplot.figure(self.figure_number)
        self.figure_number += 1
        return fig

    def tidy_axis(self, ax, subplot_grid, title='', start_time=0., legend=False, x_label='', y_label='',
                  x_axis_type='time', y_axis_type='scaled', x_sig_figs=0, y_sig_figs=0,
                  end_time=None, y_relative_limit=0.95, y_absolute_limit=None):
        """
        Method to make cosmetic changes to a set of plot axes.
        """

        # add the sub-plot title with slightly larger titles than the rest of the text on the panel
        if title:
            ax.set_title(title, fontsize=get_nice_font_size(subplot_grid) + 2.)

        # default end time for plots to end at
        if not end_time:
            end_time = self.inputs.model_constants['plot_end_time']

        # add a legend if needed
        if legend == 'for_single':
            ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., frameon=False, prop={'size': 7})
        elif legend:
            ax.legend(fontsize=get_nice_font_size(subplot_grid), frameon=False, loc=3)

        # sort x-axis
        if x_axis_type == 'time':
            ax.set_xlim((start_time, end_time))
            ax.set_xticks(find_reasonable_year_ticks(start_time, end_time))
        elif x_axis_type == 'scaled':
            vals = list(ax.get_xticks())
            max_val = max([abs(v) for v in vals])
            labels, axis_modifier = scale_axes(vals, max_val, x_sig_figs)
            ax.set_xticklabels(labels)
            ax.set_xlabel(x_label + axis_modifier, fontsize=get_nice_font_size(subplot_grid), labelpad=1)
        elif x_axis_type == 'proportion':
            ax.set_xlabel(x_label, fontsize=get_nice_font_size(subplot_grid), labelpad=1)
            ax.set_xlim((0., 1.))
        elif x_axis_type == 'individual_years':
            ax.set_xlim((start_time, end_time))
            ax.set_xticks(range(int(start_time), int(end_time, 1)))
            for tick in ax.xaxis.get_major_ticks(): tick.label.set_rotation(45)
        else:
            ax.set_xlabel(x_label, fontsize=get_nice_font_size(subplot_grid), labelpad=1)

        # sort y-axis
        ax.set_ylim(bottom=0.)
        vals = list(ax.get_yticks())
        max_val = max([abs(v) for v in vals])
        if y_axis_type == 'time':
            ax.set_ylim((start_time, end_time))
            ax.set_yticks(find_reasonable_year_ticks(start_time, end_time))
        elif y_axis_type == 'scaled':
            labels, axis_modifier = scale_axes(vals, max_val, y_sig_figs)
            ax.set_yticklabels(labels)
            ax.set_ylabel(axis_modifier + y_label, fontsize=get_nice_font_size(subplot_grid), labelpad=1)
        elif y_axis_type == 'proportion':
            ax.set_ylim((0., 1.))
            ax.set_ylabel(y_label, fontsize=get_nice_font_size(subplot_grid), labelpad=1)
        elif y_axis_type == 'limited_proportion':
            ax.set_ylim((0., 1.))
            ax.set_ylabel(y_label, fontsize=get_nice_font_size(subplot_grid), labelpad=1)
        elif not y_absolute_limit:
            ax.set_ylim(top=max_val * y_relative_limit)
            ax.set_ylabel(y_label, fontsize=get_nice_font_size(subplot_grid), labelpad=1)
        else:
            ax.set_ylim(top=y_absolute_limit)
            ax.set_ylabel(y_label, fontsize=get_nice_font_size(subplot_grid), labelpad=1)

        # set size of font for x-ticks and add a grid if requested
        for axis_to_change in [ax.xaxis, ax.yaxis]:
            for tick in axis_to_change.get_major_ticks(): tick.label.set_fontsize(get_nice_font_size(subplot_grid))
            axis_to_change.grid(self.grid)

    def save_figure(self, fig, end_figure_name):
        """
        Simple method to standardise names for output figure files.

        Args:
            end_figure_name: The part of the figure name that is variable and is input by the plotting method
            fig: Figure for saving
        """

        for file_format in self.figure_formats:
            filename = os.path.join(self.out_dir_project, self.country + end_figure_name + '.' + file_format)
            fig.savefig(filename, dpi=300)

    ''' methods for pre-processing model runner outputs to more interpretable forms '''

    def find_uncertainty_indices(self):
        """
        Quick method to create a list of the indices of interest for the runs of the uncertainty analysis.

        Updates:
            self.accepted_no_burn_in_indices: List of the uncertainty indices of interest
        """

        self.accepted_indices = self.outputs['epi_uncertainty']['accepted_indices']
        self.accepted_no_burn_in_indices = [i for i in self.accepted_indices if i >= self.gui_inputs['burn_in_runs']]

    def find_uncertainty_centiles(self, mode, output_type):
        """
        Find percentiles from uncertainty dictionaries.

        Args:
            mode: The run mode being considered
            output_type: Whether the output to be calculated is 'epi' or 'cost'
        Updates:
            self.percentiles: Adds all the required percentiles to this dictionary.
        """

        uncertainty_centiles = {}
        for scenario in self.outputs[mode][output_type]:
            uncertainty_centiles[scenario] = {}
            for output in self.outputs[mode][output_type][scenario]:
                if output != 'times':

                    # use all runs for scenario analysis (as only those that were accepted are saved)
                    if scenario:
                        matrix_to_analyse = self.outputs[mode][output_type][scenario][output]

                    # select the baseline runs for analysis from the broader set of saved results
                    else:
                        matrix_to_analyse \
                            = self.outputs[mode][output_type][scenario][output][self.accepted_no_burn_in_indices, :]

                    uncertainty_centiles[scenario][output] \
                        = numpy.percentile(matrix_to_analyse, self.model_runner.percentiles, axis=0)
        return uncertainty_centiles

    ''' methods for outputting to documents and spreadsheets and console '''

    def write_automatic_calibration_outputs(self):
        """
        Write values from automatic calibration process back to input spreadsheets, using the parameter values
        that were associated with the model run with the greatest likelihood.
        """

        try:
            path = os.path.join('autumn/xls/data_' + self.country + '.xlsx')
        except:
            self.model_runner.add_comment_to_gui_window(
                'No country input spreadsheet available for requested uncertainty parameter writing')
        else:
            self.model_runner.add_comment_to_gui_window(
                'Writing automatic calibration parameters back to input spreadsheet')

            # open workbook and sheet
            country_input_book = xl.load_workbook(path)
            country_sheet = country_input_book['constants']

            # find the integration run with the highest likelihood
            best_likelihood_index = self.outputs['epi_uncertainty']['loglikelihoods'].index(
                max(self.outputs['epi_uncertainty']['loglikelihoods']))

            # write the parameters and starting compartment sizes back in to input sheets
            for attribute in self.quantities_to_write_back:
                write_param_to_sheet(country_sheet, self.outputs['epi_uncertainty'][attribute], best_likelihood_index)

            # save
            country_input_book.save(path)

    def write_xls_by_scenario(self):
        """
        Write a spreadsheet with the sheet referring to one scenario.
        """

        # general prelims to work out what to write
        horizontal = self.gui_inputs['output_horizontally']
        result_types = ['epi_', 'raw_cost_', 'inflated_cost_', 'discounted_cost_', 'discounted_inflated_cost_']
        scenarios = [15] if self.run_mode == 'int_uncertainty' else self.scenarios

        # write a new file for each scenario and for each broad category of output
        for result_type in result_types:
            for scenario in scenarios:

                # prepare sheet
                scenario_name = t_k.find_scenario_string_from_number(scenario)
                path = os.path.join(self.out_dir_project, result_type + scenario_name) + '.xlsx'
                workbook = xl.Workbook()
                sheet = workbook.active
                sheet.title = scenario_name
                sheet.cell(row=1, column=1).value = 'Year'

                # year column
                for y, year in enumerate(self.years_to_write):
                    (row, column) = [1, y + 2] if horizontal else [y + 2, 1]
                    sheet.cell(row=row, column=column).value = year

                # epi outputs
                if result_type == 'epi_':

                    # loop over outputs
                    for out, output in enumerate(self.model_runner.epi_outputs_to_analyse):

                        # with uncertainty
                        if self.run_mode == 'epi_uncertainty' or self.run_mode == 'int_uncertainty':

                            # scenario names and confidence interval titles
                            strings_to_write = [t_k.capitalise_and_remove_underscore(output), 'Lower', 'Upper']

                            for ci in range(len(strings_to_write)):
                                (row, column) = [out * 3 + 2 + ci, 1] if horizontal else [1, out * 3 + 2 + ci]
                                sheet.cell(row=row, column=column).value = strings_to_write[ci]

                            # data columns
                            for y, year in enumerate(self.years_to_write):
                                for o in range(3):
                                    (row, column) = [out * 3 + 2 + o, y + 2] if horizontal else [y + 2, out * 3 + 2 + o]
                                    sheet.cell(row=row, column=column).value \
                                        = self.uncertainty_centiles['epi'][scenario][output][
                                        o, t_k.find_first_list_element_at_least_value(
                                            self.model_runner.outputs['manual']['epi'][scenario]['times'], year)]

                        # without uncertainty
                        else:

                            # names across top
                            (row, column) = [out + 2, 1] if horizontal else [1, out + 2]
                            sheet.cell(row=row, column=column).value = t_k.capitalise_and_remove_underscore(output)

                            # columns of data
                            for y, year in enumerate(self.years_to_write):
                                (row, column) = [out + 2, y + 2] if horizontal else [y + 2, out + 2]
                                sheet.cell(row=row, column=column).value \
                                    = self.outputs['manual']['epi'][scenario][output][
                                        t_k.find_first_list_element_at_least_value(
                                            self.outputs['manual']['epi'][scenario]['times'], year)]

                # economic outputs (uncertainty unavailable)
                elif 'cost_' in result_type:

                    # loop over interventions
                    for inter, intervention in enumerate(self.inputs.interventions_to_cost[scenario]):

                        # names across top
                        (row, column) = [inter + 2, 1] if horizontal else [1, inter + 2]
                        sheet.cell(row=row, column=column).value = t_k.capitalise_and_remove_underscore(intervention)

                        # data columns
                        for y, year in enumerate(self.years_to_write):
                            (row, column) = [inter + 2, y + 2] if horizontal else [y + 2, inter + 2]
                            sheet.cell(row=row, column=column).value \
                                = self.outputs['manual']['cost'][scenario][result_type + intervention][
                                        t_k.find_first_list_element_at_least_value(
                                            self.outputs['manual']['cost'][scenario]['times'], year)]
                workbook.save(path)

    def write_xls_by_output(self):
        """
        Write a spreadsheet with the sheet referring to one output.
        """

        # general prelims to work out what to write
        horizontal = self.gui_inputs['output_horizontally']
        scenarios = [15] if self.run_mode == 'int_uncertainty' else self.scenarios

        # write a new file for each output
        for inter in self.model_runner.epi_outputs_to_analyse:

            # prepare sheet
            path = os.path.join(self.out_dir_project, 'epi_' + inter) + '.xlsx'
            workbook = xl.Workbook()
            sheet = workbook.active
            sheet.title = inter
            sheet.cell(row=1, column=1).value = 'Year'

            # write the year column
            for y, year in enumerate(self.years_to_write):
                (row, column) = [1, y + 2] if horizontal else [y + 2, 1]
                sheet.cell(row=row, column=column).value = year

            # cycle over scenarios
            for s, scenario in enumerate(scenarios):
                scenario_name = t_k.find_scenario_string_from_number(scenario)

                # with uncertainty
                if self.run_mode == 'epi_uncertainty' or self.run_mode == 'int_uncertainty':

                    # scenario names and confidence interval titles
                    strings_to_write = [t_k.capitalise_and_remove_underscore(scenario_name), 'Lower', 'Upper']

                    # write the scenario names and confidence interval titles
                    for ci in range(len(strings_to_write)):
                        (row, column) = [s * 3 + 2 + ci, 1] if horizontal else [1, s * 3 + 2 + ci]
                        sheet.cell(row=row, column=column).value = strings_to_write[ci]

                    # write the columns of data
                    for y, year in enumerate(self.years_to_write):
                        for o in range(3):
                            (row, column) = [s * 3 + 2 + o, y + 2] if horizontal else [y + 2, s * 3 + 2 + o]
                            sheet.cell(row=row, column=column).value \
                                = self.uncertainty_centiles['epi'][scenario][inter][
                                o, t_k.find_first_list_element_at_least_value(
                                    self.model_runner.outputs['manual']['epi'][scenario]['times'], year)]

                # without uncertainty
                else:

                    # write scenario names across first row
                    (row, column) = [s + 2, 1] if horizontal else [1, s + 2]
                    sheet.cell(row=row, column=column).value = t_k.capitalise_and_remove_underscore(scenario_name)

                    # write columns of data
                    for y, year in enumerate(self.years_to_write):
                        (row, column) = [s + 2, y + 2] if horizontal else [y + 2, s + 2]
                        sheet.cell(row=row, column=column).value \
                            = self.model_runner.outputs['manual']['epi'][scenario][inter][
                                t_k.find_first_list_element_at_least_value(self.model_runner.outputs['manual']['epi'][
                                                                               scenario]['times'], year)]
            workbook.save(path)

        # code probably could bug because interventions can differ by scenario
        for inter in self.inputs.interventions_to_cost[0]:
            for cost_type in ['raw_cost_', 'inflated_cost_', 'discounted_cost_', 'discounted_inflated_cost_']:

                # make filename
                path = os.path.join(self.out_dir_project, cost_type + inter) + '.xlsx'

                # get active sheet
                workbook = xl.Workbook()
                sheet = workbook.active
                sheet.title = inter

                # write the year text cell
                sheet.cell(row=1, column=1).value = 'Year'

                # write the year text column
                for y, year in enumerate(self.years_to_write):
                    (row, column) = [1, y + 2] if horizontal else [y + 2, 1]
                    sheet.cell(row=row, column=column).value = year

                # cycle over scenarios
                for s, scenario in enumerate(scenarios):
                    scenario_name = t_k.find_scenario_string_from_number(scenario)

                    # scenario names
                    (row, column) = [s + 2, 1] if horizontal else [1, s + 2]
                    sheet.cell(row=row, column=column).value = t_k.capitalise_and_remove_underscore(scenario_name)

                    # data columns
                    for y, year in enumerate(self.years_to_write):
                        (row, column) = [s + 2, y + 2] if horizontal else [y + 2, s + 2]
                        sheet.cell(row=row, column=column).value \
                            = self.outputs['manual']['cost'][scenario][cost_type + inter][
                                t_k.find_first_list_element_at_least_value(self.outputs['manual']['cost'][scenario][
                                                                               'times'], year)]
                workbook.save(path)

    def write_opti_outputs_spreadsheet(self):

        # prelims
        path = os.path.join(self.model_runner.opti_outputs_dir, 'opti_results.xlsx')
        wb = xl.Workbook()
        sheet = wb.active
        sheet.title = 'optimisation'

        # write row names
        row_names = ['envelope', 'incidence', 'mortality']
        for row, name in enumerate(row_names):
            sheet.cell(row=row + 1, column=1).value = name
        row_index = {}
        for i, intervention in enumerate(self.model_runner.interventions_considered_for_opti):
            sheet.cell(row=i + 4, column=1).value = intervention
            row_index[intervention] = i + 4

        # populate cells with content
        for env, envelope in enumerate(self.model_runner.opti_results['annual_envelope']):
            sheet.cell(row=1, column=env + 1).value = envelope
            sheet.cell(row=2, column=env + 1).value = self.model_runner.opti_results['incidence'][env]
            sheet.cell(row=3, column=env + 1).value = self.model_runner.opti_results['mortality'][env]
            for intervention in self.model_runner.opti_results['best_allocation'][env].keys():
                sheet.cell(row=row_index[intervention], column=env + 1).value = \
                    self.model_runner.opti_results['best_allocation'][env][intervention]

        # save workbook
        wb.save(path)

    def write_docs_by_scenario(self):
        """
        Write word documents using the docx package. Writes with or without uncertainty according to whether Run
        uncertainty selected in the GUI. Currently only working for epidemiological outputs.
        """

        scenarios = [15] if self.run_mode == 'int_uncertainty' else self.scenarios
        for scenario in scenarios:

            # initialise document and table
            scenario_name = t_k.find_scenario_string_from_number(scenario)
            path = os.path.join(self.out_dir_project, scenario_name) + ".docx"
            document = Document()
            table = document.add_table(rows=len(self.years_to_write) + 1,
                                       cols=len(self.model_runner.epi_outputs_to_analyse) + 1)

            # for each epidemiological indicator
            for o, output in enumerate(self.model_runner.epi_outputs_to_analyse):

                # titles across the top
                row_cells = table.rows[0].cells
                row_cells[0].text = 'Year'
                row_cells[o + 1].text = t_k.capitalise_and_remove_underscore(output)

                # data columns
                for y, year in enumerate(self.years_to_write):

                    # write year column
                    row_cells = table.rows[y + 1].cells
                    row_cells[0].text = str(year)

                    # with uncertainty
                    if 'uncertainty' in self.run_mode:
                        point_lower_upper \
                            = tuple(self.uncertainty_centiles['epi'][scenario][output][
                                    0:3, t_k.find_first_list_element_at_least_value(
                                        self.model_runner.outputs['manual']['epi'][scenario]['times'], year)])
                        row_cells[o + 1].text = '%.1f\n(%.1f to %.1f)' % point_lower_upper

                    # without
                    else:
                        point = self.model_runner.outputs['manual']['epi'][scenario][output][
                            t_k.find_first_list_element_at_least_value(
                                self.model_runner.outputs['manual']['epi'][scenario]['times'], year)]
                        row_cells[o + 1].text = '%.1f' % point
            document.save(path)

    def write_docs_by_output(self):
        """
        Write word documents using the docx package. Writes with or without uncertainty according to whether Run
        uncertainty selected in the GUI.
        """

        # write a new file for each output
        scenarios = [15] if self.run_mode == 'int_uncertainty' else self.scenarios
        for output in self.model_runner.epi_outputs_to_analyse:

            # initialise document, years of interest and table
            path = os.path.join(self.out_dir_project, output) + ".docx"
            document = Document()
            table = document.add_table(rows=len(self.years_to_write) + 1, cols=len(self.scenarios) + 1)

            for s, scenario in enumerate(scenarios):
                scenario_name = t_k.find_scenario_string_from_number(scenario)

                # outputs across the top
                row_cells = table.rows[0].cells
                row_cells[0].text = 'Year'
                row_cells[s + 1].text = t_k.capitalise_and_remove_underscore(scenario_name)

                for y, year in enumerate(self.years_to_write):
                    row_cells = table.rows[y + 1].cells
                    row_cells[0].text = str(year)

                    # with uncertainty
                    if 'uncertainty' in self.run_mode:
                        point_lower_upper \
                            = tuple(self.uncertainty_centiles['epi'][scenario][output][0:3,
                                    t_k.find_first_list_element_at_least_value(self.model_runner.outputs['manual'][
                                                                                   'epi'][scenario]['times'], year)])
                        row_cells[s + 1].text = '%.1f\n(%.1f to %.1f)' % point_lower_upper

                    # without
                    else:
                        point = self.model_runner.outputs['manual']['epi'][scenario][output][
                            t_k.find_first_list_element_at_least_value(self.model_runner.outputs['manual'][
                                                                           'epi'][scenario]['times'], year)]
                        row_cells[s + 1].text = '%.1f' % point
            document.save(path)

    def print_int_uncertainty_relative_change(self, year):
        """
        Print some text giving percentage change in output indicators relative to baseline under intervention
        uncertainty.

        Args:
            year: Year for the comparisons to be made against
        """

        scenario, changes = 15, {}
        for output in self.model_runner.epi_outputs_to_analyse:
            absolute_values \
                = self.uncertainty_centiles['epi'][scenario][output][0:3, t_k.find_first_list_element_at_least_value(
                  self.model_runner.outputs['int_uncertainty']['epi'][scenario]['times'][0], year)]
            baseline = self.model_runner.outputs['manual']['epi'][0][output][
               t_k.find_first_list_element_at_least_value(self.model_runner.outputs['manual']['epi'][0]['times'], year)]
            changes[output] = [(i / baseline - 1.) * 1e2 for i in absolute_values]
            print(output + '\n%.1f\n(%.1f to %.1f)' % tuple(changes[output]))

    def print_average_costs(self):
        """
        Incompletely developed method to display the mean cost of an intervention over the course of its implementation
        under baseline conditions.
        """

        for scenario in self.scenarios:
            print('\n' + t_k.find_scenario_string_from_number(scenario))
            mean_cost = {}
            for inter in self.model_runner.interventions_to_cost[scenario]:
                print('\n' + inter)
                mean_cost[inter] \
                    = numpy.mean(self.model_runner.outputs['manual']['cost'][scenario]['raw_cost_' + inter])
                print('%.1f' % mean_cost[inter])
            print('total: %.1f' % sum(mean_cost.values()))

    ''' plotting methods '''

    def run_plotting(self):
        """
        Master plotting method to call all the methods that produce specific plots.
        """

        # find some general output colours
        output_colours = make_default_line_styles(5, True)
        for s, scenario in enumerate(self.scenarios):
            self.output_colours[scenario] = output_colours[s]
            self.program_colours[scenario] = {}
            for p, program in enumerate(self.interventions_to_cost[scenario]):
                # +1 is to avoid starting from black, which doesn't look as nice for programs as for baseline scenario
                self.program_colours[scenario][program] = output_colours[p + 1]

        # plot main outputs
        if self.gui_inputs['output_gtb_plots']:
            purposes = ['scenario']
            if '_uncertainty' in self.run_mode:
                purposes.extend(['ci_plot', 'progress', 'shaded'])
            for purpose in purposes:
                self.plot_outputs_against_gtb(self.gtb_available_outputs, purpose=purpose)
            if self.inputs.n_strains > 1:
                self.plot_resistant_strain_outputs(['incidence', 'mortality', 'prevalence', 'perc_incidence'])

        # plot scale-up functions - currently only doing this for the baseline model run
        if self.gui_inputs['output_scaleups']:
            if self.vars_to_view:
                self.individual_var_viewer()
            self.classify_scaleups()
            self.plot_scaleup_fns_against_data()
            self.plot_individual_scaleups_against_data()
            self.plot_programmatic_scaleups()

            # not technically a scale-up function in the same sense, but put in here anyway
            # self.plot_force_infection()

        # plot mixing matrix if relevant
        if self.inputs.is_vary_force_infection_by_riskgroup and len(self.inputs.riskgroups) > 1:
            self.plot_mixing_matrix()

        # plot economic outputs
        if self.gui_inputs['output_plot_economics']:
            self.plot_cost_coverage_curves()
            self.plot_cost_over_time()
            # self.plot_intervention_costs_by_scenario(2015, 2030)
            # self.plot_cost_over_time_stacked_bars()

        # plot compartment population sizes
        if self.gui_inputs['output_compartment_populations']: self.plot_populations()

        # plot fractions
        # if self.gui_inputs['output_fractions']: self.plot_fractions('strain')

        # plot outputs by age group
        if self.gui_inputs['output_by_subgroups']:
            self.plot_outputs_by_stratum()
            self.plot_outputs_by_stratum(strata_string='riskgroups', outputs_to_plot=['incidence', 'prevalence'])
            self.plot_proportion_cases_by_stratum()

        # plot proportions of population
        if self.gui_inputs['output_age_fractions']: self.plot_stratified_populations(age_or_risk='age')

        # plot risk group proportions
        if self.gui_inputs['output_riskgroup_fractions']: self.plot_stratified_populations(age_or_risk='risk')

        # make a flow-diagram
        if self.gui_inputs['output_flow_diagram']:
            self.model_runner.models[0].make_flow_diagram(
                os.path.join(self.out_dir_project, self.country + '_flow_diagram' + '.png'))

        # plot risk group proportions
        if self.gui_inputs['output_plot_riskgroup_checks'] and len(self.model_runner.models[0].riskgroups) > 1:
            self.plot_riskgroup_checks()

        # save figure that is produced in the uncertainty running process
        if self.run_mode == 'epi_uncertainty' and self.gui_inputs['output_param_plots']:
            self.plot_param_histograms()
            self.plot_param_timeseries()
            self.plot_priors()

        # plot popsizes for checking cost-coverage curves
        if self.gui_inputs['output_popsize_plot']:
            self.plot_popsizes()

        # plot likelihood estimates
        if self.run_mode == 'epi_uncertainty' and self.gui_inputs['output_likelihood_plot']:
            self.plot_likelihoods()

        # plot percentage of MDR for different uncertainty runs
        if self.run_mode == 'epi_uncertainty' and self.inputs.n_strains > 1:
            self.plot_perc_mdr_progress()

        # for debugging
        if self.inputs.n_strains > 1:
            self.plot_cases_by_division(['_asds', '_asmdr'],
                                        restriction_1='_mdr', restriction_2='treatment', exclusion_string='latent')

    def plot_outputs_against_gtb(self, outputs, purpose='scenario'):
        """
        Produces the plot for the main outputs, loops over multiple scenarios.

        Args:
            outputs: A list of the outputs to be plotted
            purpose: Reason for plotting or type of plot, can be either 'scenario', 'ci_plot' or 'progress'
        """

        # preliminaries
        start_time = self.inputs.model_constants['plot_start_time']
        if self.run_mode == 'int_uncertainty' or len(self.scenarios) > 1:
            start_time = self.inputs.model_constants['before_intervention_time']
        colour, indices, yaxis_label, title, patch_colour = find_standard_output_styles(outputs, lightening_factor=0.3)
        subplot_grid = find_subplot_numbers(len(outputs))
        fig = self.set_and_update_figure()

        # local variables relevant to the type of analysis requested
        if self.run_mode == 'int_uncertainty':
            uncertainty_scenario, scenarios, start_index, uncertainty_type, linewidth, linecolour, runs, \
                self.accepted_indices = 15, [0, 15], 0, 'int_uncertainty', 1., 'r', self.inputs.n_samples, []
            self.start_time_index = 0
        else:
            uncertainty_scenario, scenarios, uncertainty_type, linewidth \
                = 0, self.scenarios[::-1], 'epi_uncertainty', 1.5

        # loop through indicators
        for o, output in enumerate(outputs):

            # preliminaries
            ax = fig.add_subplot(subplot_grid[0], subplot_grid[1], o + 1)

            # overlay first so it's at the back
            gtb_ci_plot = 'hatch' if purpose == 'shaded' else 'patch'
            self.overlay_gtb_data(
                ax, o, output, start_time, indices, patch_colour, compare_gtb=False, gtb_ci_plot=gtb_ci_plot,
                plot_targets=True, uncertainty_scenario=uncertainty_scenario, alpha=1.)

            # plot with uncertainty confidence intervals
            if purpose == 'ci_plot':
                for scenario in scenarios:
                    scenario_name = t_k.find_scenario_string_from_number(scenario)
                    if not self.run_mode == 'int_uncertainty':
                        uncertainty_scenario, start_index, linecolour \
                            = scenario, self.find_start_index(scenario), self.output_colours[scenario][1]

                    # median
                    ax.plot(self.outputs[uncertainty_type]['epi'][uncertainty_scenario]['times'][0, :][start_index:],
                            self.uncertainty_centiles['epi'][uncertainty_scenario][output][0, :][start_index:],
                            color=linecolour, linestyle=self.output_colours[scenario][0],
                            linewidth=linewidth, label=t_k.capitalise_and_remove_underscore(scenario_name))

                    # upper and lower confidence bounds
                    for index in [1, 2]:
                        ax.plot(self.outputs[uncertainty_type]['epi'][uncertainty_scenario]['times'][0][start_index:],
                                self.uncertainty_centiles['epi'][uncertainty_scenario][output][index, :][start_index:],
                                color=linecolour, linestyle='--', linewidth=.5, label=None)

            # plot progressive model run outputs for uncertainty analyses
            elif purpose == 'progress':

                # get relevant data according to whether intervention or baseline uncertainty is being run
                if not self.run_mode == 'int_uncertainty':
                    runs = len(self.outputs['epi_uncertainty']['epi'][uncertainty_scenario][output])

                # plot the runs
                for run in range(runs):
                    if run in self.accepted_indices or self.plot_rejected_runs or self.run_mode == 'int_uncertainty':
                        if run in self.accepted_indices:
                            linewidth = 1.2
                            colour = str(1. - float(run) / float(len(
                                self.outputs[uncertainty_type]['epi'][uncertainty_scenario][output])))
                        elif self.run_mode == 'int_uncertainty':
                            linewidth, colour = .8, '.4'
                        else:
                            linewidth, colour = .2, 'y'
                        ax.plot(self.outputs[uncertainty_type]['epi'][uncertainty_scenario]['times'][run,
                                self.start_time_index:],
                                self.outputs[uncertainty_type]['epi'][uncertainty_scenario][output][run,
                                self.start_time_index:],
                                linewidth=linewidth, color=colour,
                                label=t_k.capitalise_and_remove_underscore('baseline'))

            elif purpose == 'shaded':

                # plot with uncertainty confidence intervals
                start_index = self.find_start_index(0)
                if self.run_mode == 'int_uncertainty': start_index = 0

                # plot shaded areas as patches
                patch_colours = [cm.Blues(x) for x in numpy.linspace(0., 1., self.model_runner.n_centiles_for_shading)]
                for i in range(self.model_runner.n_centiles_for_shading):
                    patch = create_patch_from_list(
                        self.outputs[uncertainty_type]['epi'][uncertainty_scenario]['times'][0, start_index:],
                        self.uncertainty_centiles['epi'][uncertainty_scenario][output][i + 3, :][start_index:],
                        self.uncertainty_centiles['epi'][uncertainty_scenario][output][-i - 1, :][start_index:])
                    ax.add_patch(patches.Polygon(patch, color=patch_colours[i]))

            # plot scenarios without uncertainty
            if purpose == 'scenario' or self.run_mode == 'int_uncertainty':

                if self.run_mode == 'int_uncertainty':
                    scenarios = [0]

                # plot model estimates
                for scenario in scenarios:  # reversing to ensure black baseline plotted over the top
                    start_index = self.find_start_index(scenario)

                    # work out colour depending on whether purpose is scenario analysis or incrementing comorbidities
                    colour = self.output_colours[scenario][1]
                    if self.run_mode == 'increment_comorbidity' and scenario:
                        colour = cm.Reds(.2 + .8 * self.inputs.comorbidity_prevalences[scenario])
                        label = str(int(self.inputs.comorbidity_prevalences[scenario] * 1e2)) + '%'
                    elif self.run_mode == 'increment_comorbidity' and scenario:
                        colour, label = 'k', 'Baseline'
                    else:
                        label = t_k.capitalise_and_remove_underscore(t_k.find_scenario_string_from_number(scenario))

                    # plot
                    ax.plot(self.outputs['manual']['epi'][scenario]['times'][start_index:],
                            self.outputs['manual']['epi'][scenario][output][start_index:],
                            color=colour, linestyle=self.output_colours[scenario][0], linewidth=1.5, label=label)

                # plot true mortality
                if output == 'mortality' and self.plot_true_outcomes:
                    ax.plot(self.outputs['manual']['epi'][scenario]['times'][start_index:],
                            self.outputs['manual']['epi'][scenario]['true_' + output][start_index:],
                            color=colour, linestyle=':', linewidth=1)

            # find limits to the axes
            y_absolute_limit = None
            if self.run_mode == 'int_uncertainty' or len(scenarios) > 1:
                y_absolute_limit = -1.e15  # an absurd negative value to start from
                plot_start_time_index = t_k.find_first_list_element_at_least_value(
                    self.model_runner.outputs['manual']['epi'][0]['times'], start_time)
                for scenario in self.model_runner.outputs['manual']['epi'].keys():
                    relevant_start_index = plot_start_time_index
                    if scenario != 0:
                        relevant_start_index = 0
                    y_absolute_limit_scenario = max(self.model_runner.outputs['manual']['epi'][scenario][output][relevant_start_index:])
                    if y_absolute_limit_scenario > y_absolute_limit: y_absolute_limit = y_absolute_limit_scenario
                y_absolute_limit *= 1.02  # to allow for some space between curves and top border of the box


            self.tidy_axis(ax, subplot_grid, title=title[o], start_time=start_time,
                           legend=(o == len(outputs) - 1 and len(scenarios) > 1
                                   and not self.run_mode == 'int_uncertainty'),
                           y_axis_type='raw', y_label=yaxis_label[o], y_absolute_limit=y_absolute_limit)

        # fig.suptitle(t_k.capitalise_first_letter(self.country) + ' model outputs', fontsize=self.suptitle_size)
        self.save_figure(fig, '_gtb_' + purpose)

    def overlay_gtb_data(self, ax, o, output, start_time, indices, patch_colour, compare_gtb=False, gtb_ci_plot='hatch',
                         plot_targets=True, uncertainty_scenario=0, alpha=1.):
        """
        Method to plot the data loaded directly from the GTB report in the background.

        Args:
            ax: Axis for plotting
            o: Order of output
            output: String for output
            start_time:
            indices:
            patch_colour:
            compare_gtb: Whether to plot the targets/milestones relative to GTB data rather than modelled outputs
            gtb_ci_plot: How to display the confidence intervals of the GTB data
            plot_targets: Whether to display the End TB Targets and the lines to achieve them
            uncertainty_scenario: Generally 0 to index the baseline or 15 to index the intervention uncertainty scenario
            alpha: Alpha value for patch
        """

        # prelims
        gtb_data = {}
        gtb_data_lists = {}

        # notifications
        if output == 'notifications':
            gtb_data['point_estimate'] = self.inputs.original_data['notifications']['c_newinc']
            gtb_data_lists.update(extract_dict_to_list_key_ordering(gtb_data['point_estimate'], 'point_estimate'))
            gtb_index = t_k.find_first_list_element_at_least_value(gtb_data_lists['times'], start_time)

        # extract the relevant data from the Global TB Report and use to plot a patch (for inc, prev and mortality)
        elif output in self.gtb_available_outputs:
            for level in self.level_conversion_dict:
                gtb_data[level] = self.inputs.original_data['gtb'][indices[o] + self.level_conversion_dict[level]]
                gtb_data_lists.update(extract_dict_to_list_key_ordering(gtb_data[level], level))
            gtb_index = t_k.find_first_list_element_at_least_value(gtb_data_lists['times'], start_time)
            if gtb_ci_plot == 'patch':
                colour, hatch, fill, linewidth, alpha = patch_colour[o], None, True, 1., 1.
            elif gtb_ci_plot == 'hatch':
                colour, hatch, fill, linewidth, alpha = '.3', '/', False, 0., 1.
            ax.add_patch(patches.Polygon(create_patch_from_list(gtb_data_lists['times'][gtb_index:],
                                                                gtb_data_lists['lower_limit'][gtb_index:],
                                                                gtb_data_lists['upper_limit'][gtb_index:]),
                                         color=colour, hatch=hatch, fill=fill, linewidth=linewidth))

        # plot point estimates
        if output in self.gtb_available_outputs:
            ax.plot(gtb_data['point_estimate'].keys()[gtb_index:], gtb_data['point_estimate'].values()[gtb_index:],
                    color='.3', linewidth=0.8, label=None, alpha=alpha)
            if gtb_ci_plot == 'hatch' and output != 'notifications':
                for limit in ['lower_limit', 'upper_limit']:
                    ax.plot(gtb_data[limit].keys()[gtb_index:], gtb_data[limit].values()[gtb_index:],
                            color='.3', linewidth=0.3, label=None, alpha=alpha)

            # plot the targets (and milestones) and the fitted exponential function to achieve them
            if self.run_mode == 'epi_uncertainty' and not self.run_mode == 'int_uncertainty':
                base_value = self.uncertainty_centiles['epi'][uncertainty_scenario][output][0, :][
                    t_k.find_first_list_element_at_least_value(
                        self.outputs['manual']['epi'][uncertainty_scenario]['times'], 2015.)]
            else:
                base_value = self.outputs['manual']['epi'][uncertainty_scenario][output][
                    t_k.find_first_list_element_at_least_value(
                        self.outputs['manual']['epi'][uncertainty_scenario]['times'], 2015.)]
            if compare_gtb: base_value = gtb_data['point_estimate'][2014]  # should be 2015, but data not yet inputted
            if plot_targets and (output == 'incidence' or output == 'mortality'):
                plot_endtb_targets(ax, output, base_value, '.7')

    def plot_resistant_strain_outputs(self, outputs):
        """
        Plot outputs for MDR-TB. Will extend to all resistant strains as needed, which should be pretty easy.
        Sparsely commented because largely shadows plot_outputs_against_gtb (without plotting the patch for the GTB
        outputs).

        Args:
            outputs: The outputs to be plotted (after adding the strain name to the end).
        """

        # prelims
        subplot_grid = find_subplot_numbers(len(outputs))
        fig = self.set_and_update_figure()
        colour, indices, yaxis_label, title, _ = find_standard_output_styles(outputs)

        # cycle over each output and plot
        for o, output in enumerate(outputs):
            ax = fig.add_subplot(subplot_grid[0], subplot_grid[1], o + 1)
            for scenario in self.scenarios[::-1]:
                ax.plot(self.model_runner.outputs['manual']['epi'][scenario]['times'],
                        self.model_runner.outputs['manual']['epi'][scenario][output + '_mdr'],
                        color=self.output_colours[scenario][1], linestyle=self.output_colours[scenario][0])
            self.tidy_axis(ax, subplot_grid, title=title[o], y_label=yaxis_label[o],
                           start_time=self.inputs.model_constants['recent_time'],
                           legend=(o == len(outputs) - 1))

        # finish off
        fig.suptitle(t_k.capitalise_first_letter(self.country) + ' resistant strain outputs',
                     fontsize=self.title_size)
        self.save_figure(fig, '_resistant_strain')

    def classify_scaleups(self):
        """
        Classifies the time variant parameters according to their type (e.g. programmatic, economic, demographic, etc.).
        """

        for classification in self.classifications:
            self.classified_scaleups[classification] = []
            for fn in self.model_runner.models[0].scaleup_fns:
                if classification in fn: self.classified_scaleups[classification] += [fn]

    def individual_var_viewer(self):
        """
        Function that can be used to visualise a particular var or several vars, by adding them to the function input
        list, which is now an attribute of this object (i.e. vars_to_view).
        """

        for var in self.vars_to_view:
            fig = self.set_and_update_figure()
            ax = fig.add_subplot(1, 1, 1)
            for scenario in reversed(self.scenarios):
                ax.plot(self.model_runner.models[scenario].times, self.model_runner.models[scenario].get_var_soln(var),
                        color=self.output_colours[scenario][1])
            self.tidy_axis(ax, [1, 1], start_time=self.inputs.model_constants['plot_start_time'])
            fig.suptitle(t_k.find_title_from_dictionary(var))
            self.save_figure(fig, '_var_' + var)

    def plot_scaleup_fns_against_data(self):
        """
        Plot each scale-up function as a separate panel against the data it is fitted to.
        """

        # different figure for each type of function
        for classification in self.classified_scaleups:
            if len(self.classified_scaleups[classification]) > 0:

                # find the list of the scale-up functions to work with and some x-values
                function_list = self.classified_scaleups[classification]

                # standard prelims
                fig = self.set_and_update_figure()
                subplot_grid = find_subplot_numbers(len(function_list))
                start_time, end_time \
                    = self.inputs.model_constants['plot_start_time'], self.inputs.model_constants['recent_time']
                x_vals = numpy.linspace(start_time, end_time, 1e3)

                # iterate through functions
                for f, function in enumerate(function_list):

                    # initialise axis
                    ax = fig.add_subplot(subplot_grid[0], subplot_grid[1], f + 1)

                    # iterate through the scenarios
                    for scenario in reversed(self.scenarios):

                        # line plot of scaling parameter functions
                        ax.plot(x_vals,
                                map(self.model_runner.models[scenario].scaleup_fns[function],
                                    x_vals),
                                color=self.output_colours[scenario][1],
                                label=t_k.capitalise_and_remove_underscore(
                                    t_k.find_scenario_string_from_number(scenario)))

                    # plot the raw data from which the scale-up functions were produced
                    if function in self.inputs.scaleup_data[0]:
                        data_to_plot = self.inputs.scaleup_data[0][function]
                        ax.scatter(data_to_plot.keys(), data_to_plot.values(), color='k', s=6)

                    # adjust tick font size and add panel title
                    if 'prop_' in function:
                        y_axis_type = 'proportion'
                    else:
                        y_axis_type = 'raw'

                    self.tidy_axis(ax, subplot_grid, start_time=start_time,
                                   title=t_k.capitalise_first_letter(t_k.find_title_from_dictionary(function)),
                                   legend=(f == len(function_list) - 1), y_axis_type=y_axis_type)

                # finish off
                title = self.inputs.country + ' ' + t_k.find_title_from_dictionary(classification) + ' parameter'
                if len(function_list) > 1: title += 's'
                fig.suptitle(title, fontsize=self.title_size)
                self.save_figure(fig, '_' + classification + '_scale_ups')

    def plot_individual_scaleups_against_data(self):
        """
        This method more intended for technical appendices to papers, where it is important to be comprehensive in
        presenting every fitted time-variant parameter that is used.

        Several pieces of dodgy code here in an attempt to get figures looking correct for Fiji paper supplement.
        """

        # different figure for each type of function
        for function in self.model_runner.models[0].scaleup_fns:

            # standard prelims
            fig = self.set_and_update_figure()
            fig.set_figheight(4)
            subplot_grid = [1, 2]
            end_time = 2020.

            for i in range(2):

                # initialise axis
                ax = fig.add_subplot(subplot_grid[0], subplot_grid[1], i + 1)

                start_time = self.inputs.model_constants['early_time']
                if i:
                    start_time = self.inputs.model_constants['plot_start_time']
                elif 'diabetes' in function:
                    start_time = 1900.
                elif 'vacc' in function:
                    start_time = 1920.
                x_vals = numpy.linspace(start_time, end_time, 1e3)

                # iterate through the scenarios
                for scenario in reversed(self.scenarios):

                    # line plot of scaling parameter functions
                    ax.plot(x_vals,
                            map(self.model_runner.models[scenario].scaleup_fns[function], x_vals),
                            color=self.output_colours[scenario][1],
                            label=t_k.capitalise_and_remove_underscore(t_k.find_scenario_string_from_number(scenario)))

                if function in self.inputs.scaleup_data[0]:
                    data_to_plot = self.inputs.scaleup_data[0][function]
                    ax.scatter(data_to_plot.keys(), data_to_plot.values(), color='k', s=6)

                # adjust tick font size and add panel title
                if 'prop_' in function and not i:
                    y_axis_type = 'proportion'
                    y_label = 'Proportion'
                elif 'prop_' in function and i:
                    y_axis_type = 'proportion'
                    y_label = ''
                elif 'life_expectancy' in function and not i:
                    y_axis_type = 'raw'
                    y_label = 'Years'
                elif 'life_expectancy' in function and i:
                    y_axis_type = 'raw'
                    y_label = ''
                else:
                    y_axis_type = 'raw'

                # little fudge to get height for birth rate displaying correctly for Fiji
                y_relative_limit = 0.95
                if function == 'demo_rate_birth' and i:
                    y_relative_limit = 1.1
                    y_label = ''
                elif function == 'demo_rate_birth':
                    y_relative_limit = 1.1
                    y_label = 'Births per 1,000 per year'
                elif ('program_' in function and 'death' in function) or 'diabetes' in function:
                    y_axis_type = 'limited_proportion'

                self.tidy_axis(ax, subplot_grid, start_time=start_time, legend=False, y_axis_type=y_axis_type,
                               end_time=end_time, y_label=y_label, y_relative_limit=y_relative_limit)

            # finish off
            fig.suptitle(t_k.capitalise_first_letter(t_k.find_title_from_dictionary(function)),
                         fontsize=self.title_size)
            self.save_figure(fig, '_' + function + '_scale_up')

    def plot_programmatic_scaleups(self):

        """
        Plots only the programmatic time-variant functions on a single set of axes.
        """

        # Functions to plot are those in the program_prop_ category of the classified scaleups
        functions = self.classified_scaleups['program_prop_']

        # Standard prelims
        fig = self.set_and_update_figure()
        line_styles = make_default_line_styles(len(functions), True)
        start_time = self.inputs.model_constants['plot_start_time']
        x_vals = numpy.linspace(start_time, self.inputs.model_constants['plot_end_time'], 1e3)

        # Plot functions for baseline model run only
        ax = make_single_axis(fig)
        for figure_number, function in enumerate(functions):
            ax.plot(x_vals, map(self.inputs.scaleup_fns[0][function], x_vals), line_styles[figure_number],
                    label=t_k.find_title_from_dictionary(function))

        # Finish off
        self.tidy_axis(ax, [1, 1], title=t_k.capitalise_first_letter(self.country) + ' '
                                         + t_k.find_title_from_dictionary('program_prop_') + ' parameters',
                       start_time=start_time, legend='for_single', y_axis_type='proportion')
        self.save_figure(fig, '_programmatic_scale_ups')

    def plot_cost_coverage_curves(self):
        """
        Plots cost-coverage curves at times specified in the report times inputs in control panel.
        """

        # plot figures by scenario
        for scenario in self.scenarios:
            fig = self.set_and_update_figure()

            # subplots by program
            subplot_grid = find_subplot_numbers(len(self.interventions_to_cost[scenario]))
            for p, program in enumerate(self.interventions_to_cost[scenario]):
                ax = fig.add_subplot(subplot_grid[0], subplot_grid[1], p + 1)

                # make times that each curve is produced for from control panel inputs
                times = range(int(self.inputs.model_constants['cost_curve_start_time']),
                              int(self.inputs.model_constants['cost_curve_end_time']),
                              int(self.inputs.model_constants['cost_curve_step_time']))

                for t, time in enumerate(times):
                    time_index = t_k.find_first_list_element_at_least_value(
                        self.model_runner.models[scenario].times, time)

                    # make cost coverage curve
                    x_values, y_values = [], []
                    for coverage in numpy.linspace(0, 1, 101):
                        if coverage < self.inputs.model_constants['econ_saturation_' + program]:
                            cost \
                                = economics.get_cost_from_coverage(coverage,
                                self.inputs.model_constants['econ_inflectioncost_' + program],
                                self.inputs.model_constants['econ_saturation_' + program],
                                self.inputs.model_constants['econ_unitcost_' + program],
                                self.model_runner.models[scenario].var_array[
                                    time_index, self.model_runner.models[scenario].var_labels.index(
                                        'popsize_' + program)])
                            x_values += [cost]
                            y_values += [coverage]

                    # find darkness
                    darkness = .9 - (float(t) / float(len(times))) * .9

                    # plot
                    ax.plot(x_values, y_values, color=(darkness, darkness, darkness), label=str(int(time)))

                self.tidy_axis(ax, subplot_grid, title=t_k.find_title_from_dictionary('program_prop_' + program),
                               x_axis_type='scaled', legend=(p == len(self.interventions_to_cost) - 1), y_axis_type='proportion',
                               x_label='$US ')

            # finish off with title and save file for scenario
            # fig.suptitle('Cost-coverage curves for ' + t_k.replace_underscore_with_space(scenario),
            #              fontsize=self.suptitle_size)
            self.save_figure(fig, '_' + str(scenario) + '_cost_coverage')

    def plot_cost_over_time(self):
        """
        Method that produces plots for individual and cumulative program costs for each scenario as separate figures.
        Panels of figures are the different sorts of costs (i.e. whether discounting and inflation have been applied).
        """

        # separate figures for each scenario
        for scenario in self.scenarios:

            # standard prelims, but separate for each type of plot - individual and stacked
            fig_individual = self.set_and_update_figure()
            fig_stacked = self.set_and_update_figure()
            fig_relative = self.set_and_update_figure()
            subplot_grid = find_subplot_numbers(len(self.model_runner.cost_types))

            # find the index for the first time after the current time
            reference_time_index \
                = t_k.find_first_list_element_above_value(self.outputs['manual']['cost'][scenario]['times'],
                                                          self.inputs.model_constants['reference_time'])

            # plot each type of cost to its own subplot and ensure same y-axis scale
            ax_individual = fig_individual.add_subplot(subplot_grid[0], subplot_grid[1], 1)
            ax_stacked = fig_stacked.add_subplot(subplot_grid[0], subplot_grid[1], 1)
            ax_relative = fig_relative.add_subplot(subplot_grid[0], subplot_grid[1], 1)
            ax_individual_first = copy.copy(ax_individual)
            ax_stacked_first = copy.copy(ax_stacked)
            ax_reference_first = copy.copy(ax_relative)

            for c, cost_type in enumerate(self.model_runner.cost_types):
                if c > 0:
                    ax_individual = fig_individual.add_subplot(subplot_grid[0], subplot_grid[1], c + 1,
                                                               sharey=ax_individual_first)
                    ax_stacked \
                        = fig_stacked.add_subplot(subplot_grid[0], subplot_grid[1], c + 1, sharey=ax_stacked_first)
                    ax_relative \
                        = fig_relative.add_subplot(subplot_grid[0], subplot_grid[1], c + 1, sharey=ax_reference_first)

                # create empty list for legend
                cumulative_data = [0.] * len(self.outputs['manual']['cost'][scenario]['times'])

                # plot for each intervention
                for intervention in self.inputs.interventions_to_cost[scenario]:

                    # Record the previous data for plotting as an independent object for the lower edge of the fill
                    previous_data = copy.copy(cumulative_data)

                    # Calculate the cumulative sum for the upper edge of the fill
                    for i in range(len(self.outputs['manual']['cost'][scenario]['times'])):
                        cumulative_data[i] \
                            += self.outputs['manual']['cost'][scenario][cost_type
                                                                                    + '_cost_' + intervention][i]

                    # Scale the cost data
                    individual_data \
                        = self.outputs['manual']['cost'][scenario][cost_type + '_cost_' + intervention]
                    reference_cost \
                        = self.outputs['manual']['cost'][scenario][cost_type + '_cost_' + intervention][
                            reference_time_index]
                    relative_data = [(d - reference_cost) for d in individual_data]

                    # plot lines
                    ax_individual.plot(self.outputs['manual']['cost'][scenario]['times'], individual_data,
                                       color=self.program_colours[scenario][intervention][1],
                                       label=t_k.find_title_from_dictionary(intervention))
                    ax_relative.plot(self.outputs['manual']['cost'][scenario]['times'],
                                     relative_data,
                                     color=self.program_colours[scenario][intervention][1],
                                     label=t_k.find_title_from_dictionary(intervention))

                    # plot stacked areas
                    ax_stacked.fill_between(self.model_runner.models[scenario].cost_times,
                                            previous_data, cumulative_data,
                                            color=self.program_colours[scenario][intervention][1],
                                            linewidth=0., label=t_k.find_title_from_dictionary(intervention))

                # final tidying
                for ax in [ax_individual, ax_stacked, ax_relative]:
                    self.tidy_axis(ax, subplot_grid, title=t_k.capitalise_and_remove_underscore(cost_type),
                                   start_time=self.inputs.model_constants['plot_economics_start_time'],
                                   y_label=' $US', y_axis_type='scaled', y_sig_figs=1,
                                   legend=(c == len(self.model_runner.cost_types) - 1))

            # finishing off with title and save
            fig_individual.suptitle('Individual program costs for ' + t_k.find_scenario_string_from_number(scenario),
                                    fontsize=self.title_size)
            self.save_figure(fig_individual, '_' + str(scenario) + '_timecost_individual')
            fig_stacked.suptitle('Stacked program costs for ' + t_k.find_scenario_string_from_number(scenario),
                                 fontsize=self.title_size)
            self.save_figure(fig_stacked, '_' + str(scenario) + '_timecost_stacked')
            fig_relative.suptitle('Relative program costs for ' + t_k.find_scenario_string_from_number(scenario),
                                  fontsize=self.title_size)
            self.save_figure(fig_relative, '_' + str(scenario) + '_timecost_relative')

    def plot_cost_over_time_stacked_bars(self, cost_type='raw'):
        """
        Not called, but won't be working any more because cost_outputs_integer_dict has been abandoned.

        Plotting method to plot bar graphs of spending by programs to look the way Tan gets them to look with Excel.
        That is, separated bars with costs by years.

        Args:
            cost_type: Type of cost to be plotted, i.e. whether raw, inflated, discounted or inflated-and-discounted
        """

        # separate figures for each scenario
        for scenario in self.scenarios:

            # standard prelims
            fig = self.set_and_update_figure()

            # each scenario being implemented
            for inter, intervention in enumerate(self.inputs.interventions_to_cost[scenario]):

                # find the data to plot for the current intervention
                data = self.model_runner.cost_outputs_integer_dict[
                    'manual_' + t_k.find_scenario_string_from_number(scenario)][cost_type + '_cost_' + intervention]

                # initialise the dictionaries at the first iteration
                if inter == 0:
                    base = {i: 0. for i in data}
                    upper = {i: 0. for i in data}

                # increment the upper values
                upper = {i: upper[i] + data[i] for i in data}

                # plot
                ax = make_single_axis(fig)
                ax.bar(upper.keys(), upper.values(), .6, bottom=base.values(),
                       color=self.program_colours[scenario][intervention][1],
                       label=t_k.find_title_from_dictionary(intervention))

                # increment the lower values before looping again
                base = {i: base[i] + data[i] for i in data}

            # finishing up
            self.tidy_axis(
                ax, [1, 1],
                title=t_k.capitalise_and_remove_underscore(cost_type) + ' costs by intervention for '
                      + t_k.replace_underscore_with_space(t_k.find_scenario_string_from_number(scenario)),
                start_time=self.inputs.model_constants['plot_economics_start_time'], x_axis_type='individual_years',
                y_label=' $US', y_axis_type='scaled', y_sig_figs=1, legend='for_single')
            self.save_figure(fig, '_' + t_k.find_scenario_string_from_number(scenario) + '_timecost_stackedbars')

    def plot_populations(self, strain_or_organ='organ'):
        """
        Plot population by the compartment to which they belong.

        *** Doesn't work well for any but the simplest compartmental model structure - due to number of compartments ***

        Args:
            strain_or_organ: Whether the plotting style should be done by strain or by organ
        """

        # standard prelims
        fig = self.set_and_update_figure()
        ax = make_single_axis(fig)
        colours, patterns, compartment_full_names, markers \
            = make_related_line_styles(self.model_runner.models[0].labels, strain_or_organ)

        # plot total population
        ax.plot(self.model_runner.outputs['manual']['epi'][0]['times'][self.start_time_index:],
                self.model_runner.outputs['manual']['epi'][0]['population'][self.start_time_index:],
                'k', label='total', linewidth=2)

        # plot sub-populations
        for plot_label in self.model_runner.models[0].labels:
            ax.plot(self.model_runner.outputs['manual']['epi'][0]['times'][self.start_time_index:],
                    self.model_runner.models[0].compartment_soln[plot_label][self.start_time_index:],
                    label=t_k.find_title_from_dictionary(plot_label), linewidth=1, color=colours[plot_label],
                    marker=markers[plot_label], linestyle=patterns[plot_label])

        # finishing touches
        self.tidy_axis(ax, [1, 1], title='Compartmental population distribution (baseline scenario)',
                       start_time=self.inputs.model_constants['plot_start_time'], legend='for_single',
                       y_label='Population', y_axis_type='scaled')
        self.save_figure(fig, '_population')

    def plot_fractions(self, strain_or_organ):
        """
        Plot population fractions by the compartment to which they belong.

        *** Ideally shouldn't be running directly from the model objects as is currently happening.

        *** Actually, this really isn't doing what it's supposed to at all - errors seem to be in the functions that are
        called from the took kit. ***

        Args:
            strain_or_organ: Whether the plotting style should be done by strain or by organ.
        """

        # get values to be plotted
        _, subgroup_fractions = t_k.find_fractions(self.model_runner.models[0])
        for c, category in enumerate(subgroup_fractions):
            values = subgroup_fractions[category]

            # standard prelims
            fig = self.set_and_update_figure()
            ax = make_single_axis(fig)
            colours, patterns, compartment_full_names, markers \
                = make_related_line_styles(values.keys(), strain_or_organ)

            # plot population fractions
            for plot_label in values.keys():
                ax.plot(self.model_runner.models[0].times, values[plot_label],
                        label=t_k.find_title_from_dictionary(plot_label), linewidth=1, color=colours[plot_label],
                        marker=markers[plot_label], linestyle=patterns[plot_label])

            # finishing up
            self.tidy_axis(ax, [1, 1], legend='for_single', start_time=self.inputs.model_constants['plot_start_time'],
                           y_axis_type='proportion')
            self.save_figure(fig, '_fraction')

    def plot_outputs_by_stratum(self, strata_string='agegroups', outputs_to_plot=('incidence', 'mortality')):
        """
        Plot basic epidemiological outputs either by risk stratum or by age group.
        """

        # find strata to loop over
        strata = getattr(self.inputs, strata_string)
        if len(strata) == 0:
            return

        # prelims
        fig = self.set_and_update_figure()
        subplot_grid = [len(outputs_to_plot), len(strata)]

        # loop over outputs and strata
        for o, output in enumerate(outputs_to_plot):
            for s, stratum in enumerate(strata):

                # a + 1 gives the column, o the row
                ax = fig.add_subplot(subplot_grid[0], subplot_grid[1], s + 1 + o * len(strata))

                # plot the modelled data
                for scenario in self.scenarios[::-1]:
                    start_index = self.find_start_index(scenario)
                    ax.plot(self.outputs['manual']['epi'][scenario]['times'][start_index:],
                            self.outputs['manual']['epi'][scenario][output + stratum][start_index:],
                            color=self.output_colours[scenario][1], linestyle=self.output_colours[scenario][0],
                            linewidth=1.5, label=t_k.capitalise_and_remove_underscore(
                            t_k.find_scenario_string_from_number(scenario)))

                # finish off
                if s == 0:
                    ylabel = 'Per 100,000 per year'
                else:
                    ylabel = ''
                if strata_string == 'agegroups':
                    stratum_string = t_k.turn_strat_into_label(stratum)
                else:
                    stratum_string = t_k.find_title_from_dictionary(stratum)
                self.tidy_axis(ax, subplot_grid, start_time=self.inputs.model_constants['plot_start_time'],
                               y_label=ylabel, y_axis_type='scaled',
                               title=t_k.capitalise_first_letter(output) + ', ' + stratum_string,
                               legend=(output == len(outputs_to_plot) - 1 and s == len(strata) - 1))
        fig.suptitle(t_k.capitalise_and_remove_underscore(self.country) + ' burden by sub-group',
                     fontsize=self.title_size)
        self.save_figure(fig, '_output_by_' + strata_string)

    def plot_proportion_cases_by_stratum(self, strata_string='agegroups'):
        """
        Method to plot the proportion of notifications that come from various groups of the model. Particularly intended
        to keep an eye on the proportion of notifications occurring in the paediatric population (which WHO sometimes
        say should be around 15% in well-functioning TB programs).

        Args:
            strata_string: String of the model attribute of interest - can set to 'riskgroups'
        """

        # find strata to loop over
        strata = getattr(self.inputs, strata_string)
        if len(strata) == 0: return

        colours = make_default_line_styles(len(strata), return_all=True)

        # prelims
        fig = self.set_and_update_figure()
        ax = fig.add_subplot(1, 1, 1)
        times = self.model_runner.models[0].times
        lower_plot_margin = numpy.zeros(len(times))
        upper_plot_margin = numpy.zeros(len(times))

        for s, stratum in enumerate(strata):

            # find numbers or fractions in that group
            stratum_count = t_k.calculate_proportion_list(
                self.model_runner.outputs['manual']['epi'][0]['notifications' + stratum],
                self.model_runner.outputs['manual']['epi'][0]['notifications'])

            for i in range(len(upper_plot_margin)): upper_plot_margin[i] += stratum_count[i]

            # create proxy for legend
            if strata_string == 'agegroups':
                legd_text = t_k.turn_strat_into_label(stratum)
            elif strata_string == 'riskgroups':
                legd_text = t_k.find_title_from_dictionary(stratum)

            ax.fill_between(times, lower_plot_margin, upper_plot_margin, facecolors=colours[s][1],
                            label=legd_text)

            # add group values to the lower plot range for next iteration
            for i in range(len(lower_plot_margin)): lower_plot_margin[i] += stratum_count[i]

        # tidy up plots
        self.tidy_axis(ax, [1, 1], start_time=self.inputs.model_constants['recent_time'],
                       y_axis_type='proportion', y_label='Proportion', legend=True)

        fig.suptitle('Proportion of notifications by age', fontsize=self.title_size)
        self.save_figure(fig, '_proportion_notifications_by_age')

    def plot_stratified_populations(self, age_or_risk='age'):
        """
        Function to plot population by age group both as raw numbers and as proportions,
        both from the start of the model and using the input argument.
        """

        early_time_index \
            = t_k.find_first_list_element_at_least_value(self.model_runner.outputs['manual']['epi'][0]['times'],
                                                         self.inputs.model_constants['early_time'])

        # find stratification to work with
        if age_or_risk == 'age':
            stratification = self.inputs.agegroups
        elif age_or_risk == 'risk':
            stratification = self.inputs.riskgroups
        else:
            stratification = None

        # warn if necessary
        if stratification is None:
            warnings.warn('Plotting by stratification requested, but type of stratification requested unknown')
        elif len(stratification) < 2:
            warnings.warn('No stratification to plot')
        else:

            # standard prelims
            fig = self.set_and_update_figure()
            colours = make_default_line_styles(len(stratification), return_all=True)

            # run plotting from early in the model run and from the standard start time for plotting
            for t, time in enumerate(['plot_start_time', 'early_time']):

                # initialise axes
                ax_upper = fig.add_subplot(2, 2, 1 + t)
                ax_lower = fig.add_subplot(2, 2, 3 + t)

                # find starting times
                title_time_text = t_k.find_title_from_dictionary(time)

                # initialise some variables
                times = self.outputs['manual']['epi'][0]['times']
                lower_plot_margin_count = numpy.zeros(len(times))
                upper_plot_margin_count = numpy.zeros(len(times))
                lower_plot_margin_fraction = numpy.zeros(len(times))
                upper_plot_margin_fraction = numpy.zeros(len(times))

                for s, stratum in enumerate(stratification):

                    # find numbers or fractions in that group
                    stratum_count = self.model_runner.outputs['manual']['epi'][0]['population' + stratum]
                    stratum_fraction = self.model_runner.outputs['manual']['epi'][0]['fraction' + stratum]

                    # add group values to the upper plot range for area plot
                    for i in range(len(upper_plot_margin_count)):
                        upper_plot_margin_count[i] += stratum_count[i]
                        upper_plot_margin_fraction[i] += stratum_fraction[i]

                    # create proxy for legend
                    if age_or_risk == 'age':
                        legd_text = t_k.turn_strat_into_label(stratum)
                    elif age_or_risk == 'risk':
                        legd_text = t_k.find_title_from_dictionary(stratum)

                    time_index = self.start_time_index if t == 0 else early_time_index

                    # plot total numbers
                    ax_upper.fill_between(times[time_index:], lower_plot_margin_count[time_index:],
                                          upper_plot_margin_count[time_index:], facecolors=colours[s][1])

                    # plot population proportions
                    ax_lower.fill_between(times[time_index:], lower_plot_margin_fraction[time_index:],
                                          upper_plot_margin_fraction[time_index:], facecolors=colours[s][1],
                                          label=t_k.capitalise_first_letter(legd_text))

                    # add group values to the lower plot range for next iteration
                    for i in range(len(lower_plot_margin_count)):
                        lower_plot_margin_count[i] += stratum_count[i]
                        lower_plot_margin_fraction[i] += stratum_fraction[i]

                # tidy up plots
                self.tidy_axis(ax_upper, [2, 2], start_time=self.inputs.model_constants[time],
                               title='Total numbers from ' + title_time_text, y_label='Population', y_axis_type='')
                self.tidy_axis(ax_lower, [2, 2], y_axis_type='proportion',
                               start_time=self.inputs.model_constants[time],
                               title='Proportion of population from ' + title_time_text, legend=(t == 1))

            # finish up
            fig.suptitle('Population by ' + t_k.find_title_from_dictionary(age_or_risk), fontsize=self.title_size)
            self.save_figure(fig, '_riskgroup_proportions')

    def plot_intervention_costs_by_scenario(self, year_start, year_end, horizontal=False, plot_options=None):
        """
        Not called, but won't be working any more because cost_outputs_integer_dict has been abandoned.

        Function for plotting total cost of interventions under different scenarios over a given range of years.
        Will throw error if defined year range is not present in economic model outputs.

        Args:
            year_start: Integer, start year of time frame over which to calculate total costs
            year_end: Integer, end year of time frame over which to calculate total costs (included)
            horizontal: Boolean, plot stacked bar chart horizontally
            plot_options: Dictionary, options for generating plot
        """

        # set and check options / data ranges
        intervention_names_dict \
            = {'vaccination': 'Vaccination', 'xpert': 'GeneXpert', 'xpertacf': 'GeneXpert ACF', 'smearacf': 'Smear ACF',
               'treatment_support': 'Treatment Support', 'ipt_age0to5': 'IPT 0-5 y.o.', 'ipt_age5to15': 'IPT 5-15 y.o.'}

        defaults = {
            'interventions': self.inputs.interventions_to_cost,
            'x_label_rotation': 45,
            'y_label': 'Total Cost ($)\n',
            'legend_size': 10,
            'legend_frame': False,
            'plot_style': 'ggplot',
            'title': 'Projected total costs {sy} - {ey}\n'.format(sy=year_start, ey=year_end)
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

        if options['plot_style'] is not None: style.use(options['plot_style'])

        years = range(year_start, year_end + 1)

        # make data frame (columns: interventions, rows: scenarios)
        data_frame = pandas.DataFrame(index=self.scenarios, columns=intervention_names)
        for scenario in self.scenarios:
            data_frame.loc[scenario] \
                = [sum([self.model_runner.cost_outputs_integer_dict[
                            'manual_' + t_k.find_scenario_string_from_number(scenario)][
                            'discounted_inflated_cost_' + intervention][year] for year in years])
                   for intervention in options['interventions']]
        data_frame.columns = intervention_names

        # make and style plot
        if horizontal:
            plot = data_frame.plot.barh(stacked=True, rot=options['x_label_rotation'], title=options['title'])
            plot.set_xlabel(options['y_label'])
        else:
            plot = data_frame.plot.bar(stacked=True, rot=options['x_label_rotation'], title=options['title'])
            plot.set_ylabel(options['y_label'])

        humanise_y_ticks(plot)

        handles, labels = plot.get_legend_handles_labels()
        lgd = plot.legend(handles, labels, bbox_to_anchor=(1., 0.5), loc='center left',
                          fontsize=options['legend_size'], frameon=options['legend_frame'])

        # save plot
        pyplot.savefig(os.path.join(self.out_dir_project, self.country + '_totalcost' + '.png'),
                       bbox_extra_artists=(lgd,), bbox_inches='tight')

    def plot_riskgroup_checks(self):
        """
        Plots actual risk group fractions against targets. Probably almost redundant, as this is a test of code, rather
        than really giving and epidemiological information.
        """

        # standard prelims
        fig = self.set_and_update_figure()
        ax = make_single_axis(fig)

        # plotting
        for riskgroup in self.model_runner.models[0].riskgroups:
            ax.plot(self.model_runner.models[0].times,
                    self.model_runner.models[0].actual_risk_props[riskgroup], 'g-',
                    label='Actual ' + riskgroup)
            ax.plot(self.model_runner.models[0].times,
                    self.model_runner.models[0].target_risk_props[riskgroup][1:], 'k--',
                    label='Target ' + riskgroup)

        # end bits
        self.tidy_axis(ax, [1, 1], y_axis_type='proportion', start_time=self.inputs.model_constants['plot_start_time'],
                       single_axis_room_for_legend=True)
        fig.suptitle('Population by risk group', fontsize=self.title_size)
        self.save_figure(fig, '_riskgroup_checks')

    def plot_param_histograms(self):
        """
        Simple function to plot histograms of parameter values used in uncertainty analysis.
        """

        # preliminaries
        fig = self.set_and_update_figure()
        subplot_grid = find_subplot_numbers(len(self.model_runner.outputs['epi_uncertainty']['all_parameters']))

        # loop through parameters used in uncertainty
        for p, param in enumerate(self.model_runner.outputs['epi_uncertainty']['all_parameters']):
            ax = fig.add_subplot(subplot_grid[0], subplot_grid[1], p + 1)

            # restrict to those accepted and after burn-in complete
            param_values = [self.model_runner.outputs['epi_uncertainty']['all_parameters'][param][i]
                            for i in self.accepted_no_burn_in_indices]

            # plot
            ax.hist(param_values)
            ax.set_title(t_k.find_title_from_dictionary(param))
        self.save_figure(fig, '_param_histogram')

    def plot_param_timeseries(self):
        """
        Plot accepted parameter progress over time.
        """
        fig = self.set_and_update_figure()
        subplot_grid = find_subplot_numbers(len(self.model_runner.outputs['epi_uncertainty']['all_parameters']))

        # loop through parameters used in uncertainty
        for p, param in enumerate(self.model_runner.outputs['epi_uncertainty']['all_parameters']):
            ax = fig.add_subplot(subplot_grid[0], subplot_grid[1], p + 1)

            # restrict to those accepted and after burn-in complete
            param_values = [self.model_runner.outputs['epi_uncertainty']['all_parameters'][param][i]
                            for i in self.accepted_no_burn_in_indices]

            # plot
            ax.plot(param_values)
            ax.set_title(t_k.find_title_from_dictionary(param))
        self.save_figure(fig, '_param_timeseries')

    def plot_priors(self):
        """
        Function to plot the prior distributions that are logged to get the prior contribution to the acceptance
        probability in the epidemiological uncertainty running.
        """

        fig = self.set_and_update_figure()
        subplot_grid = find_subplot_numbers(len(self.model_runner.inputs.param_ranges_unc))
        n_plot_points = 1000

        for p, param in enumerate(self.model_runner.inputs.param_ranges_unc):
            distribution, lower, upper = param['distribution'], param['bounds'][0], param['bounds'][1]
            if distribution == 'uniform':
                x_values = numpy.linspace(lower, upper, n_plot_points)
                y_values = [1. / (upper - lower)] * len(x_values)
                description = t_k.capitalise_first_letter(distribution)
            elif distribution == 'beta_2_2':
                lower, upper = 0., 1.
                x_values = numpy.linspace(lower, upper, n_plot_points)
                y_values = [scipy.stats.beta.pdf((x - lower) / (upper - lower), 2., 2.) for x in x_values]
                description = t_k.find_title_from_dictionary(distribution)
            elif distribution == 'beta_mean_stdev':
                lower, upper = 0., 1.
                x_values = numpy.linspace(lower, upper, n_plot_points)
                alpha_value = ((1. - param['additional_params'][0]) / param['additional_params'][1] ** 2. - 1.
                               / param['additional_params'][0]) * param['additional_params'][0] ** 2.
                beta_value = alpha_value * (1. / param['additional_params'][0] - 1.)
                y_values = [scipy.stats.beta.pdf(x, alpha_value, beta_value) for x in x_values]
                description = 'Beta, params:\n%.2g, %.2g' % (alpha_value, beta_value)
            elif distribution == 'beta_params':
                lower, upper = 0., 1.
                x_values = numpy.linspace(lower, upper, n_plot_points)
                y_values = [scipy.stats.beta.pdf(x, param['additional_params'][0], param['additional_params'][1])
                            for x in x_values]
                description \
                    = 'Beta, params:\n%.2g, %.2g' % (param['additional_params'][0], param['additional_params'][1])
            elif distribution == 'gamma_mean_stdev':
                x_values = numpy.linspace(lower, upper, n_plot_points)
                alpha_value = (param['additional_params'][0] / param['additional_params'][1]) ** 2.
                beta_value = param['additional_params'][1] ** 2. / param['additional_params'][0]
                y_values = [scipy.stats.gamma.pdf(x, alpha_value, scale=beta_value) for x in x_values]
                description = 'Gamma, params:\n%.2g, %.2g' % (alpha_value, beta_value)
            elif distribution == 'gamma_params':
                x_values = numpy.linspace(lower, upper, n_plot_points)
                y_values = [scipy.stats.gamma.pdf(x, param['additional_params'][0]) for x in x_values]
                description = 'Gamma, params:\n%.2g' % param['additional_params'][0]

            ax = fig.add_subplot(subplot_grid[0], subplot_grid[1], p + 1)
            ax.set_title(t_k.capitalise_first_letter(t_k.find_title_from_dictionary(param['key'])),
                         fontsize=get_nice_font_size(subplot_grid))
            ax.plot(x_values, y_values)
            ax.text(lower + .05, max(y_values) / 2., description, fontsize=get_nice_font_size(subplot_grid))
            ax.set_ylim(bottom=0.)
            for axis_to_change in [ax.xaxis, ax.yaxis]:
                for tick in axis_to_change.get_major_ticks():
                    tick.label.set_fontsize(get_nice_font_size(subplot_grid))
                axis_to_change.grid(self.grid)

        self.save_figure(fig, '_priors')

    def plot_force_infection(self):
        """
        View the force of infection vars.
        """

        # separate plot for each strain
        for strain in self.model_runner.models[0].strains:
            fig = self.set_and_update_figure()
            ax = fig.add_subplot(1, 1, 1)

            # loop over risk groups and plot line for each - now need to restrict to single age-group, as IPT modifies
            # the force of infection for some age-groups.
            for riskgroup in self.model_runner.models[0].riskgroups:
                data_to_plot = self.model_runner.models[0].get_var_soln(
                            'rate_force' + strain + riskgroup
                            + self.model_runner.models[0].agegroups[-1])[
                        self.start_time_index:]
                ax.plot(self.model_runner.models[0].times[self.start_time_index:],
                        data_to_plot * 1e2,
                        label=t_k.capitalise_first_letter(t_k.find_title_from_dictionary(riskgroup)))

            # finish off
            self.tidy_axis(ax, [1, 1], start_time=self.inputs.model_constants['plot_start_time'],
                           y_axis_type='scaled', legend=True)
            fig.suptitle('Percentage annual risk of infection, ' + t_k.find_title_from_dictionary(strain))
            self.save_figure(fig, '_rate_force' + strain)

    def plot_mixing_matrix(self):
        """
        Method to visualise the mixing matrix with bar charts.
        """

        fig = self.set_and_update_figure()
        ax = make_single_axis(fig)
        output_colours = make_default_line_styles(5, True)
        bar_width = .7
        last_data = list(numpy.zeros(len(self.inputs.riskgroups)))
        for r, to_riskgroup in enumerate(self.inputs.riskgroups):
            this_data = []
            for from_riskgroup in self.inputs.riskgroups:
                this_data.append(self.inputs.mixing[from_riskgroup][to_riskgroup])
            next_data = [last + this for last, this in zip(last_data, this_data)]
            x_positions = numpy.linspace(.5, .5 + len(next_data) - 1., len(next_data))
            ax.bar(x_positions, this_data, width=bar_width, bottom=last_data, color=output_colours[r][1],
                   label=t_k.capitalise_and_remove_underscore(t_k.find_title_from_dictionary(to_riskgroup)))
            last_data = next_data
        xlabels = [t_k.capitalise_first_letter(t_k.find_title_from_dictionary(last)) for last in self.inputs.riskgroups]
        self.tidy_axis(ax, [1, 1], y_label='Proportion',
                       x_axis_type='raw', y_axis_type='proportion', legend='for_single', title='Source of contacts')
        ax.set_xlim(0.2, max(x_positions) + 1.)
        x_label_positions = [x + bar_width / 2. for x in x_positions]
        ax.set_xticks(x_label_positions)
        ax.tick_params(length=0.)
        ax.set_xticklabels(xlabels)
        self.save_figure(fig, '_mixing')

    def plot_popsizes(self):
        """
        Plot popsizes over recent time for each program in baseline scenario.
        """

        # prelims
        fig = self.set_and_update_figure()
        ax = make_single_axis(fig)

        # plotting
        for var in self.model_runner.models[0].var_labels:
            if 'popsize_' in var:
                ax.plot(self.model_runner.models[0].times[self.start_time_index:],
                        self.model_runner.models[0].get_var_soln(var)[self.start_time_index:],
                        label=t_k.find_title_from_dictionary(var[8:]))

        # finishing up
        self.tidy_axis(ax, [1, 1], legend='for_single', start_time=self.inputs.model_constants['plot_start_time'],
                       y_label=' persons', y_axis_type='scaled')
        fig.suptitle('Population sizes for cost-coverage curves under baseline scenario')
        self.save_figure(fig, '_popsizes')

    def plot_case_detection_rate(self):
        """
        Method to visualise case detection rates across scenarios and sub-groups, to better understand the impact of
        ACF on improving detection rates relative to passive case finding alone.
        """

        # prelims
        riskgroup_styles = make_default_line_styles(5, True)
        fig = self.set_and_update_figure()
        ax_left = fig.add_subplot(1, 2, 1)
        ax_right = fig.add_subplot(1, 2, 2)
        separation = 0.
        separation_increment = .003  # artificial shifting of lines away from each other

        # specify interventions of interest for this plotting function - needs to be changed for each country
        interventions_affecting_case_detection = [0, 5, 6]

        # loop over scenarios
        for scenario in interventions_affecting_case_detection:
            scenario_name = t_k.find_scenario_string_from_number(scenario)
            manual_scenario_name = scenario
            if scenario in self.scenarios:
                start_index = self.find_start_index(scenario)

                # Repeat for each sub-group
                for r, riskgroup in enumerate(self.model_runner.models[manual_scenario_name].riskgroups[::-1]):

                    # Find weighted case detection rate over organ statuses
                    case_detection = numpy.zeros(len(self.model_runner.models[manual_scenario_name].times[
                                                     start_index:]))
                    for organ in self.inputs.organ_status:
                        case_detection_increment \
                            = [i * j for i, j in zip(self.model_runner.models[manual_scenario_name].get_var_soln(
                                                         'program_rate_detect' + organ + riskgroup)[start_index:],
                                                     self.model_runner.models[manual_scenario_name].get_var_soln(
                                                         'epi_prop' + organ)[start_index:])]
                        case_detection \
                            = model_runner.elementwise_list_addition(case_detection, case_detection_increment)
                    case_detection = [i + separation for i in case_detection]
                    delay_to_presentation = [12. / i for i in case_detection]  # Convert rate to delay
                    ax_left.plot(self.model_runner.models[manual_scenario_name].times[start_index:],
                                 case_detection,
                                 color=self.output_colours[scenario][1], linestyle=riskgroup_styles[r * 7][:-1])
                    ax_right.plot(self.model_runner.models[manual_scenario_name].times[start_index:],
                                  delay_to_presentation,
                                  label=t_k.capitalise_first_letter(t_k.find_title_from_dictionary(riskgroup)) + ', '
                                        + t_k.replace_underscore_with_space(scenario_name),
                                  color=self.output_colours[scenario][1], linestyle=riskgroup_styles[r * 7][:-1])
                    separation += separation_increment

            # Finish off
            self.tidy_axis(ax_left, [1, 2], start_time=self.inputs.model_constants['plot_start_time'],
                           y_axis_type='raw', title='Case detection rates', y_label='Presentation rate (per year)')
            self.tidy_axis(ax_right, [1, 2], legend=True, start_time=self.inputs.model_constants['plot_start_time'],
                           y_axis_type='raw', title='Delay to presentation', y_label='Months')
            ax_right.legend(prop={'size': 8}, frameon=False)
        self.save_figure(fig, '_case_detection')

    def plot_likelihoods(self):
        """
        Method to plot likelihoods over runs, differentiating accepted and rejected runs to illustrate progression.
        """

        # plotting prelims
        fig = self.set_and_update_figure()
        ax = fig.add_subplot(1, 1, 1)

        # find accepted likelihoods
        accepted_log_likelihoods = [self.outputs['epi_uncertainty']['loglikelihoods'][i] for i in self.accepted_indices]

        # plot the rejected values
        for i in self.outputs['epi_uncertainty']['rejected_indices']:

            # Find the index of the last accepted index before the rejected one we're currently interested in
            last_acceptance_before = [j for j in self.accepted_indices if j < i][-1]

            # Plot from the previous acceptance to the current rejection
            ax.plot([last_acceptance_before, i],
                    [self.outputs['epi_uncertainty']['loglikelihoods'][last_acceptance_before],
                     self.outputs['epi_uncertainty']['loglikelihoods'][i]], marker='o', linestyle='--', color='.5')

        # plot the accepted values
        ax.plot(self.accepted_indices, accepted_log_likelihoods, marker='o', color='k')

        # finishing up
        fig.suptitle('Progression of likelihood', fontsize=self.title_size)
        ax.set_xlabel('All runs', fontsize=get_nice_font_size([1, 1]), labelpad=1)
        ax.set_ylabel('Likelihood', fontsize=get_nice_font_size([1, 1]), labelpad=1)
        self.save_figure(fig, '_likelihoods')

    def plot_perc_mdr_progress(self):
        """
        Plot the percentage of MDR-TB among incident TB cases over time, for the different accepted runs.
        """

        fig = self.set_and_update_figure()
        ax = make_single_axis(fig)

        # plot the target
        ax.plot(self.inputs.model_constants['current_time'], self.inputs.model_constants['tb_perc_mdr_prevalence'],
                marker='o', markersize=5, markeredgewidth=0., linewidth=0.)

        # plot the runs
        for run in range(len(self.outputs['epi_uncertainty']['epi'][0]['perc_incidence_mdr'])):
            if run in self.accepted_indices or self.plot_rejected_runs:
                if run in self.accepted_indices:
                    linewidth, colour = 1.2, str(
                        1. - float(run) / float(len(self.outputs['epi_uncertainty']['epi'][0]['perc_incidence_mdr'])))
                elif self.run_mode == 'int_uncertainty':
                    linewidth, colour = .8, '.4'
                else:
                    linewidth, colour = .2, 'y'
                ax.plot(self.outputs['epi_uncertainty']['epi'][0]['times'][run,
                        self.start_time_index:],
                        self.outputs['epi_uncertainty']['epi'][0]['perc_incidence_mdr'][run,
                        self.start_time_index:],
                        linewidth=linewidth, color=colour,
                        label=t_k.capitalise_and_remove_underscore('baseline'))

        # finishing up
        fig.suptitle('Progression of MDR-TB proportion', fontsize=self.title_size)
        ax.set_xlabel('', fontsize=get_nice_font_size([1, 1]), labelpad=1)
        ax.set_ylabel('% of MDR-TB in TB incidence', fontsize=get_nice_font_size([1, 1]), labelpad=1)
        self.save_figure(fig, '_perc_mdr_progress')

    def plot_cases_by_division(self, divisions, restriction_1='', restriction_2='',
                               exclusion_string='we all love futsal'):
        """
        Plot the number cases in across various categories, within the population specified in the restriction string.
        Mostly for debugging purposes.
        """

        fig = self.set_and_update_figure()
        divisions, compartment_types \
            = self.model_runner.models[0].calculate_aggregate_compartment_divisions_from_strings(
                divisions, required_string_1=restriction_1, required_string_2=restriction_2,
                exclusion_string=exclusion_string)
        subplot_grid = find_subplot_numbers(len(compartment_types))
        for c, compartment_type in enumerate(compartment_types):
            if divisions[compartment_type]:
                ax = fig.add_subplot(subplot_grid[0], subplot_grid[1], c + 1)
                ax.plot(self.model_runner.models[0].times, divisions[compartment_type])
                self.tidy_axis(ax, subplot_grid, start_time=self.inputs.model_constants['start_time'],
                               title=compartment_type + restriction_1 + restriction_2)
        self.save_figure(fig, '_mdr_by_compartment_type')

    def plot_optimised_epi_outputs(self):
        """
        Plot incidence and mortality over funding. This corresponds to the outputs obtained under optimal allocation.
        """

        fig = self.set_and_update_figure()
        left_ax = make_single_axis(fig)
        right_ax = left_ax.twinx()
        plots = {'incidence': [left_ax, 'b^', 'TB incidence per 100,000 per year'],
                'mortality': [right_ax, 'r+', 'TB mortality per 100,000 per year']}
        for plot in plots:
            plots[plot][0].plot(self.model_runner.opti_results['annual_envelope'],
                                 self.model_runner.opti_results[plot], plots[plot][1], linewidth=2.0, label=plot)
            self.tidy_axis(plots[plot][0], [1, 1], y_axis_type='raw', y_label=plots[plot][2], x_sig_figs=1,
                           title='Annual funding (US$)', x_axis_type='scaled', x_label='$US ', legend='for_single')
        self.save_figure(fig, '_optimised_outputs')

    def plot_piecharts_opti(self):

        n_envelopes = len(self.model_runner.opti_results['annual_envelope'])
        subplot_grid = find_subplot_numbers(n_envelopes + 1)
        font_size = get_nice_font_size(subplot_grid)

        fig = self.set_and_update_figure()
        colors = ['#000037', '#7398B5', '#D94700', '#DBE4E9', '#62000E', '#3D5F00', '#240445', 'black', 'red',
                  'yellow', 'blue']  # AuTuMN colours
        color_dict = {}
        for i, intervention in enumerate(self.model_runner.interventions_considered_for_opti):
            color_dict[intervention] = colors[i]

        interventions_for_legend = []
        for i, funding in enumerate(self.model_runner.opti_results['annual_envelope']):
            ax = fig.add_subplot(subplot_grid[0], subplot_grid[1], i+1)
            temp_dict = self.model_runner.opti_results['best_allocation'][i]
            temp_dict = {key: val for key, val in temp_dict.iteritems() if val > 0.0001}
            labels = temp_dict.keys()
            fracs = temp_dict.values()
            dynamic_colors = [color_dict[lab] for lab in labels]
            ax.pie(fracs, autopct='%1.1f%%', startangle=90, pctdistance=0.8, radius=0.8, colors=dynamic_colors, \
                   textprops={'backgroundcolor': 'white', 'fontsize': font_size})
            ax.axis('equal')
            circle = pyplot.Circle((0, 0), 0.4, color='w')
            ax.add_artist(circle)
            ax.text(0, 0, get_string_for_funding(funding), horizontalalignment='center', verticalalignment='center')
            if len(interventions_for_legend) == 0:
                # The legend will contain interventions sorted by proportion of funding for the smallest funding
                interventions_for_legend = sorted(temp_dict, key=temp_dict.get, reverse=True)
            else:
                # we need to add interventions that were not selected for lower funding amounts
                for intervention in temp_dict.keys():
                    if intervention not in interventions_for_legend:
                        interventions_for_legend.append(intervention)

        # Generate a gost pie chart that include all interventions to be able to build the full legend
        ax = fig.add_subplot(subplot_grid[0], subplot_grid[1], n_envelopes + 1)
        fracs = numpy.random.uniform(0, 1, size=len(interventions_for_legend))
        dynamic_colors = [color_dict[lab] for lab in interventions_for_legend]
        patches, texts = ax.pie(fracs, colors=dynamic_colors)
        ax.cla()  # Clear the gost pie chart

        ax = fig.add_subplot(subplot_grid[0], subplot_grid[1], n_envelopes + 1)
        ax.legend(patches, interventions_for_legend, loc='right', fontsize=2.0*font_size)
        ax.axis('off')
        fig.tight_layout()  # reduces the margins to maximize the size of the pies
        fig.suptitle('Optimal allocation of resource')
        self.save_figure(fig, '_optimal_allocation')

    ''' miscellaneous '''

    def open_output_directory(self):
        """
        Opens the directory into which all the outputs have been placed.
        """

        operating_system = platform.system()
        if 'Windows' in operating_system:
            os.system('start  ' + self.out_dir_project)
        elif 'Darwin' in operating_system:
            os.system('open  ' + self.out_dir_project)
