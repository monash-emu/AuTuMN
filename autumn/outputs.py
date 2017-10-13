
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
    while working_time < end_time:
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


def plot_endtb_targets(ax, output, base_value, plot_colour):
    """
    Plot the End TB Targets and the direction that we need to head to achieve them.

    Args:
        ax: The axis to be plotted on to
        o: Output number
        output: Output string
        gtb_data: GTB data values
        plot_colour: List of colours for plotting
    """

    times = [2015., 2020., 2025., 2030., 2035.]
    if output == 'mortality':

        # hard coded to the End TB Targets
        target_values \
            = [base_value, base_value * .65, base_value * .25, base_value * .1, base_value * .05]

        # plot the individual targets themselves
        ax.plot(times[1:], target_values[1:],
                marker='o', markersize=4, color=plot_colour, markeredgewidth=0., linewidth=0.)

        # cycle through times and plot
        for t in range(len(times) - 1):
            times_to_plot, output_to_reach_target = find_times_from_exp_function(t, times, target_values)
            ax.plot(times_to_plot, output_to_reach_target, color=plot_colour, linewidth=.5)

    elif output == 'incidence':

        # hard coded to the End TB Targets
        target_values = [base_value, base_value * .8, base_value * .5, base_value * .2, base_value * .1]

        # plot the individual targets themselves
        ax.plot(times[1:], target_values[1:],
                marker='o', markersize=4, color=plot_colour, markeredgewidth=0., linewidth=0.)

        # cycle through times and plot
        for t in range(len(times) - 1):
            times_to_plot, output_to_reach_target = find_times_from_exp_function(t, times, target_values)
            ax.plot(times_to_plot, output_to_reach_target, color=plot_colour, linewidth=.5)


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

    # Should be redundant once Project module complete

    if png is not None:
        pylab.savefig(png, dpi=300)


def plot_comparative_age_parameters(data_strat_list, data_value_list, model_value_list, model_strat_list,
                                    parameter_name):

    # Get good tick labels from the stratum lists
    data_strat_labels = []
    for i in range(len(data_strat_list)):
        data_strat_labels += [t_k.turn_strat_into_label(data_strat_list[i])]
    model_strat_labels = []
    for i in range(len(model_strat_list)):
        model_strat_labels += [t_k.turn_strat_into_label(model_strat_list[i])]

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
            n = n + 1
            answer = find_smallest_factors_of_integer(n)
        i = i + 1

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
        y_sig_figs: The number of significant figures for the ticks
    Returns:
        labels: List of the modified tick labels
        axis_modifier: The text to be added to the axis
    """

    y_number_format = '%.' + str(y_sig_figs) + 'f'
    if max_val < 5e-9:
        labels = [y_number_format % (v * 1e12) for v in vals]
        axis_modifier = 'Trillionth '
    elif max_val < 5e-6:
        labels = [y_number_format % (v * 1e9) for v in vals]
        axis_modifier = 'Billionth '
    elif max_val < 5e-3:
        labels = [y_number_format % (v * 1e6) for v in vals]
        axis_modifier = 'Millionth '
    elif max_val < 5:
        labels = [y_number_format % (v * 1e3) for v in vals]
        axis_modifier = 'Thousandth '
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

        # find value to write from list and index
        value = working_list[param][median_run_index]

        # over-write existing parameter value if present
        param_found = False
        for row in country_sheet.rows:
            if row[0].value == param:
                row[1].value = value
                param_found = True

        # if parameter not found in existing spreadsheet, write into new row at the bottom
        if not param_found:
            max_row = country_sheet.max_row
            country_sheet.cell(row=max_row + 1, column=1).value = param
            country_sheet.cell(row=max_row + 1, column=2).value = value


class Project:
    def __init__(self, model_runner, gui_inputs):
        """
        Initialises an object of class Project, that will contain all the information (data + outputs) for writing a
        report for a country.

        Args:
            models: dictionary such as: models = {'baseline': model, 'scenario_1': model_1,  ...}
        """

        self.model_runner = model_runner
        self.inputs = self.model_runner.inputs
        self.gui_inputs = gui_inputs
        self.country = self.gui_inputs['country'].lower()
        self.name = 'test_' + self.country
        self.out_dir_project = os.path.join('projects', self.name)
        if not os.path.isdir(self.out_dir_project):
            os.makedirs(self.out_dir_project)

        self.figure_number = 1
        self.classifications = ['demo_', 'econ_', 'epi_prop_smear', 'epi_rr', 'program_prop_', 'program_timeperiod_',
                                'program_prop_novel', 'program_prop_treatment', 'program_prop_detect',
                                'int_prop_vaccination', 'program_prop_treatment_success',
                                'program_prop_treatment_death', 'transmission_modifier']
        self.output_colours = {}
        self.uncertainty_output_colours = {}
        self.program_colours = {}
        self.suptitle_size = 13
        self.classified_scaleups = {}
        self.grid = False
        self.plot_rejected_runs = False

        # Extract some characteristics from the models within model runner
        self.scenarios = self.gui_inputs['scenarios_to_run']
        self.scenario_names = self.gui_inputs['scenario_names_to_run']
        self.programs = self.inputs.interventions_to_cost
        self.gtb_available_outputs = ['incidence', 'mortality', 'prevalence', 'notifications']
        self.level_conversion_dict = {'lower_limit': '_lo', 'upper_limit': '_hi', 'point_estimate': ''}

        # to have a look at some individual vars scaling over time
        self.vars_to_view = []

        # comes up so often that we need to find this index, that best to do once in instantiation
        self.start_time_index \
            = t_k.find_first_list_element_at_least_value(self.model_runner.epi_outputs['manual_baseline']['times'],
                                                         self.inputs.model_constants['plot_start_time'])

        self.plot_true_outcomes = False

    ''' General methods for use below '''

    def find_years_to_write(self, scenario, output, epi=True):
        """
        Find years that need to be written into a spreadsheet or document.

        Args:
            scenario: Model scenario to be written.
            output: Epidemiological or economic output.
        """

        if epi:
            integer_dict = self.model_runner.epi_outputs_integer_dict[scenario][output]
        else:
            integer_dict = self.model_runner.cost_outputs_integer_dict[scenario]['raw_cost_' + output]

        requested_years = range(int(self.inputs.model_constants['report_start_time']),
                                int(self.inputs.model_constants['report_end_time']),
                                int(self.inputs.model_constants['report_step_time']))
        years = []
        for y in integer_dict:
            if y in requested_years: years += [y]
        return years

    def find_var_index(self, var):

        """
        Finds the index number for a var in the var arrays. (Arbitrarily uses the baseline model from the model runner.)

        Args:
            var: String for the var that we're looking for.

        Returns:
            The var's index (unnamed).

        """

        return self.model_runner.model_dict['manual_baseline'].var_labels.index(var)

    def find_start_index(self, scenario=None):

        """
        Very simple, but commonly used bit of code to determine whether to start from the start of the epidemiological
        outputs, or - if we're dealing with the baseline scenario - to start from the appropriate time index.

        Args:
            scenario: Scenario number (i.e. None for baseline or single integer for scenario)
        Returns:
            Index that can be used to find starting point in epidemiological output lists
        """

        if scenario:
            return 0
        else:
            return self.start_time_index

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
        Create axes for a figure with a single plot with a reasonable amount of space around.

        Returns:
            ax: The axes that can be plotted on
        """

        ax = fig.add_axes([0.1, 0.1, 0.6, 0.75])
        return ax

    def make_legend_to_single_axis(self, ax, scenario_handles, scenario_labels):

        """
        Standardised format to legend at side of single axis plot.

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
        Produces a standard set of line styles that isn't adapted to the data being plotted.

        Args:
            n: The number of line-styles
            return_all: Whether to return all of the styles up to n or just the last one
        Returns:
            line_styles: A list of standard line-styles, or if return_all is False,
                then the single item (for methods that are iterating through plots.
        """

        # iterate through a standard set of line styles
        for i in range(n):
            line_styles = []
            for line in ["-", ":", "-.", "--"]:
                for colour in "krbgmcy":
                    line_styles.append(line + colour)

        if return_all:
            return line_styles
        else:
            return line_styles[n - 1]

    def tidy_axis(self, ax, subplot_grid, title='', start_time=0., legend=False, x_label='', y_label='',
                  x_axis_type='time', y_axis_type='scaled', x_sig_figs=0, y_sig_figs=0):
        """
        Method to make cosmetic changes to a set of plot axes.
        """

        # add the sub-plot title with slightly larger titles than the rest of the text on the panel
        if title: ax.set_title(title, fontsize=get_nice_font_size(subplot_grid) + 2.)

        # add a legend if needed
        if legend == 'for_single':
            ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., frameon=False, prop={'size': 7})
        elif legend:
            ax.legend(fontsize=get_nice_font_size(subplot_grid), frameon=False)

        # sort x-axis
        if x_axis_type == 'time':
            ax.set_xlim((start_time, self.inputs.model_constants['plot_end_time']))
            ax.set_xticks(find_reasonable_year_ticks(start_time, self.inputs.model_constants['plot_end_time']))
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
            ax.set_xlim((start_time, self.inputs.model_constants['plot_end_time']))
            ax.set_xticks(range(int(start_time), int(self.inputs.model_constants['plot_end_time']), 1))
            for tick in ax.xaxis.get_major_ticks():
                tick.label.set_rotation(45)
        else:
            ax.set_xlabel(x_label, fontsize=get_nice_font_size(subplot_grid), labelpad=1)

        # sort y-axis
        ax.set_ylim(bottom=0.)
        vals = list(ax.get_yticks())
        max_val = max([abs(v) for v in vals])
        if y_axis_type == 'time':
            ax.set_ylim((start_time, self.inputs.model_constants['plot_end_time']))
            ax.set_yticks(find_reasonable_year_ticks(start_time, self.inputs.model_constants['plot_end_time']))
        elif y_axis_type == 'scaled':
            labels, axis_modifier = scale_axes(vals, max_val, y_sig_figs)
            ax.set_yticklabels(labels)
            ax.set_ylabel(axis_modifier + y_label, fontsize=get_nice_font_size(subplot_grid), labelpad=1)
        elif y_axis_type == 'proportion':
            ax.set_ylim(top=1.)
            ax.set_ylabel(y_label, fontsize=get_nice_font_size(subplot_grid), labelpad=1)
        else:
            ax.set_ylim(top=max_val * .95)
            ax.set_ylabel(y_label, fontsize=get_nice_font_size(subplot_grid), labelpad=1)

        # set size of font for x-ticks and add a grid if requested
        for axis_to_change in [ax.xaxis, ax.yaxis]:
            for tick in axis_to_change.get_major_ticks():
                tick.label.set_fontsize(get_nice_font_size(subplot_grid))
            axis_to_change.grid(self.grid)

    def save_figure(self, fig, last_part_of_name_for_figure):
        """
        Simple method to standardise names for output figure files.

        Args:
            last_part_of_name_for_figure: The part of the figure name that is variable and input from the
                plotting method.
        """

        png = os.path.join(self.out_dir_project, self.country + last_part_of_name_for_figure + '.png')
        fig.savefig(png, dpi=300)

    def save_opti_figure(self, fig, last_part_of_name_for_figure):

        """
        Same as previous method, when applied to optimisation outputs.
        """

        png = os.path.join(self.model_runner.opti_outputs_dir, self.country + last_part_of_name_for_figure + '.png')
        fig.savefig(png, dpi=300)

    ''' Methods for outputting to Office applications '''

    def master_outputs_runner(self):
        """
        Method to work through all the fundamental output methods, which then call all the specific output
        methods for plotting and writing as required.
        """

        # write automatic calibration values back to sheets
        if self.gui_inputs['output_uncertainty'] and self.gui_inputs['write_uncertainty_outcome_params']:
            self.write_automatic_calibration_outputs()

        # write spreadsheets - with sheet for each scenario or each output
        if self.gui_inputs['output_spreadsheets']:
            if self.gui_inputs['output_by_scenario']:
                print('Writing scenario spreadsheets')
                self.write_xls_by_scenario()
            else:
                print('Writing output indicator spreadsheets')
                self.write_xls_by_output()

        # write documents - with document for each scenario or each output
        if self.gui_inputs['output_documents']:
            if self.gui_inputs['output_by_scenario']:
                print('Writing scenario documents')
                self.write_docs_by_scenario()
            else:
                print('Writing output indicator documents')
                self.write_docs_by_output()

        # write optimisation spreadsheets
        self.write_opti_outputs_spreadsheet()

        # master plotting method
        self.run_plotting()

        # open the directory to which everything has been written to save the user a click or two
        self.open_output_directory()

    def write_automatic_calibration_outputs(self):
        """
        Write median values from automatic calbiration process back to input spreadsheets, using the parameter values
        that were associated with the median model output value.
        """

        try:
            path = os.path.join('autumn/xls/data_' + self.country + '.xlsx')
        except:
            print('No country input spreadsheet for requested uncertainty parameter writing')
        else:
            print('Writing calibration parameters back to input spreadsheet')

            # open workbook and sheet
            country_input_book = xl.load_workbook(path)
            country_sheet = country_input_book['constants']

            # find the position of the median value by incidence
            list_to_find_median \
                = list(self.model_runner.epi_outputs_uncertainty['uncertainty_baseline']['incidence'][
                           self.model_runner.accepted_indices,
                           t_k.find_first_list_element_at_least_value(
                               self.model_runner.epi_outputs_uncertainty['uncertainty_baseline']['times'],
                               self.inputs.model_constants['current_time'])])

            # find index of the median value - not very happy with this code yet
            median_run_index \
                = t_k.find_list_element_equal_to(
                list_to_find_median, numpy.percentile(list_to_find_median, 50, interpolation='nearest'))

            # write the parameters and starting compartment sizes back in to input sheets
            write_param_to_sheet(country_sheet, self.model_runner.all_parameters_tried, median_run_index)
            write_param_to_sheet(country_sheet, self.model_runner.all_compartment_values_tried, median_run_index)
            write_param_to_sheet(country_sheet, self.model_runner.all_other_adjustments_made, median_run_index)

            # save
            country_input_book.save(path)

    def write_xls_by_scenario(self):
        """
        Write a spreadsheet with the sheet referring to one scenario.
        """

        # whether to write horizontally
        horizontal = self.gui_inputs['output_horizontally']

        # write a new file for each scenario and for each general type of output
        for result_type in ['epi_', 'raw_cost_', 'inflated_cost_', 'discounted_cost_', 'discounted_inflated_cost_']:
            for scenario in self.scenario_names:

                # make filename
                path = os.path.join(self.out_dir_project, result_type + scenario)
                path += '.xlsx'

                # get active sheet
                wb = xl.Workbook()
                sheet = wb.active
                sheet.title = scenario

                # write the year text cell
                sheet.cell(row=1, column=1).value = 'Year'

                # for epidemiological outputs (for which uncertainty is fully finished)
                if result_type == 'epi_':
                    for output in self.model_runner.epi_outputs_to_analyse:

                        # Find years to write
                        years = self.find_years_to_write('manual_' + scenario, output, epi=True)

                        # Write the year column
                        for y, year in enumerate(years):
                            row = y + 2
                            column = 1
                            if horizontal: column, row = row, column
                            sheet.cell(row=row, column=column).value = year

                        # Cycle over outputs
                        for o, output in enumerate(self.model_runner.epi_outputs_to_analyse):

                            # Without uncertainty
                            if not self.gui_inputs['output_uncertainty']:

                                # Write output names across first row
                                row = 1
                                column = o + 2
                                if horizontal: column, row = row, column
                                sheet.cell(row=row, column=column).value = t_k.capitalise_and_remove_underscore(output)

                                # Write columns of data
                                for y, year in enumerate(years):
                                    row = y + 2
                                    column = o + 2
                                    if horizontal: column, row = row, column
                                    sheet.cell(row=row, column=column).value \
                                        = self.model_runner.epi_outputs_integer_dict['manual_' + scenario][output][
                                        year]

                            # with uncertainty
                            else:

                                # 1, 0, 2 indicates point estimate, lower limit, upper limit
                                order_to_write = [1, 0, 2]

                                # write the scenario names and confidence interval titles
                                row = 1
                                column = o * 3 + 2
                                if horizontal: column, row = row, column
                                sheet.cell(row=row, column=column).value = t_k.capitalise_and_remove_underscore(output)
                                row = 1
                                column = o * 3 + 3
                                if horizontal: column, row = row, column
                                sheet.cell(row=row, column=column).value = 'Lower'
                                row = 1
                                column = o * 3 + 4
                                if horizontal: column, row = row, column
                                sheet.cell(row=row, column=column).value = 'Upper'

                                # write the columns of data
                                for y, year in enumerate(years):
                                    year_index \
                                        = t_k.find_first_list_element_at_least_value(
                                        self.model_runner.epi_outputs['uncertainty_' + scenario]['times'], year)
                                    for ord, order in enumerate(order_to_write):
                                        row = y + 2
                                        column = o * 3 + 2 + ord
                                        if horizontal: column, row = row, column
                                        sheet.cell(row=row, column=column).value \
                                            = self.model_runner.epi_outputs_uncertainty_centiles['uncertainty_'
                                                                                                 + scenario][
                                            output][order,
                                                    year_index]

                # for economic outputs (uncertainty not yet fully finished)
                elif 'cost_' in result_type:
                    for output in self.inputs.interventions_to_cost:

                        # find years to write
                        years = self.find_years_to_write('manual_' + scenario, output, epi=False)

                        # write the year cell
                        sheet.cell(row=1, column=1).value = 'Year'

                        # Write the year text column
                        for y, year in enumerate(years):
                            row = y + 2
                            column = 1
                            if horizontal: column, row = row, column
                            sheet.cell(row=row, column=column).value = year

                        # For each intervention
                        for o, output in enumerate(self.inputs.interventions_to_cost):

                            # Write output names across the top
                            row = 1
                            column = o + 2
                            if horizontal: column, row = row, column
                            sheet.cell(row=row, column=column).value = t_k.capitalise_and_remove_underscore(output)

                            # Write the columns of data
                            for y, year in enumerate(years):
                                row = y + 2
                                column = o + 2
                                if horizontal: column, row = row, column
                                sheet.cell(row=row, column=column).value \
                                    = self.model_runner.cost_outputs_integer_dict['manual_' + scenario][result_type + output][year]

                # save workbook
                wb.save(path)

    def write_xls_by_output(self):
        """
        Write a spreadsheet with the sheet referring to one output.
        """

        # Whether to write horizontally
        horizontal = self.gui_inputs['output_horizontally']

        # Write a new file for each output
        for output in self.model_runner.epi_outputs_to_analyse:

            # Make filename
            path = os.path.join(self.out_dir_project, 'epi_' + output)
            path += '.xlsx'

            # Get active sheet
            wb = xl.Workbook()
            sheet = wb.active
            sheet.title = output

            # For each scenario
            for scenario in self.scenario_names:

                # Find years to write
                years = self.find_years_to_write('manual_' + scenario, output, epi=True)

                # Write the year text cell
                sheet.cell(row=1, column=1).value = 'Year'

                # Write the year column
                for y, year in enumerate(years):
                    row = y + 2
                    column = 1
                    if horizontal: column, row = row, column
                    sheet.cell(row=row, column=column).value = year

                # Cycle over scenarios
                for s, scenario in enumerate(self.scenario_names):

                    # Without uncertainty
                    if not self.gui_inputs['output_uncertainty']:

                        # Write scneario names across first row
                        row = 1
                        column = s + 2
                        if horizontal: column, row = row, column
                        sheet.cell(row=row, column=column).value = \
                            t_k.capitalise_and_remove_underscore(scenario)

                        # Write columns of data
                        for y, year in enumerate(years):
                            row = y + 2
                            column = s + 2
                            if horizontal: column, row = row, column
                            sheet.cell(row=row, column=column).value \
                                = self.model_runner.epi_outputs_integer_dict['manual_' + scenario][output][year]

                    # With uncertainty
                    else:
                        # 1, 0, 2 indicates point estimate, lower limit, upper limit
                        order_to_write = [1, 0, 2]

                        # Write the scenario names and confidence interval titles
                        row = 1
                        column = s * 3 + 2
                        if horizontal: column, row = row, column
                        sheet.cell(row=row, column=column).value = \
                            t_k.capitalise_and_remove_underscore(scenario)
                        row = 1
                        column = s * 3 + 3
                        if horizontal: column, row = row, column
                        sheet.cell(row=row, column=column).value = 'Lower'
                        row = 1
                        column = s * 3 + 4
                        if horizontal: column, row = row, column
                        sheet.cell(row=row, column=column).value = 'Upper'

                        # Write the columns of data
                        for y, year in enumerate(years):
                            year_index \
                                = t_k.find_first_list_element_at_least_value(
                                self.model_runner.epi_outputs['uncertainty_' + scenario]['times'], year)
                            for o, order in enumerate(order_to_write):
                                row = y + 2
                                column = s * 3 + 2 + o
                                if horizontal: column, row = row, column
                                sheet.cell(row=row, column=column).value \
                                    = self.model_runner.epi_outputs_uncertainty_centiles[
                                    'uncertainty_' + scenario][output][order, year_index]

            # Save workbook
            wb.save(path)

        for output in self.inputs.interventions_to_cost[None]:

            years = self.find_years_to_write('manual_baseline', output, epi=False)

            for cost_type in ['raw_cost_', 'inflated_cost_', 'discounted_cost_', 'discounted_inflated_cost_']:

                # Make filename
                path = os.path.join(self.out_dir_project, cost_type + output)
                path += '.xlsx'

                # Get active sheet
                wb = xl.Workbook()
                sheet = wb.active
                sheet.title = output

                # Write the year text cell
                sheet.cell(row=1, column=1).value = 'Year'

                # Write the year text column
                for y, year in enumerate(years):
                    row = y + 2
                    column = 1
                    if horizontal: column, row = row, column
                    sheet.cell(row=row, column=column).value = year

                # Cycle over scenarios
                for s, scenario in enumerate(self.scenario_names):

                    # Write the scenario names
                    row = 1
                    column = s + 2
                    if horizontal: column, row = row, column
                    sheet.cell(row=row, column=column).value \
                        = t_k.capitalise_and_remove_underscore(scenario)

                    # Write the columns of data
                    for y, year in enumerate(years):
                        row = y + 2
                        column = s + 2
                        if horizontal: column, row = row, column
                        sheet.cell(row=row, column=column).value \
                            = self.model_runner.cost_outputs_integer_dict['manual_'
                                                                          + scenario][cost_type + output][year]

                # Save workbook
                wb.save(path)

    def write_docs_by_output(self):
        """
        Write word documents using the docx package. Writes with or without uncertainty according to whether Run
        uncertainty selected in the GUI.
        """

        # Write a new file for each output
        for output in self.model_runner.epi_outputs_to_analyse:

            # Initialise document
            path = os.path.join(self.out_dir_project, output)
            path += ".docx"

            # Find years to write
            years = self.find_years_to_write('manual_baseline', output, epi=True)

            # Make table
            document = Document()
            table = document.add_table(rows=len(years) + 1, cols=len(self.scenario_names) + 1)

            for s, scenario in enumerate(self.scenario_names):

                # Write outputs across the top
                row_cells = table.rows[0].cells
                row_cells[0].text = 'Year'
                row_cells[s + 1].text = t_k.capitalise_and_remove_underscore(scenario)

                for y, year in enumerate(years):
                    year_index \
                        = t_k.find_first_list_element_at_least_value(
                        self.model_runner.epi_outputs['uncertainty_' + scenario]['times'], year)
                    row_cells = table.rows[y + 1].cells
                    row_cells[0].text = str(year)
                    if self.gui_inputs['output_uncertainty']:
                        (lower_limit, point_estimate, upper_limit) = self.model_runner.epi_outputs_uncertainty_centiles[
                            'uncertainty_' + scenario][output][0:3, year_index]
                        row_cells[s + 1].text = '%.2f (%.2f to %.2f)' % (point_estimate, lower_limit, upper_limit)
                    else:
                        point_estimate = self.model_runner.epi_outputs_integer_dict['manual_' + scenario][output][year]
                        row_cells[s + 1].text = '%.2f' % point_estimate

            # Save document
            document.save(path)

    def write_docs_by_scenario(self):
        """
        Write word documents using the docx package. Writes with or without uncertainty according to whether Run
        uncertainty selected in the GUI.
        """

        for scenario in self.scenario_names:

            # Initialise document
            path = os.path.join(self.out_dir_project, scenario)
            path += ".docx"

            # Find years to write
            years = self.find_years_to_write('manual_' + scenario, 'population', epi=True)

            # Make table
            document = Document()
            table = document.add_table(rows=len(years) + 1, cols=len(self.model_runner.epi_outputs_to_analyse) + 1)

            # Only working for epidemiological outputs
            for o, output in enumerate(self.model_runner.epi_outputs_to_analyse):

                # Write outputs across the top
                row_cells = table.rows[0].cells
                row_cells[0].text = 'Year'
                row_cells[o + 1].text = t_k.capitalise_and_remove_underscore(output)

                for y, year in enumerate(years):
                    year_index = t_k.find_first_list_element_at_least_value(
                        self.model_runner.epi_outputs['uncertainty_' + scenario]['times'], year)

                    row_cells = table.rows[y + 1].cells
                    row_cells[0].text = str(year)
                    if self.gui_inputs['output_uncertainty']:
                        (lower_limit, point_estimate, upper_limit) = self.model_runner.epi_outputs_uncertainty_centiles[
                            'uncertainty_' + scenario][output][0:3, year_index]
                        row_cells[o + 1].text = '%.2f (%.2f to %.2f)' % (point_estimate, lower_limit, upper_limit)
                    else:
                        point_estimate = self.model_runner.epi_outputs_integer_dict['manual_' + scenario][output][year]
                        row_cells[o + 1].text = '%.2f' % point_estimate

            # Save document
            document.save(path)

    def write_opti_outputs_spreadsheet(self):

        if self.model_runner.optimisation:

            # Make filename
            path = os.path.join(self.model_runner.opti_outputs_dir, 'opti_results.xlsx')

            # Get active sheet
            wb = xl.Workbook()
            sheet = wb.active
            sheet.title = 'optimisation'

            # Write row names
            sheet.cell(row=1, column=1).value = 'envelope'
            sheet.cell(row=2, column=1).value = 'incidence'
            sheet.cell(row=3, column=1).value = 'mortality'
            n_row = 3
            row_index = {}
            for intervention in self.model_runner.interventions_considered_for_opti:
                n_row += 1
                sheet.cell(row=n_row, column=1).value = intervention
                row_index[intervention] = n_row

            # Populate cells with content
            n_col = 1
            for env, envelope in enumerate(self.model_runner.opti_results['annual_envelope']):
                n_col += 1
                sheet.cell(row=1, column=n_col).value = envelope
                sheet.cell(row=2, column=n_col).value = self.model_runner.opti_results['incidence'][env]
                sheet.cell(row=3, column=n_col).value = self.model_runner.opti_results['mortality'][env]
                for intervention in self.model_runner.opti_results['best_allocation'][env].keys():
                    sheet.cell(row=row_index[intervention], column=n_col).value = \
                        self.model_runner.opti_results['best_allocation'][env][intervention]

            # Save workbook
            wb.save(path)

    ''' Plotting methods '''

    def run_plotting(self):
        """
        Master plotting method to call all the methods that produce specific plots.
        """

        # find some general output colours
        output_colours = self.make_default_line_styles(5, True)
        for s, scenario in enumerate(self.scenarios):
            self.output_colours[scenario] = output_colours[s]
            self.program_colours[scenario] = {}
            for p, program in enumerate(self.programs[scenario]):
                # +1 is to avoid starting from black, which doesn't look as nice for programs as for baseline scenario
                self.program_colours[scenario][program] = output_colours[p + 1]

        # plot main outputs
        if self.gui_inputs['output_gtb_plots']:
            self.plot_outputs_against_gtb(self.gtb_available_outputs, ci_plot=None)
            if self.gui_inputs['output_uncertainty'] or self.inputs.intervention_uncertainty:
                self.plot_outputs_against_gtb(self.gtb_available_outputs, ci_plot=True)
                self.plot_outputs_against_gtb(self.gtb_available_outputs, ci_plot=False)
                self.plot_shaded_outputs_gtb(self.gtb_available_outputs)
            if self.gui_inputs['n_strains'] > 1:
                self.plot_resistant_strain_outputs(['incidence', 'mortality', 'prevalence', 'perc_incidence'])

        # plot scale-up functions - currently only doing this for the baseline model run
        if self.gui_inputs['output_scaleups']:
            if self.vars_to_view:
                self.individual_var_viewer()
            self.classify_scaleups()
            self.plot_scaleup_fns_against_data()
            self.plot_programmatic_scaleups()

            # not technically a scale-up function in the same sense, but put in here anyway
            self.plot_force_infection()

        # plot mixing matrix if relevant
        # if self.model_runner.model_dict['manual_baseline'].vary_force_infection_by_riskgroup \
        #         and len(self.inputs.riskgroups) > 1:
        #     self.plot_mixing_matrix()

        # plot economic outputs
        if self.gui_inputs['output_plot_economics']:
            self.plot_cost_coverage_curves()
            self.plot_cost_over_time()
            # self.plot_intervention_costs_by_scenario(2015, 2030)
            self.plot_cost_over_time_stacked_bars()

        # plot compartment population sizes
        if self.gui_inputs['output_compartment_populations']:
            self.plot_populations()

        # plot fractions
        if self.gui_inputs['output_fractions']:
            self.plot_fractions('strain')

        # plot outputs by age group
        if self.gui_inputs['output_by_subgroups']:
            self.plot_outputs_by_stratum()
            self.plot_outputs_by_stratum(strata_string='riskgroups', outputs_to_plot=['incidence', 'prevalence'])
            self.plot_proportion_cases_by_stratum()

        # plot proportions of population
        if self.gui_inputs['output_age_fractions']:
            self.plot_stratified_populations(age_or_risk='age')

        # plot risk group proportions
        if self.gui_inputs['output_riskgroup_fractions']:
            self.plot_stratified_populations(age_or_risk='risk')

        # make a flow-diagram
        if self.gui_inputs['output_flow_diagram']:
            png = os.path.join(self.out_dir_project, self.country + '_flow_diagram' + '.png')
            self.model_runner.model_dict['manual_baseline'].make_flow_diagram(png)

        # plot risk group proportions
        if self.gui_inputs['output_plot_riskgroup_checks'] \
                and len(self.model_runner.model_dict['manual_baseline'].riskgroups) > 1:
            self.plot_riskgroup_checks()

        # save figure that is produced in the uncertainty running process
        if self.gui_inputs['output_param_plots']:
            param_tracking_figure = self.set_and_update_figure()
            param_tracking_figure = self.model_runner.plot_progressive_parameters_tk(from_runner=False,
                                                                                     input_figure=param_tracking_figure)
            self.save_figure(param_tracking_figure, '_param_tracking')
            self.plot_param_histograms()

        # plot popsizes for checking cost-coverage curves
        if self.gui_inputs['output_popsize_plot']:
            self.plot_popsizes()

        # plot likelihood estimates
        if self.gui_inputs['output_likelihood_plot']:
            self.plot_likelihoods()

        # optimisation plotting
        if self.model_runner.optimisation:
            self.plot_optimised_epi_outputs()
            self.plot_piecharts_opti()

    def plot_outputs_against_gtb(self, outputs, ci_plot=None, plot_targets=True):
        """
        Produces the plot for the main outputs, loops over multiple scenarios.

        Args:
            outputs: A list of the outputs to be plotted.
            ci_plot: Whether to plot uncertainty intervals around the estimates generated from uncertainty runs.
        """

        # standard preliminaries
        start_time = self.inputs.model_constants['plot_start_time']
        colour, indices, yaxis_label, title, patch_colour = find_standard_output_styles(outputs, lightening_factor=0.3)
        subplot_grid = find_subplot_numbers(len(outputs))
        fig = self.set_and_update_figure()

        # loop through indicators
        for o, output in enumerate(outputs):

            # preliminaries
            ax = fig.add_subplot(subplot_grid[0], subplot_grid[1], o + 1)

            ''' plotting GTB data in background '''

            gtb_data = {}
            gtb_data_lists = {}

            # notifications
            if output == 'notifications':
                gtb_data['point_estimate'] = self.inputs.original_data['notifications']['c_newinc']

            # extract the relevant data from the Global TB Report and use to plot a patch (for inc, prev and mortality)
            elif output in self.gtb_available_outputs:
                for level in self.level_conversion_dict:
                    gtb_data[level] = self.inputs.original_data['tb'][indices[o] + self.level_conversion_dict[level]]
                    gtb_data_lists.update(extract_dict_to_list_key_ordering(gtb_data[level], level))
                ax.add_patch(patches.Polygon(
                    create_patch_from_list(
                        gtb_data_lists['times'], gtb_data_lists['lower_limit'], gtb_data_lists['upper_limit']),
                    color=patch_colour[o]))

            # plot point estimates
            if output in self.gtb_available_outputs:
                ax.plot(gtb_data['point_estimate'].keys(), gtb_data['point_estimate'].values(), color=colour[o],
                        linewidth=0.5, label=None)

                # plot the targets (and milestones) and the fitted exponential function to achieve them
                base_value = gtb_data['point_estimate'][2014]  # should be 2015, but data not yet incorporated
                if plot_targets: plot_endtb_targets(ax, output, base_value, patch_colour[o])

            ''' plotting modelled data '''

            # plot scenarios without uncertainty
            if ci_plot is None:

                # plot model estimates
                for scenario in self.scenarios[::-1]:  # reversing to ensure black baseline plotted over top
                    scenario_name = t_k.find_scenario_string_from_number(scenario)
                    start_index = self.find_start_index(scenario)
                    ax.plot(self.model_runner.epi_outputs['manual_' + scenario_name]['times'][start_index:],
                            self.model_runner.epi_outputs['manual_' + scenario_name][output][start_index:],
                            color=self.output_colours[scenario][1], linestyle=self.output_colours[scenario][0],
                            linewidth=1.5, label=t_k.capitalise_and_remove_underscore(scenario_name))

                # plot "true" mortality
                if output == 'mortality' and self.plot_true_outcomes:
                    ax.plot(self.model_runner.epi_outputs['manual_' + scenario_name]['times'][start_index:],
                            self.model_runner.epi_outputs['manual_' + scenario_name]['true_' + output][start_index:],
                            color=self.output_colours[scenario][1], linestyle=':', linewidth=1)
                end_filename = '_scenario'

            # plot with uncertainty confidence intervals
            elif ci_plot and (self.gui_inputs['output_uncertainty'] or self.inputs.intervention_uncertainty):
                if self.inputs.intervention_uncertainty:
                    scenarios = [None, 15]
                    uncertainty_type = 'intervention_uncertainty'
                    start_index = 0
                    linewidth = 1.
                    linecolour = 'r'
                else:
                    scenarios = self.scenarios
                    linewidth = 1.5

                for scenario in scenarios[::-1]:
                    scenario_name = t_k.find_scenario_string_from_number(scenario)
                    if not self.inputs.intervention_uncertainty:
                        uncertainty_type = 'uncertainty_' + scenario_name
                        start_index = self.find_start_index(scenario)
                        linecolour = self.output_colours[scenario][1]

                    # median
                    ax.plot(
                        self.model_runner.epi_outputs_uncertainty[uncertainty_type]['times'][start_index:],
                        self.model_runner.epi_outputs_uncertainty_centiles[uncertainty_type][output][
                            self.model_runner.percentiles.index(50), :][start_index:],
                        color=linecolour, linestyle=self.output_colours[scenario][0],
                        linewidth=linewidth, label=t_k.capitalise_and_remove_underscore(scenario_name))

                    # upper and lower confidence bounds
                    for centile in [2.5, 97.5]:
                        ax.plot(
                            self.model_runner.epi_outputs_uncertainty[uncertainty_type]['times'][start_index:],
                            self.model_runner.epi_outputs_uncertainty_centiles[uncertainty_type][output][
                                self.model_runner.percentiles.index(centile), :][start_index:],
                            color=linecolour, linestyle='--', linewidth=.5, label=None)
                    end_filename = '_ci'

            # plot progressive model run outputs for uncertainty analyses
            elif self.gui_inputs['output_uncertainty'] or self.inputs.intervention_uncertainty:

                # get relevant data according to whether intervention or baseline uncertainty is being run
                if self.inputs.intervention_uncertainty:
                    runs = self.inputs.n_samples
                    self.start_time_index = 0
                    uncertainty_type = 'intervention_uncertainty'
                else:
                    uncertainty_type = 'uncertainty_baseline'
                    runs = len(self.model_runner.epi_outputs_uncertainty[uncertainty_type][output])

                # plot the runs
                for run in range(runs):
                    if run not in self.model_runner.accepted_indices and self.plot_rejected_runs:
                        ax.plot(self.model_runner.epi_outputs_uncertainty[
                                    uncertainty_type]['times'][self.start_time_index:],
                                self.model_runner.epi_outputs_uncertainty[
                                    uncertainty_type][output][run, self.start_time_index:],
                                linewidth=.2, color='y', label=t_k.capitalise_and_remove_underscore('baseline'))
                    else:
                        ax.plot(self.model_runner.epi_outputs_uncertainty[
                                    uncertainty_type]['times'][self.start_time_index:],
                                self.model_runner.epi_outputs_uncertainty[
                                    uncertainty_type][output][run, self.start_time_index:],
                                linewidth=1.2,
                                color=str(1. - float(run) / float(len(
                                    self.model_runner.epi_outputs_uncertainty[uncertainty_type][output]))),
                                label=t_k.capitalise_and_remove_underscore('baseline'))
                    end_filename = '_progress'

            if self.inputs.intervention_uncertainty:
                ax.plot(self.model_runner.epi_outputs['manual_baseline']['times'][self.start_time_index:],
                        self.model_runner.epi_outputs['manual_baseline'][output][self.start_time_index:],
                        color='k', linewidth=1.2)

            self.tidy_axis(ax, subplot_grid, title=title[o], start_time=start_time,
                           legend=(o == len(outputs) - 1 and len(self.scenarios) > 1
                                   and not self.inputs.intervention_uncertainty),
                           y_axis_type='raw', y_label=yaxis_label[o])

        # add main title and save
        fig.suptitle(t_k.capitalise_first_letter(self.country) + ' model outputs', fontsize=self.suptitle_size)
        self.save_figure(fig, '_gtb' + end_filename)

    def plot_shaded_outputs_gtb(self, outputs, ci_plot=False, gtb_ci_plot='hatch', plot_targets=True):
        """
        Creates visualisation of uncertainty outputs with density of shading proportional to the number of model runs
        that went through a certain output value. Similar to our plotting approach for the award-winning figure in
        American Journal of Epidemiology.

        Args:
            outputs: The types of output to plot
            ci_plot: Whether to add dotted lines at the edges of the shaded modelled output areas
            gtb_ci_plot: How to overlay the GTB data - either 'hatch' for hatched area, or 'patch' for translucent area
        """

        # standard preliminaries
        start_time = self.inputs.model_constants['plot_start_time']
        colour, indices, yaxis_label, title, patch_colour = find_standard_output_styles(outputs, lightening_factor=0.3)
        subplot_grid = find_subplot_numbers(len(outputs))
        fig = self.set_and_update_figure()

        # loop through indicators
        for o, output in enumerate(outputs):

            # preliminaries
            ax = fig.add_subplot(subplot_grid[0], subplot_grid[1], o + 1)

            ''' plot modelled data '''

            # plot with uncertainty confidence intervals
            scenario_name = 'baseline'
            start_index = self.find_start_index(None)

            if self.inputs.intervention_uncertainty:
                uncertainty_type = 'intervention_uncertainty'
                start_index = 0
            else:
                uncertainty_type = 'uncertainty_baseline'

            # overlay median and upper and lower CIs if requested
            if ci_plot:
                ax.plot(
                    self.model_runner.epi_outputs_uncertainty[uncertainty_type]['times'][start_index:],
                    self.model_runner.epi_outputs_uncertainty_centiles[uncertainty_type][
                        output][self.model_runner.percentiles.index(50), :][start_index:],
                    color=self.output_colours[None][1], linestyle=self.output_colours[None][0],
                    linewidth=1.5, label=t_k.capitalise_and_remove_underscore(scenario_name))
                for centile in [2.5, 97.5]:
                    ax.plot(
                        self.model_runner.epi_outputs_uncertainty[uncertainty_type]['times'][
                            start_index:],
                        self.model_runner.epi_outputs_uncertainty_centiles[uncertainty_type][output][
                        self.model_runner.percentiles.index(centile), :][start_index:],
                        color=self.output_colours[None][1], linestyle='--', linewidth=.5, label=None)

            # plot shaded areas as patches
            progressive_patch_colours \
                = [cm.Blues(x) for x in numpy.linspace(0., 1., self.model_runner.n_centiles_for_shading)]
            for i in range(self.model_runner.n_centiles_for_shading):
                patch = create_patch_from_list(
                    self.model_runner.epi_outputs_uncertainty[uncertainty_type]['times'][start_index:],
                    self.model_runner.epi_outputs_uncertainty_centiles[uncertainty_type][output][i+3, :][start_index:],
                    self.model_runner.epi_outputs_uncertainty_centiles[uncertainty_type][output][-i-1, :][start_index:])
                ax.add_patch(patches.Polygon(patch, color=progressive_patch_colours[i]))

            if self.inputs.intervention_uncertainty:
                ax.plot(self.model_runner.epi_outputs['manual_baseline']['times'][self.start_time_index:],
                        self.model_runner.epi_outputs['manual_baseline'][output][self.start_time_index:],
                        color='k', linewidth=1.2)

            ''' plotting GTB data in background '''

            gtb_data = {}
            if gtb_ci_plot == 'hatch':
                alpha = 1.
            elif gtb_ci_plot == 'patch':
                alpha = .2

            # notifications
            if output == 'notifications':
                gtb_data['point_estimate'] = self.inputs.original_data['notifications']['c_newinc']

            # extract the relevant data from the Global TB Report and use to plot a patch (for inc, prev and mortality)
            elif output in self.gtb_available_outputs:
                gtb_data_lists = {}
                for level in self.level_conversion_dict:
                    gtb_data[level] = self.inputs.original_data['tb'][indices[o] + self.level_conversion_dict[level]]
                    gtb_data_lists.update(extract_dict_to_list_key_ordering(gtb_data[level], level))
                if gtb_ci_plot == 'patch':
                    ax.add_patch(patches.Polygon(
                        create_patch_from_list(gtb_data_lists['times'],
                                               gtb_data_lists['lower_limit'],
                                               gtb_data_lists['upper_limit']),
                        color=patch_colour[o], alpha=alpha))
                elif gtb_ci_plot == 'hatch':
                    ax.add_patch(patches.Polygon(
                        create_patch_from_list(gtb_data_lists['times'],
                                               gtb_data_lists['lower_limit'],
                                               gtb_data_lists['upper_limit']),
                        color='.3', hatch='/', fill=False, linewidth=0.))

            # plot point estimates
            if output in self.gtb_available_outputs:
                ax.plot(gtb_data['point_estimate'].keys(), gtb_data['point_estimate'].values(),
                        color='.3', linewidth=0.8, label=None, alpha=alpha)
                if gtb_ci_plot == 'hatch' and output != 'notifications':
                    for limit in ['lower_limit', 'upper_limit']:
                        ax.plot(gtb_data['upper_limit'].keys(), gtb_data[limit].values(),
                                color='.3', linewidth=0.3, label=None, alpha=alpha)

                # plot the targets (and milestones) and the fitted exponential function to achieve them
                base_value = gtb_data['point_estimate'][2014]  # should be 2015, but data not yet available
                if plot_targets: plot_endtb_targets(ax, output, base_value, '.7')

            self.tidy_axis(ax, subplot_grid, title=title[o], start_time=start_time,
                           legend=(o == len(outputs) - 1 and len(self.scenarios) > 1),
                           y_axis_type='raw', y_label=yaxis_label[o])

        # add main title and save
        fig.suptitle(t_k.capitalise_first_letter(self.country) + ' model outputs', fontsize=self.suptitle_size)
        self.save_figure(fig, '_gtb_shaded')

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
                scenario_name = t_k.find_scenario_string_from_number(scenario)
                ax.plot(self.model_runner.epi_outputs['manual_' + scenario_name]['times'],
                        self.model_runner.epi_outputs['manual_' + scenario_name][output + '_mdr'],
                        color=self.output_colours[scenario][1], linestyle=self.output_colours[scenario][0])
            self.tidy_axis(ax, subplot_grid, title=title[o], y_label=yaxis_label[o],
                           start_time=self.inputs.model_constants['recent_time'],
                           legend=(o == len(outputs) - 1))

        # finish off
        fig.suptitle(t_k.capitalise_first_letter(self.country) + ' resistant strain outputs',
                     fontsize=self.suptitle_size)
        self.save_figure(fig, '_resistant_strain')

    def classify_scaleups(self):
        """
        Classifies the time variant parameters according to their type (e.g. programmatic, economic, demographic, etc.).
        """

        for classification in self.classifications:
            self.classified_scaleups[classification] = []
            for fn in self.model_runner.model_dict['manual_baseline'].scaleup_fns:
                if classification in fn:
                    self.classified_scaleups[classification] += [fn]
        self.classified_scaleups['misc'] = ['program_prop_firstline_dst', 'program_prop_algorithm_sensitivity']

    def individual_var_viewer(self):

        """
        Function that can be used to visualise a particular var or several vars, by adding them to the function input
        list, which is now an attribute of this object (i.e. vars_to_view).
        """

        for function in self.vars_to_view:
            fig = self.set_and_update_figure()
            ax = fig.add_subplot(1, 1, 1)
            for scenario in reversed(self.scenarios):
                scenario_name = t_k.find_scenario_string_from_number(scenario)
                ax.plot(self.model_runner.model_dict['manual_' + scenario_name].times,
                        self.model_runner.model_dict['manual_' + scenario_name].get_var_soln(function),
                        color=self.output_colours[scenario][1])
            self.tidy_axis(ax, [1, 1], start_time=self.inputs.model_constants['plot_start_time'])
            fig.suptitle(t_k.find_title_from_dictionary(function))
            self.save_figure(fig, '_var_' + function)

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
                    = self.inputs.model_constants['plot_start_time'], self.inputs.model_constants['plot_end_time']
                x_vals = numpy.linspace(start_time, end_time, 1e3)

                # iterate through functions
                for f, function in enumerate(function_list):

                    # initialise axis
                    ax = fig.add_subplot(subplot_grid[0], subplot_grid[1], f + 1)

                    # iterate through the scenarios
                    for scenario in reversed(self.scenarios):
                        scenario_name = t_k.find_scenario_string_from_number(scenario)

                        # line plot of scaling parameter functions
                        ax.plot(x_vals,
                                map(self.model_runner.model_dict['manual_' + scenario_name].scaleup_fns[function],
                                    x_vals),
                                color=self.output_colours[scenario][1],
                                label=t_k.capitalise_and_remove_underscore(scenario_name))

                    # plot the raw data from which the scale-up functions were produced
                    if function in self.inputs.scaleup_data[None]:
                        data_to_plot = self.inputs.scaleup_data[None][function]
                        ax.scatter(data_to_plot.keys(), data_to_plot.values(), color='k', s=6)

                    # adjust tick font size and add panel title
                    if 'prop_treatment_death' in function:
                        y_axis_type = 'raw'
                    elif 'prop_' in function:
                        y_axis_type = 'proportion'
                    else:
                        y_axis_type = 'raw'
                    self.tidy_axis(ax, subplot_grid, start_time=start_time,
                                   title=t_k.capitalise_first_letter(t_k.find_title_from_dictionary(function)),
                                   legend=(f == len(function_list) - 1), y_axis_type=y_axis_type)

                # finish off
                title = self.inputs.country + ' ' + t_k.find_title_from_dictionary(classification) + ' parameter'
                if len(function_list) > 1: title += 's'
                fig.suptitle(title, fontsize=self.suptitle_size)
                self.save_figure(fig, '_' + classification + '_scale_ups')

    def plot_programmatic_scaleups(self):

        """
        Plots only the programmatic time-variant functions on a single set of axes.
        """

        # Functions to plot are those in the program_prop_ category of the classified scaleups
        functions = self.classified_scaleups['program_prop_']

        # Standard prelims
        fig = self.set_and_update_figure()
        line_styles = self.make_default_line_styles(len(functions), True)
        start_time = self.inputs.model_constants['plot_start_time']
        x_vals = numpy.linspace(start_time, self.inputs.model_constants['plot_end_time'], 1e3)

        # Plot functions for baseline model run only
        ax = self.make_single_axis(fig)
        for figure_number, function in enumerate(functions):
            ax.plot(x_vals, map(self.inputs.scaleup_fns[None][function], x_vals), line_styles[figure_number],
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

        # Plot figures by scenario
        for scenario in self.scenario_names:
            fig = self.set_and_update_figure()

            # Subplots by program
            subplot_grid = find_subplot_numbers(len(self.programs[t_k.find_scenario_number_from_string(scenario)]))
            for p, program in enumerate(self.programs[t_k.find_scenario_number_from_string(scenario)]):
                ax = fig.add_subplot(subplot_grid[0], subplot_grid[1], p + 1)

                # Make times that each curve is produced for from control panel inputs
                times = range(int(self.inputs.model_constants['cost_curve_start_time']),
                              int(self.inputs.model_constants['cost_curve_end_time']),
                              int(self.inputs.model_constants['cost_curve_step_time']))

                for t, time in enumerate(times):
                    time_index = t_k.find_first_list_element_at_least_value(
                        self.model_runner.model_dict['manual_' + scenario].times, time)

                    # Make cost coverage curve
                    x_values, y_values = [], []
                    for coverage in numpy.linspace(0, 1, 101):
                        if coverage < self.inputs.model_constants['econ_saturation_' + program]:
                            cost \
                                = economics.get_cost_from_coverage(coverage,
                                self.inputs.model_constants['econ_inflectioncost_' + program],
                                self.inputs.model_constants['econ_saturation_' + program],
                                self.inputs.model_constants['econ_unitcost_' + program],
                                self.model_runner.model_dict['manual_' + scenario].var_array[
                                    time_index, self.model_runner.model_dict['manual_' + scenario].var_labels.index(
                                        'popsize_' + program)])
                            x_values += [cost]
                            y_values += [coverage]

                    # Find darkness
                    darkness = .9 - (float(t) / float(len(times))) * .9

                    # Plot
                    ax.plot(x_values, y_values, color=(darkness, darkness, darkness), label=str(int(time)))

                self.tidy_axis(ax, subplot_grid, title=t_k.find_title_from_dictionary('program_prop_' + program),
                               x_axis_type='scaled', legend=(p == len(self.programs) - 1), y_axis_type='proportion',
                               x_label='$US ')

            # Finish off with title and save file for scenario
            fig.suptitle('Cost-coverage curves for ' + t_k.replace_underscore_with_space(scenario),
                         fontsize=self.suptitle_size)
            self.save_figure(fig, '_' + scenario + '_cost_coverage')

    def plot_cost_over_time(self):
        """
        Method that produces plots for individual and cumulative program costs for each scenario as separate figures.
        Panels of figures are the different sorts of costs (i.e. whether discounting and inflation have been applied).
        """

        # separate figures for each scenario
        for scenario in self.scenario_names:

            # standard prelims, but separate for each type of plot - individual and stacked
            fig_individual = self.set_and_update_figure()
            fig_stacked = self.set_and_update_figure()
            fig_relative = self.set_and_update_figure()
            subplot_grid = find_subplot_numbers(len(self.model_runner.cost_types))

            # find the index for the first time after the current time
            reference_time_index \
                = t_k.find_first_list_element_above_value(self.model_runner.cost_outputs['manual_' + scenario]['times'],
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
                cumulative_data = [0.] * len(self.model_runner.cost_outputs['manual_' + scenario]['times'])

                # plot for each intervention
                for intervention in self.inputs.interventions_to_cost[t_k.find_scenario_number_from_string(scenario)]:

                    # Record the previous data for plotting as an independent object for the lower edge of the fill
                    previous_data = copy.copy(cumulative_data)

                    # Calculate the cumulative sum for the upper edge of the fill
                    for i in range(len(self.model_runner.cost_outputs['manual_' + scenario]['times'])):
                        cumulative_data[i] \
                            += self.model_runner.cost_outputs['manual_' + scenario][cost_type
                                                                                    + '_cost_' + intervention][i]

                    # Scale the cost data
                    individual_data \
                        = self.model_runner.cost_outputs['manual_' + scenario][cost_type + '_cost_' + intervention]
                    reference_cost \
                        = self.model_runner.cost_outputs['manual_' + scenario][cost_type + '_cost_'
                                                                               + intervention][reference_time_index]
                    relative_data = [(d - reference_cost) for d in individual_data]

                    # Plot lines
                    ax_individual.plot(self.model_runner.cost_outputs['manual_' + scenario]['times'],
                                       individual_data,
                                       color=self.program_colours[
                                           t_k.find_scenario_number_from_string(scenario)][intervention][1],
                                       label=t_k.find_title_from_dictionary(intervention))
                    ax_relative.plot(self.model_runner.cost_outputs['manual_' + scenario]['times'],
                                     relative_data,
                                     color=self.program_colours[
                                         t_k.find_scenario_number_from_string(scenario)][intervention][1],
                                     label=t_k.find_title_from_dictionary(intervention))

                    # Plot stacked areas
                    ax_stacked.fill_between(self.model_runner.model_dict['manual_' + scenario].cost_times,
                                            previous_data, cumulative_data,
                                            color=self.program_colours[
                                                t_k.find_scenario_number_from_string(scenario)][intervention][1],
                                            linewidth=0., label=t_k.find_title_from_dictionary(intervention))

                # final tidying
                for ax in [ax_individual, ax_stacked, ax_relative]:
                    self.tidy_axis(ax, subplot_grid, title=t_k.capitalise_and_remove_underscore(cost_type),
                                   start_time=self.inputs.model_constants['plot_economics_start_time'],
                                   y_label=' $US', y_axis_type='scaled', y_sig_figs=1,
                                   legend=(c == len(self.model_runner.cost_types) - 1))

            # finishing off with title and save
            fig_individual.suptitle('Individual program costs for ' + t_k.find_title_from_dictionary(scenario),
                                    fontsize=self.suptitle_size)
            self.save_figure(fig_individual, '_' + scenario + '_timecost_individual')
            fig_stacked.suptitle('Stacked program costs for ' + t_k.find_title_from_dictionary(scenario),
                                 fontsize=self.suptitle_size)
            self.save_figure(fig_stacked, '_' + scenario + '_timecost_stacked')
            fig_relative.suptitle('Relative program costs for ' + t_k.find_title_from_dictionary(scenario),
                                  fontsize=self.suptitle_size)
            self.save_figure(fig_relative, '_' + scenario + '_timecost_relative')

    def plot_cost_over_time_stacked_bars(self, cost_type='raw'):
        """
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
            for int, intervention in enumerate(self.inputs.interventions_to_cost[scenario]):

                # find the data to plot for the current intervention
                data = self.model_runner.cost_outputs_integer_dict[
                    'manual_' + t_k.find_scenario_string_from_number(scenario)][cost_type + '_cost_' + intervention]

                # initialise the dictionaries at the first iteration
                if int == 0:
                    base = {i: 0. for i in data}
                    upper = {i: 0. for i in data}

                # increment the upper values
                upper = {i: upper[i] + data[i] for i in data}

                # plot
                ax = self.make_single_axis(fig)
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
        ax = self.make_single_axis(fig)
        colours, patterns, compartment_full_names, markers \
            = make_related_line_styles(self.model_runner.model_dict['manual_baseline'].labels, strain_or_organ)

        # plot total population
        ax.plot(self.model_runner.epi_outputs['manual_baseline']['times'][self.start_time_index:],
                self.model_runner.epi_outputs['manual_baseline']['population'][self.start_time_index:],
                'k', label='total', linewidth=2)

        # plot sub-populations
        for plot_label in self.model_runner.model_dict['manual_baseline'].labels:
            ax.plot(self.model_runner.epi_outputs['manual_baseline']['times'][self.start_time_index:],
                    self.model_runner.model_dict['manual_baseline'].compartment_soln[plot_label][
                        self.start_time_index:],
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

        # Get values to be plotted
        _, subgroup_fractions = t_k.find_fractions(self.model_runner.model_dict['manual_baseline'])
        for c, category in enumerate(subgroup_fractions):
            values = subgroup_fractions[category]

            # Standard prelims
            fig = self.set_and_update_figure()
            ax = self.make_single_axis(fig)
            colours, patterns, compartment_full_names, markers \
                = make_related_line_styles(values.keys(), strain_or_organ)

            # Plot population fractions
            for plot_label in values.keys():
                ax.plot(self.model_runner.model_dict['manual_baseline'].times, values[plot_label],
                        label=t_k.find_title_from_dictionary(plot_label), linewidth=1, color=colours[plot_label],
                        marker=markers[plot_label], linestyle=patterns[plot_label])

            # Finishing up
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
                    scenario_name = t_k.find_scenario_string_from_number(scenario)
                    start_index = self.find_start_index(scenario)
                    ax.plot(
                        self.model_runner.epi_outputs['manual_' + scenario_name]['times'][start_index:],
                        self.model_runner.epi_outputs['manual_' + scenario_name][output + stratum][start_index:],
                        color=self.output_colours[scenario][1], linestyle=self.output_colours[scenario][0],
                        linewidth=1.5, label=t_k.capitalise_and_remove_underscore(scenario_name))

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
                     fontsize=self.suptitle_size)
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

        colours = self.make_default_line_styles(len(strata), return_all=True)

        # prelims
        fig = self.set_and_update_figure()
        ax = fig.add_subplot(1, 1, 1)
        times = self.model_runner.model_dict['manual_baseline'].times
        lower_plot_margin = numpy.zeros(len(times))
        upper_plot_margin = numpy.zeros(len(times))

        for s, stratum in enumerate(strata):

            # find numbers or fractions in that group
            stratum_count = t_k.calculate_proportion_list(
                self.model_runner.epi_outputs['manual_baseline']['notifications' + stratum],
                self.model_runner.epi_outputs['manual_baseline']['notifications'])

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

        fig.suptitle('Proportion of notifications by age', fontsize=self.suptitle_size)
        self.save_figure(fig, '_proportion_notifications_by_age')

    def plot_stratified_populations(self, age_or_risk='age'):
        """
        Function to plot population by age group both as raw numbers and as proportions,
        both from the start of the model and using the input argument.
        """

        early_time_index \
            = t_k.find_first_list_element_at_least_value(self.model_runner.epi_outputs['manual_baseline']['times'],
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
            colours = self.make_default_line_styles(len(stratification), return_all=True)

            # run plotting from early in the model run and from the standard start time for plotting
            for t, time in enumerate(['plot_start_time', 'early_time']):

                # find starting times
                title_time_text = t_k.find_title_from_dictionary(time)

                # initialise some variables
                times = self.model_runner.model_dict['manual_baseline'].times
                lower_plot_margin_count = numpy.zeros(len(times))
                upper_plot_margin_count = numpy.zeros(len(times))
                lower_plot_margin_fraction = numpy.zeros(len(times))
                upper_plot_margin_fraction = numpy.zeros(len(times))

                for s, stratum in enumerate(stratification):

                    # find numbers or fractions in that group
                    stratum_count = self.model_runner.epi_outputs['manual_baseline']['population' + stratum]
                    stratum_fraction = self.model_runner.epi_outputs['manual_baseline']['fraction' + stratum]

                    # add group values to the upper plot range for area plot
                    for i in range(len(upper_plot_margin_count)):
                        upper_plot_margin_count[i] += stratum_count[i]
                        upper_plot_margin_fraction[i] += stratum_fraction[i]

                    # create proxy for legend
                    if age_or_risk == 'age':
                        legd_text = t_k.turn_strat_into_label(stratum)
                    elif age_or_risk == 'risk':
                        legd_text = t_k.find_title_from_dictionary(stratum)

                    if t == 0:
                        time_index = self.start_time_index
                    else:
                        time_index = early_time_index

                    # plot total numbers
                    ax_upper = fig.add_subplot(2, 2, 1 + t)
                    ax_upper.fill_between(times[time_index:], lower_plot_margin_count[time_index:],
                                          upper_plot_margin_count[time_index:], facecolors=colours[s][1])

                    # plot population proportions
                    ax_lower = fig.add_subplot(2, 2, 3 + t)
                    ax_lower.fill_between(times[time_index:], lower_plot_margin_fraction[time_index:],
                                          upper_plot_margin_fraction[time_index:],
                                          facecolors=colours[s][1], label=legd_text)

                    # tidy up plots
                    self.tidy_axis(ax_upper, [2, 2], start_time=self.inputs.model_constants[time],
                                   title='Total numbers from ' + title_time_text, y_label='population')
                    self.tidy_axis(ax_lower, [2, 2], y_axis_type='proportion',
                                   start_time=self.inputs.model_constants[time],
                                   title='Proportion of population from ' + title_time_text, legend=(t == 1))

                    # add group values to the lower plot range for next iteration
                    for i in range(len(lower_plot_margin_count)):
                        lower_plot_margin_count[i] += stratum_count[i]
                        lower_plot_margin_fraction[i] += stratum_fraction[i]

            # finish up
            fig.suptitle('Population by ' + t_k.find_title_from_dictionary(age_or_risk), fontsize=self.suptitle_size)
            self.save_figure(fig, '_riskgroup_proportions')

    def plot_intervention_costs_by_scenario(self, year_start, year_end, horizontal=False, plot_options=None):
        """
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

        # Make data frame (columns: interventions, rows: scenarios)
        data_frame = pandas.DataFrame(index=self.scenarios, columns=intervention_names)
        for scenario in self.scenarios:
            data_frame.loc[scenario] \
                = [sum([self.model_runner.cost_outputs_integer_dict[
                            'manual_' + t_k.find_scenario_string_from_number(scenario)][
                            'discounted_inflated_cost_' + intervention][year] for year in years])
                   for intervention in options['interventions']]
        data_frame.columns = intervention_names

        # Make and style plot
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

        # Save plot
        pyplot.savefig(os.path.join(self.out_dir_project, self.country + '_totalcost' + '.png'),
                       bbox_extra_artists=(lgd,), bbox_inches='tight')

    def plot_riskgroup_checks(self):
        """
        Plots actual risk group fractions against targets. Probably almost redundant, as this is a test of code, rather
        than really giving and epidemiological information.
        """

        # standard prelims
        fig = self.set_and_update_figure()
        ax = self.make_single_axis(fig)

        # plotting
        for riskgroup in self.model_runner.model_dict['manual_baseline'].riskgroups:
            ax.plot(self.model_runner.model_dict['manual_baseline'].times,
                    self.model_runner.model_dict['manual_baseline'].actual_risk_props[riskgroup], 'g-',
                    label='Actual ' + riskgroup)
            ax.plot(self.model_runner.model_dict['manual_baseline'].times,
                    self.model_runner.model_dict['manual_baseline'].target_risk_props[riskgroup][1:], 'k--',
                    label='Target ' + riskgroup)

        # end bits
        self.tidy_axis(ax, [1, 1], y_axis_type='proportion', start_time=self.inputs.model_constants['plot_start_time'],
                       single_axis_room_for_legend=True)
        fig.suptitle('Population by risk group', fontsize=self.suptitle_size)
        self.save_figure(fig, '_riskgroup_checks')

    def plot_param_histograms(self):
        """
        Simple function to plot histograms of parameter values used in uncertainty analysis.
        """

        # preliminaries
        fig = self.set_and_update_figure()
        subplot_grid = find_subplot_numbers(len(self.model_runner.all_parameters_tried))

        # loop through parameters used in uncertainty
        for p, param in enumerate(self.model_runner.all_parameters_tried):
            ax = fig.add_subplot(subplot_grid[0], subplot_grid[1], p + 1)

            # restrict to those accepted and after burn-in complete
            param_values = [self.model_runner.all_parameters_tried[param][i]
                            for i in self.model_runner.accepted_no_burn_in_indices]

            # plot
            ax.hist(param_values)
            ax.set_title(t_k.find_title_from_dictionary(param))
        self.save_figure(fig, '_param_histogram')

    def plot_force_infection(self):
        """
        View the force of infection vars.
        """

        # separate plot for each strain
        for strain in self.model_runner.model_dict['manual_baseline'].strains:
            fig = self.set_and_update_figure()
            ax = fig.add_subplot(1, 1, 1)

            # loop over risk groups and plot line for each - now need to restrict to single age-group, as IPT modifies
            # the force of infection for some age-groups.
            for riskgroup in self.model_runner.model_dict['manual_baseline'].riskgroups:
                data_to_plot = self.model_runner.model_dict['manual_baseline'].get_var_soln(
                            'rate_force' + strain + riskgroup
                            + self.model_runner.model_dict['manual_baseline'].agegroups[-1])[
                        self.start_time_index:]
                ax.plot(self.model_runner.model_dict['manual_baseline'].times[self.start_time_index:],
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
        ax = self.make_single_axis(fig)
        output_colours = self.make_default_line_styles(5, True)
        bar_width = .7
        last_data = list(numpy.zeros(len(self.model_runner.model_dict['manual_baseline'].riskgroups)))
        for r, to_riskgroup in enumerate(self.model_runner.model_dict['manual_baseline'].riskgroups):
            data = []
            for from_riskgroup in self.model_runner.model_dict['manual_baseline'].riskgroups:
                data += [self.model_runner.model_dict['manual_baseline'].mixing[from_riskgroup][to_riskgroup]]
            next_data = [i + j for i, j in zip(last_data, data)]
            x_positions = numpy.linspace(.5, .5 + len(next_data) - 1., len(next_data))
            ax.bar(x_positions, data,
                   width=bar_width, bottom=last_data, color=output_colours[r][1],
                   label=t_k.capitalise_and_remove_underscore(t_k.find_title_from_dictionary(to_riskgroup)))
            last_data = next_data
        xlabels = [t_k.capitalise_first_letter(t_k.find_title_from_dictionary(i))
                   for i in self.model_runner.model_dict['manual_baseline'].riskgroups]
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

        # Prelims
        fig = self.set_and_update_figure()
        ax = self.make_single_axis(fig)

        # Plotting
        for var in self.model_runner.model_dict['manual_baseline'].var_labels:
            if 'popsize_' in var:
                ax.plot(self.model_runner.model_dict['manual_baseline'].times[self.start_time_index:],
                        self.model_runner.model_dict['manual_baseline'].get_var_soln(var)[self.start_time_index:],
                        label=t_k.find_title_from_dictionary(var[8:]))

        # Finishing up
        self.tidy_axis(ax, [1, 1], legend='for_single', start_time=self.inputs.model_constants['plot_start_time'],
                       y_label=' persons', y_axis_type='scaled')
        fig.suptitle('Population sizes for cost-coverage curves under baseline scenario')
        self.save_figure(fig, '_popsizes')

    def plot_case_detection_rate(self):

        """
        Method to visualise case detection rates across scenarios and sub-groups, to better understand the impact of
        ACF on improving detection rates relative to passive case finding alone.
        """

        # Prelims
        riskgroup_styles = self.make_default_line_styles(5, True)
        fig = self.set_and_update_figure()
        ax_left = fig.add_subplot(1, 2, 1)
        ax_right = fig.add_subplot(1, 2, 2)
        separation = 0.
        separation_increment = .003  # Artificial shifting of lines away from each other

        # Specif interventions of interest for this plotting function - needs to be changed for each country
        interventions_affecting_case_detection = [None, 5, 6]

        # Loop over scenarios
        for scenario in interventions_affecting_case_detection:
            scenario_name = t_k.find_scenario_string_from_number(scenario)
            manual_scenario_name = 'manual_' + scenario_name
            if scenario in self.scenarios:
                start_index = self.find_start_index(scenario)

                # Repeat for each sub-group
                for r, riskgroup in enumerate(self.model_runner.model_dict[manual_scenario_name].riskgroups[::-1]):

                    # Find weighted case detection rate over organ statuses
                    case_detection = numpy.zeros(len(self.model_runner.model_dict[manual_scenario_name].times[
                                                     start_index:]))
                    for organ in self.inputs.organ_status:
                        case_detection_increment \
                            = [i * j for i, j in zip(self.model_runner.model_dict[manual_scenario_name].get_var_soln(
                                                         'program_rate_detect' + organ + riskgroup)[start_index:],
                                                     self.model_runner.model_dict[manual_scenario_name].get_var_soln(
                                                         'epi_prop' + organ)[start_index:])]
                        case_detection \
                            = model_runner.elementwise_list_addition(case_detection, case_detection_increment)
                    case_detection = [i + separation for i in case_detection]
                    delay_to_presentation = [12. / i for i in case_detection]  # Convert rate to delay
                    ax_left.plot(self.model_runner.model_dict[manual_scenario_name].times[start_index:],
                                 case_detection,
                                 color=self.output_colours[scenario][1], linestyle=riskgroup_styles[r * 7][:-1])
                    ax_right.plot(self.model_runner.model_dict[manual_scenario_name].times[start_index:],
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

        # Plotting prelims
        fig = self.set_and_update_figure()
        ax = fig.add_subplot(1, 1, 1)

        # Find accepted likelihoods
        accepted_log_likelihoods = [self.model_runner.loglikelihoods[i] for i in self.model_runner.accepted_indices]

        # Plot the rejected values
        for i in self.model_runner.rejected_indices:

            # Find the index of the last accepted index before the rejected one we're currently interested in
            last_acceptance_before = [j for j in self.model_runner.accepted_indices if j < i][-1]

            # Plot from the previous acceptance to the current rejection
            ax.plot([last_acceptance_before, i],
                    [self.model_runner.loglikelihoods[last_acceptance_before],
                     self.model_runner.loglikelihoods[i]], marker='o', linestyle='--', color='.5')

        # Plot the accepted values
        ax.plot(self.model_runner.accepted_indices, accepted_log_likelihoods, marker='o', color='k')

        # Finishing up
        fig.suptitle('Progression of likelihood', fontsize=self.suptitle_size)
        ax.set_xlabel('All runs', fontsize=get_nice_font_size([1, 1]), labelpad=1)
        ax.set_ylabel('Likelihood', fontsize=get_nice_font_size([1, 1]), labelpad=1)
        self.save_figure(fig, '_likelihoods')

    def plot_optimised_epi_outputs(self):

        """
        Plot incidence and mortality over funding. This corresponds to the outputs obtained under optimal allocation.
        """

        fig = self.set_and_update_figure()
        left_ax = self.make_single_axis(fig)
        right_ax = left_ax.twinx()
        plots = {'incidence': [left_ax, 'b^', 'TB incidence per 100,000 per year'],
                'mortality': [right_ax, 'r+', 'TB mortality per 100,000 per year']}
        for plot in plots:
            plots[plot][0].plot(self.model_runner.opti_results['annual_envelope'],
                                 self.model_runner.opti_results[plot], plots[plot][1], linewidth=2.0, label=plot)
            self.tidy_axis(plots[plot][0], [1, 1], y_axis_type='raw', y_label=plots[plot][2], x_sig_figs=1,
                           title='Annual funding (US$)', x_axis_type='scaled', x_label='$US ', legend='for_single')
        self.save_opti_figure(fig, '_optimised_outputs')

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
        self.save_opti_figure(fig, '_optimal_allocation')

    ############################
    ### Miscellaneous method ###
    ############################

    def open_output_directory(self):

        """
        Opens the directory into which all the outputs have been placed.
        """

        operating_system = platform.system()
        if 'Windows' in operating_system:
            os.system('start ' + ' ' + self.out_dir_project)
        elif 'Darwin' in operating_system:
            os.system('open ' + ' ' + self.out_dir_project)


