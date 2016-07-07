

import pylab
import numpy
from matplotlib import pyplot, patches
import base_analyses
import os


"""

Module for plotting population systems

"""

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
                prop={'size':7})
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


def find_subplot_numbers(n):

    # Find a nice number of subplots for a panel plot
    answer = find_smallest_factors_of_integer(n)
    i = 0
    while i < 10:
        if abs(answer[0] - answer[1]) > 3:
            n = n + 1
            answer = find_smallest_factors_of_integer(n)
        i = i + 1

    return answer


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


def get_nice_font_size(subplot_grid):

    # Simple function to return a reasonable font size
    # as appropriate to the number of rows of subplots in the figure
    return 3. + 9. / subplot_grid[0]


def truncate_data(model, left_xlimit):

    # Not going to argue that the following code is the most elegant approach
    right_xlimit_index = len(model.times) - 1
    left_xlimit_index = 0
    for i in range(len(model.times)):
        if model.times[i] > left_xlimit:
            left_xlimit_index = i
            break
    return right_xlimit_index, left_xlimit_index


def find_reasonable_year_ticks(start_time, end_time):

    """
    Simple method to find some reasonably spaced x-ticks

    Args:
        start_time: Plotting start time
        end_time: Plotting end time

    Returns:
        xticsk: List of where the x ticks should go
    """

    # If the range is divisible by 15
    if (start_time - end_time) % 15 == 0:
        xticks = numpy.arange(start_time, end_time + 15, 15)
    # Otherwise if it's divisible by 10
    elif (start_time - end_time) % 10 == 0:
        xticks = numpy.arange(start_time, end_time + 10, 10)
    # Otherwise just give up on having ticks along axis
    else:
        xticks = [start_time, end_time]

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


def make_default_line_styles(n, return_all=True):

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
            for colour in "rbmgk":
                line_styles.append(line + colour)

    if return_all:
        return line_styles
    else:
        return line_styles[n-1]


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


def plot_populations(model, labels, values, left_xlimit, strain_or_organ, png=None):

    right_xlimit_index, left_xlimit_index = truncate_data(model, left_xlimit)
    colours, patterns, compartment_full_names, markers\
        = make_related_line_styles(labels, strain_or_organ)
    ax = make_axes_with_room_for_legend()
    axis_labels = []
    ax.plot(
        model.times[left_xlimit_index: right_xlimit_index],
        model.get_var_soln("population")[left_xlimit_index: right_xlimit_index],
        'k',
        label="total", linewidth=2)
    axis_labels.append("Number of persons")

    for i_plot, plot_label in enumerate(labels):
        ax.plot(
            model.times[left_xlimit_index: right_xlimit_index],
            values[plot_label][left_xlimit_index: right_xlimit_index],
            label=plot_label, linewidth=1,
            color=colours[plot_label],
            marker=markers[plot_label],
            linestyle=patterns[plot_label])
        axis_labels.append(compartment_full_names[plot_label])

    title = make_plot_title(model, labels)

    set_axes_props(ax, 'Year', 'Persons',
                   'Population, ' + title, True,
                   axis_labels)
    save_png(png)


def plot_fractions(model, values, left_xlimit, strain_or_organ, png=None, figure_number=30):

    right_xlimit_index, left_xlimit_index = truncate_data(model, left_xlimit)
    colours, patterns, compartment_full_names, markers\
        = make_related_line_styles(values.keys(), strain_or_organ)
    fig = pyplot.figure(figure_number)
    ax = fig.add_axes([0.1, 0.1, 0.6, 0.75])
    axis_labels = []
    for i_plot, plot_label in enumerate(values.keys()):
        ax.plot(
            model.times[left_xlimit_index: right_xlimit_index],
            values[plot_label][left_xlimit_index: right_xlimit_index],
            label=plot_label, linewidth=1,
            color=colours[plot_label],
            marker=markers[plot_label],
            linestyle=patterns[plot_label])
        axis_labels.append(compartment_full_names[plot_label])
    title = make_plot_title(model, values.keys())
    set_axes_props(ax, 'Year', 'Proportion of population',
        'Population, ' + title, True, axis_labels)
    save_png(png)


def plot_outputs_against_gtb(model,
                             labels,
                             start_time,
                             end_time_str='current_time',
                             png=None,
                             country='',
                             scenario=None,
                             gtb=True,
                             figure_number=11,
                             final_run=True):

    """
    Produces the plot for the main outputs, can handle multiple scenarios (if required).
    Save as png at the end.
    Note that if running a series of scenarios, it is expected that the last scenario to
    be run will be baseline, which should have scenario set to None.

    Args:
        model: The entire model object
        labels: A list of the outputs to be plotted
        start_time: Starting time
        end_time_str: String to access end time from data
        png:
        country: Country being plotted (just need for title)
        scenario: The scenario being run, number needed for line colour

    """

    # Get standard colours for plotting GTB data against
    colour, indices, yaxis_label, title, patch_colour = \
        find_standard_output_styles(labels, lightening_factor=0.3)

    # Get the colours for the model outputs
    if scenario is None:
        # Last scenario to run should be baseline and should be run last
        # to lay a black line over the top for comparison
        output_colour = ['-k'] * len(labels)
    else:
        # Otherwise cycling through colours
        output_colour = [make_default_line_styles(scenario, False)] * len(labels)

    # Extract the plotting data of interest
    plotting_data = []
    for i in range(len(indices)):
        plotting_data += [{}]
        for j in model.inputs['tb']:
            if indices[i] in j and '_lo' in j:
                plotting_data[i]['lower_limit'] = model.inputs['tb'][j]
            elif indices[i] in j and '_hi' in j:
                plotting_data[i]['upper_limit'] = model.inputs['tb'][j]
            elif indices[i] in j:
                plotting_data[i]['point_estimate'] = model.inputs['tb'][j]

    # Truncate data to what you want to look at (rather than going back to the dawn of time)
    right_xlimit_index, left_xlimit_index = truncate_data(model, start_time)

    subplot_grid = find_subplot_numbers(len(labels))

    # Time to plot until
    end_time = model.inputs['model_constants'][end_time_str]

    # Not sure whether we have to specify a figure number
    fig = pyplot.figure(figure_number)

    # Overall title
    fig.suptitle(country + ' model outputs', fontsize=12)

    # Truncate notification data to years of interest
    notification_data = {}
    for i in model.inputs['notifications']['c_newinc']:
        if i > start_time:
            notification_data[i] = \
                model.inputs['notifications']['c_newinc'][i]

    for i, outcome in enumerate(labels):

        ax = fig.add_subplot(subplot_grid[0], subplot_grid[1], i + 1)

        # Plot the modelled data
        ax.plot(
            model.times[left_xlimit_index: right_xlimit_index],
            model.get_var_soln(labels[i])[left_xlimit_index: right_xlimit_index],
            color=output_colour[i][1],
            linestyle=output_colour[i][0],
            linewidth=1.5)

        # This is supposed to mean if it's the last scenario, which is the baseline
        # (provided the function has been called as intended).
        if scenario is None:

            if gtb:
            # Plot the GTB data
            # Notifications are just plotted against raw reported notifications,
            # as there are no confidence intervals around these values.
                if outcome == 'notifications':
                    ax.plot(notification_data.keys(), notification_data.values(),
                            color=colour[i], linewidth=0.5)
                    ax.set_ylim((0., max(notification_data)))
                else:
                    # Central point-estimate
                    ax.plot(plotting_data[i]['point_estimate'].keys(), plotting_data[i]['point_estimate'].values(),
                            # label=labels[i],
                            color=colour[i], linewidth=0.5)

                    # Create the patch array
                    patch_array = create_patch_from_dictionary(plotting_data[i])

                    # Create the patch image and plot it
                    patch = patches.Polygon(patch_array, color=patch_colour[i])
                    ax.add_patch(patch)

                    # Make y-axis range extend downwards to zero
                    ax.set_ylim((0., max(plotting_data[i]['upper_limit'].values())))

            # Set x-ticks
            xticks = find_reasonable_year_ticks(start_time, end_time)
            ax.set_xticks(xticks)

            # Adjust size of labels of x-ticks
            for axis_to_change in [ax.xaxis, ax.yaxis]:
                for tick in axis_to_change.get_major_ticks():
                    tick.label.set_fontsize(get_nice_font_size(subplot_grid))

            # Add the sub-plot title with slightly larger titles than the rest of the text on the panel
            ax.set_title(title[i], fontsize=get_nice_font_size(subplot_grid) + 2.)

            # Label the y axis with the smaller text size
            ax.set_ylabel(yaxis_label[i], fontsize=get_nice_font_size(subplot_grid))

            # Get the handles, except for the last one, which plots the data
            scenario_handles = ax.lines[:-1]

            # Make some string labels for these handles
            # (this code could probably be better)
            scenario_labels = []
            for i in range(len(scenario_handles)):
                if i < len(scenario_handles) - 1:
                    scenario_labels += ['Scenario ' + str(i + 1)]
                else:
                    scenario_labels += ['Baseline']

            # Draw the legend
            ax.legend(scenario_handles,
                      scenario_labels,
                      fontsize=get_nice_font_size(subplot_grid) - 2.,
                      frameon=False)

    if final_run:
        # Save
        save_png(png)


def plot_outputs_by_age(model,
                             labels,
                             start_time,
                             end_time_str='current_time',
                             png=None,
                             country='',
                             scenario=None,
                             figure_number=11,
                             final_run=True):

    """
    Produces the plot for the main outputs, can handle multiple scenarios (if required).
    Save as png at the end.
    Note that if running a series of scenarios, it is expected that the last scenario to
    be run will be baseline, which should have scenario set to None.

    Args:
        model: The entire model object
        labels: A list of the outputs to be plotted
        start_time: Starting time
        end_time_str: String to access end time from data
        png:
        country: Country being plotted (just need for title)
        scenario: The scenario being run, number needed for line colour

    """

    # Get standard colours for plotting GTB data against
    colour, indices, _, _, _ = \
        find_standard_output_styles(labels, lightening_factor=0.3)

    # Get the colours for the model outputs
    if scenario is None:
        # Last scenario to run should be baseline and should be run last
        # to lay a black line over the top for comparison
        output_colour = ['-k'] * len(labels)
    else:
        # Otherwise cycling through colours
        output_colour = [make_default_line_styles(scenario, False)] * len(labels)

    # Truncate data to what you want to look at (rather than going back to the dawn of time)
    right_xlimit_index, left_xlimit_index = truncate_data(model, start_time)

    subplot_grid = find_subplot_numbers(len(model.agegroups) + 1)

    # Time to plot until
    end_time = model.inputs['model_constants'][end_time_str]

    # Not sure whether we have to specify a figure number
    fig = pyplot.figure(figure_number)

    # Overall title
    fig.suptitle(country + ' burden by age group', fontsize=14)


    # Find the highest incidence value in the time period considered across all age groups
    ymax = 0.
    for agegroup in model.agegroups:
        new_ymax = max(model.get_var_soln('incidence' + agegroup)[left_xlimit_index: right_xlimit_index])
        if new_ymax > ymax:
            ymax = new_ymax

    for i, agegroup in enumerate(model.agegroups + ['']):

        ax = fig.add_subplot(subplot_grid[0], subplot_grid[1], i + 1)

        # Plot the modelled data
        ax.plot(
            model.times[left_xlimit_index: right_xlimit_index],
            model.get_var_soln('incidence' + agegroup)[left_xlimit_index: right_xlimit_index],
            color=output_colour[i][1],
            linestyle=output_colour[i][0],
            linewidth=1.5)

        # This is supposed to mean if it's the last scenario, which is the baseline
        # (provided this function has been called as intended).
        if scenario is None:

            # Set x-ticks
            xticks = find_reasonable_year_ticks(start_time, end_time)
            ax.set_xticks(xticks)

            # Adjust size of labels of x-ticks
            for axis_to_change in [ax.xaxis, ax.yaxis]:
                for tick in axis_to_change.get_major_ticks():
                    tick.label.set_fontsize(get_nice_font_size(subplot_grid)-4.)

            # Add the sub-plot title with slightly larger titles than the rest of the text on the panel
            ax.set_title(base_analyses.turn_strat_into_label(agegroup), fontsize=get_nice_font_size(subplot_grid))

            # Label the y axis with the smaller text size
            if i == 0:
                ax.set_ylabel('Per 100,000 per year', fontsize=get_nice_font_size(subplot_grid)-4.)

            # Set upper y-limit to the maximum value for any age group during the period of interest
            ax.set_ylim(bottom=0., top=ymax)

            # Get the handles, except for the last one, which plots the data
            scenario_handles = ax.lines[:-1]

            # Make some string labels for these handles
            # (this code could probably be better)
            scenario_labels = []
            for i in range(len(scenario_handles)):
                if i < len(scenario_handles) - 1:
                    scenario_labels += ['Scenario ' + str(i + 1)]
                else:
                    scenario_labels += ['Baseline']

            # Draw the legend
            ax.legend(scenario_handles,
                      scenario_labels,
                      fontsize=get_nice_font_size(subplot_grid) - 2.,
                      frameon=False)

    if final_run:
        # Save
        save_png(png)


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


def plot_scaleup_fns(model, functions, png=None,
                     start_time_str='start_time', end_time_str='',
                     parameter_type='', country=u'', figure_number=1):

    line_styles = make_default_line_styles(len(functions), True)
    if start_time_str == 'recent_time':
        start_time = model.inputs['model_constants'][start_time_str]
    else:
        start_time = model.inputs['model_constants'][start_time_str]
    end_time = model.inputs['model_constants'][end_time_str]
    x_vals = numpy.linspace(start_time, end_time, 1E3)

    pyplot.figure(figure_number)

    ax = make_axes_with_room_for_legend()
    for figure_number, function in enumerate(functions):
        ax.plot(x_vals,
                map(model.scaleup_fns[function],
                    x_vals), line_styles[figure_number],
                label=function)

    plural = ''
    if len(functions) > 1:
        plural += 's'
    title = str(country) + ' ' + \
            replace_underscore_with_space(parameter_type) + \
            ' parameter' + plural + ' from ' + replace_underscore_with_space(start_time_str)
    set_axes_props(ax, 'Year', 'Parameter value',
                   title, True, functions)

    ylims = relax_y_axis(ax)
    ax.set_ylim(bottom=ylims[0], top=ylims[1])

    save_png(png)


def plot_all_scaleup_fns_against_data(model, functions, png=None,
                                      start_time_str='start_time',
                                      end_time_str='',
                                      parameter_type='',
                                      scenario=None,
                                      figure_number=2):

    # Get the colours for the model outputs
    if scenario is None:
        # Last scenario to run should be baseline and should be run last
        # to lay a black line over the top for comparison
        output_colour = ['k'] * len(functions)
    else:
        # Otherwise cycling through colours
        output_colour = [make_default_line_styles(scenario, False)[1]] * len(functions)

    # Determine how many subplots to have
    subplot_grid = find_subplot_numbers(len(functions))

    # Set x-values
    if start_time_str == 'recent_time':
        start_time = model.inputs['model_constants'][start_time_str]
    else:
        start_time = model.inputs['model_constants'][start_time_str]
    end_time = model.inputs['model_constants'][end_time_str]
    x_vals = numpy.linspace(start_time, end_time, 1E3)

    # Initialise figure
    fig = pyplot.figure(figure_number)

    # Upper title for whole figure
    plural = ''
    if len(functions) > 1:
        plural += 's'
    title = model.inputs['model_constants']['country'] + ' ' + \
            base_analyses.replace_underscore_with_space(parameter_type) + \
            ' parameter' + plural + ' from ' + base_analyses.replace_underscore_with_space(start_time_str)
    fig.suptitle(title)

    # Iterate through functions
    for figure_number, function in enumerate(functions):

        # Initialise subplot areas
        ax = fig.add_subplot(subplot_grid[0], subplot_grid[1], figure_number + 1)

        # Line plot scaling parameters
        ax.plot(x_vals,
                map(model.scaleup_fns[function],
                    x_vals),
                # line_styles[i],
                # label=function,
                color=output_colour[figure_number])

        if scenario is None:
            data_to_plot = {}
            for j in model.scaleup_data[function]:
                if j > start_time:
                    data_to_plot[j] = model.scaleup_data[function][j]

            # Scatter plot data from which they are derived
            ax.scatter(data_to_plot.keys(),
                       data_to_plot.values(),
                       color=output_colour[figure_number],
                       s=6)

            # Adjust tick font size
            ax.set_xticks([start_time, end_time])
            for axis_to_change in [ax.xaxis, ax.yaxis]:
                for tick in axis_to_change.get_major_ticks():
                    tick.label.set_fontsize(get_nice_font_size(subplot_grid))

            # Truncate parameter names depending on whether it is a
            # treatment success/death proportion
            title = base_analyses.capitalise_first_letter(replace_underscore_with_space(function))
            ax.set_title(title, fontsize=get_nice_font_size(subplot_grid))

            ylims = relax_y_axis(ax)
            ax.set_ylim(bottom=ylims[0], top=ylims[1])

            save_png(png)

    fig.suptitle('Scale-up functions')

def plot_classified_scaleups(model, base):

    # Classify scale-up functions for plotting
    classified_scaleups = {'program_prop': [],
                           'program_other': [],
                           'birth': [],
                           'cost': [],
                           'econ': [],
                           'non_program': []}
    for fn in model.scaleup_fns:
        if 'program_prop' in fn:
            classified_scaleups['program_prop'] += [fn]
        elif 'program' in fn:
            classified_scaleups['program_other'] += [fn]
        elif 'demo_rate_birth' in fn:
            classified_scaleups['birth'] += [fn]
        elif 'cost' in fn:
            classified_scaleups['cost'] += [fn]
        elif 'econ' in fn:
            classified_scaleups['econ'] += [fn]
        else:
            classified_scaleups['non_program'] += [fn]

    times_to_plot = ['start_', 'recent_']

    # Plot them from the start of the model and from "recent_time"
    for i, classification in enumerate(classified_scaleups):
        if len(classified_scaleups[classification]) > 0:
            for j, start_time in enumerate(times_to_plot):
                plot_all_scaleup_fns_against_data(model,
                                                  classified_scaleups[classification],
                                                  base + '_' + classification + '_datascaleups_from' + start_time[:-1] + '.png',
                                                  start_time + 'time',
                                                  'current_time',
                                                  classification,
                                                  figure_number=i + j * len(classified_scaleups) + 2)
                if classification == 'program_prop':
                    plot_scaleup_fns(model,
                                     classified_scaleups[classification],
                                     base + '_' + classification + 'scaleups_from' + start_time[:-1] + '.png',
                                     start_time + 'time',
                                     'current_time',
                                     classification,
                                     figure_number=i + j * len(classified_scaleups) + 2 + len(classified_scaleups) * len(times_to_plot))


def plot_comparative_age_parameters(data_strat_list,
                                    data_value_list,
                                    model_value_list,
                                    model_strat_list,
                                    parameter_name):

    # Get good tick labels from the stratum lists
    data_strat_labels = []
    for i in range(len(data_strat_list)):
        data_strat_labels += [base_analyses.turn_strat_into_label(data_strat_list[i])]
    model_strat_labels = []
    for i in range(len(model_strat_list)):
        model_strat_labels += [base_analyses.turn_strat_into_label(model_strat_list[i])]

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
    fig.suptitle(base_analyses.capitalise_first_letter(base_analyses.replace_underscore_with_space(parameter_name))
                 + ' adjustment',
                 fontsize=15)

    # Find directory and save
    out_dir = 'fullmodel_graphs'
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    base = os.path.join(out_dir, parameter_name)
    save_png(base + '_param_adjustment.png')


def save_png(png):

    if png is not None:
        pylab.savefig(png, dpi=300)


def open_pngs(pngs):

    import platform
    import os
    operating_system = platform.system()
    if 'Windows' in operating_system:
        os.system("start " + " ".join(pngs))
    elif 'Darwin' in operating_system:
        os.system('open ' + " ".join(pngs))



