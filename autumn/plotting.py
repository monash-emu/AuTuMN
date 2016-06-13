

import pylab
import numpy
from matplotlib import pyplot, patches

"""

Module for plotting population systems

"""

def make_axes_with_room_for_legend():
    fig = pyplot.figure()
    ax = fig.add_axes([0.1, 0.1, 0.6, 0.75])
    return ax


def set_axes_props(
        ax, xlabel=None, ylabel=None, title=None, is_legend=True,
        axis_labels=None):

    frame_color = "grey"

    # hide top and right border of plot
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
            text.set_color(frame_color)

    if title is not None:
        t = ax.set_title(title)
        t.set_color(frame_color)

    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontname('Arial')
        label.set_fontsize(8)

    ax.tick_params(color=frame_color, labelcolor=frame_color)
    for spine in ax.spines.values():
        spine.set_edgecolor(frame_color)
    ax.xaxis.label.set_color(frame_color)
    ax.yaxis.label.set_color(frame_color)

    humanise_y_ticks(ax)


def humanise_y_ticks(ax):
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


def replace_underscore_with_space(original_string):

    # Just a simple method to remove underscores and replace with
    # spaces for titles of plots.

    replaced_string = ''
    for i in range(len(original_string)):
        if original_string[i] == '_':
            replaced_string += ' '
        else:
            replaced_string += original_string[i]

    return replaced_string


def capitalise_first_letter(old_string):

    new_string = ''
    for i in range(len(old_string)):
        if i == 0:
            new_string += old_string[i].upper()
        else:
            new_string += old_string[i]

    return new_string


def find_smallest_factors_of_integer(n):

    # Simple function to find the smallest whole number factors of an integer
    answer = [1E3, 1E3]
    for i in range(1, n + 1):
        if n % i == 0 and i+(n/i) < sum(answer):
            answer = [i, n/i]
    return answer


def relax_y_axis(ax):

    # Simple algorithm to move yaxis limits a little further from the data
    # (Just my preferences really)
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


def plot_fractions(model, labels, values, left_xlimit, strain_or_organ, png=None):
    right_xlimit_index, left_xlimit_index = truncate_data(model, left_xlimit)
    colours, patterns, compartment_full_names, markers\
        = make_related_line_styles(labels, strain_or_organ)
    ax = make_axes_with_room_for_legend()
    axis_labels = []
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
    set_axes_props(ax, 'Year', 'Proportion of population',
        'Population, ' + title, True, axis_labels)
    save_png(png)


def plot_outputs(model, labels, left_xlimit, png=None):

    # Truncate data to what you want to look at (rather than going back to the dawn of time)
    right_xlimit_index, left_xlimit_index = truncate_data(model, left_xlimit)
    colours = {}
    patterns = {}
    full_names = {}
    axis_labels = []
    yaxis_label = "Per 100,000 (per year as applicable)"

    # Sort out the plotting patterns
    for label in labels:
        patterns[label] = get_line_pattern(label, "strain")
        if "incidence" in label:
            colours[label] = (0, 0, 0)
            full_names[label] = "Incidence"
        elif "notification" in label:
            colours[label] = (0, 0, 1)
            full_names[label] = "Notifications"
        elif "mortality" in label:
            colours[label] = (1, 0, 0)
            full_names[label] = "Mortality"
        elif "prevalence" in label:
            colours[label] = (0, 0.5, 0)
            full_names[label] = "Prevalence"
        elif "proportion" in label:
            colours[label] = (0, 0, 0)
            full_names[label] = "Proportion"
            yaxis_label = "Percentage"

        if "_ds" in label:
            full_names[label] += ", DS-TB"
        elif "_mdr" in label:
            full_names[label] += ", MDR-TB"
    ax = make_axes_with_room_for_legend()

    for i_plot, var_label in enumerate(labels):
        ax.plot(
            model.times[left_xlimit_index: right_xlimit_index],
            model.get_var_soln(var_label)[left_xlimit_index: right_xlimit_index],
            color=colours[var_label],
            label=var_label, linewidth=1, linestyle=patterns[var_label]
        )
        axis_labels.append(full_names[var_label])

    set_axes_props(ax, 'Year', yaxis_label, "Main epidemiological outputs", True,
        axis_labels)
    save_png(png)


def plot_outputs_against_gtb(model, label, left_xlimit, png=None, country_data=None):

    # Sort out the plotting patterns
    if label == "incidence":
        colour = (0, 0, 0)
        index = 'e_inc_100k'
        yaxis_label = 'Per 100,000 per year'
        title = "Incidence"
    elif label == "mortality":
        colour = (1, 0, 0)
        index = 'e_mort_exc_tbhiv_100k'
        yaxis_label = 'Per 100,000 per year'
        title = "Mortality"
    elif label == "prevalence":
        colour = (0, 0.5, 0)
        index = 'e_prev_100k'
        yaxis_label = 'Per 100,000'
        title = "Prevalence"

    # Create a colour half-way between the line colour and white
    patch_colour = []
    for i in range(len(colour)):
        patch_colour += [1. - (1. - colour[i]) / 2.]

    # Extract the plotting data you're interested in
    plotting_data = {}
    for i in country_data['tb']:
        if index in i and '_lo' in i:
            plotting_data['lower_limit'] = country_data['tb'][i]
        elif index in i and '_hi' in i:
            plotting_data['upper_limit'] = country_data['tb'][i]
        elif index in i:
            plotting_data['point_estimate'] = country_data['tb'][i]
    plotting_data['year'] = country_data['tb'][u'year']

    # Truncate data to what you want to look at (rather than going back to the dawn of time)
    right_xlimit_index, left_xlimit_index = truncate_data(model, left_xlimit)
    axis_labels = []

    # Prepare axes
    ax = make_axes_with_room_for_legend()

    # Plot the modelled data
    ax.plot(
        model.times[left_xlimit_index: right_xlimit_index],
        model.get_var_soln(label)[left_xlimit_index: right_xlimit_index],
        color=colour,
        label=label, linewidth=1.5,
    )
    axis_labels.append("Modelled " + label)

    # Plot the GTB data

    # Central point-estimate
    ax.plot(plotting_data['year'], plotting_data['point_estimate'],
            label=label, color=colour, linewidth=0.5)
    axis_labels.append("Reported " + label)

    # Create the patch array
    patch_array = numpy.zeros(shape=(len(plotting_data['lower_limit']) * 2, 2))
    for i in range(len(plotting_data['lower_limit'])):
        patch_array[i][0] = plotting_data['year'][i]
        patch_array[-(i+1)][0] = plotting_data['year'][i]
        patch_array[i][1] = plotting_data['lower_limit'][i]
        patch_array[-(i+1)][1] = plotting_data['upper_limit'][i]
    # Create the patch image and plot it
    patch = patches.Polygon(patch_array, color=patch_colour)
    ax.add_patch(patch)

    ax.set_ylim([0,
                 max(plotting_data['upper_limit'] +
                     model.get_var_soln(label)[left_xlimit_index: right_xlimit_index].tolist())])

    # Set axes
    set_axes_props(ax, 'Year', yaxis_label, title, True,
        axis_labels)
    save_png(png)


def plot_all_outputs_against_gtb(model,
                                 labels,
                                 start_time,
                                 end_time_str='current_time',
                                 png=None,
                                 country='',
                                 scenario=None):

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
        output_colour = ['k'] * len(labels)
    else:
        # Otherwise cycling through colours
        output_colour = [make_default_line_styles(scenario, False)[1]] * len(labels)

    # Extract the plotting data you're interested in
    plotting_data = []
    for i in range(len(indices)):
        plotting_data += [{}]
        for j in model.data['tb']:
            if indices[i] in j and '_lo' in j:
                plotting_data[i]['lower_limit'] = model.data['tb'][j]
            elif indices[i] in j and '_hi' in j:
                plotting_data[i]['upper_limit'] = model.data['tb'][j]
            elif indices[i] in j:
                plotting_data[i]['point_estimate'] = model.data['tb'][j]
        plotting_data[i]['year'] = model.data['tb'][u'year']

    # Truncate data to what you want to look at (rather than going back to the dawn of time)
    right_xlimit_index, left_xlimit_index = truncate_data(model, start_time)

    subplot_grid = find_subplot_numbers(len(labels))

    # Time to plot until
    end_time = model.data['attributes'][end_time_str]

    # Not sure whether we have to specify a figure number
    fig = pyplot.figure(11)

    # Overall title
    fig.suptitle(country + ' model outputs', fontsize=12)

    for i in range(len(model.data['notifications'][u'year'])):
        if model.data['notifications'][u'year'][i] > start_time:
            notification_start_index = i
            break
    notification_year_data = model.data['notifications'][u'year'][notification_start_index:]
    notification_data = model.data['notifications'][u'c_newinc'][notification_start_index:]

    for i, outcome in enumerate(labels):

        ax = fig.add_subplot(subplot_grid[0], subplot_grid[1], i + 1)

        # Plot the modelled data
        ax.plot(
            model.times[left_xlimit_index: right_xlimit_index],
            model.get_var_soln(labels[i])[left_xlimit_index: right_xlimit_index],
            color=output_colour[i],
            label=labels[i], linewidth=1.5)

        # This is supposed to mean if it's the last scenario, which is the baseline
        # (provided the function has been called as intended).
        if scenario is None:

            # Plot the GTB data
            # Notifications are just plotted against raw reported notifications,
            # as there are no confidence intervals around these values.
            if outcome == 'notifications':
                ax.plot(notification_year_data, notification_data,
                        color=colour[i], linewidth=0.5)
                ax.set_ylim((0., max(notification_data)))
            else:
                # Central point-estimate
                ax.plot(plotting_data[i]['year'], plotting_data[i]['point_estimate'],
                        label=labels[i], color=colour[i], linewidth=0.5)

                # Create the patch array
                patch_array = create_patch_from_dictionary(plotting_data[i])

                # Create the patch image and plot it
                patch = patches.Polygon(patch_array, color=patch_colour[i])
                ax.add_patch(patch)

                # Make y-axis range extend downwards to zero
                ax.set_ylim((0., max(plotting_data[i]['upper_limit'])))

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
    for i in range(len(dict['lower_limit'])):
        patch_array[i][0] = dict['year'][i]
        patch_array[-(i + 1)][0] = dict['year'][i]
        patch_array[i][1] = dict['lower_limit'][i]
        patch_array[-(i + 1)][1] = dict['upper_limit'][i]

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
                     parameter_type='', country=u''):

    line_styles = make_default_line_styles(len(functions), True)
    start_time = model.data['attributes'][start_time_str]
    end_time = model.data['attributes'][end_time_str]
    x_vals = numpy.linspace(start_time, end_time, 1E3)
    ax = make_axes_with_room_for_legend()
    for i, function in enumerate(functions):
        ax.plot(x_vals,
                map(model.scaleup_fns[function],
                    x_vals), line_styles[i],
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
                                      parameter_type='', country=u''):

    # Get some styles for the lines
    line_styles = make_default_line_styles(len(functions), True)

    # Determine how many subplots to have
    subplot_grid = find_subplot_numbers(len(functions))

    # Set x-values
    start_time = model.data['attributes'][start_time_str]
    end_time = model.data['attributes'][end_time_str]
    x_vals = numpy.linspace(start_time, end_time, 1E3)

    # Initialise figure
    fig = pyplot.figure()

    # Upper title for whole figure
    plural = ''
    if len(functions) > 1:
        plural += 's'
    title = str(country) + ' ' + \
            replace_underscore_with_space(parameter_type) + \
            ' parameter' + plural + ' from ' + replace_underscore_with_space(start_time_str)
    fig.suptitle(title)

    # Iterate through functions
    for i, function in enumerate(functions):

        # Initialise subplot areas
        ax = fig.add_subplot(subplot_grid[0], subplot_grid[1], i + 1)

        # Line plot scaling parameters
        ax.plot(x_vals,
                map(model.scaleup_fns[function],
                    x_vals), line_styles[i],
                    label=function)
        data_to_plot = {}
        for j in model.scaleup_data[function]:
            if j > start_time:
                data_to_plot[j] = model.scaleup_data[function][j]

        # Scatter plot data from which they are derived
        ax.scatter(data_to_plot.keys(),
                    data_to_plot.values(),
                   color=line_styles[i][-1],
                   s=6)

        # Adjust tick font size
        ax.set_xticks([start_time, end_time])
        for axis_to_change in [ax.xaxis, ax.yaxis]:
            for tick in axis_to_change.get_major_ticks():
                tick.label.set_fontsize(get_nice_font_size(subplot_grid))

        # Truncate parameter names depending on whether it is a
        # treatment success/death proportion
        title = capitalise_first_letter(replace_underscore_with_space(function))
        ax.set_title(title, fontsize=get_nice_font_size(subplot_grid))

        ylims = relax_y_axis(ax)
        ax.set_ylim(bottom=ylims[0], top=ylims[1])

        save_png(png)

    fig.suptitle('Scale-up functions')


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



