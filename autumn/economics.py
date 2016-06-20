
import pylab
import numpy
from matplotlib import patches
import matplotlib.pyplot as plt


"""
Module for estimating cost of a program
"""



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


def plot_outputs_against_gtb(model,
                             labels,
                             start_time,
                             end_time_str='current_time',
                             png=None,
                             country='',
                             scenario=None,
                             gtb=True,
                             figure_number=11):

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
    fig = pyplot.figure(figure_number)

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
            linewidth=1.5)

        # This is supposed to mean if it's the last scenario, which is the baseline
        # (provided the function has been called as intended).
        if scenario is None:

            if gtb:
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
                            # label=labels[i],
                            color=colour[i], linewidth=0.5)

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
                     parameter_type='', country=u'', figure_number=1):

    line_styles = make_default_line_styles(len(functions), True)
    if start_time_str == 'recent_time':
        start_time = model.data['attributes'][start_time_str]
    else:
        start_time = model.data['country_constants'][start_time_str]
    end_time = model.data['attributes'][end_time_str]
    x_vals = numpy.linspace(start_time, end_time, end_time - start_time + 1)
    #print(x_vals)

    pyplot.figure(figure_number)

    def outcome_coverage_fx(cov, outcome_zerocov, outcome_fullcov):
        y = (outcome_fullcov - outcome_zerocov) * numpy.array (cov) + outcome_zerocov
        return y

   # cost_values = []


    ax = make_axes_with_room_for_legend()
    for figure_number, function in enumerate(functions):
        #print(function)
        if function == str("program_prop_vaccination"):
            y_vals = map(model.scaleup_fns[function], x_vals)
            cov = y_vals

            print(x_vals, y_vals)
            import matplotlib.pyplot as plt
            plt.plot(x_vals, y_vals)
            plt.show()



            cost_values = outcome_coverage_fx(cov, 10., 80.)
            import matplotlib.pyplot as plt
            plt.plot(y_vals, cost_values)
            plt.show()


        ax.plot(x_vals,
                map(model.scaleup_fns[function],
                    x_vals), line_styles[figure_number],
                label=function)
        #print (x_vals, )


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
                                      parameter_type='', country=u'',
                                      scenario=None,
                                      figure_number=2):

    # Get some styles for the lines
    line_styles = make_default_line_styles(len(functions), True)

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
        start_time = model.data['attributes'][start_time_str]
    else:
        start_time = model.data['country_constants'][start_time_str]
    end_time = model.data['attributes'][end_time_str]
    x_vals = numpy.linspace(start_time, end_time, 1E3)

    # Initialise figure
    fig = pyplot.figure(figure_number)

    # Upper title for whole figure
    plural = ''
    if len(functions) > 1:
        plural += 's'
    title = str(country) + ' ' + \
            replace_underscore_with_space(parameter_type) + \
            ' parameter' + plural + ' from ' + replace_underscore_with_space(start_time_str)
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



