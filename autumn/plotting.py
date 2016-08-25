

import pylab
import numpy
from matplotlib import pyplot, patches
import tool_kit
import os
import warnings
import write_outputs


"""

Module for plotting population systems

"""


def plot_populations(model, labels, values, left_xlimit, strain_or_organ, png=None):

    right_xlimit_index, left_xlimit_index = write_outputs.find_truncation_points(model, left_xlimit)
    colours, patterns, compartment_full_names, markers\
        = write_outputs.make_related_line_styles(labels, strain_or_organ)
    ax = write_outputs.make_axes_with_room_for_legend()
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

    title = write_outputs.make_plot_title(model, labels)

    write_outputs.set_axes_props(ax, 'Year', 'Persons',
                   'Population, ' + title, True,
                   axis_labels)
    save_png(png)


def plot_fractions(model, values, left_xlimit, strain_or_organ, png=None, figure_number=30):

    right_xlimit_index, left_xlimit_index = write_outputs.find_truncation_points(model, left_xlimit)
    colours, patterns, compartment_full_names, markers\
        = write_outputs.make_related_line_styles(values.keys(), strain_or_organ)
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
    title = write_outputs.make_plot_title(model, values.keys())
    write_outputs.set_axes_props(ax, 'Year', 'Proportion of population',
        'Population, ' + title, True, axis_labels)
    save_png(png)


def plot_stratified_populations(model, png=None, age_or_comorbidity='age', start_time='start_time'):

    """
    Function to plot population by age group both as raw numbers and as proportions,
    both from the start of the model and using the input argument

    Args:
        model: The entire model object being interrogated
        left_xlimit: Float value representing the time to plot from for the recent plot
        png: The name of the file to be saved

    """

    if age_or_comorbidity == 'age':
        stratification = model.agegroups
    elif age_or_comorbidity == 'comorbidity':
        stratification = model.comorbidities
    else:
        raise NameError('Stratification not permitted')

    if len(stratification) < 2:
        warnings.warn('No stratification to plot')
    else:
        # Open figure
        fig = pyplot.figure()

        # Extract data
        stratified_soln, denominator = tool_kit.sum_over_compartments(model, stratification)
        stratified_fraction = tool_kit.get_fraction_soln(stratified_soln.keys(), stratified_soln, denominator)

        colours = write_outputs.make_default_line_styles(len(stratification), return_all=True)

        # Loop over starting from the model start and the specified starting time
        for i_time, plot_left_time in enumerate(['recent_time', start_time]):

            # Find starting times
            right_xlimit_index, left_xlimit_index \
                = write_outputs.find_truncation_points(model,
                                         model.inputs['model_constants'][plot_left_time])
            title_time_text = tool_kit.find_title_from_dictionary(plot_left_time)

            # Initialise some variables
            times = model.times[left_xlimit_index: right_xlimit_index]
            lower_plot_margin_count = numpy.zeros(len(times))
            upper_plot_margin_count = numpy.zeros(len(times))
            lower_plot_margin_fraction = numpy.zeros(len(times))
            upper_plot_margin_fraction = numpy.zeros(len(times))
            legd_text = []

            for i, stratum in enumerate(stratification):

                # Find numbers or fractions in that group
                stratum_count = stratified_soln[stratum][left_xlimit_index: right_xlimit_index]
                stratum_fraction = stratified_fraction[stratum][left_xlimit_index: right_xlimit_index]

                # Add group values to the upper plot range for area plot
                for j in range(len(upper_plot_margin_count)):
                    upper_plot_margin_count[j] += stratum_count[j]
                    upper_plot_margin_fraction[j] += stratum_fraction[j]

                # Plot
                ax = fig.add_subplot(2, 2, 1 + i_time)
                ax.fill_between(times, lower_plot_margin_count, upper_plot_margin_count, facecolors=colours[i][1])

                # Create proxy for legend
                ax.plot([], [], color=colours[i][1], linewidth=6)
                if age_or_comorbidity == 'age':
                    legd_text += [tool_kit.turn_strat_into_label(stratum)]
                elif age_or_comorbidity == 'comorbidity':
                    print(tool_kit.find_title_from_dictionary(stratum))
                    legd_text += [tool_kit.find_title_from_dictionary(stratum)]

                # Cosmetic changes at the end
                if i == len(stratification)-1:
                    ax.set_ylim((0., max(upper_plot_margin_count) * 1.1))
                    ax.set_xlim(int(model.times[left_xlimit_index]),
                                model.times[right_xlimit_index])
                    ax.set_title('Total numbers' + title_time_text, fontsize=8)
                    xticks = write_outputs.find_reasonable_year_ticks(int(model.times[left_xlimit_index]),
                                                        model.times[right_xlimit_index])
                    ax.set_xticks(xticks)
                    for axis_to_change in [ax.xaxis, ax.yaxis]:
                        for tick in axis_to_change.get_major_ticks():
                            tick.label.set_fontsize(write_outputs.get_nice_font_size([2]))
                    if i_time == 1:
                        ax.legend(reversed(ax.lines), reversed(legd_text), loc=2, frameon=False, fontsize=8)

                # Plot popuation proportions
                ax = fig.add_subplot(2, 2, 3 + i_time)
                ax.fill_between(times, lower_plot_margin_fraction, upper_plot_margin_fraction, facecolors=colours[i][1])

                # Cosmetic changes at the end
                if i == len(stratification)-1:
                    ax.set_ylim((0., 1.))
                    ax.set_xlim(int(model.times[left_xlimit_index]),
                                model.times[right_xlimit_index])
                    ax.set_title('Proportion of population' + title_time_text, fontsize=8)
                    xticks = write_outputs.find_reasonable_year_ticks(int(model.times[left_xlimit_index]),
                                                        model.times[right_xlimit_index])
                    ax.set_xticks(xticks)
                    for axis_to_change in [ax.xaxis, ax.yaxis]:
                        for tick in axis_to_change.get_major_ticks():
                            tick.label.set_fontsize(write_outputs.get_nice_font_size([2]))

                # Add group values to the lower plot range for next iteration
                for j in range(len(lower_plot_margin_count)):
                    lower_plot_margin_count[j] += stratum_count[j]
                    lower_plot_margin_fraction[j] += stratum_fraction[j]

        # Finish up
        fig.suptitle('Population by ' + tool_kit.find_title_from_dictionary(age_or_comorbidity),
                     fontsize=13)
        save_png(png)


def plot_outputs_against_gtb(model,
                             labels,
                             start_time,
                             end_time_str='current_time',
                             png=None,
                             country='',
                             scenario=None,
                             gtb=True,
                             figure_number=31,
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
        write_outputs.find_standard_output_styles(labels, lightening_factor=0.3)

    # Get the colours for the model outputs
    if scenario is None:
        # Last scenario to run should be baseline and should be run last
        # to lay a black line over the top for comparison
        output_colour = ['-k'] * len(labels)
    else:
        # Otherwise cycling through colours
        output_colour = [write_outputs.make_default_line_styles(scenario, False)] * len(labels)

    # Extract the plotting data of interest
    plotting_data = []
    for i in range(len(indices)):
        plotting_data += [{}]
        for j in model.inputs.original_data['tb']:
            if indices[i] in j and '_lo' in j:
                plotting_data[i]['lower_limit'] = model.inputs.original_data['tb'][j]
            elif indices[i] in j and '_hi' in j:
                plotting_data[i]['upper_limit'] = model.inputs.original_data['tb'][j]
            elif indices[i] in j:
                plotting_data[i]['point_estimate'] = model.inputs.original_data['tb'][j]

    # Truncate data to what you want to look at (rather than going back to the dawn of time)
    right_xlimit_index, left_xlimit_index = write_outputs.find_truncation_points(model, start_time)

    subplot_grid = write_outputs.find_subplot_numbers(len(labels))

    # Time to plot until
    end_time = model.inputs.model_constants[end_time_str]

    # Not sure whether we have to specify a figure number
    fig = pyplot.figure(figure_number)

    # Overall title
    fig.suptitle(country + ' model outputs', fontsize=12)

    # Truncate notification data to years of interest
    notification_data = {}
    for i in model.inputs.original_data['notifications']['c_newinc']:
        if i > start_time:
            notification_data[i] = \
                model.inputs.original_data['notifications']['c_newinc'][i]

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

                max_modelled_output = max(model.get_var_soln(labels[i])[left_xlimit_index: right_xlimit_index])

                if outcome == 'notifications':
                    ax.plot(notification_data.keys(), notification_data.values(),
                            color=colour[i], linewidth=0.5)
                    max_notifications = max(notification_data.values())
                    if max_modelled_output > max_notifications:
                        max_notifications = max_modelled_output
                    ax.set_ylim((0., max_notifications * 1.1))

                else:
                    # Central point-estimate
                    ax.plot(plotting_data[i]['point_estimate'].keys(), plotting_data[i]['point_estimate'].values(),
                            color=colour[i], linewidth=0.5)

                    # Create the patch array
                    patch_array = write_outputs.create_patch_from_dictionary(plotting_data[i])

                    # Create the patch image and plot it
                    patch = patches.Polygon(patch_array, color=patch_colour[i])
                    ax.add_patch(patch)

                    max_output = max(plotting_data[i]['upper_limit'].values())
                    if max_modelled_output > max_output:
                        max_output = max_modelled_output

                    # Make y-axis range extend downwards to zero
                    ax.set_ylim((0., max_output * 1.1))

            # Set x-ticks
            xticks = write_outputs.find_reasonable_year_ticks(start_time, end_time)
            ax.set_xticks(xticks)

            # Adjust size of labels of x-ticks
            for axis_to_change in [ax.xaxis, ax.yaxis]:
                for tick in axis_to_change.get_major_ticks():
                    tick.label.set_fontsize(write_outputs.get_nice_font_size(subplot_grid))

            # Add the sub-plot title with slightly larger titles than the rest of the text on the panel
            ax.set_title(title[i], fontsize=write_outputs.get_nice_font_size(subplot_grid) + 2.)

            # Label the y axis with the smaller text size
            ax.set_ylabel(yaxis_label[i], fontsize=write_outputs.get_nice_font_size(subplot_grid))

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
                      fontsize=write_outputs.get_nice_font_size(subplot_grid) - 2.,
                      frameon=False)

    if final_run:
        # Save
        save_png(png)


def plot_outputs_by_age(model,
                        start_time,
                        end_time_str='current_time',
                        png=None,
                        country='',
                        scenario=None,
                        figure_number=21,
                        final_run=True):

    """
    Produces the plot for the main outputs by age, can handle multiple scenarios (if required).
    Save as png at the end.
    Note that if running a series of scenarios, it is expected that the last scenario to
    be run will be baseline, which should have scenario set to None.
    This function is a bit less flexible than plot_outputs_against_gtb, in which you can select the
    outputs you want to plot. This one is constrained to incidence and mortality (which are the only
    ones currently calculated in the model object.

    Args:
        model: The entire model object
        start_time: Starting time
        end_time_str: String to access end time from data
        png: The filename
        country: Country being plotted (just needed for title)
        scenario: The scenario being run, number needed for line colour

    """

    # Get the colours for the model outputs
    if scenario is None:
        # Last scenario to run should be baseline and should be run last
        # to lay a black line over the top for comparison
        output_colour = ['-k']
    else:
        # Otherwise cycling through colours
        output_colour = [write_outputs.make_default_line_styles(scenario, False)]

    # Truncate data to what you want to look at (rather than going back to the dawn of time)
    right_xlimit_index, left_xlimit_index = write_outputs.find_truncation_points(model, start_time)

    subplot_grid = write_outputs.find_subplot_numbers(len(model.agegroups) * 2 + 1)

    # Time to plot until
    end_time = model.inputs.model_constants[end_time_str]

    # Not sure whether we have to specify a figure number
    fig = pyplot.figure(figure_number)

    # Overall title
    fig.suptitle(country + ' burden by age group', fontsize=14)

    for output_no, output in enumerate(['incidence', 'mortality']):

        # Find the highest incidence value in the time period considered across all age groups
        ymax = 0.
        for agegroup in model.agegroups:
            new_ymax = max(model.get_var_soln(output + agegroup)[left_xlimit_index: right_xlimit_index])
            if new_ymax > ymax:
                ymax = new_ymax

        for i, agegroup in enumerate(model.agegroups + ['']):

            ax = fig.add_subplot(subplot_grid[0], subplot_grid[1], i + 1 + output_no * (len(model.agegroups)+1))

            # Plot the modelled data
            ax.plot(
                model.times[left_xlimit_index: right_xlimit_index],
                model.get_var_soln(output + agegroup)[left_xlimit_index: right_xlimit_index],
                color=output_colour[0][1],
                linestyle=output_colour[0][0],
                linewidth=1.5)

            # This is supposed to mean if it's the last scenario, which is the baseline
            # (provided this function has been called as intended).
            if scenario is None:

                # Set x-ticks
                xticks = write_outputs.find_reasonable_year_ticks(start_time, end_time)
                ax.set_xticks(xticks)

                # Adjust size of labels of x-ticks
                for axis_to_change in [ax.xaxis, ax.yaxis]:
                    for tick in axis_to_change.get_major_ticks():
                        tick.label.set_fontsize(write_outputs.get_nice_font_size(subplot_grid))

                # Add the sub-plot title with slightly larger titles than the rest of the text on the panel
                ax.set_title(tool_kit.capitalise_first_letter(output) + ', '
                             + tool_kit.turn_strat_into_label(agegroup), fontsize=write_outputs.get_nice_font_size(subplot_grid))

                # Label the y axis with the smaller text size
                if i == 0:
                    ax.set_ylabel('Per 100,000 per year', fontsize=write_outputs.get_nice_font_size(subplot_grid))

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
                          fontsize=write_outputs.get_nice_font_size(subplot_grid) - 2.,
                          frameon=False)

    # Save
    if final_run:
        save_png(png)


def plot_flows(model, labels, png=None):

    colours, patterns, compartment_full_names\
        = write_outputs.make_related_line_styles(labels)
    ax = write_outputs.make_axes_with_room_for_legend()
    axis_labels = []
    for i_plot, plot_label in enumerate(labels):
        ax.plot(
            model.times,
            model.get_flow_soln(plot_label) / 1E3,
            label=plot_label, linewidth=1,
            color=colours[plot_label],
            linestyle=patterns[plot_label])
        axis_labels.append(compartment_full_names[plot_label])
    write_outputs.set_axes_props(ax, 'Year', 'Change per year, thousands',
                   'Aggregate flows in/out of compartment',
                   True, axis_labels)
    save_png(png)


def plot_scaleup_fns(model, functions, png=None,
                     start_time_str='start_time', end_time_str='',
                     parameter_type='', country=u'', figure_number=1):

    line_styles = write_outputs.make_default_line_styles(len(functions), True)
    if start_time_str == 'recent_time':
        start_time = model.inputs.model_constants[start_time_str]
    else:
        start_time = model.inputs.model_constants[start_time_str]
    end_time = model.inputs.model_constants[end_time_str]
    x_vals = numpy.linspace(start_time, end_time, 1E3)

    pyplot.figure(figure_number)

    ax = write_outputs.make_axes_with_room_for_legend()
    for figure_number, function in enumerate(functions):
        ax.plot(x_vals,
                map(model.scaleup_fns[function],
                    x_vals), line_styles[figure_number],
                label=function)

    plural = ''
    if len(functions) > 1:
        plural += 's'
    title = str(country) + ' ' + \
            tool_kit.find_title_from_dictionary(parameter_type) + \
            ' parameter' + plural + tool_kit.find_title_from_dictionary(start_time_str)
    write_outputs.set_axes_props(ax, 'Year', 'Parameter value',
                   title, True, functions)

    ylims = write_outputs.relax_y_axis(ax)
    ax.set_ylim(bottom=ylims[0], top=ylims[1])

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
                  fontsize=write_outputs.get_nice_font_size(subplot_grid))
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



