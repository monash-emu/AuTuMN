

import math
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


def truncate_data(model, left_xlimit):
    # Not going to argue that the following code is the most elegant approach
    right_xlimit_index = len(model.times) - 1
    left_xlimit_index = 0
    for i in range(len(model.times)):
        if model.times[i] > left_xlimit:
            left_xlimit_index = i
            break
    return right_xlimit_index, left_xlimit_index


def make_default_line_styles():
    """
    Now inactive
    """
    line_styles = []
    for line in ["-", ":", "-.", "--"]:
        for color in "rbmgk":
            line_styles.append(line + color)
    return line_styles


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


def plot_scaleup_fns(model, functions, png=None, start_time=1900.):

    end_time = 2015.
    x_vals = numpy.linspace(start_time, end_time, 1E3)
    ax = make_axes_with_room_for_legend()
    for function in functions:
        ax.plot(x_vals,
                map(model.scaleup_fns[function],
                    x_vals),
                label=function)
    set_axes_props(ax, 'Year', 'Proportion',
                   'Scaling parameter values', True, functions)
    save_png(png)


def plot_scaleup_fns_against_data(model, function, png=None, start_time=1900.):

    end_time = 2015.
    x_vals = numpy.linspace(start_time, end_time, 1E3)
    ax = make_axes_with_room_for_legend()
    ax.plot(x_vals,
            map(model.scaleup_fns[function],
                x_vals),
            label=function)
    data_to_plot = {}
    for i in model.data['programs'][function]:
        if i > start_time:
            data_to_plot[i] = model.data['programs'][function][i]
    ax.scatter(data_to_plot.keys(),
            data_to_plot.values())
    set_axes_props(ax, 'Year', 'Proportion',
                   'Scaling of ' + function, True, function)
    save_png(png)


def plot_all_scaleup_fns_against_data(model, functions, png=None, start_time=1900.):

    n = 1
    for program in functions:
        if 'prop' in program:
            end_time = 2015.
            x_vals = numpy.linspace(start_time, end_time, 1E3)
            fig = pyplot.figure(10)
            fig.suptitle('Scaling parameter values over time (calendar years)')
            ax = fig.add_subplot(4, 4, n)
            n += 1
            ax.plot(x_vals,
                    map(model.scaleup_fns[program],
                        x_vals),
                    label=program)
            data_to_plot = {}
            for i in model.data['programs'][program]:
                if i > start_time:
                    data_to_plot[i] = model.data['programs'][program][i]
            ax.scatter(data_to_plot.keys(),
                    data_to_plot.values())
            ax.set_xticks([start_time, end_time])
            for tick in ax.xaxis.get_major_ticks():
                tick.label.set_fontsize(6)
            for tick in ax.yaxis.get_major_ticks():
                tick.label.set_fontsize(5)
            ax.legend([program[13:], 'data'], loc=2, frameon=False, prop={'size': 5})
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


