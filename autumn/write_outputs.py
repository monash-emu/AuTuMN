
import os
import glob
import datetime
import autumn.model
import autumn.plotting
from autumn.spreadsheet import read_input_data_xls
import numpy as np
import openpyxl as xl
import tool_kit
from docx import Document
from matplotlib import pyplot, patches
import numpy


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

    autumn.plotting.humanise_y_ticks(ax)


def indices(a, func):
    return [i for (i, val) in enumerate(a) if func(val)]


class Project:

    def __init__(self, country, inputs):

        """
        Initialises an object of class Project, that will contain all the information (data + outputs) for writing a
        report for a country
        Args:
            models: dictionary such as: models = {'baseline': model, 'scenario_1': model_1,  ...}
        """

        self.country = country.lower()
        self.name = 'test_' + self.country
        self.scenarios = []
        self.models = {}
        self.full_output_dict = {}
        self.integer_output_dict = {}
        self.inputs = inputs

    #################################
    # General methods for use below #
    #################################

    def find_or_make_directory(self):

        out_dir_project = os.path.join('projects', self.name)
        if not os.path.isdir(out_dir_project):
            os.makedirs(out_dir_project)
        return out_dir_project

    def make_path(self, filename):

        # Sort out directory if not already sorted
        out_dir_project = os.path.join(filename, self.name)
        if not os.path.isdir(out_dir_project):
            os.makedirs(out_dir_project)
        return out_dir_project

    def find_years_to_write(self, scenario, output, minimum=0, maximum=3000, step=1):

        requested_years = range(minimum, maximum, step)
        years = []
        for y in self.integer_output_dict[scenario][output].keys():
            if y in requested_years:
                years += [y]
        return years

    #########################################
    # Methods to collect data for later use #
    #########################################

    def create_output_dicts(self, outputs=['incidence', 'mortality', 'prevalence', 'notifications']):

        """
        Works through all the methods to this object that are required to populate the output dictionaries.
        First the "full" ones with all time point included, then the abbreviated ones.

        Args:
            outputs: The outputs to be populated to the dictionaries
        """

        self.create_full_output_dict(outputs)
        self.add_full_economics_dict()
        self.extract_integer_dict()

    def create_full_output_dict(self, outputs):

        """
        Creates a dictionary for each requested output at every time point in that model's times attribute
        """

        for scenario in self.scenarios:
            self.full_output_dict[scenario] = {}
            for label in outputs:
                times = self.models[scenario].times
                solution = self.models[scenario].get_var_soln(label)
                self.full_output_dict[scenario][label] = dict(zip(times, solution))

    def add_full_economics_dict(self):

        """
        Creates an economics dictionary structure that mirrors that of the epi dictionaries and adds
        this to the main outputs (epi) dictionary
        """

        for model in self.models:
            economics_dict = {}
            for intervention in self.models[model].costs:
                if intervention != 'cost_times':
                    economics_dict['cost_' + intervention] = {}
                    for t in range(len(self.models[model].costs['cost_times'])):
                        economics_dict['cost_' + intervention][self.models[model].costs['cost_times'][t]] \
                            = self.models[model].costs[intervention]['raw_cost'][t]
            self.full_output_dict[model].update(economics_dict)

    def extract_integer_dict(self):

        """
        Extracts a dictionary from full_output_dict with only integer years, using the first time value greater than
        the integer year in question.
        """

        for model in self.models:
            self.integer_output_dict[model] = {}
            for output in self.full_output_dict[model]:
                self.integer_output_dict[model][output] = {}
                times = self.full_output_dict[model][output].keys()
                times.sort()
                start = np.floor(times[0])
                finish = np.floor(times[-1])
                float_years = np.linspace(start, finish, finish - start + 1.)
                for year in float_years:
                    key = [t for t in times if t >= year][0]
                    self.integer_output_dict[model][output][int(key)] \
                        = self.full_output_dict[model][output][key]

    #################################################
    # Methods for outputting to Office applications #
    #################################################

    def write_spreadsheets(self):

        """
        Determine whether to write to spreadsheets by scenario or by output
        """

        if self.inputs.model_constants['output_spreadsheets']:
            if self.inputs.model_constants['output_by_scenario']:
                print('Writing scenario spreadsheets')
                self.write_xls_by_scenario()
            else:
                print('Writing output indicator spreadsheets')
                self.write_xls_by_output()

    def write_xls_by_output(self):

        # Find directory to write to
        out_dir_project = self.find_or_make_directory()

        # Write a new file for each output
        outputs = self.integer_output_dict['baseline'].keys()
        for output in outputs:

            # Make filename
            path = os.path.join(out_dir_project, output)
            path += ".xlsx"

            # Get active sheet
            wb = xl.Workbook()
            sheet = wb.active
            sheet.title = output

            # Write a new file for each epidemiological indicator
            for scenario in self.integer_output_dict.keys():
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
        if self.inputs.model_constants['output_horizontally']:
            self.write_horizontally_by_scenario(sheet, output, years)
        else:
            self.write_vertically_by_scenario(sheet, output, years)

    def write_xls_by_scenario(self):

        # Find directory to write to
        out_dir_project = self.find_or_make_directory()

        # Write a new file for each scenario
        scenarios = self.integer_output_dict.keys()
        for scenario in scenarios:

            # Make filename
            path = os.path.join(out_dir_project, scenario)
            path += '.xlsx'

            # Get active sheet
            wb = xl.Workbook()
            sheet = wb.active
            sheet.title = scenario

            for output in self.integer_output_dict['baseline'].keys():
                self.write_xls_column_or_row(sheet, scenario, output)

            # Save workbook
            wb.save(path)

    def write_horizontally_by_scenario(self, sheet, output, years):

        sheet.cell(row=1, column=1).value = 'Year'

        col = 1
        for y in years:
            col += 1
            sheet.cell(row=1, column=col).value = y

        r = 1
        for scenario in self.scenarios:
            r += 1
            sheet.cell(row=r, column=1).value = \
                tool_kit.replace_underscore_with_space(
                    tool_kit.capitalise_first_letter(scenario))
            col = 1
            for y in years:
                col += 1
                if y in self.integer_output_dict[scenario][output]:
                    sheet.cell(row=r, column=col).value \
                        = self.integer_output_dict[scenario][output][y]

    def write_horizontally_by_output(self, sheet, scenario, years):

        sheet.cell(row=1, column=1).value = 'Year'

        col = 1
        for y in years:
            col += 1
            sheet.cell(row=1, column=col).value = y

        r = 1
        for output in self.integer_output_dict['baseline'].keys():
            r += 1
            sheet.cell(row=r, column=1).value = \
                tool_kit.replace_underscore_with_space(
                    tool_kit.capitalise_first_letter(output))
            col = 1
            for y in years:
                col += 1
                if y in self.integer_output_dict[scenario][output]:
                    sheet.cell(row=r, column=col).value \
                        = self.integer_output_dict[scenario][output][y]

    def write_vertically_by_scenario(self, sheet, output, years):

        sheet.cell(row=1, column=1).value = 'Year'

        row = 1
        for y in years:
            row += 1
            sheet.cell(row=row, column=1).value = y

        col = 1
        for scenario in self.scenarios:
            col += 1
            sheet.cell(row=1, column=col).value = \
                tool_kit.replace_underscore_with_space(
                    tool_kit.capitalise_first_letter(scenario))
            row = 1
            for y in years:
                row += 1
                if y in self.integer_output_dict[scenario][output]:
                    sheet.cell(row=row, column=col).value = self.integer_output_dict[scenario][output][y]

    def write_vertically_by_output(self, sheet, scenario, years):

        sheet.cell(row=1, column=1).value = 'Year'

        row = 1
        for y in years:
            row += 1
            sheet.cell(row=row, column=1).value = y

        col = 1
        for output in self.integer_output_dict['baseline'].keys():
            col += 1
            sheet.cell(row=1, column=col).value = \
                tool_kit.replace_underscore_with_space(
                    tool_kit.capitalise_first_letter(output))
            row = 1
            for y in years:
                row += 1
                if y in self.integer_output_dict[scenario][output]:
                    sheet.cell(row=row, column=col).value \
                        = self.integer_output_dict[scenario][output][y]

    def write_documents(self):

        """
        Determine whether to write to documents by scenario or by output
        """

        if self.inputs.model_constants['output_documents']:
            if self.inputs.model_constants['output_by_scenario']:
                print('Writing scenario documents')
                self.write_docs_by_scenario()
            else:
                print('Writing output indicator documents')
                self.write_docs_by_output()

    def write_docs_by_output(self):

        # Find directory to write to
        out_dir_project = self.find_or_make_directory()

        # Write a new file for each output
        outputs = self.integer_output_dict['baseline'].keys()

        for output in outputs:

            # Initialise document
            path = os.path.join(out_dir_project, output)
            path += ".docx"
            document = Document()
            table = document.add_table(rows=1, cols=len(self.scenarios) + 1)

            # Write headers
            header_cells = table.rows[0].cells
            header_cells[0].text = 'Year'
            for scenario_no, scenario in enumerate(self.scenarios):
                header_cells[scenario_no + 1].text \
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

                for sc, scenario in enumerate(self.scenarios):
                    if year in self.integer_output_dict[scenario][output]:
                        row_cells[sc + 1].text = '%.2f' % self.integer_output_dict[scenario][output][year]

            # Save document
            document.save(path)

    def write_docs_by_scenario(self):

        # Find directory to write to
        out_dir_project = self.find_or_make_directory()

        # Write a new file for each output
        outputs = self.integer_output_dict['baseline'].keys()

        for scenario in self.scenarios:

            # Initialise document
            path = os.path.join(out_dir_project, scenario)
            path += ".docx"
            document = Document()
            table = document.add_table(rows=1, cols=len(outputs) + 1)

            # Write headers
            header_cells = table.rows[0].cells
            header_cells[0].text = 'Year'
            for output_no, output in enumerate(outputs):
                header_cells[output_no + 1].text \
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

                for out, output in enumerate(outputs):
                    if year in self.integer_output_dict[scenario][output]:
                        row_cells[out + 1].text = '%.2f' % self.integer_output_dict[scenario][output][year]

            # Save document
            document.save(path)

    def run_plotting(self):

        # Plot scale-up functions - currently only doing this for the baseline model run
        if self.inputs.model_constants['output_scaleups']:
            self.plot_classified_scaleups(self.models['baseline'])

    def plot_classified_scaleups(self, model):

        # Classify scale-up functions
        classifications = ['demo_', 'econ_', 'epi_', 'program_prop_', 'program_timeperiod']
        classified_scaleups = {}
        for classification in classifications:
            classified_scaleups[classification] = []
            for fn in model.scaleup_fns:
                if classification in fn:
                    classified_scaleups[classification] += [fn]

        out_dir_project = self.find_or_make_directory()
        base = os.path.join(out_dir_project, self.country + '_baseline_')

        # Time periods to perform the plots over
        times_to_plot = ['start_', 'recent_']

        # Plot them from the start of the model and from "recent_time"
        for c, classification in enumerate(classified_scaleups):
            if len(classified_scaleups[classification]) > 0:
                for j, start_time in enumerate(times_to_plot):
                    self.plot_all_scaleup_fns_against_data(model,
                                                           classified_scaleups[classification],
                                                           base + classification + '_datascaleups_from' + start_time[:-1] + '.png',
                                                           start_time + 'time',
                                                           'current_time',
                                                           classification,
                                                           figure_number=c + j * len(classified_scaleups) + 2)
                    if classification == 'program_prop':
                        autumn.plotting.plot_scaleup_fns(model,
                                                         classified_scaleups[classification],
                                                         base + classification + 'scaleups_from' + start_time[:-1] + '.png',
                                                         start_time + 'time',
                                                         'current_time',
                                                         classification,
                                                         figure_number=c + j * len(classified_scaleups) + 2 + len(classified_scaleups) * len(times_to_plot))

    def plot_all_scaleup_fns_against_data(self, model, functions, png=None,
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
            output_colour = [autumn.plotting.make_default_line_styles(scenario, False)[1]] * len(functions)

        # Determine how many subplots to have
        subplot_grid = autumn.plotting.find_subplot_numbers(len(functions))

        # Set x-values
        if start_time_str == 'recent_time':
            start_time = model.inputs.model_constants[start_time_str]
        else:
            start_time = model.inputs.model_constants[start_time_str]
        end_time = model.inputs.model_constants[end_time_str]
        x_vals = numpy.linspace(start_time, end_time, 1E3)

        # Initialise figure
        fig = pyplot.figure(figure_number)

        # Upper title for whole figure
        plural = ''
        if len(functions) > 1:
            plural += 's'
        title = model.inputs.model_constants['country'] + ' ' + \
                tool_kit.find_title_from_dictionary(parameter_type) + \
                ' parameter' + plural + tool_kit.find_title_from_dictionary(start_time_str)
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
                for j in model.inputs.scaleup_data[scenario][function]:
                    if j > start_time:
                        data_to_plot[j] = model.inputs.scaleup_data[scenario][function][j]

                # Scatter plot data from which they are derived
                ax.scatter(data_to_plot.keys(),
                           data_to_plot.values(),
                           color=output_colour[figure_number],
                           s=6)

                # Adjust tick font size
                ax.set_xticks([start_time, end_time])
                for axis_to_change in [ax.xaxis, ax.yaxis]:
                    for tick in axis_to_change.get_major_ticks():
                        tick.label.set_fontsize(autumn.plotting.get_nice_font_size(subplot_grid))

                # Truncate parameter names depending on whether it is a
                # treatment success/death proportion
                title = tool_kit.find_title_from_dictionary(function)
                ax.set_title(title, fontsize=autumn.plotting.get_nice_font_size(subplot_grid))

                ylims = autumn.plotting.relax_y_axis(ax)
                ax.set_ylim(bottom=ylims[0], top=ylims[1])

        fig.suptitle('Scale-up functions')

        autumn.plotting.save_png(png)

