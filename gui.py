import collections
import threading

import autumn.model_runner
import autumn.outputs as outputs
import autumn.tool_kit as tool_kit

from Tkinter import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


def find_button_name_from_string(working_string):
    button_name_dictionary = {
        'output_uncertainty':
            'Run uncertainty',
        'write_uncertainty_outcome_params':
            'Record parameters',
        'output_spreadsheets':
            'Write to spreadsheets',
        'output_documents':
            'Write to documents',
        'output_by_scenario':
            'Output by scenario',
        'output_horizontally':
            'Write horizontally',
        'output_gtb_plots':
            'Plot outcomes',
        'output_compartment_populations':
            'Plot compartment sizes',
        'output_by_subgroups':
            'Plot outcomes by sub-groups',
        'output_age_fractions':
            'Plot proportions by age',
        'output_riskgroup_fractions':
            'Plot proportions by risk group',
        'output_flow_diagram':
            'Draw flow diagram',
        'output_fractions':
            'Plot compartment fractions',
        'output_scaleups':
            'Plot scale-up functions',
        'output_plot_economics':
            'Plot economics graphs',
        'output_plot_riskgroup_checks':
            'Plot risk group checks',
        'output_age_calculations':
            'Plot age calculation weightings',
        'output_param_plots':
            'Plot parameter progression',
        'output_popsize_plot':
            'Plot "popsizes" for cost-coverage curves',
        'output_likelihood_plot':
            'Plot log likelihoods over runs',
        'riskgroup_diabetes':
            'Type II diabetes',
        'riskgroup_hiv':
            'HIV',
        'riskgroup_prison':
            'Prison',
        'riskgroup_urbanpoor':
            'Urban poor',
        'riskgroup_ruralpoor':
            'Rural poor',
        'riskgroup_indigenous':
            'Indigenous',
        'is_lowquality':
            'Low quality care',
        'is_amplification':
            'Resistance amplification',
        'is_timevariant_organs':
            'Time-variant organ status',
        'is_misassignment':
            'Strain mis-assignment',
        'is_vary_detection_by_organ':
            'Vary case detection by organ status',
        'n_organs':
            'Number of organ strata',
        'n_strains':
            'Number of strains',
        'is_vary_force_infection_by_riskgroup':
            'Heterogeneous mixing',
        'is_treatment_history':
            'Treatment history'}

    if working_string in button_name_dictionary:
        return button_name_dictionary[working_string]
    elif 'scenario_' in working_string:
        return tool_kit.capitalise_first_letter(tool_kit.replace_underscore_with_space(working_string))
    else:
        return working_string


def get_autumn_params():
    params = collections.OrderedDict()

    bool_keys = [
        'output_flow_diagram', 'output_compartment_populations', 'output_riskgroup_fractions',
        'output_age_fractions', 'output_by_subgroups', 'output_fractions', 'output_scaleups',
        'output_gtb_plots', 'output_plot_economics', 'output_plot_riskgroup_checks',
        'output_param_plots', 'output_popsize_plot', 'output_likelihood_plot',
        'output_uncertainty', 'write_uncertainty_outcome_params', 'output_spreadsheets',
        'output_documents', 'output_by_scenario', 'output_horizontally',
        'output_age_calculations', 'riskgroup_diabetes', 'riskgroup_hiv',
        'riskgroup_prison', 'riskgroup_indigenous', 'riskgroup_urbanpoor',
        'riskgroup_ruralpoor',
        'is_lowquality', 'is_amplification', 'is_misassignment', 'is_vary_detection_by_organ',
        'is_timevariant_organs', 'is_treatment_history',
        'is_vary_force_infection_by_riskgroup']

    for i in range(1, 15):
        bool_keys.append('scenario_' + str(i))

    for key in bool_keys:
        params[key] = {
            'value': False,
            'type': "boolean",
            'label': find_button_name_from_string(key),
        }

    default_boolean_keys = [
        # 'output_uncertainty',
        # 'write_uncertainty_outcome_params',
        'output_param_plots',
        'is_amplification',
        'is_misassignment',
        # 'is_lowquality',
        'output_riskgroup_fractions',
        'is_vary_detection_by_organ',
        'is_treatment_history',
        'riskgroup_prison',
        # 'riskgroup_urbanpoor',
        'output_scaleups',
        # 'output_by_subgroups',
        'riskgroup_ruralpoor',
        'output_gtb_plots',
        'is_vary_force_infection_by_riskgroup',
        'riskgroup_diabetes',
        # 'riskgroup_hiv',
        # 'riskgroup_indigenous',
        # 'is_timevariant_organs'
    ]

    for k in default_boolean_keys:
        params[k]['value'] = True

    # Model running options
    options = [
        'Afghanistan', 'Albania', 'Angola', 'Argentina', 'Armenia', 'Australia', 'Austria',
        'Azerbaijan', 'Bahrain', 'Bangladesh', 'Belarus', 'Belgium', 'Benin',
        'Bhutan', 'Botswana', 'Brazil', 'Bulgaria', 'Burundi', 'Cameroon', 'Chad',
        'Chile', 'Croatia', 'Djibouti', 'Ecuador', 'Estonia', 'Ethiopia', 'Fiji', 'Gabon', 'Georgia',
        'Ghana', 'Guatemala', 'Guinea',
        'Philippines', 'Romania'
    ]
    params['country'] = {
        'type': 'drop_down',
        'options': options,
        'value': 'Bulgaria'
    }

    options = ['Runge Kutta', 'Explicit']
    params['integration_method'] = {
        'type': 'drop_down',
        'options': options,
        'value': options[1]
    }

    options = ['Method 1', 'Method 2', 'Method 3', 'Method 4', 'Method 5']
    params['fitting_method'] = {
        'type': 'drop_down',
        'options': options,
        'value': options[-1]
    }

    params['default_smoothness'] = {
        'type': 'slider',
        'label': 'Default fitting smoothness',
        'value': 1.0,
        'interval': 0.1,
        'min': 0.0,
        'max': 5.0,
    }
    params['time_step'] = {
        'type': 'slider',
        'label': 'Integration time step',
        'value': 0.5,
        'min': 0.005,
        'max': 0.5,
        'interval': 0.005
    }

    # Model stratifications options
    options = ['Pos / Neg / Extra', 'Pos / Neg', 'Unstratified']
    params['n_organs'] = {
        'type': 'drop_down',
        'options': options,
        'value': options[0]
    }

    options = ['Single strain', 'DS / MDR', 'DS / MDR / XDR']
    params['n_strains'] = {
        'type': 'drop_down',
        'options': options,
        'value': options[1]
    }

    # Uncertainty options
    params['uncertainty_runs'] = {
        'type': 'integer',
        'value': 5,
        'label': 'Number of uncertainty runs'
    }
    params['burn_in_runs'] = {
        'type': 'integer',
        'value': 0,
        'label': 'Number of burn-in runs'
    }
    params['search_width'] = {
        'type': 'double',
        'value': 0.08,
        'label': 'Relative search width'
    }
    options = ['No saving or loading', 'Load', 'Save']
    params['pickle_uncertainty'] = {
        'type': 'drop_down',
        'options': options,
        'value': options[0]
    }

    for key, value in params.items():
        if not value.get('label'):
            value['label'] = key

    param_groups = [
        {'keys': [], 'name': 'Model running'},
        {'keys': [], 'name': 'Model Stratifications'},
        {'keys': [], 'name': 'Elaborations'},
        {'keys': [], 'name': 'Scenarios to run'},
        {'keys': [], 'name': 'Uncertainty'},
        {'keys': [], 'name': 'Plotting'},
        {'keys': [], 'name': 'MS Office outputs'}
    ]

    for key in bool_keys:
        name = params[key]['label']
        if ('Plot' in name or 'Draw' in name):
            param_groups[5]['keys'].append(key)
        elif ('uncertainty' in name or 'uncertainty' in key):
            param_groups[4]['keys'].append(key)
        elif 'is_' in key:
            param_groups[2]['keys'].append(key)
        elif ('riskgroup_' in key or 'n_' in key):
            param_groups[1]['keys'].append(key)
        elif 'scenario_' in key:
            param_groups[3]['keys'].append(key)
        else:
            param_groups[6]['keys'].append(key)

    for k in ['country', 'integration_method', 'fitting_method',
              'default_smoothness', 'time_step']:
        param_groups[0]['keys'].append(k)

    for k in ['n_organs', 'n_strains']:
        param_groups[1]['keys'].append(k)

    for k in ['uncertainty_runs', 'burn_in_runs',
              'search_width', 'pickle_uncertainty']:
        param_groups[4]['keys'].append(k)

    return {
        'params': params,
        'param_groups': param_groups
    }


def convert_params_to_inputs(params):
    organ_stratification_keys = {
        'Pos / Neg / Extra': 3,
        'Pos / Neg': 2,
        'Unstratified': 0}
    strain_stratification_keys = {
        'Single strain': 0,
        'DS / MDR': 2,
        'DS / MDR / XDR': 3}

    inputs = {}
    inputs['scenarios_to_run'] = [0]
    inputs['scenario_names_to_run'] = ['baseline']
    for key, param in params.iteritems():
        value = param['value']
        if param['type'] == "boolean":
            if 'scenario_' not in key:
                inputs[key] = param['value']
            elif param['value']:
                i_scenario = int(key[9:])
                inputs['scenarios_to_run'] \
                    += [i_scenario]
                inputs['scenario_names_to_run'] \
                    += [tool_kit.find_scenario_string_from_number(i_scenario)]
        elif param['type'] == "drop_down":
            if key == 'fitting_method':
                inputs[key] = int(value[-1])
            elif key == 'n_organs':
                inputs[key] = organ_stratification_keys[value]
            elif key == 'n_strains':
                inputs[key] = strain_stratification_keys[value]
            else:
                inputs[key] = value
        else:
            inputs[key] = value

    return inputs


class App:
    def __init__(self, master):
        """
        All processes involved in setting up the first basic GUI frame are collated here.

        Args:
            master: The GUI
        """

        # prepare data structures
        self.gui_outputs = {}
        self.thread_number = 0

        # set up first frame
        self.master = master
        self.frame = Frame(master)
        self.frame.pack()
        self.figure_frame = Toplevel(master)
        self.figure_frame.title('Tracking parameters over model runs')
        self.master.minsize(1500, 500)
        self.master.title('AuTuMN - version 1.0')

        self.title_font = 'Helvetica 10 bold italic'
        self.label_font = 'Helvetica 9 bold italic'

        autumn_params = get_autumn_params()

        self.params = autumn_params['params']
        self.make_tk_controls_in_params()

        self.param_groups = autumn_params['param_groups']
        self.set_tk_controls_in_frame()

    def make_tk_controls_in_params(self):
        for key, param in self.params.iteritems():
            if param['type'] == "boolean":
                param['tk_var'] = IntVar()
                if param['value'] == True:
                    param['tk_var'].set(True)
                param['tk_control'] = Checkbutton(
                    self.frame,
                    text=param['label'],
                    variable=param['tk_var'],
                    pady=5)
            if param['type'] == 'drop_down':
                param['tk_var'] = StringVar()
                param['tk_var'].set(param['value'])
                param['tk_control'] = OptionMenu(
                    self.frame,
                    param['tk_var'],
                    *param['options'])
            if param['type'] == 'integer':
                param['tk_var'] = IntVar()
                param['tk_var'].set(param['value'])
                param['tk_label'] = Label(
                    self.frame,
                    text=param['label'],
                    font=self.label_font)
                param['tk_control'] = Entry(
                    self.frame,
                    textvariable=param['tk_var'])
            if param['type'] == 'double':
                param['tk_var'] = DoubleVar()
                param['tk_var'].set(param['value'])
                param['tk_label'] = Label(
                    self.frame,
                    text=param['label'],
                    font=self.label_font)
                param['tk_control'] = Entry(
                    self.frame,
                    textvariable=param['tk_var'])
            if param['type'] == 'slider':
                param['tk_var'] = DoubleVar()
                param['tk_var'].set(param['value'])
                param['tk_label'] = Label(
                    self.frame,
                    text=param['label'],
                    font=self.label_font)
                param['tk_control'] = Scale(
                    self.frame,
                    from_=param['min'],
                    to=param['max'],
                    resolution=param['interval'],
                    orient=HORIZONTAL,
                    width=10,
                    length=130,
                    sliderlength=20,
                    variable=param['tk_var'])

    def set_tk_controls_in_frame(self):
        for column, param_group in enumerate(self.param_groups):
            row = 0
            title = Label(self.frame, text=param_group['name'])
            title.grid(row=row, column=column, sticky=NW, pady=10)
            title.config(font=self.title_font)
            self.frame.grid_columnconfigure(column, minsize=200)
            row += 1
            if column == 0:
                # model running button
                self.run = Button(self.frame, text='Run', command=self.execute, width=10)
                self.run.grid(row=1, column=0, sticky=W, padx=2)
                self.run.config(font=self.title_font, fg='darkgreen', bg='grey')
                row += 1
            for key in param_group['keys']:
                param = self.params[key]
                if 'tk_label' in param:
                    param['tk_label'].grid(row=row, column=column, sticky=SW)
                    row += 1
                param['tk_control'].grid(row=row, column=column, sticky=W)
                row += 1
            if column == 0:
                # creating main output window
                console_label = Label(
                    self.frame,
                    text='Runtime outputs console',
                    font=self.label_font)
                console_label.grid(row=row, column=0, sticky=SW)
                row += 1
                self.runtime_outputs = Text(self.frame)
                self.runtime_outputs.grid(row=row, column=0, rowspan=5, columnspan=3)
                self.runtime_outputs.config(height=9)

    def execute(self):
        """
        This is the main method to run the model. It replaces test_full_models.py
        """

        for param in self.params.values():
            if param['type'] == "boolean":
                param['value'] = bool(param['tk_var'].get())
            else:
                param['value'] = param['tk_var'].get()

        self.gui_outputs = convert_params_to_inputs(self.params)

        with open('bgui_outputs.json', 'w') as f:
            import json
            json.dump(self.gui_outputs, f, indent=2)

        # record thread number
        self.thread_number += 1

        # indicate processing has started
        self.runtime_outputs.insert(
            END,
            '_____________________________________________\n'
                + 'Model run commenced using thread #%d.\n'
                % self.thread_number)
        self.runtime_outputs.see(END)

        # use multiple threads to deal with multiple user calls to run the model through the run button
        is_romain = False  # set True for Romain's computer, otherwise leave False
        if is_romain:
            self.run_model()
        else:
            execution_threads = []
            execution_thread = threading.Thread(target=self.run_model)
            execution_threads.append(execution_thread)
            execution_thread.start()

    def run_model(self):
        """
        Here the the objects to actually perform the model run are called after the thread has been initialised in the
        execute method above.

        """

        # for some unknown reason, this code has started bugging - commented out
        # if not self.gui_outputs['output_uncertainty']:
        #     self.figure_frame.withdraw()

        self.model_runner = autumn.model_runner.ModelRunner(
            self.gui_outputs, self.runtime_outputs, js_gui=self.handle_message)
        self.model_runner.master_runner()
        project = outputs.Project(self.model_runner, self.gui_outputs)
        project.master_outputs_runner()

    def handle_message(self, command, data={}):
        if command == "console":
            self.runtime_outputs.insert(END, data["message"] + '\n')
            self.runtime_outputs.see(END)
        elif command == "graph":
            self.graph(data)

    def graph(self, data, input_figure=None):
        import json
        with open('graph_data.json', 'wt') as f:
            json.dump(data, f, indent=2)

        # initialise plotting
        if not input_figure:
            param_tracking_figure = plt.Figure()
            parameter_plots = FigureCanvasTkAgg(param_tracking_figure, master=self.figure_frame)

        else:
            param_tracking_figure = input_figure

        subplot_grid = outputs.find_subplot_numbers(len(data["all_parameters_tried"]))

        # cycle through parameters with one subplot for each parameter
        for p, param in enumerate(data["all_parameters_tried"]):

            # extract accepted params from all tried params
            accepted_params = list(
                p for p, a in zip(
                    data["all_parameters_tried"][param],
                    data["whether_accepted_list"])
                if a)

            # plot
            ax = param_tracking_figure.add_subplot(subplot_grid[0], subplot_grid[1], p + 1)
            ax.plot(range(1, len(accepted_params) + 1), accepted_params, linewidth=2, marker='o', markersize=4,
                    mec='b', mfc='b')
            ax.set_xlim((1., len(data["accepted_indices"]) + 1))

            # find the y-limits from the parameter bounds and the parameter values tried
            for param_number in range(len(data["param_ranges_unc"])):
                if data["param_ranges_unc"][param_number]['key'] == param:
                    bounds = data["param_ranges_unc"][param_number]['bounds']
            ylim_margins = .1
            min_ylimit = min(accepted_params + [bounds[0]])
            max_ylimit = max(accepted_params + [bounds[1]])
            ax.set_ylim((min_ylimit * (1 - ylim_margins), max_ylimit * (1 + ylim_margins)))

            # indicate the prior bounds
            ax.plot([1, len(data["accepted_indices"]) + 1], [min_ylimit, min_ylimit], color='0.8')
            ax.plot([1, len(data["accepted_indices"]) + 1], [max_ylimit, max_ylimit], color='0.8')

            # plot rejected parameters
            for run, rejected_params in data["rejection_dict"][param].items():
                if data["rejection_dict"][param][run]:
                    ax.plot([run + 1] * len(rejected_params), rejected_params, marker='o', linestyle='None',
                            mec='0.5', mfc='0.5', markersize=3)
                    for r in range(len(rejected_params)):
                        ax.plot([run, run + 1], [data["acceptance_dict"][param][run], rejected_params[r]], color='0.5',
                                linestyle='--')

            # label
            ax.set_title(data["names"][param])
            if p > len(data["all_parameters_tried"]) - subplot_grid[1] - 1:
                ax.set_xlabel('Accepted runs')

            if not input_figure:
                # output to GUI window
                parameter_plots.show()
                parameter_plots.draw()
                parameter_plots.get_tk_widget().grid(row=1, column=1)

        if input_figure:
            return param_tracking_figure



if __name__ == '__main__':
    root = Tk()
    app = App(root)
    root.mainloop()
