
from Tkinter import *
import autumn.model_runner
import autumn.outputs
import autumn.tool_kit
import threading


def find_button_name_from_string(working_string):

    button_name_dictionary = {'output_uncertainty':
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
        return autumn.tool_kit.capitalise_first_letter(autumn.tool_kit.replace_underscore_with_space(working_string))
    else:
        return working_string


class App:
    def __init__(self, master):
        """
        All processes involved in setting up the first basic GUI frame are collated here.

        Args:
            master: The GUI
        """

        # prepare data structures
        self.gui_outputs = {}
        self.raw_outputs = {}
        self.drop_downs = {}
        self.thread_number = 0

        # set up first frame
        self.master = master
        self.frame = Frame(master)
        self.frame.pack()
        self.figure_frame = Toplevel(master)
        self.figure_frame.title('Tracking parameters over model runs')
        self.master.minsize(1500, 500)
        self.master.title('AuTuMN - version 1.0')

        # model running button
        self.run = Button(self.frame, text='Run', command=self.execute, width=10)
        self.run.grid(row=1, column=0, sticky=W, padx=2)
        self.run.config(font='Helvetica 10 bold italic', fg='darkgreen', bg='grey')

        # creating main output window
        self.runtime_outputs = Text(self.frame)
        self.runtime_outputs.grid(row=10, column=0, rowspan=5, columnspan=3)
        self.runtime_outputs.config(height=9)

        # prepare Boolean data structures
        self.boolean_dictionary = {}
        self.boolean_inputs = ['output_flow_diagram', 'output_compartment_populations', 'output_riskgroup_fractions',
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
            self.boolean_inputs += ['scenario_' + str(i)]

        for boolean in self.boolean_inputs:
            self.boolean_dictionary[boolean] = IntVar()

        # set and collate checkboxes
        self.boolean_toggles = {}
        for boolean in self.boolean_inputs:
            self.boolean_toggles[boolean] = Checkbutton(self.frame,
                                                        text=find_button_name_from_string(boolean),
                                                        variable=self.boolean_dictionary[boolean],
                                                        pady=5)
        plot_row = 1
        option_row = 1
        uncertainty_row = 1
        riskgroup_row = 1
        elaboration_row = 1
        running_row = 2
        scenario_row = 1
        for boolean in self.boolean_inputs:
            if 'Plot ' in find_button_name_from_string(boolean) \
                    or 'Draw ' in find_button_name_from_string(boolean):
                self.boolean_toggles[boolean].grid(row=plot_row, column=5, sticky=W)
                plot_row += 1
            elif 'uncertainty' in find_button_name_from_string(boolean) or 'uncertainty' in boolean:
                self.boolean_toggles[boolean].grid(row=uncertainty_row, column=4, sticky=W)
                uncertainty_row += 1
            elif 'is_' in boolean:
                self.boolean_toggles[boolean].grid(row=elaboration_row, column=2, sticky=W)
                elaboration_row += 1
            elif 'riskgroup_' in boolean or 'n_' in boolean:
                self.boolean_toggles[boolean].grid(row=riskgroup_row, column=1, sticky=W)
                riskgroup_row += 1
            elif 'scenario_' in boolean:
                self.boolean_toggles[boolean].grid(row=scenario_row, column=3, sticky=W)
                scenario_row += 1
            else:
                self.boolean_toggles[boolean].grid(row=option_row, column=6, sticky=W)
                option_row += 1

        # drop down menus for multiple options
        running_dropdown_list = ['country', 'integration_method', 'fitting_method']
        for dropdown in running_dropdown_list:
            self.raw_outputs[dropdown] = StringVar()
        self.raw_outputs['country'].set('Fiji')
        self.raw_outputs['integration_method'].set('Explicit')
        self.raw_outputs['fitting_method'].set('Method 5')

        self.drop_downs['country'] \
            = OptionMenu(self.frame, self.raw_outputs['country'],
                         'Afghanistan', 'Albania', 'Angola', 'Argentina', 'Armenia', 'Australia', 'Austria',
                         'Azerbaijan', 'Bahrain', 'Bangladesh', 'Belarus', 'Belgium', 'Benin',
                         'Bhutan', 'Botswana', 'Brazil', 'Bulgaria', 'Burundi', 'Cameroon', 'Chad',
                         'Chile', 'Croatia', 'Djibouti', 'Ecuador', 'Estonia', 'Ethiopia', 'Fiji', 'Gabon', 'Georgia',
                         'Ghana', 'Guatemala', 'Guinea',
                         'Philippines', 'Romania')
        self.drop_downs['integration_method'] \
            = OptionMenu(self.frame, self.raw_outputs['integration_method'],
                         'Runge Kutta', 'Explicit')
        self.drop_downs['fitting_method'] \
            = OptionMenu(self.frame, self.raw_outputs['fitting_method'],
                         'Method 1', 'Method 2', 'Method 3', 'Method 4', 'Method 5')
        for drop_down in running_dropdown_list:
            self.drop_downs[drop_down].grid(row=running_row, column=0, sticky=W)
            running_row += 1

        # model stratifications options
        numerical_stratification_inputs = ['n_organs', 'n_strains']
        for option in numerical_stratification_inputs:
            self.raw_outputs[option] = StringVar()
        self.raw_outputs['n_organs'].set('Pos / Neg / Extra')
        self.raw_outputs['n_strains'].set('Single strain')
        self.drop_downs['n_organs'] = OptionMenu(self.frame, self.raw_outputs['n_organs'],
                                                 'Pos / Neg / Extra',
                                                 'Pos / Neg',
                                                 'Unstratified')
        self.drop_downs['n_strains'] = OptionMenu(self.frame, self.raw_outputs['n_strains'],
                                                  'Single strain', 'DS / MDR', 'DS / MDR / XDR')
        for option in numerical_stratification_inputs:
            self.drop_downs[option].grid(row=riskgroup_row, column=1, sticky=W)
            riskgroup_row += 1

        # consistent width to drop-down menus
        for drop_down in self.drop_downs:
            self.drop_downs[drop_down].config(width=15)

        # column titles (order important)
        column_titles = ['Model running', 'Model stratifications', 'Elaborations', 'Scenarios to run', 'Uncertainty',
                         'Plotting', 'MS Office outputs']
        for i in range(len(column_titles)):
            title = Label(self.frame, text=column_titles[i])
            title.grid(row=0, column=i, sticky=NW, pady=10)
            title.config(font='Helvetica 10 bold italic')
            self.frame.grid_columnconfigure(i, minsize=200)

        # sliders
        slider_list = ['default_smoothness', 'time_step']
        sliders = {}
        for slide in slider_list:
            self.raw_outputs[slide] = DoubleVar()
        self.raw_outputs['default_smoothness'].set(1.)
        self.raw_outputs['time_step'].set(.5)
        label_font = 'Helvetica 9 bold italic'
        slider_labels = {'default_smoothness':
                             Label(self.frame, text='Default fitting smoothness', font=label_font),
                         'time_step':
                             Label(self.frame, text='Integration time step', font=label_font)}
        sliders['default_smoothness'] = Scale(self.frame, from_=0, to=5, resolution=.1, orient=HORIZONTAL,
                                              width=10, length=130, sliderlength=20,
                                              variable=self.raw_outputs['default_smoothness'])
        sliders['time_step'] = Scale(self.frame, from_=0.005, to=.5, resolution=.005, orient=HORIZONTAL,
                                     width=10, length=130, sliderlength=20, variable=self.raw_outputs['time_step'])
        for l, label in enumerate(slider_list):
            slider_labels[label].grid(row=running_row, column=0, sticky=SW)
            running_row += 1
            sliders[label].grid(row=running_row, column=0, sticky=NW)
            running_row += 1
        console_label = Label(self.frame, text='Runtime outputs console', font=label_font)
        console_label.grid(row=running_row, column=0, sticky=SW)

        # numeric entry box
        uncertainty_numeric_list = {'uncertainty_runs': ['Number of uncertainty runs', 5],
                                    'burn_in_runs': ['Number of burn-in runs', 0],
                                    'search_width': ['Relative search width', .08]}
        # self.boolean_dictionary['output_uncertainty'].set(True)
        # self.boolean_dictionary['write_uncertainty_outcome_params'].set(True)
        self.boolean_dictionary['output_param_plots'].set(True)
        # self.boolean_dictionary['output_documents'].set(True)
        # self.boolean_dictionary['is_lowquality'].set(True)
        self.boolean_dictionary['is_vary_detection_by_organ'].set(True)
        self.boolean_dictionary['is_treatment_history'].set(True)
        # self.boolean_dictionary['riskgroup_prison'].set(True)
        # self.boolean_dictionary['riskgroup_urbanpoor'].set(True)
        # self.boolean_dictionary['riskgroup_ruralpoor'].set(True)
        self.boolean_dictionary['output_gtb_plots'].set(True)
        self.boolean_dictionary['is_vary_force_infection_by_riskgroup'].set(True)
        self.boolean_dictionary['riskgroup_diabetes'].set(True)
        # self.boolean_dictionary['riskgroup_hiv'].set(True)
        # self.boolean_dictionary['riskgroup_indigenous'].set(True)
        # self.boolean_dictionary['is_timevariant_organs'].set(True)

        for numeric in uncertainty_numeric_list.keys():
            numeric_label = Label(self.frame, text=uncertainty_numeric_list[numeric][0], font=label_font)
            numeric_label.grid(row=uncertainty_row, column=4, sticky=SW)
            uncertainty_row += 1
            if numeric == 'search_width':
                self.raw_outputs[numeric] = DoubleVar()
            else:
                self.raw_outputs[numeric] = IntVar()
            self.raw_outputs[numeric].set(uncertainty_numeric_list[numeric][1])
            runs = Entry(self.frame, textvariable=self.raw_outputs[numeric])
            runs.grid(row=uncertainty_row, column=4, sticky=NW)
            uncertainty_row += 1

        # pickling options
        uncertainty_dropdown_list = ['pickle_uncertainty']
        for dropdown in uncertainty_dropdown_list:
            self.raw_outputs[dropdown] = StringVar()
        self.raw_outputs['pickle_uncertainty'].set('No saving or loading')
        self.drop_downs['pickle_uncertainty'] \
            = OptionMenu(self.frame, self.raw_outputs['pickle_uncertainty'],
                         'No saving or loading', 'Load', 'Save')
        for drop_down in uncertainty_dropdown_list:
            self.drop_downs[drop_down].grid(row=uncertainty_row, column=4, sticky=W)
            uncertainty_row += 1

    def execute(self):
        """
        This is the main method to run the model. It replaces test_full_models.py
        """

        # collate check-box boolean options
        self.gui_outputs['scenarios_to_run'] = [0]
        self.gui_outputs['scenario_names_to_run'] = ['baseline']
        for boolean in self.boolean_inputs:
            if 'scenario_' in boolean:
                if self.boolean_dictionary[boolean].get() == 1:
                    self.gui_outputs['scenarios_to_run'] \
                        += [int(boolean[9:])]
                    self.gui_outputs['scenario_names_to_run'] \
                        += [autumn.tool_kit.find_scenario_string_from_number(int(boolean[9:]))]
            else:
                self.gui_outputs[boolean] = bool(self.boolean_dictionary[boolean].get())

        # collate drop-down box options
        organ_stratification_keys = {'Pos / Neg / Extra': 3,
                                     'Pos / Neg': 2,
                                     'Unstratified': 0}
        strain_stratification_keys = {'Single strain': 0,
                                      'DS / MDR': 2,
                                      'DS / MDR / XDR': 3}

        for option in self.raw_outputs:
            if option == 'fitting_method':
                self.gui_outputs[option] = int(self.raw_outputs[option].get()[-1])
            elif option == 'n_organs':
                self.gui_outputs[option] = organ_stratification_keys[self.raw_outputs[option].get()]
            elif option == 'n_strains':
                self.gui_outputs[option] = strain_stratification_keys[self.raw_outputs[option].get()]
            else:
                self.gui_outputs[option] = self.raw_outputs[option].get()

        # record thread number
        self.thread_number += 1

        # indicate processing has started
        self.runtime_outputs.insert(END, '_____________________________________________\n' +
                                    'Model run commenced using thread #%d.\n' % self.thread_number)
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

        model_runner = autumn.model_runner.ModelRunner(self.gui_outputs, self.runtime_outputs, self.figure_frame)
        model_runner.master_runner()
        project = autumn.outputs.Project(model_runner, self.gui_outputs)
        project.master_outputs_runner()


if __name__ == '__main__':

    root = Tk()
    app = App(root)
    root.mainloop()

