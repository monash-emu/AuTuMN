from Tkinter import *
import autumn.model_runner
import autumn.outputs
import datetime


def find_button_name_from_string(working_string):

    button_name_dictionary = {'output_uncertainty':
                                  'Run uncertainty',
                              'output_spreadsheets':
                                  'Write to spreadsheets',
                              'output_documents':
                                  'Write to documents',
                              'output_by_scenario':
                                  'Output by scenario (as opposed to program)',
                              'output_horizontally':
                                  'Write horizontally (if writing to Excel sheets)',
                              'output_gtb_plots':
                                  'Plot outcomes against GTB data',
                              'output_compartment_populations':
                                  'Plot compartment sizes',
                              'output_by_age':
                                  'Plot output by age groups',
                              'output_age_fractions':
                                  'Plot proportion of population by age group',
                              'output_comorbidity_fractions':
                                  'Plot proportion of population by risk group',
                              'output_flow_diagram':
                                  'Draw flow diagram of model compartments',
                              'output_fractions':
                                  'Plot compartment fractions',
                              'output_scaleups':
                                  'Plot scale-up functions (two ways)',
                              'output_plot_economics':
                                  'Plot economics graphs (cost-coverage and cost-time)',
                              'output_age_calculations':
                                  'Plot age calculation weightings',
                              'comorbidity_diabetes':
                                  'Type II diabetes',
                              'is_lowquality':
                                  'Low quality care',
                              'is_amplification':
                                  'Resistance amplification',
                              'is_misassignment':
                                  'Strain mis-assignment'}

    if working_string in button_name_dictionary:
        return button_name_dictionary[working_string]
    else:
        return working_string


class App:

    def __init__(self, master):

        """
        All processes involved in setting up the first basic GUI frame are collated here.

        Args:
            master: The GUI

        """

        # Prepare data structures
        self.output_options = {}
        self.multi_option = {}

        # Set up first frame
        self.master = master
        frame = Frame(master)
        frame.pack()
        self.master.minsize(1550, 400)
        self.master.title('AuTuMN (version 1.0)')

        # Model running button
        self.run = Button(frame, text='Run', command=self.execute, width=10)
        self.run.grid(row=1, column=0, sticky=W, padx=6)
        self.run.config(font='Helvetica 10 bold italic', fg='red', bg='grey')

        # Prepare Boolean data structures
        self.boolean_dictionary = {}
        self.boolean_inputs = ['output_flow_diagram', 'output_compartment_populations', 'output_comorbidity_fractions',
                               'output_age_fractions', 'output_by_age', 'output_fractions', 'output_scaleups',
                               'output_gtb_plots', 'output_plot_economics', 'output_uncertainty', 'output_spreadsheets',
                               'output_documents', 'output_by_scenario', 'output_horizontally',
                               'output_age_calculations', 'comorbidity_diabetes',
                               'is_lowquality', 'is_amplification', 'is_misassignment']
        for boolean in self.boolean_inputs:
            self.boolean_dictionary[boolean] = IntVar()

        # Set and collate checkboxes
        self.boolean_toggles = {}
        for boolean in self.boolean_inputs:
            self.boolean_toggles[boolean] = Checkbutton(frame,
                                                        text=find_button_name_from_string(boolean),
                                                        variable=self.boolean_dictionary[boolean],
                                                        pady=5)
        plot_row = 1
        option_row = 1
        uncertainty_row = 1
        comorbidity_row = 1
        elaboration_row = 1
        for boolean in self.boolean_inputs:
            if 'Plot ' in find_button_name_from_string(boolean) \
                    or 'Draw ' in find_button_name_from_string(boolean):
                self.boolean_toggles[boolean].grid(row=plot_row, column=4, sticky=W)
                plot_row += 1
            elif 'uncertainty' in find_button_name_from_string(boolean):
                self.boolean_toggles[boolean].grid(row=uncertainty_row, column=3, sticky=W)
                uncertainty_row += 1
            elif 'comorbidity_' in boolean:
                self.boolean_toggles[boolean].grid(row=comorbidity_row, column=1, sticky=W)
            elif 'is_' in boolean:
                self.boolean_toggles[boolean].grid(row=elaboration_row, column=2, sticky=W)
                elaboration_row += 1
            else:
                self.boolean_toggles[boolean].grid(row=option_row, column=5, sticky=W)
                option_row += 1

        # Column titles
        column_titles = {0: 'Model running',
                         1: 'Risk groups',
                         2: 'Elaborations',
                         3: 'Uncertainty',
                         4: 'Plotting',
                         5: 'MS Office outputs'}
        for i in range(len(column_titles)):
            title = Label(frame, text=column_titles[i])
            title.grid(row=0, column=i, sticky=NW, pady=10)
            title.config(font='Helvetica 10 bold italic')
            frame.grid_columnconfigure(i, minsize=250)

        self.multi_option['integration_method'] = StringVar()
        self.multi_option['fitting_method'] = IntVar()
        self.multi_option['integration_method'].set('Runge Kutta')
        self.multi_option['fitting_method'].set(5)
        self.drop_downs = {}
        self.drop_downs['integration_menu'] \
            = OptionMenu(frame, self.multi_option['integration_method'],
                         'Runge Kutta', 'Scipy', 'Explicit')
        self.drop_downs['fitting_menu'] \
            = OptionMenu(frame, self.multi_option['fitting_method'],
                         1, 2, 3, 4, 5)
        for d, drop_down in enumerate(self.drop_downs):
            self.drop_downs[drop_down].config(width=12)
            self.drop_downs[drop_down].grid(row=d+2, column=0, sticky=W, padx=4)

    def execute(self):

        """
        This is the main method to run the model. It replaces test_full_models.py

        """

        # Collate check-box boolean options
        for boolean in self.boolean_inputs:
            self.output_options[boolean] = bool(self.boolean_dictionary[boolean].get())

        # Collate drop-down box options
        for option in self.multi_option:
            self.output_options[option] = self.multi_option[option].get()

        # Start timer
        start_realtime = datetime.datetime.now()

        # Run everything
        model_runner = autumn.model_runner.ModelRunner(self.output_options)
        model_runner.master_runner()
        project = autumn.outputs.Project(model_runner, self.output_options)
        project.master_outputs_runner()

        # Report time
        print('Time elapsed in running script is ' + str(datetime.datetime.now() - start_realtime))

if __name__ == '__main__':

    root = Tk()
    app = App(root)
    root.mainloop()

