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
                                  'Plot economics graphs (cost-coverage and cost-time)'}

    if working_string in button_name_dictionary:
        return button_name_dictionary[working_string]
    else:
        return working_string


class App:

    def __init__(self, master):

        self.output_options = {}
        frame = Frame(master)
        frame.pack()

        root.title('AuTuMN (version 1.0)')
        self.run = Button(frame, text='Run', command=self.execute)
        self.run.grid(row=1, column=0, sticky=W, padx=4)

        self.boolean_dictionary = {}
        self.boolean_inputs = ['output_flow_diagram', 'output_compartment_populations', 'output_comorbidity_fractions',
                               'output_age_fractions', 'output_by_age', 'output_fractions', 'output_scaleups',
                               'output_gtb_plots', 'output_plot_economics', 'output_uncertainty', 'output_spreadsheets',
                               'output_documents', 'output_by_scenario', 'output_horizontally']
        for boolean in self.boolean_inputs:
            self.boolean_dictionary[boolean] = IntVar()

        self.boolean_toggles = {}
        for boolean in self.boolean_inputs:
            self.boolean_toggles[boolean] = Checkbutton(frame,
                                                        text=find_button_name_from_string(boolean),
                                                        variable=self.boolean_dictionary[boolean])

        plot_row = 1
        option_row = 1
        uncertainty_row = 1
        for boolean in self.boolean_inputs:
            if 'Plot ' in find_button_name_from_string(boolean) \
                    or 'Draw ' in find_button_name_from_string(boolean):
                self.boolean_toggles[boolean].grid(row=plot_row, column=1, sticky=W)
                plot_row += 1
            elif 'uncertainty' in find_button_name_from_string(boolean):
                self.boolean_toggles[boolean].grid(row=uncertainty_row, column=3, sticky=W)
                uncertainty_row += 1
            else:
                self.boolean_toggles[boolean].grid(row=option_row, column=2, sticky=W)
                option_row += 1

        column_titles = {0: 'Run model',
                         1: 'Plotting options',
                         2: 'MS Office options',
                         3: 'Uncertainty options'}
        for i in range(4):
            title = Label(frame, text=column_titles[i])
            title.grid(row=0, column=i, sticky=NW, pady=10)
            title.config(font='Helvetica 10 bold italic')
            frame.grid_columnconfigure(i, minsize=250)

    def execute(self):

        for boolean in self.boolean_inputs:
            self.output_options[boolean] = bool(self.boolean_dictionary[boolean].get())

        # Start timer
        start_realtime = datetime.datetime.now()

        # Run everything
        model_runner = autumn.model_runner.ModelRunner(self.output_options)
        model_runner.master_runner()
        project = autumn.outputs.Project(model_runner, self.output_options)
        project.master_outputs_runner()

        print('Time elapsed in running script is ' + str(datetime.datetime.now() - start_realtime))

if __name__ == '__main__':

    root = Tk()
    app = App(root)
    root.mainloop()

