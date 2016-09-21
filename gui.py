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
                                  'Plot outcomes against GTB data'}

    if working_string in button_name_dictionary:
        return button_name_dictionary[working_string]
    else:
        return working_string

class App:

    def __init__(self, master):

        frame = Frame(master)
        frame.pack()
        root.minsize(500, 200)
        self.run = Button(frame, text='Run', command=self.execute)
        self.run.grid(row=1, column=0)

        self.boolean_dictionary = {}
        self.boolean_inputs = ['output_uncertainty', 'output_spreadsheets', 'output_documents', 'output_by_scenario',
                               'output_horizontally', 'output_gtb_plots']
        for boolean in self.boolean_inputs:
            self.boolean_dictionary[boolean] = IntVar()

        self.boolean_toggles = {}
        for boolean in self.boolean_inputs:
            self.boolean_toggles[boolean] = Checkbutton(frame,
                                                        text=find_button_name_from_string(boolean),
                                                        variable=self.boolean_dictionary[boolean])

        for b, boolean in enumerate(self.boolean_inputs):
            self.boolean_toggles[boolean].grid(row=b, column=2, sticky=W)

        self.output_options = {}

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

