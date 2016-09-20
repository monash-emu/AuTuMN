from Tkinter import *
import autumn.model_runner
import autumn.outputs
import datetime

class App:

    def __init__(self, master):

        frame = Frame(master)
        frame.pack()
        root.minsize(500, 200)
        self.run = Button(frame, text='Run', command=self.execute)
        self.run.pack(side=LEFT)
        self.button = Button(frame, text='Quit', fg='red', command=frame.quit)
        self.button.pack(side=LEFT)
        self.output_uncertainty = IntVar()
        self.uncertainty_toggle = Checkbutton(frame,
                                              text='Output uncertainty',
                                              variable=self.output_uncertainty)
        self.uncertainty_toggle.pack(side=TOP)
        self.output_spreadsheet = IntVar()
        self.spreadsheet_toggle = Checkbutton(frame,
                                              text='Write to spreadsheets',
                                              variable=self.output_spreadsheet)
        self.spreadsheet_toggle.pack(side=TOP)

    def execute(self):

        self.output_options = {}
        self.output_options['output_uncertainty'] = bool(self.output_uncertainty.get())
        self.output_options['output_spreadsheet'] = bool(self.output_spreadsheet.get())

        # Start timer
        start_realtime = datetime.datetime.now()

        # Run everything
        model_runner = autumn.model_runner.ModelRunner(self.output_options)
        model_runner.master_runner()
        project = autumn.outputs.Project(model_runner)
        project.master_outputs_runner()

        print('Time elapsed in running script is ' + str(datetime.datetime.now() - start_realtime))

if __name__ == '__main__':

    root = Tk()
    app = App(root)
    root.mainloop()

