
import threading

import autumn.model_runner
import autumn.outputs as outputs
import autumn.gui_params as gui_params

from Tkinter import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


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

        autumn_params = gui_params.get_autumn_params()

        self.params = autumn_params['params']
        self.make_tk_controls_in_params()

        self.param_groups = autumn_params['param_groups']
        self.set_tk_controls_in_frame()

    def make_tk_controls_in_params(self):
        for key, param in self.params.iteritems():
            if param['type'] == 'boolean':
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
            if param['type'] == 'boolean':
                param['value'] = bool(param['tk_var'].get())
            else:
                param['value'] = param['tk_var'].get()

        self.gui_outputs = gui_params.convert_params_to_inputs(self.params)

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

        self.model_runner = autumn.model_runner.TbRunner(
            self.gui_outputs, self.runtime_outputs, js_gui=self.handle_message)
        self.model_runner.master_runner()
        project = outputs.Project(self.model_runner, self.gui_outputs)
        project.master_outputs_runner()

    def handle_message(self, command, data={}):
        if command == 'console':
            self.runtime_outputs.insert(END, data['message'] + '\n')
            self.runtime_outputs.see(END)
        elif command == 'graph':
            self.graph(data)

    def graph(self, data, input_figure=None):
        # initialise plotting
        if not input_figure:
            param_tracking_figure = plt.Figure()
            parameter_plots = FigureCanvasTkAgg(param_tracking_figure, master=self.figure_frame)

        else:
            param_tracking_figure = input_figure

        subplot_grid = outputs.find_subplot_numbers(len(data['all_parameters_tried']))

        # cycle through parameters with one subplot for each parameter
        for p, param in enumerate(data['all_parameters_tried']):

            # extract accepted params from all tried params
            accepted_params = list(
                p for p, a in zip(
                    data['all_parameters_tried'][param],
                    data['whether_accepted_list'])
                if a)

            # plot
            ax = param_tracking_figure.add_subplot(subplot_grid[0], subplot_grid[1], p + 1)
            ax.plot(range(1, len(accepted_params) + 1), accepted_params, linewidth=2, marker='o', markersize=4,
                    mec='b', mfc='b')
            ax.set_xlim((1., len(data['accepted_indices']) + 1))

            # find the y-limits from the parameter bounds and the parameter values tried
            for param_number in range(len(data['param_ranges_unc'])):
                if data['param_ranges_unc'][param_number]['key'] == param:
                    bounds = data['param_ranges_unc'][param_number]['bounds']
            ylim_margins = .1
            min_ylimit = min(accepted_params + [bounds[0]])
            max_ylimit = max(accepted_params + [bounds[1]])
            ax.set_ylim((min_ylimit * (1 - ylim_margins), max_ylimit * (1 + ylim_margins)))

            # indicate the prior bounds
            ax.plot([1, len(data['accepted_indices']) + 1], [min_ylimit, min_ylimit], color='0.8')
            ax.plot([1, len(data['accepted_indices']) + 1], [max_ylimit, max_ylimit], color='0.8')

            # plot rejected parameters
            for run, rejected_params in data['rejection_dict'][param].items():
                if data['rejection_dict'][param][run]:
                    ax.plot([run + 1] * len(rejected_params), rejected_params, marker='o', linestyle='None',
                            mec='0.5', mfc='0.5', markersize=3)
                    for r in range(len(rejected_params)):
                        ax.plot([run, run + 1], [data['acceptance_dict'][param][run], rejected_params[r]], color='0.5',
                                linestyle='--')

            # label
            ax.set_title(data['names'][param])
            if p > len(data['all_parameters_tried']) - subplot_grid[1] - 1:
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

