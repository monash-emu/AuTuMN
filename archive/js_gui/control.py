import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

import autumn.model_runner
import autumn.outputs
from flask_socketio import emit
import time

import eventlet
eventlet.monkey_patch()


class Autumn:

    """ Autumn class to handle input from Autumn Application """

    def __init__(self, gui_settings):

        """
        Initialize class Autumn with a dictionary of setting from the GUI.
        The initialization defines translations from the parameters in the GUI to the parameters
        necessary to use ModelRunner and Outputs.

        The private method _parse_settings() is called at the end of initialization to translate
        and store settings in model_settings.

        The private method _run() is also called at the end of initialization to initiate the
        ModelRunner and Output modules and start the model.

        The private method _emit_message(comment, target), emits a message to the specified target
        element in the GUI (default is 'console', which should take care of everything for now) -
        there are two message at the end of __init__ and a few more than can be changed in _run.

        Note that a call of the monkey-patched time.sleep() with emit_delay follows emission of messages
        to prevent accumulation in the socket queue in long-running functions. The delay of 0.1 seconds
        can probably be adjusted, but should be good for now.

        Verbose prints translated settings and data types.

        :param gui_settings: dict, dictionary of settings from inputs in GUI

        """

        self.gui_settings = gui_settings

        self.model_settings = None

        self.emit_delay = 0.1
        self.verbose = True

        self.bool = {"false": False, "true": True}

        self.translations = {
            "fittingSmoothness": "default_smoothness",
            "documentsCheck": "output_documents",
            "hivCheck": "comorbidity_hiv",
            "adaptiveCheck": "adaptive_uncertainty",
            "explicitSelection": "integration_method",
            "diabetesCheck": "comorbidity_diabetes",
            "careCheck": "is_lowquality",
            "misassignmentCheck": "is_misassignment",
            "horizontalCheck": "output_horizontally",
            "burninNumber": "burn_in_runs",
            "uncertaintyNumber": "uncertainty_runs",
            "methodSelection": "fitting_method",
            "uncertaintyCheck": "output_uncertainty",
            "strainSelection": "n_strains",
            "byscenarioCheck": "output_by_scenario",
            "ioSelection": "pickle_uncertainty",
            "integrationTime": "time_step",
            "spreadsheetCheck": "output_spreadsheets",
            "amplificationCheck": "is_amplification",
            "countrySelection": "country",
            "searchNumber": "search_width",
            "smearSelection": "n_organs"
        }

        self.organ_dict = {"Unstratified": 0, "Positive/Negative": 2, "Positive/Negative/Extra": 3}
        self.pickle_dict = {"No Saving / Loading": "No saving or loading", "Load": "Load", "Save": "Save"}
        self.strain_dict = {"Single": 0, "DR/MDR": 2, "DR/MDR/XDR": 3}

        self.plot_options = {
            'Outcomes': 'output_gtb_plots',
            'Compartment Sizes': 'output_compartment_populations',
            'Outcomes by Subgroups': 'output_by_subgroups',
            'Proportions by Age': 'output_age_fractions',
            'Proportions by Risk Group': 'output_comorbidity_fractions',
            'Flowdiagram': 'output_flow_diagram',
            'Compartment Fractions': 'output_fractions',
            'Scale-up Functions': 'output_scaleups',
            'Economics Graph': 'output_plot_economics',
            'Comorbidity Checks': 'output_plot_comorbidity_checks',
            'Age Calculation Weights': 'output_age_calculations',
            'Parameter Progression': 'output_param_plots',
            'Popsize for Cost-coverage': 'output_popsize_plot',
            'Log Likelihood over Runs': 'output_likelihood_plot'
        }

        self._emit_message("--------------------------------------------------")
        self._emit_message("Inititated control class Autumn for ModelRunner...")

        self._parse_settings()
        self._run()

    def _parse_settings(self):

        """ Convert settings into format for ModelRunner """

        plot_selections = self.gui_settings.pop("plotSelection")
        scenario_selections = self.gui_settings.pop("scenarioSelection")

        self.gui_settings["smearSelection"] = self.organ_dict[self.gui_settings["smearSelection"]]
        self.gui_settings["ioSelection"] = self.pickle_dict[self.gui_settings["ioSelection"]]
        self.gui_settings["strainSelection"] = self.strain_dict[self.gui_settings["strainSelection"]]

        self.gui_settings["methodSelection"] = int(self.gui_settings["methodSelection"].split()[1])
        self.gui_settings["integrationTime"] = float(self.gui_settings["integrationTime"])
        self.gui_settings["uncertaintyNumber"] = int(self.gui_settings["uncertaintyNumber"])
        self.gui_settings["fittingSmoothness"] = float(self.gui_settings["fittingSmoothness"])
        self.gui_settings["searchNumber"] = float(self.gui_settings["searchNumber"])
        self.gui_settings["burninNumber"] = int(self.gui_settings["burninNumber"])
        # First, translate all boolean strings to Booleans

        updated_settings = {(self.translations[key]): (self.bool[value] if value in self.bool.keys() else value)
                            for key, value in self.gui_settings.items()}

        # Second, get and set plotting options from array (GUI, MultiSelection)

        plot_settings = {value: (True if key in plot_selections else False)
                         for key, value in self.plot_options.items()}

        # Third, sort scenario options from array (GUI, MultiSelection) -
        # not strictly necessary, since scenarios are already ordered from MultiSelection

        scenario_names = sorted(scenario_selections, key=lambda x: int(x.split()[1]))
        scenario_numbers = [int(scenario.split()[1]) for scenario in scenario_names]

        scenario_settings = {"scenario_names_to_run": ["baseline"] + scenario_names,
                             "scenarios_to_run": [None] + scenario_numbers}

        self.model_settings = merge_dicts(updated_settings, plot_settings, scenario_settings)

        if self.verbose:
            for k, v in self.model_settings.items():
                print "Set", k, "to", v, "which is", type(v)

        self._emit_message("Parsed settings from GUI...")

    def _emit_message(self, comment, target='console'):

        """ Emits a message to the specified element in the GUI """

        print "Emitting", comment

        emit(target, {'message': comment})
        time.sleep(self.emit_delay)

    def _run(self):

        """ Starts the model run and outputs in ModelRunner / Project"""

        self._emit_message("Running model...")
        self._emit_message("--------------------------------------------------")

        model_runner = autumn.model_runner.TbRunner(self.model_settings, runtime_outputs=None,
                                                    figure_frame=None, js_gui=True)
        model_runner.master_runner()
        model_outputs = autumn.outputs.Project(model_runner, self.model_settings)
        model_outputs.master_outputs_runner()

        self._emit_message("--------------------------------------------------")

### Helper Functions ###


def merge_dicts(*dict_args):
    """
    Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts.
    """

    result = {}
    for dictionary in dict_args:
        result.update(dictionary)

    return result