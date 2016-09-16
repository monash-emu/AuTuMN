from numpy import isfinite
import copy
import tool_kit
import model
import os
import outputs
import data_processing

def is_positive_definite(v):
    return isfinite(v) and v > 0.0


class ModelRunner:

    def __init__(self, model=None):

        self.model = copy.deepcopy(model)
        self.is_last_run_success = False
        self.nb_accepted = 0

    def set_model_with_params(self, param_dict):

        n_set = 0
        for key in param_dict:
            if key in self.model.params:
                n_set += 1
                self.model.set_parameter(key, param_dict[key])
            else:
                raise ValueError("%s not in model params" % key)

    def convert_param_list_to_dict(self, params):

        param_dict = {}
        for val, props in zip(params, self.model.param_ranges_unc):
            param_dict[props['key']] = val
        return param_dict

    def run_with_params(self, params):

        for i, p in enumerate(params):
            if not is_positive_definite(p):
                print "Warning: parameter%d=%f is invalid for model" % (i, p)
                self.is_last_run_success = False
                return
            bounds = self.model.param_ranges_unc[i]["bounds"]
            if (p < bounds[0]) or (p > bounds[1]):
                #print "Warning: parameter%d=%f is outside of the allowed bounds" % (i, p)
                self.is_last_run_success = False
                return

        param_dict = self.convert_param_list_to_dict(params)

        self.set_model_with_params(param_dict)
        self.is_last_run_success = True
        try:
            self.model.integrate()
        except:
            print "Warning: parameters=%s failed with model" % params
            self.is_last_run_success = False


class ModelRunnerNew:

    def __init__(self, inputs):

        self.inputs = inputs
        self.inputs.read_and_load_data()
        self.project = outputs.Project(self.inputs.country, self.inputs)
        self.model_dict = {}
        self.is_last_run_success = False
        self.nb_accepted = 0

    def run_scenarios(self):

        for scenario in self.inputs.model_constants['scenarios_to_run']:

            # Name and initialise model
            scenario_name = tool_kit.find_scenario_string_from_number(scenario)
            self.model_dict[scenario_name] = model.ConsolidatedModel(scenario, self.inputs)

            # Create an outputs object for use later
            self.project.scenarios.append(scenario_name)

            # Introduce model at first run
            tool_kit.introduce_model(self.model_dict, scenario_name)

            # Sort out times for scenario runs
            if scenario is None:
                self.model_dict[scenario_name].start_time = self.inputs.model_constants['start_time']
            else:
                scenario_start_time_index = \
                    self.model_dict['baseline'].find_time_index(self.inputs.model_constants['recent_time'])
                self.model_dict[scenario_name].start_time = \
                    self.model_dict['baseline'].times[scenario_start_time_index]
                self.model_dict[scenario_name].loaded_compartments = \
                    self.model_dict['baseline'].load_state(scenario_start_time_index)

            # Describe model
            print('Running model "' + scenario_name + '".')
            tool_kit.describe_model(self.model_dict, scenario_name)

            # Integrate and add result to outputs object
            self.model_dict[scenario_name].integrate()
            self.project.models[scenario_name] = self.model_dict[scenario_name]

    def run_uncertainty(self):

        if self.inputs.model_constants['output_uncertainty']:

            # Prepare directory for eventual pickling
            out_dir = 'pickles'
            if not os.path.isdir(out_dir):
                os.makedirs(out_dir)
            filename = os.path.join(out_dir, 'uncertainty.pkl')

            # Don't run uncertainty but load a saved simulation
            if self.project.models['baseline'].pickle_uncertainty == 'read':
                self.project.models['baseline'].uncertainty_results = tool_kit.pickle_load(filename)
                print "Uncertainty results loaded from previous simulation"

            # Run uncertainty
            else:
                self.project.models['baseline'].run_uncertainty()

            # Write uncertainty if requested
            if self.project.models['baseline'].pickle_uncertainty == 'write':
                tool_kit.pickle_save(self.project.models['baseline'].uncertainty_results, filename)
                print "Uncertainty results written on the disc"

            self.project.rearrange_uncertainty()

    def set_model_with_params(self, param_dict):

        n_set = 0
        for key in param_dict:
            if key in self.model.params:
                n_set += 1
                self.model.set_parameter(key, param_dict[key])
            else:
                raise ValueError("%s not in model params" % key)

    def convert_param_list_to_dict(self, params):

        param_dict = {}
        for val, props in zip(params, self.model.param_ranges_unc):
            param_dict[props['key']] = val
        return param_dict

    def run_with_params(self, params):

        for i, p in enumerate(params):
            if not is_positive_definite(p):
                print "Warning: parameter%d=%f is invalid for model" % (i, p)
                self.is_last_run_success = False
                return
            bounds = self.model.param_ranges_unc[i]["bounds"]
            if (p < bounds[0]) or (p > bounds[1]):
                # print "Warning: parameter%d=%f is outside of the allowed bounds" % (i, p)
                self.is_last_run_success = False
                return

        param_dict = self.convert_param_list_to_dict(params)

        self.set_model_with_params(param_dict)
        self.is_last_run_success = True
        try:
            self.model.integrate()
        except:
            print "Warning: parameters=%s failed with model" % params
            self.is_last_run_success = False























