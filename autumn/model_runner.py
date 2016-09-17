from numpy import isfinite
import copy
import tool_kit
import model
import os
import outputs
import data_processing
import numpy
import datetime
from scipy.stats import norm, beta


def is_positive_definite(v):
    return isfinite(v) and v > 0.0


def generate_candidates(n_candidates, param_ranges_unc):

    """
    Function for generating candidate parameters

    """

    # Dictionary for storing candidates
    param_candidates = {}
    for param_dict in param_ranges_unc:

        # Find bounds of parameter
        bound_low, bound_high = param_dict['bounds'][0], param_dict['bounds'][1]

        # Draw from distribution
        if param_dict['distribution'] == 'beta':
            x = numpy.random.beta(2., 2., n_candidates)
            x = bound_low + x * (bound_high - bound_low)
        elif param_dict['distribution'] == 'uniform':
            x = numpy.random.uniform(bound_low, bound_high, n_candidates)

            # Return values
        param_candidates[param_dict['key']] = x
    return param_candidates


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
        for val, props in zip(params, self.model.inputs.param_ranges_unc):
            param_dict[props['key']] = val
        return param_dict

    def run_with_params(self, params):

        for i, p in enumerate(params):
            if not is_positive_definite(p):
                print "Warning: parameter%d=%f is invalid for model" % (i, p)
                self.is_last_run_success = False
                return
            bounds = self.model.inputs.param_ranges_unc[i]["bounds"]
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

    def __init__(self):

        self.inputs = data_processing.Inputs(True)
        self.inputs.read_and_load_data()
        self.project = outputs.Project(self.inputs.country, self.inputs)
        self.model_dict = {}
        self.is_last_run_success = False
        self.nb_accepted = 0
        self.adaptive_search = True  # If True, next candidate generated according to previous position
        self.accepted_parameters = {}
        self.interventions_to_cost = ['vaccination', 'xpert', 'treatment_support', 'smearacf', 'xpertacf',
                                      'ipt_age0to5', 'ipt_age5to15', 'decentralisation']
        self.labels = ['incidence', 'mortality', 'prevalence', 'notifications']
        self.cost_types = ['raw_cost', 'discounted_cost', 'inflated_cost', 'discounted_inflated_cost']
        self.n_runs = 5  # Number of accepted runs per scenario
        self.burn_in = 0  # Number of accepted runs that we burn
        self.loglikelihoods = []
        self.outputs_unc = [
            {
                'key': 'incidence',
                'posterior_width': None,
                'width_multiplier': 2.0  # for incidence for ex. Width of Normal posterior relative to CI width in data
            }]
        # Width of the interval in which next parameter value is likely (95%) to be drawn.
        # (Expressed as a proportion of the width defined in bounds.)
        self.search_width = 0.2
        self.results = {}
        self.all_parameters_tried = {}
        self.whether_accepted_list = []

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

            # Store
            self.results['scenarios'] = {}
            self.store_scenario_results(scenario_name)

    def store_scenario_results(self, scenario, scenarios_or_uncertainty='scenarios'):

        """
        This method is designed to store all the results that will be needed for later analysis in separate
        attributes to the individual models, to avoid them being over-written during the uncertainty process.
        Args:
            scenario: The name of the model run.

        """

        self.results[scenarios_or_uncertainty][scenario] = {}

        self.results[scenarios_or_uncertainty][scenario]['compartment_soln'] \
            = self.model_dict[scenario].compartment_soln
        self.results[scenarios_or_uncertainty][scenario]['costs'] \
            = self.model_dict[scenario].costs
        self.results[scenarios_or_uncertainty][scenario]['flow_array'] \
            = self.model_dict[scenario].flow_array
        self.results[scenarios_or_uncertainty][scenario]['fraction_array'] \
            = self.model_dict[scenario].fraction_array
        self.results[scenarios_or_uncertainty][scenario]['fraction_soln'] \
            = self.model_dict[scenario].fraction_soln
        self.results[scenarios_or_uncertainty][scenario]['soln_array'] \
            = self.model_dict[scenario].soln_array
        self.results[scenarios_or_uncertainty][scenario]['var_array'] \
            = self.model_dict[scenario].var_array

    def master_uncertainty(self):

        print('Uncertainty analysis')
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
                # self.prepare_uncertainty_storage()
                self.run_uncertainty()

            # Write uncertainty if requested
            if self.project.models['baseline'].pickle_uncertainty == 'write':
                tool_kit.pickle_save(self.project.models['baseline'].uncertainty_results, filename)
                print "Uncertainty results written on the disc"

            self.project.rearrange_uncertainty()

    def prepare_uncertainty_storage(self):

        """
        Prepare structures to store uncertainty outcomes.

        """

        # Epidemiological outputs
        for scenario in self.inputs.model_constants['scenarios_to_run']:
            scenario_name = tool_kit.find_scenario_string_from_number(scenario)
            self.uncertainty_results[scenario_name] = {}
            for label in self.labels:
                self.uncertainty_results[scenario_name][label] = {}

            # Economic outputs
            self.uncertainty_results[scenario_name]['costs'] = {}
            for intervention in self.interventions_to_cost:
                self.uncertainty_results[scenario_name]['costs'][intervention] = {}
                for cost_type in self.cost_types:
                    self.uncertainty_results[scenario_name]['costs'][intervention][cost_type] = {}

    def run_uncertainty(self):

        """
        Main method to run all the uncertainty processes.

        """

        # If not doing an adaptive search, only need to start with a single parameter set
        if self.adaptive_search:
            n_candidates = 1
        else:
            n_candidates = self.n_runs * 10

        # Define an initial set of parameter candidates only
        param_candidates = generate_candidates(n_candidates, self.inputs.param_ranges_unc)
        normal_char = self.get_normal_char()

        # Prepare for loop
        for param_dict in self.inputs.param_ranges_unc:
            self.accepted_parameters[param_dict['key']] = []
            self.all_parameters_tried[param_dict['key']] = []
        n_accepted = 0
        i_candidates = 0
        run = 0
        prev_log_likelihood = -1e10
        params = []
        self.results['uncertainty'] = {}
        self.results['uncertainty']['baseline'] = {}
        self.results['uncertainty']['baseline']['compartment_soln'] = []
        self.results['uncertainty']['baseline']['costs'] = []
        self.results['uncertainty']['baseline']['flow_array'] = []
        self.results['uncertainty']['baseline']['fraction_array'] = []
        self.results['uncertainty']['baseline']['fraction_soln'] = []
        self.results['uncertainty']['baseline']['soln_array'] = []
        self.results['uncertainty']['baseline']['var_array'] = []

        # Until a sufficient number of parameters are accepted
        while n_accepted < self.n_runs + self.burn_in:

            # Set timer
            start_timer_run = datetime.datetime.now()

            # Not exactly sure what this does
            new_params = []
            if self.adaptive_search:
                if i_candidates == 0:
                    new_params = []
                    for param_dict in self.inputs.param_ranges_unc:
                        new_params.append(param_candidates[param_dict['key']][run])
                        params.append(param_candidates[param_dict['key']][run])
                else:
                    new_params = self.update_param(params)
            else:
                for param_dict in self.inputs.param_ranges_unc:
                    new_params.append(param_candidates[param_dict['key']][run])

            # Run the integration
            # (includes checking parameters, setting parameters and recording success/failure of run)
            self.run_with_params(new_params)

            # Store results
            self.results['uncertainty']['baseline']['compartment_soln'].append(
                self.model_dict['baseline'].compartment_soln)
            self.results['uncertainty']['baseline']['costs'].append(
                self.model_dict['baseline'].costs)
            self.results['uncertainty']['baseline']['flow_array'].append(
                self.model_dict['baseline'].flow_array)
            self.results['uncertainty']['baseline']['fraction_array'].append(
                self.model_dict['baseline'].fraction_array)
            self.results['uncertainty']['baseline']['fraction_soln'].append(
                self.model_dict['baseline'].fraction_soln)
            self.results['uncertainty']['baseline']['soln_array'].append(
                self.model_dict['baseline'].soln_array)
            self.results['uncertainty']['baseline']['var_array'].append(
                self.model_dict['baseline'].var_array)

            # Record results in accepted parameter dictionary
            for p, param_dict in enumerate(self.inputs.param_ranges_unc):
                self.all_parameters_tried[param_dict['key']].append(new_params[p])

            # Calculate prior
            prior_log_likelihood = 0.
            for p, param_dict in enumerate(self.inputs.param_ranges_unc):
                param_val = new_params[p]

                # Calculate the density of param_val
                bound_low, bound_high = param_dict['bounds'][0], param_dict['bounds'][1]

                # Normalise value and find log of PDF from beta distribution
                if param_dict['distribution'] == 'beta':
                    prior_log_likelihood \
                        += beta.logpdf((param_val - bound_low) / (bound_high - bound_low),
                                       2., 2.)

                # Find log of PDF from uniform distribution
                elif param_dict['distribution'] == 'uniform':
                    prior_log_likelihood \
                        += numpy.log(1. / (bound_high - bound_low))

            # Calculate posterior
            posterior_log_likelihood = 0.
            for output_dict in self.outputs_unc:
                working_output_dictionary = normal_char[output_dict['key']]
                for year in working_output_dictionary.keys():
                    year_index \
                        = tool_kit.find_first_list_element_at_least_value(self.model_dict['baseline'].times,
                                                                          year)
                    model_result_for_output \
                        = self.model_dict['baseline'].get_var_soln(output_dict['key'])[year_index]
                    mu, sd = working_output_dictionary[year][0], working_output_dictionary[year][1]
                    posterior_log_likelihood += norm.logpdf(model_result_for_output, mu, sd)

            # Sum for overall likelihood of run
            log_likelihood = prior_log_likelihood + posterior_log_likelihood

            # Determine acceptance
            if log_likelihood >= prev_log_likelihood:
                accepted = 1
            else:
                accepted = numpy.random.binomial(n=1, p=numpy.exp(log_likelihood - prev_log_likelihood))

            # Record information for accepted runs
            if accepted != 1:
                self.whether_accepted_list.append(False)
            elif accepted == 1:
                self.whether_accepted_list.append(True)
                n_accepted += 1

                # Record results in accepted parameter dictionary
                for p, param_dict in enumerate(self.inputs.param_ranges_unc):
                    self.accepted_parameters[param_dict['key']].append(new_params[p])

                # Update likelihood and parameter set for next run
                prev_log_likelihood = log_likelihood
                params = new_params

                # Store outputs once burn-in complete
                if n_accepted > self.burn_in:

                    # Model storage
                    params_dict = {}
                    for p, param_dict in enumerate(self.inputs.param_ranges_unc):
                        params_dict[param_dict['key']] = new_params[p]
                    self.loglikelihoods.append(log_likelihood)

                    # Run scenarios other than baseline and store uncertainty
                    for scenario in self.inputs.model_constants['scenarios_to_run']:
                        scenario_name = tool_kit.find_scenario_string_from_number(scenario)
                        if scenario is not None:
                            scenario_start_time_index = \
                                self.model_dict['baseline'].find_time_index(self.inputs.model_constants['recent_time'])
                            self.model_dict[scenario_name].start_time = \
                                self.model_dict['baseline'].times[scenario_start_time_index]
                            self.model_dict[scenario_name].loaded_compartments = \
                                self.model_dict['baseline'].load_state(scenario_start_time_index)
                            self.model_dict[scenario_name].integrate()

                            self.results['uncertainty'][scenario_name] = {}
                            self.results['uncertainty'][scenario_name]['compartment_soln'] = []
                            self.results['uncertainty'][scenario_name]['costs'] = []
                            self.results['uncertainty'][scenario_name]['flow_array'] = []
                            self.results['uncertainty'][scenario_name]['fraction_array'] = []
                            self.results['uncertainty'][scenario_name]['fraction_soln'] = []
                            self.results['uncertainty'][scenario_name]['soln_array'] = []
                            self.results['uncertainty'][scenario_name]['var_array'] = []

                            self.results['uncertainty'][scenario_name]['compartment_soln'].append(
                                self.model_dict[scenario_name].compartment_soln)
                            self.results['uncertainty'][scenario_name]['costs'].append(
                                self.model_dict[scenario_name].costs)
                            self.results['uncertainty'][scenario_name]['flow_array'].append(
                                self.model_dict[scenario_name].flow_array)
                            self.results['uncertainty'][scenario_name]['fraction_array'].append(
                                self.model_dict[scenario_name].fraction_array)
                            self.results['uncertainty'][scenario_name]['fraction_soln'].append(
                                self.model_dict[scenario_name].fraction_soln)
                            self.results['uncertainty'][scenario_name]['soln_array'].append(
                                self.model_dict[scenario_name].soln_array)
                            self.results['uncertainty'][scenario_name]['var_array'].append(
                                self.model_dict[scenario_name].var_array)

            i_candidates += 1
            run += 1

            # Generate more candidates if required
            if not self.adaptive_search and run >= len(param_candidates.keys()):
                param_candidates = generate_candidates(n_candidates, self.inputs.param_ranges_unc)
                run = 0
            print(str(n_accepted) + ' accepted / ' + str(i_candidates) + ' candidates @@@@@@@@ Running time: '
                  + str(datetime.datetime.now() - start_timer_run))

    def set_model_with_params(self, param_dict):

        """
        Populates baseline model with params from uncertainty calculations.

        Args:
            param_dict: Dictionary of the parameters to be set within the model (keys parameter name strings and values
                parameter values).

        """

        n_set = 0
        for key in param_dict:
            if key in self.model_dict['baseline'].params:
                n_set += 1
                self.model_dict['baseline'].set_parameter(key, param_dict[key])
            else:
                raise ValueError("%s not in model params" % key)

    def convert_param_list_to_dict(self, params):

        param_dict = {}
        for names, vals in zip(self.inputs.param_ranges_unc, params):
            param_dict[names['key']] = vals
        return param_dict

    def get_normal_char(self):

        """
        Define the characteristics (mean and standard deviation) of the normal distribution for model outputs
        (incidence, mortality).

        Returns:
            normal_char: Dictionary with keys outputs and values dictionaries. Sub-dictionaries have keys years
                and values lists, with first element of list means and second standard deviations.

        """

        # Dictionary storing the characteristics of the normal distributions
        normal_char = {}
        for output_dict in self.inputs.outputs_unc:
            normal_char[output_dict['key']] = {}
            if output_dict['key'] == 'mortality':
                sd = output_dict['posterior_width'] / (2.0 * 1.96)
                for year in self.inputs.data_to_fit[output_dict['key']].keys():
                    mu = self.inputs.data_to_fit[output_dict['key']][year]
                    normal_char[output_dict['key']][year] = [mu, sd]

            elif output_dict['key'] == 'incidence':
                for year in self.inputs.data_to_fit[output_dict['key']].keys():
                    low = self.inputs.data_to_fit['incidence_low'][year]
                    high = self.inputs.data_to_fit['incidence_high'][year]
                    sd = output_dict['width_multiplier'] * (high - low) / (2.0 * 1.96)
                    mu = (high + low) / 2.
                    normal_char[output_dict['key']][year] = [mu, sd]

        return normal_char

    def update_param(self, pars):

        # pars is the former position for the different parameters
        new_pars = []
        i = 0
        for par_dict in self.inputs.param_ranges_unc:
            bounds = par_dict['bounds']
            sd = self.search_width * (bounds[1] - bounds[0]) / (2.0 * 1.96)
            random = -100.
            while random < bounds[0] or random > bounds[1]:
                random = norm.rvs(loc=pars[i], scale=sd, size=1)
            new_pars.append(random)
            i += 1
        return new_pars

    def run_with_params(self, params):

        # Check whether parameter values are acceptable
        for p, param in enumerate(params):

            # Whether the parameter value is valid
            if not is_positive_definite(param):
                print 'Warning: parameter%d=%f is invalid for model' % (p, param)
                self.is_last_run_success = False
                return
            bounds = self.inputs.param_ranges_unc[p]['bounds']

            # Whether the parameter value is within acceptable ranges
            if (param < bounds[0]) or (param > bounds[1]):
                # print "Warning: parameter%d=%f is outside of the allowed bounds" % (p, param)
                self.is_last_run_success = False
                return

        param_dict = self.convert_param_list_to_dict(params)

        self.set_model_with_params(param_dict)
        self.is_last_run_success = True
        try:
            self.model_dict['baseline'].integrate()
        except:
            print "Warning: parameters=%s failed with model" % params
            self.is_last_run_success = False
