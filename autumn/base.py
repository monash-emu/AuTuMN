
# external imports
import os
import numpy
from autumn import tool_kit
import scipy.stats
from graphviz import Digraph
import re
import json

# AuTuMN imports
from autumn.economics import get_cost_from_coverage, get_coverage_from_cost
from autumn.inputs import NumpyEncoder


def add_unique_tuple_to_list(a_list, a_tuple):
    """
    Adds or modifies a list of tuples, comparing only the items before the last in the tuples
    (i.e. the compartments), with the last value in the tuple assumed to be the value for the flow rate.
    """

    for i, test_tuple in enumerate(a_list):
        if test_tuple[:-1] == a_tuple[:-1]:
            a_list[i] = a_tuple
            break
    else:
        a_list.append(a_tuple)


class BaseModel:
    """
    General class not specific to any particular infection, upon which disease-specific transmission models can be
    built.
    """

    def __init__(self):
        (self.labels, self.relevant_interventions, self.times) \
            = [[] for _ in range(3)]
        (self.init_compartments, self.compartments, self.params, self.vars, self.scaleup_fns, self.flows, \
         self.compartment_soln, self.flows_by_type) \
            = [{} for _ in range(8)]
        (self.time, self.start_time) \
            = [0. for _ in range(2)]
        (self.inputs, self.var_labels, self.var_array, self.flow_array, self.fraction_array, self.graph,
         self.loaded_compartments, self.integration_method) \
            = [None for _ in range(8)]
        self.scenario = 0
        self.time_step = 1.
        self.run_costing = True
        flow_types \
            = ['var_entry', 'fixed_infection_death', 'var_infection_death', 'fixed_transfer', 'var_transfer',
               'linked_transfer']
        for flow_type in flow_types:
            self.flows_by_type[flow_type] = []
        self.flow_type_index = {
            'var_entry': {'to': 0, 'rate': 1},
            'fixed_infection_death': {'from': 0, 'rate': 1},
            'var_infection_death': {'from': 0, 'rate': 1},
            'fixed_transfer': {'from': 0, 'to': 1, 'rate': 2},
            'var_transfer': {'from': 0, 'to': 1, 'rate': 2},
            'linked_transfer': {'from': 0, 'to': 1, 'rate': 2}}

        # list for removing labels
        self.remove_labels = []


    ''' time-related functions '''

    def make_times(self, start, end, delta):
        """
        Simple method to create time steps for reporting of outputs.

        Args:
            start: Start time for integration.
            end: End time for integration.
            delta: Step size.
        """

        self.times, time = [], start
        while time <= end:
            self.times.append(time)
            time += delta
        if self.times[-1] < end: self.times.append(end)

    def find_time_index(self, time):
        """
        Method to find first time point in times list at or after a certain specified time point.

        Args:
            time: Float for the time point of interest.
        """

        return [i for i, j in enumerate(self.times) if j >= time][0] - 1

    ''' methods to set values to aspects of the model '''

    def set_parameter(self, label, val):
        """
        Almost to simple to need a method. Sets a single parameter value.

        Args:
            label: Parameter name
            val: Parameter value
        """

        self.params[label] = val

    def get_constant_or_variable_param(self, param):
        """
        Now obsolete with new approach to determining whether parameters are constant or time variant in the data
        processing module.

        Simple function to look first in vars then params for a parameter value and
        raise an error if the parameter is not found.

        Args:
            param: String for the parameter (should be the same in either vars or params)
        Returns:
            param_value: The value of the parameter
        """

        if param in self.vars:
            param_value = self.vars[param]
        elif param in self.params:
            param_value = self.params[param]
        else:
            raise NameError('Parameter "' + param + '" not found in either vars or params.')
        return param_value

    def set_compartment(self, label, init_val=0.):
        """
        Assign an initial value to a compartment.

        Args:
            label: The name of the compartment
            init_val: The starting size of this compartment
        """

        #changes for discrding labels based on init_val =-1
        assert init_val >= 0., 'Start with negative compartment not permitted'

        # search the label with patterns in self.remove_labels
        match = 0
        for pattern in self.remove_labels:
            pattern = '.*' + pattern.replace('_', '.*')
            search_result = re.compile(pattern).match(label)
            if search_result:
                match = match + 1

        if match == 0:
            if label not in self.labels: self.labels.append(label)
            self.init_compartments[label] = init_val
        else:
            pass
            #print('Skipping label : ', label)

    def remove_compartment(self, label):
        """
        Remove a compartment from the model. Generally intended to be used in cases where the loops make it easier to
        create compartments and then remove them again.

        Args:
            label: Name fo the compartment to be removed
        """

        #assert label in self.init_compartments, 'Tried to remove non-existent compartment'
        if label in self.init_compartments:
            self.labels.remove(label)
            del self.init_compartments[label]

    def initialise_compartments(self):
        """
        Initialise compartments to starting values.
        """

        pass

    ''' methods to manipulate compartment data items '''

    def convert_list_to_compartments(self, compartment_vector):
        """
        Uses self.labels to convert list of compartments to dictionary.

        Args:
            compartment_vector: List of compartment values.
        Returns:
            A dictionary with keys being the compartment names (from the strings in self.labels) and values the elements
                of compartment_vector
        """

        return {l: compartment_vector[i] for i, l in enumerate(self.labels)}

    def convert_compartments_to_list(self, compartment_dict):
        """
        Reverse of previous method. Converts

        Args:
            compartment_dict: Dictionary with keys strings of compartment names
        Returns:
            List of compartment values ordered according to self.labels
        """

        return [compartment_dict[l] for l in self.labels]

    def get_init_list(self):
        """
        Sets starting state for model run according to whether initial conditions are specified, or
        whether we are taking up from where a previous run left off.

        Returns:
            List of compartment values.
        """

        if self.loaded_compartments:
            return self.convert_compartments_to_list(self.loaded_compartments)
        else:
            return self.convert_compartments_to_list(self.init_compartments)

    ''' methods to add intercompartmental flows to the model '''

    def set_var_entry_rate_flow(self, label, var_label):
        """
        Set variable entry/birth/recruitment flow.

        Args:
            label: String for the compartment to which the entry rate applies.
            var_label: String to index the parameters dictionary.
        """
        match = 0
        for pattern in self.remove_labels:
            pattern = '.*' + pattern.replace('_', '.*')
            search_result = re.compile(pattern).match(label)  #
            if search_result:
                match = match + 1

        if match == 0:
            add_unique_tuple_to_list(self.flows_by_type['var_entry'], (label, var_label))
        else:
            pass
            #print('skipping var entry flow from label : ' + label + ' to ' + var_label)

    def set_fixed_infection_death_rate_flow(self, label, param_label):
        """
        Set fixed infection death rate flow.

        Args:
            label: String for the compartment to which the death rate applies.
            param_label: String to index the parameters dictionary.
        """
        match = 0
        for pattern in self.remove_labels:
            pattern = '.*' + pattern.replace('_', '.*')
            search_result = re.compile(pattern).match(label)  #
            if search_result:
                match = match + 1

        if match == 0:
            add_unique_tuple_to_list(self.flows_by_type['fixed_infection_death'], (label, self.params[param_label]))
        else:
            pass
            #print('skipping fixed infection death flow from label  : ' + label + ' to ' + str(self.params[param_label]))

    def set_var_infection_death_rate_flow(self, label, var_label):
        """
        Set variable infection death rate flow.

        Args:
            label: String for the compartment to which the death rate applies.
            var_label: String to index the parameters dictionary.
        """
        match = 0
        for pattern in self.remove_labels:
            pattern = '.*' + pattern.replace('_', '.*')
            search_result = re.compile(pattern).match(label)  #
            if search_result:
                match = match + 1

        if match == 0:
            add_unique_tuple_to_list(self.flows_by_type['var_infection_death'], (label, var_label))
        else:
            pass
            #print('skipping var infection death rate flow from label  : ' + label + ' to  ' + var_label)

    def set_fixed_transfer_rate_flow(self, from_label, to_label, param_label):
        """
        Set fixed inter-compartmental transfer rate flows.

        Args:
            from_label: String for the compartment from which this flow comes.
            to_label: String for the compartment to which this flow goes.
            param_label: String to index the parameters dictionary.
        """
        match = 0
        for pattern in self.remove_labels:
            pattern = '.*' + pattern.replace('_', '.*')
            search_result = re.compile(pattern).match(from_label)  #
            if search_result:
                match = match + 1

        for pattern in self.remove_labels:
            pattern = '.*' + pattern.replace('_', '.*')
            search_result = re.compile(pattern).match(to_label)  #
            if search_result:
                match = match + 1

        prop_ageing_diabetes = 0.082  # provisional hard-code for Bulgaria
        prop_ageing_prison = 0.0014  # provisional hard-code for Bulgaria

        # from_label and to_label does not contain any of the removed compartments
        if match == 0:
            # check for only ageing flows, to exclude other flows such as ipt
            if 'ageing_rate_age' in param_label:
                # check for split, prison_age_min
                if '_norisk' in from_label and '_age5to15' in from_label:
                    # for prison_age_min
                    if '_prison' in to_label:
                        add_unique_tuple_to_list(self.flows_by_type['fixed_transfer'],
                                                 (from_label, to_label,
                                                  self.params[param_label] * prop_ageing_prison))
                    # for norisk ageing flow  age5to15 to age15to25
                    if '_norisk' in to_label:
                        add_unique_tuple_to_list(self.flows_by_type['fixed_transfer'],
                                                 (from_label, to_label,
                                                  self.params[param_label] * (1. - prop_ageing_prison)))
                elif '_norisk'  in from_label and '_age15to25' in from_label:
                    # for diabetes_age_min
                    if '_diabetes' in to_label:
                        add_unique_tuple_to_list(self.flows_by_type['fixed_transfer'],
                                                 (from_label, to_label,
                                                  self.params[param_label] * prop_ageing_diabetes))
                    # for norisk ageing flow age15to25 to age25up
                    if'_norisk' in to_label:
                        add_unique_tuple_to_list(self.flows_by_type['fixed_transfer'],
                                             (from_label, to_label,
                                              self.params[param_label] * (1. - prop_ageing_diabetes)))
                else:
                    add_unique_tuple_to_list(self.flows_by_type['fixed_transfer'],
                                             (from_label, to_label, self.params[param_label]))

            else:
                add_unique_tuple_to_list(self.flows_by_type['fixed_transfer'],
                                         (from_label, to_label, self.params[param_label]))



    def set_linked_transfer_rate_flow(self, from_label, to_label, var_label):
        """
        Set linked inter-compartmental transfer rate flows, where the flow between two compartments is dependent upon
        a flow between another two compartments.

        Args:
            from_label: String for the compartment from which this flow comes.
            to_label: String for the compartment to which this flow goes.
            var_label: String to index the vars dictionary.
        """
        match = 0
        for pattern in self.remove_labels:
            pattern = '.*' + pattern.replace('_', '.*')
            search_result = re.compile(pattern).match(from_label)  #
            if search_result:
                match = match + 1

        for pattern in self.remove_labels:
            pattern = '.*' + pattern.replace('_', '.*')
            search_result = re.compile(pattern).match(to_label)  #
            if search_result:
                match = match + 1

        if match == 0:
            add_unique_tuple_to_list(self.flows_by_type['linked_transfer'], (from_label, to_label, var_label))
        else:
            pass
            #print('skipping linked transfer rate flow from label  : ' + from_label + ' to  ' + to_label)

    def set_var_transfer_rate_flow(self, from_label, to_label, var_label):
        """
        Set variable inter-compartmental transfer rate flows.

        Args:
            from_label: String for the compartment from which this flow comes.
            to_label: String for the compartment to which this flow goes.
            var_label: String to index the vars dictionary.
        """
        match = 0
        for pattern in self.remove_labels:
            pattern = '.*' + pattern.replace('_', '.*')
            search_result = re.compile(pattern).match(from_label)  #
            if search_result:
                match = match + 1

        for pattern in self.remove_labels:
            pattern = '.*' + pattern.replace('_', '.*')
            search_result = re.compile(pattern).match(to_label)  #
            if search_result:
                match = match + 1

        if match == 0:
            add_unique_tuple_to_list(self.flows_by_type['var_transfer'], (from_label, to_label, var_label))
            #print('++++++++++++++ adding var transfer rate flow from label  : ' + from_label + ' to  ' + to_label )
        else:
            pass
            #print('skipping var transfer rate flow from label  : ' + from_label + ' to  ' + to_label)

    ''' variable and flow-related methods '''

    def set_scaleup_fn(self, label, fn):
        """
        Simple method to add a scale-up function to the dictionary of scale-ups.

        Args:
            label: String for name of function
            fn: The function to be added
        """

        self.scaleup_fns[label] = fn

    def clear_vars(self):
        """
        Clear previously populated vars dictionary. Method over-written in economics structures in next tier of model
        object up.
        """

        self.vars.clear()

    def calculate_scaleup_vars(self):
        """
        Find the values of the scale-up functions at a specific point in time. Called within the integration process.
        """

        for label, fn in self.scaleup_fns.items(): self.vars[label] = fn(self.time)

    def calculate_vars(self):
        """
        Calculate the self.vars that depend on current model conditions (compartment sizes) rather than scale-up
        functions (model-specific).
        """

        pass

    def calculate_vars_from_spending(self):
        """
        Method that can be used in economic calculations to calculate coverage of interventions from the spending
        directed to them.
        """

        pass

    def calculate_flows(self):
        """
        Calculate flows, which should only depend on compartment values and self.vars calculated in
        calculate_variable_rates.
        """

        for label in self.labels: self.flows[label] = 0.

        # birth flows
        for label, vars_label in self.flows_by_type['var_entry']:
            self.flows[label] += self.vars[vars_label]

        # dynamic transmission flows
        for from_label, to_label, vars_label in self.flows_by_type['var_transfer']:
            val = self.compartments[from_label] * self.vars[vars_label]
            self.flows[from_label] -= val
            self.flows[to_label] += val

        # fixed-rate flows
        for from_label, to_label, rate in self.flows_by_type['fixed_transfer']:
            val = self.compartments[from_label] * rate
            self.flows[from_label] -= val
            self.flows[to_label] += val

        # linked flows
        for from_label, to_label, vars_label in self.flows_by_type['linked_transfer']:
            val = self.vars[vars_label]
            self.flows[from_label] -= val
            self.flows[to_label] += val

        # normal death flows - note that there has to be a param or a var with the label 'demo_life_expectancy'
        self.vars['rate_death'] = 0.
        for label in self.labels:
            val = self.compartments[label] / self.get_constant_or_variable_param('demo_life_expectancy')
            self.flows[label] -= val

        # extra death flows
        self.vars['rate_infection_death'] = 0.
        for label, rate in self.flows_by_type['fixed_infection_death']:
            val = self.compartments[label] * rate
            self.flows[label] -= val
            self.vars['rate_infection_death'] += val
        for label, rate in self.flows_by_type['var_infection_death']:
            val = self.compartments[label] * self.vars[vars_label]
            self.flows[label] -= val
            self.vars['rate_infection_death'] += val

    def prepare_vars_flows(self):
        """
        This function collects some other functions that previously led to a bug because not all of them were called
        in the diagnostics round.
        """

        self.clear_vars()
        self.calculate_scaleup_vars()
        self.calculate_vars_from_spending()
        self.calculate_vars()
        self.calculate_flows()

    def set_flows(self):
        """
        Main method to work through setting all intercompartmental flows.
        """

        pass

    ''' main integration methods '''

    def init_run(self):
        """
        Works through the main methods in needed for the integration process. Contains more code that is dependent on
        correct naming of inputs, but should be universal to models based on this class (i.e. scenario_end_time).
        """

        self.make_times(self.start_time, self.inputs.model_constants['scenario_end_time'], self.time_step)
        self.initialise_compartments()
        self.set_flows()
        assert self.times is not None, 'Times have not been set yet'

    def make_derivative_fn(self):
        """
        Create the main derivative function.
        """

        def derivative_fn(y, t):
            self.time = t
            self.compartments = self.convert_list_to_compartments(y)
            self.prepare_vars_flows()
            flow_vector = self.convert_compartments_to_list(self.flows)
            self.checks()
            return flow_vector
        return derivative_fn

    def integrate(self):
        """
        Numerical integration. This version also includes storage of compartment / vars / flows solutions which was
        previously done in calculate_diagnostics.
        Currently implemented for Explicit Euler and Runge-Kutta 4 methods
        """

        self.process_uncertainty_params()
        self.init_run()
        y = self.get_init_list()  # get initial conditions (loaded compartments for scenarios)
        y = self.make_adjustments_during_integration(y)

        # prepare storage objects
        n_compartment = len(y)
        n_time = len(self.times)
        self.flow_array = numpy.zeros((n_time, len(self.labels)))

        derivative = self.make_derivative_fn()

        # previously done in calculate_diagnostics
        for i, label in enumerate(self.labels):
            self.compartment_soln[label] = [None] * n_time  # initialise lists
            self.compartment_soln[label][0] = y[i]  # store initial state

        # need to run derivative here to get the initial vars
        k1 = derivative(y, self.times[0])

        # 'make_adjustments_during_integration' was already run but needed to be done again now that derivative
        # has been run. Indeed, derivative allows new vars to be created and these vars are used in
        # 'make_adjustments_during_integration'
        y = self.make_adjustments_during_integration(y)

        #coercing to list for python3
        self.var_labels = list(self.vars.keys())
        self.var_array = numpy.zeros((n_time, len(self.var_labels)))

        # populate arrays for initial state
        for i_label, var_label in enumerate(self.var_labels):
            self.var_array[0, i_label] = self.vars[var_label]
        for i_label, label in enumerate(self.labels):
            self.flow_array[0, i_label] = self.flows[label]

        # initialisation of iterative objects that will be used during integration
        y_candidate = numpy.zeros((len(y)))
        prev_time = self.times[0]  # time of the latest successful integration step (not necessarily stored)
        dt_is_ok = True  # Boolean to indicate whether previous proposed integration time was successfully passed

        # for each time as stored in self.times, except the first one
        for i_time, next_time in enumerate(self.times[1:]):
            store_step = False  # whether the calculated time is to be stored (i.e. appears in self.times)
            cpt_reduce_step = 0  # counts the number of times that the time step needs to be reduced

            while store_step is False:
                if not dt_is_ok:  # previous proposed time step was too wide
                    adaptive_dt /= 2.
                    is_temp_time_in_times = False  # whether the upcoming calculation step corresponds to next_time
                else:  # Previous time step was accepted
                    adaptive_dt = next_time - prev_time
                    is_temp_time_in_times = True  # upcoming attempted integration step corresponds to next_time
                    k1 = numpy.asarray(derivative(y, prev_time))  # evaluate function at previous successful step

                temp_time = prev_time + adaptive_dt  # new attempted calculation time

                # explicit Euler integration
                if self.integration_method == 'Explicit':
                    for i in range(n_compartment):
                        y_candidate[i] = y[i] + adaptive_dt * k1[i]

                # Runge-Kutta 4 integration
                elif self.integration_method == 'Runge Kutta':
                    y_k2 = y + 0.5 * adaptive_dt * k1
                    if (y_k2 >= 0.).all():
                        k2 = numpy.asarray(derivative(y_k2, prev_time + 0.5 * adaptive_dt))
                    else:
                        dt_is_ok = False
                        continue
                    y_k3 = y + 0.5 * adaptive_dt * k2
                    if (y_k3 >= 0.).all():
                        k3 = numpy.asarray(derivative(y_k3, prev_time + 0.5 * adaptive_dt))
                    else:
                        dt_is_ok = False
                        continue
                    y_k4 = y + adaptive_dt * k3
                    if (y_k4 >= 0.).all():
                        k4 = numpy.asarray(derivative(y_k4, temp_time))
                    else:
                        dt_is_ok = False
                        continue

                    y_candidate = []
                    for i in range(n_compartment):
                        y_candidate.append(y[i] + (adaptive_dt / 6.) * (k1[i] + 2. * k2[i] + 2. * k3[i] + k4[i]))

                if (numpy.asarray(y_candidate) >= 0.).all():  # accept the new integration step temp_time
                    dt_is_ok = True
                    prev_time = temp_time
                    cpt_reduce_step = 0
                    for i in range(n_compartment):
                        y[i] = y_candidate[i]
                    if is_temp_time_in_times:
                        store_step = True  # to end the while loop and update i_time
                else:  # if integration failed at proposed step, reduce time step
                    dt_is_ok = False
                    cpt_reduce_step += 1
                    if cpt_reduce_step > 50:
                        print('integration did not complete. The following compartments became negative:')
                        print([self.labels[i] for i in range(len(y_candidate)) if y_candidate[i] < 0.])
                        break

            # adjustments for risk groups
            y = self.make_adjustments_during_integration(y)

            # for stored steps only, store compartment state, vars and intercompartmental flows
            for i, label in enumerate(self.labels):
                self.compartment_soln[label][i_time + 1] = y[i]
            for i_label, var_label in enumerate(self.var_labels):
                self.var_array[i_time + 1, i_label] = self.vars[var_label]
            for i_label, label in enumerate(self.labels):
                self.flow_array[i_time + 1, i_label] = self.flows[label]
            if self.inputs.debug == True:
                with open("compartments_py36.json", "a") as json_file:
                    json_file.write(json.dumps(self.compartments, cls=NumpyEncoder))
                    json_file.write(',\n')

        if self.run_costing:
            self.calculate_economics_diagnostics()

    def process_uncertainty_params(self):
        """
        Perform some simple parameter processing - just for those that are used as uncertainty parameters and so can't
        be processed in the inputs module.
        """

        pass

    def make_adjustments_during_integration(self, y):
        """
        Only relevant to stratified models at this stage, because this is the only adjustment made.
        """

        pass

    def checks(self):
        """
        Assertion(s) run during simulation. Currently only checks that compartments are positive.
        """

        for label in self.labels:
            assert self.compartments[label] >= 0.

    ''' output/diagnostic calculations '''

    def get_compartment_soln(self, label):
        """
        Extract the column of the compartment array pertaining to a particular compartment.

        Args:
            label: String of the compartment.
        Returns:
            The solution for the compartment.
        """

        for i in range(len(self.compartment_soln[label])):
            if type(self.compartment_soln[label][i]) == numpy.array \
                    or type(self.compartment_soln[label][i]) == numpy.ndarray:
                self.compartment_soln[label][i] = self.compartment_soln[label][i][0]
        return numpy.array(self.compartment_soln[label])

    def get_var_soln(self, label):
        """
        Extract the column of var_array that pertains to a particular var.

        Args:
            label: String of the var
        Returns:
            The solution for the var
        """

        i_label = self.var_labels.index(label)
        return self.var_array[:, i_label]

    def get_flow_soln(self, label):
        """
        Extract the column of flow_array that pertains to a particular intercompartmental flow.

        Args:
            label: String of the flow
        Returns:
            The solution for the flow
        """

        i_label = self.labels.index(label)
        return self.flow_array[:, i_label]

    def load_state(self, i_time):
        """
        Returns the recorded compartment values at a particular point in time for the model.

        Args:
            i_time: Time from which the compartment values are to be loade
        Returns:
            state_compartments: The compartment values from that time in the model's integration
        """

        state_compartments = {}
        for label in self.labels:
            state_compartments[label] = self.compartment_soln[label][i_time]
        return state_compartments

    def reset_compartment_values(self):
        """
        Simple method to reset the sizes of all compartments back to their original values.
        """

        self.compartments = {compartment: value for compartment, value in self.init_compartments.items()}

    def calculate_outgoing_compartment_flows(self, from_compartment, to_compartment=''):
        """
        Method to sum the total flows emanating from a set of compartments containing a particular string, restricting
        to the flows entering a particular compartment of interest, if required.

        Args:
            from_compartment: The string of the compartment that the flows are coming out of
            to_compartment: The string of the compartment of interest that flows should be going in to, if any
                (otherwise '' for all flows)
        Returns:
            outgoing_flows: Dictionary of all the compartments of interest and the sum of their outgoing flows
        """

        outgoing_flows = {}
        for label in self.labels:
            if from_compartment in label:
                outgoing_flows[label] = 0.
                for flow in self.flows_by_type['fixed_transfer']:
                    if flow[0] == label and to_compartment in flow[1]:
                        outgoing_flows[label] += flow[2]
                for flow in self.flows_by_type['var_transfer']:
                    if flow[0] == label and to_compartment in flow[1]:
                        outgoing_flows[label] += self.vars[flow[2]]
                for flow in self.flows_by_type['fixed_infection_death']:
                    if flow[0] == label and to_compartment == '':
                        outgoing_flows[label] += flow[1]
                for flow in self.flows_by_type['var_infection_death']:
                    if flow[0] == label and to_compartment == '':
                        outgoing_flows[label] += self.vars[flow[1]]
                if to_compartment == '':
                    outgoing_flows[label] += 1. / self.get_constant_or_variable_param('demo_life_expectancy')
        return outgoing_flows

    def calculate_outgoing_proportion(self, from_compartment, to_compartment=''):
        """
        Method that uses the previous method (calculate_outgoing_compartment_flows) to determine the proportion of all
        flows coming out of a compartment that go a specific compartment.

        Args:
            from_compartment: Origin compartment
            to_compartment: Destination compartment that you want to know the proportions for
        Returns:
            proportion_to_specific_compartment: Dictionary with keys of all the compartments containing the
                from_compartment string and values float proportions of the amount of flows going in that direction.
        """

        outgoing_flows_all \
            = self.calculate_outgoing_compartment_flows(from_compartment=from_compartment)
        outgoing_flows_to_specific_compartment \
            = self.calculate_outgoing_compartment_flows(from_compartment=from_compartment,
                                                        to_compartment=to_compartment)
        proportion_to_specific_compartment = {}
        for compartment in outgoing_flows_all:
            proportion_to_specific_compartment[compartment] \
                = outgoing_flows_to_specific_compartment[compartment] / outgoing_flows_all[compartment]
        return proportion_to_specific_compartment

    def calculate_aggregate_outgoing_proportion(self, from_compartment, to_compartment):

        numerator = 0.
        denominator = 0.
        for flow in self.flows_by_type['fixed_transfer']:
            if from_compartment in flow[0]:
                denominator += flow[2]
                if to_compartment in flow[1]:
                    numerator += flow[2]
        for flow in self.flows_by_type['var_transfer']:
            if from_compartment in flow[0]:
                denominator += self.vars[flow[2]]
                if to_compartment in flow[1]:
                    numerator += self.vars[flow[2]]
        for flow in self.flows_by_type['fixed_infection_death']:
            if from_compartment in flow[0]:
                denominator += flow[1]
        for flow in self.flows_by_type['var_infection_death']:
            if from_compartment in flow[0]:
                denominator += self.vars[flow[1]]
        return numerator / denominator

    def calculate_aggregate_compartment_sizes_from_strings(self, strings):
        """
        Calculate aggregate compartment sizes for the compartments that have a name containing the elements of
        the list strings. This is a diagnostic method to be run after integration. Initially all compartments and then
        becomes the required subset.

        Args:
            strings: a list of strings
        Returns:
            A time-series providing the aggregate size for the different times of self.times.
        """

        aggregate_sizes = numpy.zeros(len(self.times))
        for string in strings: compartments_to_aggregate = [c for c in self.labels if string in c]
        for compartment in compartments_to_aggregate:
            aggregate_sizes = tool_kit.elementwise_list_addition(aggregate_sizes, self.compartment_soln[compartment])
        return aggregate_sizes

    def calculate_economics_diagnostics(self):

        pass

    ''' flow diagram '''

    def make_flow_diagram(self, png):
        """
        Use graphviz module to create flow diagram of compartments and intercompartmental flows.
        """

        styles = {
            'graph': {'label': 'Dynamic Transmission Model',
                      'fontsize': '16',},
            'nodes': {'fontname': 'Helvetica',
                      'shape': 'box',
                      'style': 'filled',
                      'fillcolor': '#CCDDFF',},
            'edges': {'style': 'dotted',
                      'arrowhead': 'open',
                      'fontname': 'Courier',
                      'fontsize': '10',}
        }

        def apply_styles(graph, styles):
            graph.graph_attr.update(('graph' in styles and styles['graph']) or {})
            graph.node_attr.update(('nodes' in styles and styles['nodes']) or {})
            graph.edge_attr.update(('edges' in styles and styles['edges']) or {})
            return graph

        def num_str(f):
            abs_f = abs(f)
            if abs_f > 1e9: return '%.1fB' % (f/1E9)
            if abs_f > 1e6: return '%.1fM' % (f/1E6)
            if abs_f > 1e3: return '%.1fK' % (f/1E3)
            if abs_f > 100.: return '%.0f' % f
            if abs_f > 0.5: return '%.1f' % f
            if abs_f > 0.05: return '%.2f' % f
            if abs_f > 0.0005: return '%.4f' % f
            if abs_f > 0.000005: return '%.6f' % f
            return str(f)

        self.graph = Digraph(format='png')
        for label in self.labels:
            self.graph.node(label)
        self.graph.node('tb_death')
        for from_label, to_label, var_label in self.flows_by_type['var_transfer']:
            self.graph.edge(from_label, to_label, label=var_label[:4])
        for from_label, to_label, rate in self.flows_by_type['fixed_transfer']:
            self.graph.edge(from_label, to_label, label=num_str(rate))
        for from_label, to_label, rate in self.flows_by_type['linked_transfer']:
            self.graph.edge(from_label, to_label, label='link')
        for label, rate in self.flows_by_type['fixed_infection_death']:
            self.graph.edge(label, 'tb_death', label=num_str(rate))
        for label, rate in self.flows_by_type['var_infection_death']:
            self.graph.edge(label, 'tb_death', label=var_label[:4])
        base, ext = os.path.splitext(png)
        if ext.lower() != '.png': base = png
        self.graph = apply_styles(self.graph, styles)
        self.graph.render(base)


class EconomicModel(BaseModel):
    """
    Class containing all the economic-related methods, although the BaseModel needs to have structure for placeholders
    for these methods when economics is not being utilised.
    """

    def __init__(self):
        BaseModel.__init__(self)

        self.cost_times = []
        self.eco_drives_epi = False
        self.costs = None
        self.end_period_costing = 2035.
        self.interventions_to_cost = []
        self.interventions_considered_for_opti = []
        self.available_funding = {}
        self.annual_available_funding = {}
        self.startups_apply = {}

    def clear_vars(self):

        # before clearing vars, need to save the popsize vars for economics calculations
        saved_vars = {}
        if self.eco_drives_epi:
            for key in self.vars.keys():
                if 'popsize' in key: saved_vars[key] = self.vars[key]

        # clear previously populated vars dictionary
        self.vars.clear()

        # re-populated from saved vars
        self.vars = saved_vars

    def calculate_economics_diagnostics(self):
        """
        Run the economics diagnostics associated with a model run. Note that integration has been completed by this
        point. Only the raw costs are stored in the model object, while the other costs will be calculated when
        generating outputs.
        """

        start_index_for_costs = tool_kit.find_first_list_element_at_least(
            self.times, self.inputs.model_constants['recent_time'])
        self.cost_times = self.times[start_index_for_costs:]
        self.costs = numpy.zeros((len(self.cost_times), len(self.interventions_to_cost)))

        for i, inter in enumerate(self.interventions_to_cost):
            for t, time in enumerate(self.cost_times):

                # costs from cost-coverage curves
                cost = get_cost_from_coverage(self.scaleup_fns['int_prop_' + inter](time),
                                              self.inputs.model_constants['econ_inflectioncost_' + inter],
                                              self.inputs.model_constants['econ_saturation_' + inter],
                                              self.inputs.model_constants['econ_unitcost_' + inter],
                                              self.var_array[start_index_for_costs + t, self.var_labels.index('popsize_' + inter)])

                # start-up costs
                if 'econ_startupcost_' + inter in self.inputs.model_constants \
                        and 'econ_startupduration_' + inter in self.inputs.model_constants \
                        and self.inputs.model_constants['econ_startupduration_' + inter] > 0.:
                    cost += self.calculate_startup_cost(time, inter)
                self.costs[t, i] = cost

    def calculate_startup_cost(self, time, intervention):
        """
        Finds a smoothed out amount of start-up costs to the relevant times. Uses the beta PDF to smooth out scale-up
        costs. Note that the beta PDF of scipy returns zeros if its first argument is not between zero and one, so the
        code should still run if there are no start-up costs.

        Args:
            time: Float for the calendar time being costed
            intervention: String for the intervention being costed
        Returns:
            Costs at the time point considered for the intervention in question updated for start-ups as required
        """

        return scipy.stats.beta.pdf((time - self.inputs.intervention_startdates[self.scenario][intervention])
                                    / self.inputs.model_constants['econ_startupduration_' + intervention], 2., 5.) \
            / self.inputs.model_constants['econ_startupduration_' + intervention] \
            * self.inputs.model_constants['econ_startupcost_' + intervention]

    def calculate_vars_from_spending(self):
        """
        Update parameter values according to the funding allocated to each interventions. This process is done during
        integration.
        """

        if self.eco_drives_epi and self.time > self.inputs.model_constants['recent_time']:
            interventions = self.interventions_considered_for_opti
            for i in interventions:

                # should ultimately remove this code
                if i in ['ipt_age0to5', 'ipt_age5to15'] and len(self.agegroups) < 2: continue

                int_key = 'int_prop_' + i

                # update intervention_startdates if starting now
                if self.inputs.intervention_startdates[self.scenario][i] is None:
                    self.inputs.intervention_startdates[self.scenario][i] = self.time

                # starting cost has already been taken into account in 'distribute_funding_across_years'
                self.vars[int_key] \
                    = get_coverage_from_cost(self.annual_available_funding[i],
                                             self.inputs.model_constants['econ_inflectioncost_' + i],
                                             self.inputs.model_constants['econ_saturation_' + i],
                                             self.inputs.model_constants['econ_unitcost_' + i],
                                             self.vars['popsize_' + i] if 'popsize_' + i in self.vars.keys() else 0.,
                                             alpha=1.)

    def distribute_funding_across_years(self):

        # number of years to fund
        n_years = self.end_period_costing - self.inputs.model_constants['scenario_start_time']
        for inter in self.interventions_considered_for_opti:
            self.annual_available_funding[inter] = 0.
            # if intervention hasn't started
            if self.inputs.intervention_startdates[self.scenario][inter] is None:
                if self.available_funding[inter] < self.inputs.model_constants['econ_startupcost_' + inter]:
                    # print 'available_funding insufficient to cover starting costs of ' + int
                    pass
                else:
                    self.inputs.intervention_startdates[self.scenario][inter] \
                        = self.inputs.model_constants['scenario_start_time']
                    self.annual_available_funding[inter] \
                        = (self.available_funding[inter] - self.inputs.model_constants['econ_startupcost_' + inter]) \
                          / n_years
            else:
                self.annual_available_funding[inter] = (self.available_funding[inter])/n_years


class StratifiedModel(EconomicModel):
    """
    Adds the stratifications for the entire populations and some methods that are universally applicable to them. Note
    that the stratifications into disease classes (e.g. pulmonary/extrapulmonary for TB) is done in the disease-specific
    model module.
    """

    def __init__(self):
        BaseModel.__init__(self)

        self.agegroups = []
        self.riskgroups = []
        self.actual_risk_props = {}
        self.target_risk_props = {}

    def get_relevant_riskgroup_denominators(self):
        """
        Calculate the population by age-group for the age-groups that are relevant to each risk-group. For example, the
        25up population will be returned associated with the diabetes risk-group.

        return: a dictionary keyed with the names of risk-groups and containing the relevant population sizes
        """
        denominators = {}
        for riskgroup in self.riskgroups:
            strings_to_dodge = []  # will contain the age-related strings of the compartments that had been removed
            for label in self.remove_labels:
                if riskgroup[1:] in label:
                    strings_to_dodge.append(label.split('_')[1])

            popsize = 0.
            if self.compartments:
                for label in self.compartments.keys():
                    counts = True  # will become False if one of the dodged strings is detected in the compartment name
                    for dodged_label in strings_to_dodge:
                        if dodged_label in label:
                            counts = False
                    if counts:
                        popsize += self.compartments[label]
            denominators[riskgroup] = popsize

        return denominators

    def make_adjustments_during_integration(self, y):
        """
        Adjusts the proportions of the population in each risk group according to the calculations
        made in assess_risk_props above.

        Args:
            y: The original compartment vector y to be adjusted
        Returns:
            The adjusted compartment vector (y).
        """

        risk_adjustment_factor = {}

        # find the target proportions for each risk group stratum
        if len(self.riskgroups) > 1:
            for riskgroup in self.riskgroups:
                if riskgroup not in self.target_risk_props:
                    self.target_risk_props[riskgroup] = []
            self.target_risk_props['_norisk'].append(1.)
            for riskgroup in self.riskgroups:
                if riskgroup != '_norisk':
                    self.target_risk_props[riskgroup].append(
                        self.get_constant_or_variable_param('riskgroup_prop' + riskgroup))
                    self.target_risk_props['_norisk'][-1] \
                        -= self.target_risk_props[riskgroup][-1]

            # if integration has started properly
            if self.compartments:

                # find the actual proportions in each risk group stratum
                population = sum(self.compartments.values())
                relevant_riskgroup_denominators = self.get_relevant_riskgroup_denominators()
                for riskgroup in self.riskgroups:
                    if riskgroup not in self.actual_risk_props:
                        self.actual_risk_props[riskgroup] = []
                    self.actual_risk_props[riskgroup].append(0.)
                    for c in self.compartments:
                        if riskgroup in c:
                            self.actual_risk_props[riskgroup][-1] += self.compartments[c]

                    self.actual_risk_props[riskgroup][-1] /= relevant_riskgroup_denominators[riskgroup]

                    if self.inputs.debug:
                        if 2015. < self.time < 2015.5:
                            print("############")
                            print("2015 percentage of " + riskgroup)
                            print(100. * self.actual_risk_props[riskgroup][-1])


                # find the scaling factor for the risk group in question
                for riskgroup in self.riskgroups:
                    if self.actual_risk_props[riskgroup][-1] > 0.:
                        risk_adjustment_factor[riskgroup] = self.target_risk_props[riskgroup][-1] \
                                                                     / self.actual_risk_props[riskgroup][-1]
                    else:
                        risk_adjustment_factor[riskgroup] = 1.
        else:

            # otherwise, it's just a list of ones
            if '' not in self.target_risk_props: self.target_risk_props[''] = []
            self.target_risk_props[''].append(1.)

        if risk_adjustment_factor != {}:
            compartments = self.convert_list_to_compartments(y)
            # for c in compartments:
            #     for riskgroup in self.riskgroups:
            #         if riskgroup in c: compartments[c] *= risk_adjustment_factor[riskgroup]
            return self.convert_compartments_to_list(compartments)
        else:
            return y

    def set_ageing_flows(self):
        """
        Set ageing flows for any number of age groups.
        """

        for label in self.labels:
            for n_agegroup, agegroup in enumerate(self.agegroups):
                if agegroup in label and n_agegroup < len(self.agegroups) - 1:
                    self.set_fixed_transfer_rate_flow(label,
                                                      label[0: label.find('_age')] + self.agegroups[n_agegroup + 1],
                                                      'ageing_rate' + self.agegroups[n_agegroup])

        # for ageing flow if labels _age5to15 the set the ageing flow for _age15to25  be setting match to zero
        for riskgroups in self.riskgroups:
            if riskgroups != '_norisk' and riskgroups[1:] in self.inputs.params_to_age_weight:
                # print('------' + riskgroups)
                riskgroups_age_min = riskgroups[1:] + '_age_min'
                if self.inputs.model_constants[riskgroups_age_min]:
                    for label in self.labels:
                        if re.search('_norisk', label):
                            if str(int(self.inputs.model_constants[riskgroups_age_min])) == label[-2:]:
                                norisk_agegroup = label[label.find('_age'):]
                                next_risk_agegroup = self.agegroups[self.agegroups.index(norisk_agegroup) + 1]
                                new_label = label.replace('_norisk', riskgroups)
                                new_label = new_label.replace(norisk_agegroup, next_risk_agegroup)
                                self.set_fixed_transfer_rate_flow(label, new_label, 'ageing_rate' + norisk_agegroup)

                riskgroups_age_max = riskgroups[1:] + '_age_max'
                if self.inputs.model_constants[riskgroups_age_max]:
                    for label in self.labels:
                        if re.search(riskgroups, label):
                            if str(int(self.inputs.model_constants[riskgroups_age_max])) == label[-2:]:
                                risk_agegroup = label[label.find('_age'):]
                                next_norisk_agegroup = self.agegroups[self.agegroups.index(risk_agegroup) + 1]
                                new_label = label.replace(riskgroups, '_norisk')
                                new_label = new_label.replace(risk_agegroup, next_norisk_agegroup)
                                self.set_fixed_transfer_rate_flow(label, new_label, 'ageing_rate' + risk_agegroup)