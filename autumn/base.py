
import os
import numpy
from scipy.integrate import odeint
import tool_kit
from autumn.economics import get_cost_from_coverage, get_coverage_from_cost
import scipy.stats


def add_unique_tuple_to_list(a_list, a_tuple):

    """
    Adds or modifies a list of tuples, compares only the items
    before the last in the tuples, the last value in the tuple
    is assumed to be a value.

    """

    for i, test_tuple in enumerate(a_list):
        if test_tuple[:-1] == a_tuple[:-1]:
            a_list[i] = a_tuple
            break
    else:
        a_list.append(a_tuple)


class BaseModel:

    def __init__(self):

        self.inputs = None
        self.gui_inputs = {}
        self.labels = []
        self.init_compartments = {}
        self.compartments = {}
        self.params = {}
        self.times = []
        self.time = 0.
        self.start_time = 0.
        self.cost_times = []
        self.scaleup_fns = {}
        self.vars = {}
        self.soln_array = None
        self.var_labels = None
        self.var_array = None
        self.flow_array = None
        self.fraction_array = None
        self.flows = {}
        self.fixed_transfer_rate_flows = []
        self.linked_transfer_rate_flows = []
        self.fixed_infection_death_rate_flows = []
        self.var_transfer_rate_flows = []
        self.var_entry_rate_flows = []
        self.var_infection_death_rate_flows = []
        self.agegroups = []
        self.costs = None
        self.intervention = None
        self.run_costing = True
        self.end_period_costing = 2035
        self.interventions_to_cost = ['vaccination', 'xpert', 'treatment_support', 'smearacf', 'xpertacf',
                                     'ipt_age0to5', 'ipt_age5to15', 'decentralisation']
        self.eco_drives_epi = False
        self.available_funding = {}
        self.annual_available_funding = {}
        self.startups_apply = {}
        self.intervention_startdates = {}
        self.graph = None
        self.comorbidities = []
        self.actual_comorb_props = {}
        self.target_comorb_props = {}
        self.loaded_compartments = None

    ##############################
    ### Time-related functions ###
    ##############################

    def make_times(self, start, end, delta):

        """
        Simple method to create time steps for reporting of outputs.

        Args:
            start: Start time for integration.
            end: End time for integration.
            delta: Step size.

        Creates:
            self.times: List of model time steps.

        """

        self.times = []
        step = start
        while step <= end:
            self.times.append(step)
            step += delta
        if self.times[-1] < end:
            self.times.append(end)

    def find_time_index(self, time):

        """
        Method to find first time point in times list after a certain specified time point.

        Args:
            time: The time point for interrogation.

        Returns:
            index: The index of the self.times list that refers to the time point argument.

        """

        for index, model_time in enumerate(self.times):
            if model_time > time:
                return index

        raise ValueError('Time not found')

    #####################################################
    ### Methods to set values to aspects of the model ###
    #####################################################

    def set_parameter(self, label, val):

        """
        Almost to simple to need a method. Sets a single parameter value.

        Args:
            label: Parameter name.
            val: Parameter value.

        Modifies:
            self.params: Adds a parameter value to this dictionary.

        """

        self.params[label] = val

    def get_constant_or_variable_param(self, param):

        """
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
            label: The name of the compartment.
            init_val: The starting size of this compartment.

        Modifies:
            self.init_compartments: Assigns init_val to the compartment specified.

        """

        assert init_val >= 0., 'Start with negative compartment not permitted'
        if label not in self.labels:
            self.labels.append(label)
        self.init_compartments[label] = init_val

    def initialise_compartments(self):

        """
        Initialise compartments to starting values.

        """

        pass

    ####################################################
    ### Methods to manipulate compartment data items ###
    ####################################################

    def convert_list_to_compartments(self, compartment_vector):

        """
        Uses self.labels to convert list of compartments to dictionary.

        Args:
            compartment_vector: List of compartment values.

        Returns:
            Dictionary with keys the compartment names (from the strings in self.labels) and values the elements
                from compartment_vector.
        """

        return {l: compartment_vector[i] for i, l in enumerate(self.labels)}

    def convert_compartments_to_list(self, compartment_dict):

        """
        Reverse of previous method. Converts

        Args:
            compartment_dict: Dictionary with keys strings of compartment names.

        Returns:
            List of compartment values ordered according to self.labels.

        """

        return [compartment_dict[l] for l in self.labels]

    def get_init_list(self):

        """
        Sets starting state for model run according to whether initial conditions are specified, or
        whether we are taking up from where a previous run left off.

        Returns:
            List of compartment values.

        """

        if self.loaded_compartments is None:
            return self.convert_compartments_to_list(self.init_compartments)
        else:
            return self.convert_compartments_to_list(self.loaded_compartments)

    ############################################################
    ### Methods to add intercompartmental flows to the model ###
    ############################################################

    def set_var_entry_rate_flow(self, label, var_label):

        """
        Set variable entry/birth/recruitment flow.

        Args:
            label: String for the compartment to which the entry rate applies.
            var_label: String to index the parameters dictionary.

        Returns:
            Adds to self.var_flows, which apply variable birth rates to susceptible (generally) compartments.

        """

        add_unique_tuple_to_list(
            self.var_entry_rate_flows,
            (label, var_label))

    def set_fixed_infection_death_rate_flow(self, label, param_label):

        """
        Set fixed infection death rate flow.

        Args:
            label: String for the compartment to which the death rate applies.
            param_label: String to index the parameters dictionary.

        Returns:
            Adds to self.fixed_infection_death_rate_flows, which apply single death rates to active infection
                compartments.

        """

        add_unique_tuple_to_list(
            self.fixed_infection_death_rate_flows,
            (label, self.params[param_label]))

    def set_var_infection_death_rate_flow(self, label, var_label):

        """
        Set variable infection death rate flow.

        Args:
            label: String for the compartment to which the death rate applies.
            var_label: String to index the parameters dictionary.

        Returns:
            Adds to self.var_infection_death_rate_flows, which apply single variable death rates to active infection
                compartments.

        """

        add_unique_tuple_to_list(
            self.var_infection_death_rate_flows,
            (label, var_label))

    def set_fixed_transfer_rate_flow(self, from_label, to_label, param_label):

        """
        Set fixed inter-compartmental transfer rate flows.

        Args:
            from_label: String for the compartment from which this flow comes.
            to_label: String for the compartment to which this flow goes.
            param_label: String to index the parameters dictionary.

        Returns:
            Adds to self.fixed_transfer_rate_flows, which apply single fixed intercompartmental transfer rates to
                two compartments.

        """

        add_unique_tuple_to_list(
            self.fixed_transfer_rate_flows,
            (from_label, to_label, self.params[param_label]))

    def set_linked_transfer_rate_flow(self, from_label, to_label, var_label):

        """
        Set linked inter-compartmental transfer rate flows, where the flow between two compartments is dependent upon
        a flow between another two compartments.

        Args:
            from_label: String for the compartment from which this flow comes.
            to_label: String for the compartment to which this flow goes.
            var_label: String to index the vars dictionary.

        Returns:
            Adds to self.linked_transfer_rate_flows, which apply variable, linked intercompartmental transfer rates to
                two compartments.

        """

        add_unique_tuple_to_list(
            self.linked_transfer_rate_flows,
            (from_label, to_label, var_label))

    def set_var_transfer_rate_flow(self, from_label, to_label, var_label):

        """
        Set variable inter-compartmental transfer rate flows.

        Args:
            from_label: String for the compartment from which this flow comes.
            to_label: String for the compartment to which this flow goes.
            var_label: String to index the vars dictionary.

        Returns:
            Adds to self.var_transfer_rate_flows, which apply variable intercompartmental transfer rates to
                two compartments.

        """

        add_unique_tuple_to_list(
            self.var_transfer_rate_flows,
            (from_label, to_label, var_label))

    #########################################
    ### Variable and flow-related methods ###
    #########################################

    def set_scaleup_fn(self, label, fn):

        """
        Simple method to add a scale-up function to the dictionary of scale-ups.

        Args:
            label: String for name of function.
            fn: The function to be added.

        """

        self.scaleup_fns[label] = fn

    def calculate_scaleup_vars(self):

        """
        Find the values of the scale-up functions at a specific point in time.
        Called within the integration process.

        """

        for label, fn in self.scaleup_fns.iteritems():
            self.vars[label] = fn(self.time)

    def calculate_vars(self):

        """
        Calculate the self.vars that depend on current model conditions (compartment sizes) rather than scale-up
        functions. (Model-specific, so currently just "pass".)

        """

        pass

    def calculate_flows(self):

        """
        Calculate flows, which should only depend on compartment values
        and self.vars calculated in calculate_variable_rates.

        """

        for label in self.labels:
            self.flows[label] = 0.

        # Birth flows
        for label, vars_label in self.var_entry_rate_flows:
            self.flows[label] += self.vars[vars_label]

        # Dynamic transmission flows
        for from_label, to_label, vars_label in self.var_transfer_rate_flows:
            val = self.compartments[from_label] * self.vars[vars_label]
            self.flows[from_label] -= val
            self.flows[to_label] += val

        # Fixed-rate flows
        for from_label, to_label, rate in self.fixed_transfer_rate_flows:
            val = self.compartments[from_label] * rate
            self.flows[from_label] -= val
            self.flows[to_label] += val

        # Linked flows
        for from_label, to_label, vars_label in self.linked_transfer_rate_flows:
            val = self.vars[vars_label]
            self.flows[from_label] -= val
            self.flows[to_label] += val

        # Normal death flows
        # Note that there has to be a param or a var with the label 'demo_life_expectancy'
        # (which has to have this name). Saves on creating a separate model attribute
        # just for population death. I think it makes more sense for it to be just
        # another parameter.
        self.vars['rate_death'] = 0.
        for label in self.labels:
            val = self.compartments[label] / self.get_constant_or_variable_param('demo_life_expectancy')
            self.flows[label] -= val
            self.vars['rate_death'] += val

        # Extra death flows
        self.vars['rate_infection_death'] = 0.
        for label, rate in self.fixed_infection_death_rate_flows:
            val = self.compartments[label] * rate
            self.flows[label] -= val
            self.vars['rate_infection_death'] += val
        for label, rate in self.var_infection_death_rate_flows:
            val = self.compartments[label] * self.vars[vars_label]
            self.flows[label] -= val
            self.vars['rate_infection_death'] += val

    def prepare_vars_flows(self):

        """
        This function collects some other functions that previously led to a bug because not all of them were called
        in the diagnostics round.

        """

        # Before clearing vars, we need to save the popsize vars for economics calculations
        saved_vars = {}
        if self.eco_drives_epi:
            for key in self.vars.keys():
                if 'popsize' in key: saved_vars[key] = self.vars[key]

        # Clear previously populated vars dictionary
        self.vars.clear()

        # Re-populated from saved vars
        self.vars = saved_vars

        # Calculate vars and flows sequentially
        self.calculate_scaleup_vars()
        self.calculate_vars()
        self.calculate_flows()

    def set_flows(self):

        """
        Main method to work through setting all intercompartmental flows.

        """

        pass

    def set_ageing_flows(self):

        """
        Set ageing flows for any number of age stratifications.

        """

        for label in self.labels:
            for number_agegroup in range(len(self.agegroups)):
                if self.agegroups[number_agegroup] in label and number_agegroup < len(self.agegroups) - 1:
                    self.set_fixed_transfer_rate_flow(
                        label, label[0: label.find('_age')] + self.agegroups[number_agegroup + 1],
                        'ageing_rate' + self.agegroups[number_agegroup])

    ################################
    ### Main integration methods ###
    ################################

    def init_run(self):

        """
        Works through the main methods in needed for the integration process.

        """

        self.make_times(self.start_time,
                        self.inputs.model_constants['scenario_end_time'],
                        self.gui_inputs['time_step'])
        self.initialise_compartments()
        if len(self.agegroups) > 0: self.set_ageing_flows()
        self.set_flows()
        assert self.times is not None, 'Times have not been set yet'

    def make_derivative_fn(self):

        """
        Create the main derivative function

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
        Method to select integration approach, using request from GUI.

        """

        dt_max = 2.
        if self.gui_inputs['integration_method'] == 'Explicit':
            self.integrate_explicit(dt_max)
        elif self.gui_inputs['integration_method'] == 'Runge Kutta':
            self.integrate_runge_kutta(dt_max)

    def integrate_explicit(self, dt_max=0.05):

        """
        Integrate with Euler Explicit method.

        Input:
            min_dt: represents the time step for calculation points. The attribute self.times will also be used to make
                sure that a solution is affected to the time points known by the model

        """

        self.init_run()
        y = self.get_init_list()
        y_candidate = numpy.zeros((len(y)))
        n_compartment = len(y)
        n_time = len(self.times)
        self.soln_array = numpy.zeros((n_time, n_compartment))

        derivative = self.make_derivative_fn()
        old_time = self.times[0]
        time = old_time
        self.soln_array[0, :] = y
        dt_is_ok = True
        for i_time, new_time in enumerate(self.times):
            while time < new_time:
                if not dt_is_ok:
                    adaptive_dt_max = dt / 2.
                else:
                    adaptive_dt_max = dt_max
                    old_time = time
                dt_is_ok = True

                f = derivative(y, time)
                time = old_time + adaptive_dt_max
                dt = adaptive_dt_max
                if time > new_time:
                    dt = new_time - old_time
                    time = new_time

                for i in range(n_compartment):
                    y_candidate[i] = y[i] + dt * f[i]

                if (numpy.asarray(y_candidate) >= 0).all():
                    dt_is_ok = True
                    for i in range(n_compartment):
                        y[i] = y_candidate[i]
                else:
                    dt_is_ok = False

            if i_time < n_time - 1:
                self.soln_array[i_time+1, :] = y
                self.prepare_comorb_adjustments()
                y = self.adjust_compartment_size(y)

        self.calculate_diagnostics()
        if self.run_costing:
            self.calculate_economics_diagnostics()

    def integrate_runge_kutta(self, dt_max=0.05):

        """
        Uses Runge-Kutta 4 method.

        Input:
            min_dt: represents the time step for calculation points. The attribute self.times will also be used to make
                sure that a solution is affected to the time points known by the model

        """

        self.init_run()
        y = self.get_init_list()
        n_compartment = len(y)
        n_time = len(self.times)
        self.soln_array = numpy.zeros((n_time, n_compartment))

        derivative = self.make_derivative_fn()
        old_time = self.times[0]
        time = self.times[0]
        self.soln_array[0, :] = y
        dt_is_ok = True
        for i_time, new_time in enumerate(self.times):
            while time < new_time:
                if not dt_is_ok:
                    adaptive_dt_max = dt/2.0
                else:
                    old_time = time
                    adaptive_dt_max = dt_max
                dt_is_ok = True
                time = old_time + adaptive_dt_max
                dt = adaptive_dt_max
                if time > new_time:
                    dt = new_time - old_time
                    time = new_time
                k1 = numpy.asarray(derivative(y, old_time))
                y_k2 = y + 0.5 * dt * k1
                if (y_k2 >= 0).all():
                    k2 = numpy.asarray(derivative(y_k2, old_time + 0.5*dt))
                else:
                    dt_is_ok = False
                    continue
                y_k3 = y + 0.5 * dt * k2
                if (y_k3 >= 0).all():
                    k3 = numpy.asarray(derivative(y_k3,  old_time + 0.5*dt))
                else:
                    dt_is_ok = False
                    continue
                y_k4 = y + dt*k3
                if (y_k4 >= 0).all():
                    k4 = numpy.asarray(derivative(y_k4, time))
                else:
                    dt_is_ok = False
                    continue

                y_candidate = []
                for i in range(n_compartment):
                    y_candidate.append(y[i] + (dt/6.0) * (k1[i] + 2.0*k2[i] + 2.0*k3[i] + k4[i]))

                if (numpy.asarray(y_candidate) >= 0).all():
                    y = y_candidate
                else:
                    dt_is_ok = False
                    continue

            if i_time < n_time - 1:
                self.soln_array[i_time + 1, :] = y
                self.prepare_comorb_adjustments()
                y = self.adjust_compartment_size(y)

        self.calculate_diagnostics()
        if self.run_costing:
            self.calculate_economics_diagnostics()

    def checks(self):

        """
        Assertion run during the simulation.

        Args:
            error_margin: acceptable difference between target invariants

        """

        # Check all compartments are positive
        for label in self.labels:
            assert self.compartments[label] >= 0.

    ######################################
    ### Output/diagnostic calculations ###
    ######################################

    def calculate_output_vars(self):

        """
        Calculate diagnostic vars that can depend on self.flows, as well as self.vars calculated in calculate_vars.

        """

        pass

    def calculate_diagnostics(self):

        # Populate the self.compartment_soln dictionary
        self.compartment_soln = {}
        for label in self.labels:
            if label in self.compartment_soln:
                continue
            self.compartment_soln[label] = self.get_compartment_soln(label)

        # Run through the integration times
        n_time = len(self.times)
        for t in range(n_time):

            # Replicate the times that occurred during integration
            self.time = self.times[t]

            # Replicate the compartment values that occurred during integration
            for label in self.labels:
                self.compartments[label] = self.compartment_soln[label][t]

            # Prepare the vars as during integration
            self.prepare_vars_flows()
            self.calculate_output_vars()

            # Initialise arrays if not already done
            if self.var_labels is None:
                self.var_labels = self.vars.keys()
                self.var_array = numpy.zeros((n_time, len(self.var_labels)))
                self.flow_array = numpy.zeros((n_time, len(self.labels)))

            # Populate arrays
            for i_label, label in enumerate(self.var_labels):
                self.var_array[t, i_label] = self.vars[label]
            for i_label, label in enumerate(self.labels):
                self.flow_array[t, i_label] = self.flows[label]

        # Thinking of getting rid of this section - should be possible to calculate in model_runner rather than model
        self.fraction_array = numpy.zeros((n_time, len(self.labels)))
        self.fraction_soln = {}
        for i_label, label in enumerate(self.labels):
            self.fraction_soln[label] = [
                v / t
                for v, t
                in zip(
                    self.compartment_soln[label],
                    self.get_var_soln('population'))]
            self.fraction_array[:, i_label] = self.fraction_soln[label]

    def calculate_economics_diagnostics(self):

        """
        Run the economics diagnostics associated with a model run.
        Integration has been completed by this point.
        Only the raw costs are stored in the model object. The other costs will be calculated when generating outputs

        """

        self.determine_whether_startups_apply()

        # Find start and end indices for economics calculations
        start_index = tool_kit.find_first_list_element_at_least_value(self.times,
                                                                      self.inputs.model_constants['recent_time'])
        end_index = tool_kit.find_first_list_element_at_least_value(self.times,
                                                                    self.inputs.model_constants['scenario_end_time'])
        self.cost_times = self.times[start_index:]
        self.costs = numpy.zeros((len(self.cost_times), len(self.interventions_to_cost)))

        # Loop over interventions to be costed
        for int_index, intervention in enumerate(self.interventions_to_cost):
            # for each step time. We may want to change this bit. No need for all time steps
            # Just add a third argument if you want to decrease the frequency of calculation
            for i, t in enumerate(self.cost_times):
                cost = get_cost_from_coverage(self.scaleup_fns['program_prop_' + intervention](t),
                                              self.inputs.model_constants['econ_inflectioncost_' + intervention],
                                              self.inputs.model_constants['econ_saturation_' + intervention],
                                              self.inputs.model_constants['econ_unitcost_' + intervention],
                                              self.var_array[i, self.var_labels.index('popsize_' + intervention)])
                # Start-up costs
                if self.startups_apply[intervention] \
                        and self.inputs.model_constants['scenario_start_time'] < t \
                        and t < self.inputs.model_constants['scenario_start_time'] \
                                + self.inputs.model_constants['econ_startupduration_' + intervention]:

                    # New code with beta PDF used to smooth out scale-up costs
                    cost += scipy.stats.beta.pdf((t - self.inputs.model_constants['scenario_start_time'])
                                                 / self.inputs.model_constants['econ_startupduration_' + intervention],
                                                 2.,
                                                 5.) \
                            / self.inputs.model_constants['econ_startupduration_' + intervention] \
                            * self.inputs.model_constants['econ_startupcost_' + intervention]

                self.costs[i, int_index] = cost

    def update_vars_from_cost(self):

        """
        update parameter values according to the funding allocated to each interventions. This process is done during
        integration
        Returns:
        Nothing
        """
        interventions = self.interventions_to_cost
        for int in interventions:
            if (int in ['ipt_age0to5', 'ipt_age5to15']) and (len(self.agegroups) < 2):
                continue

            vars_key = 'program_prop_' + int
            cost = self.annual_available_funding[int]
            if cost == 0:
                coverage = 0
            else:
                unit_cost = self.inputs.model_constants['econ_unitcost_' + int]
                c_inflection_cost = self.inputs.model_constants['econ_inflectioncost_' + int]
                saturation = self.inputs.model_constants['econ_saturation_' + int]
                popsize_key = 'popsize_' + int
                if popsize_key in self.vars.keys():
                    pop_size = self.vars[popsize_key]
                else:
                    pop_size = 0

                # starting costs
                # is a programm starting right now? In that case, update intervention_startdates
                if self.intervention_startdates[int] is None: # means intervention hadn't started yet
                    self.intervention_startdates[int] = self.time

                # starting cost has already been taken into account in 'distribute_funding_across_years'
                coverage = get_coverage_from_cost(cost, c_inflection_cost, saturation, unit_cost, pop_size, alpha=1.0)
            self.vars[vars_key] = coverage

    def get_compartment_soln(self, label):

        """
        Get the column of soln_array that pertains to a particular compartment.

        Args:
            label: String of the compartment.

        Returns:
            The solution for the compartment.

        """

        assert self.soln_array is not None, 'calculate_diagnostics has not been run'
        i_label = self.labels.index(label)
        return self.soln_array[:, i_label]

    def get_var_soln(self, label):

        """
        Get the column of var_array that pertains to a particular compartment.

        Args:
            label: String of the var.

        Returns:
            The solution for the var.

        """

        assert self.var_array is not None, 'calculate_diagnostics has not been run'
        i_label = self.var_labels.index(label)
        return self.var_array[:, i_label]

    def get_flow_soln(self, label):

        """
        Get the column of flow_array that pertains to a particular compartment.

        Args:
            label: String of the flow.

        Returns:
            The solution for the flow.

        """

        assert self.flow_array is not None, 'calculate_diagnostics has not been run'
        i_label = self.labels.index(label)
        return self.flow_array[:, i_label]

    def load_state(self, i_time):

        """
        Returns the recorded compartment values at a particular point in time for the model.

        Args:
            i_time: Time from which the compartment values are to be loaded.

        Returns:
            state_compartments: The compartment values from that time in the model's integration.

        """

        state_compartments = {}
        for i_label, label in enumerate(self.labels):
            state_compartments[label] = self.soln_array[i_time, i_label]
        return state_compartments

    ###############################
    ### Flow diagram production ###
    ###############################

    def make_flow_diagram(self, png):

        from graphviz import Digraph

        styles = {
            'graph': {
                'label': 'Dynamic Transmission Model',
                'fontsize': '16',
            },
            'nodes': {
                'fontname': 'Helvetica',
                'shape': 'box',
                'style': 'filled',
                'fillcolor': '#CCDDFF',
            },
            'edges': {
                'style': 'dotted',
                'arrowhead': 'open',
                'fontname': 'Courier',
                'fontsize': '10',
            }
        }

        def apply_styles(graph, styles):
            graph.graph_attr.update(
                ('graph' in styles and styles['graph']) or {}
            )
            graph.node_attr.update(
                ('nodes' in styles and styles['nodes']) or {}
            )
            graph.edge_attr.update(
                ('edges' in styles and styles['edges']) or {}
            )
            return graph

        def num_str(f):
            abs_f = abs(f)
            if abs_f > 1E9:
                return '%.1fB' % (f/1E9)
            if abs_f > 1E6:
                return '%.1fM' % (f/1E6)
            if abs_f > 1E3:
                return '%.1fK' % (f/1E3)
            if abs_f > 100:
                return '%.0f' % f
            if abs_f > 0.5:
                return '%.1f' % f
            if abs_f > 0.05:
                return '%.2f' % f
            if abs_f > 0.0005:
                return '%.4f' % f
            if abs_f > 0.000005:
                return '%.6f' % f
            return str(f)

        self.graph = Digraph(format='png')
        for label in self.labels:
            self.graph.node(label)
        self.graph.node('tb_death')
        for from_label, to_label, var_label in self.var_transfer_rate_flows:
            self.graph.edge(from_label, to_label, label=var_label[:4])
        for from_label, to_label, rate in self.fixed_transfer_rate_flows:
            self.graph.edge(from_label, to_label, label=num_str(rate))
        for from_label, to_label, rate in self.linked_transfer_rate_flows:
            self.graph.edge(from_label, to_label, label='link')
        for label, rate in self.fixed_infection_death_rate_flows:
            self.graph.edge(label, 'tb_death', label=num_str(rate))
        for label, rate in self.var_infection_death_rate_flows:
            self.graph.edge(label, 'tb_death', label=var_label[:4])
        base, ext = os.path.splitext(png)
        if ext.lower() != '.png':
            base = png

        self.graph = apply_styles(self.graph, styles)

        self.graph.render(base)

    ####################################
    ### Intervention-related methods ###
    ####################################

    def check_list_of_interventions(self):

        """
        Find list of feasible interventions given model structure.

        """

        # If model is not age-structured, age-specific IPT does not make sense
        if len(self.agegroups) < 2:
            self.interventions_to_cost = [inter for inter in self.interventions_to_cost
                                          if inter not in ['ipt_age0to5', 'ipt_age5to15']]

    def determine_whether_startups_apply(self):

        """
        Determine whether an intervention is applied and has start-up costs in the scenario being run.

        """

        # Start assuming each costed intervention has no start-up costs in this scenario
        for program in self.interventions_to_cost:
            self.startups_apply[program] = False

            # If the program reaches values greater than zero and start-up costs are greater than zero, change to True
            if self.inputs.intervention_applied[self.scenario]['program_prop_' + program] \
                    and self.inputs.model_constants['econ_startupcost_' + program] > 0.:
                self.startups_apply[program] = True

    def find_intervention_startdates(self):

        """
        Find the dates when the different interventions start and populate self.intervention_startdates

        """

        scenario = self.scenario
        for intervention in self.interventions_to_cost:
            self.intervention_startdates[intervention] = None
            param_key = 'program_prop_' + intervention
            param_key2 = 'program_perc_' + intervention
            param_dict = self.inputs.scaleup_data[scenario][param_key]
            years_pos_coverage = [key for (key, value) in param_dict.items() if value > 0.]  # years after start
            if len(years_pos_coverage) > 0:  # some coverage present at baseline
                self.intervention_startdates[intervention] = min(years_pos_coverage)

    def distribute_funding_across_years(self):

        # Number of years to fund
        n_years = self.end_period_costing - self.inputs.model_constants['scenario_start_time']
        for int in self.interventions_to_cost:
            self.annual_available_funding[int] = 0
            # If intervention hasn't started
            if self.intervention_startdates[int] is None:
                if self.available_funding[int] < self.inputs.model_constants['econ_startupcost_' + int]:
                    print 'available_funding insufficient to cover starting costs of ' + int
                else:
                    self.intervention_startdates[int] = self.inputs.model_constants['scenario_start_time']
                    self.annual_available_funding[int] \
                        = (self.available_funding[int] - self.inputs.model_constants['econ_startupcost_' + int]) \
                          / n_years
            else:
                self.annual_available_funding[int] = (self.available_funding[int])/n_years


class StratifiedModel(BaseModel):

    def prepare_comorb_adjustments(self):

        """
        Find the target and actual proportion of the population in the risk groups/comorbidities being run in the model.

        """

        # Find the target proportions for each comorbidity stratum
        if len(self.comorbidities) > 1:
            for comorbidity in self.comorbidities:
                if comorbidity not in self.target_comorb_props:
                    self.target_comorb_props[comorbidity] = []
            self.target_comorb_props['_nocomorb'].append(1.)
            for comorbidity in self.comorbidities:
                if comorbidity != '_nocomorb':
                    self.target_comorb_props[comorbidity].append(
                        self.get_constant_or_variable_param('comorb_prop' + comorbidity))
                    self.target_comorb_props['_nocomorb'][-1] \
                        -= self.target_comorb_props[comorbidity][-1]
            # If integration has started properly
            if self.compartments:

                # Find the actual proportions in each comorbidity stratum
                population = sum(self.compartments.values())
                for comorbidity in self.comorbidities:
                    if comorbidity not in self.actual_comorb_props:
                        self.actual_comorb_props[comorbidity] = []
                    self.actual_comorb_props[comorbidity].append(0.)
                    for c in self.compartments:
                        if comorbidity in c:
                            self.actual_comorb_props[comorbidity][-1] += self.compartments[c] / population

                # Find the scaling factor for the risk group in question
                self.comorb_adjustment_factor = {}
                for comorbidity in self.comorbidities:
                    if self.actual_comorb_props[comorbidity][-1] > 0.:
                        self.comorb_adjustment_factor[comorbidity] = self.target_comorb_props[comorbidity][-1] \
                                                                     / self.actual_comorb_props[comorbidity][-1]
                    else:
                        self.comorb_adjustment_factor[comorbidity] = 1.
        else:
            # Otherwise, it's just a list of ones
            if '' not in self.target_comorb_props:
                self.target_comorb_props[''] = []
            self.target_comorb_props[''].append(1.)

    def adjust_compartment_size(self, y):

        """
        Adjusts the proportions of the population in each comorbidity group according to the calculations
        made in assess_comorbidity_props above.

        Args:
            y: The original compartment vector y to be adjusted.

        Returns:
            The adjusted compartment vector (y).

        """

        if hasattr(self, 'comorb_adjustment_factor'):
            compartments = self.convert_list_to_compartments(y)
            for c in compartments:
                for comorbidity in self.comorbidities:
                    if comorbidity in c:
                        compartments[c] *= self.comorb_adjustment_factor[comorbidity]

            return self.convert_compartments_to_list(compartments)
        else:
            return y



