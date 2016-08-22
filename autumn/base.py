import os

import numpy
import scipy
from scipy.integrate import odeint
from tool_kit import indices

from autumn.curve import make_two_step_curve

class BaseModel:

    def __init__(self):

        self.labels = []
        self.init_compartments = {}
        self.params = {}
        self.times = None

        self.scaleup_fns = {}
        self.vars = {}

        self.soln_array = None
        self.var_labels = None
        self.var_array = None
        self.flow_array = None
        self.fraction_array = None

        self.is_additional_diagnostics = False

        self.flows = {}
        self.fixed_transfer_rate_flows = []
        self.linked_transfer_rate_flows = []
        self.fixed_infection_death_rate_flows = []
        self.var_transfer_rate_flows = []
        self.var_flows = []
        self.var_infection_death_rate_flows = []

        self.costs = {}
        self.run_costing = True
        self.end_period_costing = 2035
        self.interventions_to_cost = ['vaccination', 'xpert', 'treatment_support', 'smearacf', 'xpertacf']

        self.eco_drives_epi = True

    def make_times(self, start, end, delta):

        "Return steps between start and end every delta"

        self.times = []
        step = start
        while step <= end:
            self.times.append(step)
            step += delta
        if self.times[-1] < end:
            self.times.append(end)

    def make_times_with_n_step(self, start, end, n):
        "Return steps between start and in n increments"
        self.times = []
        step = start
        delta = (end - start) / float(n)
        while step <= end:
            self.times.append(step)
            step += delta
        if self.times[-1] < end:
            self.times.append(end)

    def find_time_index(self, time):

        for index, model_time in enumerate(self.times):
            if model_time > time:
                return index

        raise ValueError('Time not found')

    def set_compartment(self, label, init_val=0.0):
        if label not in self.labels:
            self.labels.append(label)
        self.init_compartments[label] = init_val
        assert init_val >= 0, 'Start with negative compartment not permitted'

    def set_parameter(self, label, val):
        self.params[label] = val

    def convert_list_to_compartments(self, vec):
        return {l: vec[i] for i, l in enumerate(self.labels)}

    def convert_compartments_to_list(self, compartments):
        return [compartments[l] for l in self.labels]

    def get_init_list(self):

        if self.loaded_compartments is None:
            return self.convert_compartments_to_list(self.init_compartments)
        else:
            return self.convert_compartments_to_list(self.loaded_compartments)

    def set_population_death_rate(self, death_label):

        # Currently inactive (although Bosco might not be pleased about this)
        self.death_rate = self.params[death_label]

    def set_fixed_infection_death_rate_flow(self, label, param_label):
        add_unique_tuple_to_list(
            self.fixed_infection_death_rate_flows,
            (label, self.params[param_label]))

    def set_var_infection_death_rate_flow(self, label, vars_label):
        add_unique_tuple_to_list(
            self.var_infection_death_rate_flows,
            (label, vars_label))

    def set_fixed_transfer_rate_flow(self, from_label, to_label, param_label):
        add_unique_tuple_to_list(
            self.fixed_transfer_rate_flows,
            (from_label, to_label, self.params[param_label]))

    def set_linked_transfer_rate_flow(self, from_label, to_label, vars_label):
        add_unique_tuple_to_list(
            self.linked_transfer_rate_flows,
            (from_label, to_label, vars_label))

    def set_var_transfer_rate_flow(self, from_label, to_label, vars_label):
        add_unique_tuple_to_list(
            self.var_transfer_rate_flows,
            (from_label, to_label, vars_label))

    def set_scaleup_fn(self, label, fn):
        self.scaleup_fns[label] = fn

    def set_var_entry_rate_flow(self, label, vars_label):
        add_unique_tuple_to_list(
            self.var_flows,
            (label, vars_label))

    def calculate_vars_of_scaleup_fns(self):
        for label, fn in self.scaleup_fns.iteritems():
            self.vars[label] = fn(self.time)

    def calculate_vars(self):
        """
        Calculate self.vars that only depend on compartment values
        """
        pass

    def calculate_flows(self):
        """
        Calculate flows, which should only depend on compartment values
        and self.vars calculated in calculate_variable_rates.
        """
        for label in self.labels:
            self.flows[label] = 0.0

        # birth flows
        for label, vars_label in self.var_flows:
            self.flows[label] += self.vars[vars_label]

        # dynamic transmission flows
        for from_label, to_label, vars_label in self.var_transfer_rate_flows:
            val = self.compartments[from_label] * self.vars[vars_label]
            self.flows[from_label] -= val
            self.flows[to_label] += val

        # fixed-rate flows
        for from_label, to_label, rate in self.fixed_transfer_rate_flows:
            val = self.compartments[from_label] * rate
            self.flows[from_label] -= val
            self.flows[to_label] += val

        # linked flows
        for from_label, to_label, vars_label in self.linked_transfer_rate_flows:
            val = self.vars[vars_label]
            self.flows[from_label] -= val
            self.flows[to_label] += val

        # normal death flows
        # This might be naughty - but now changed to access one of the parameters
        # (which has to have this name). Saves on creating a separate model attribute
        # just for population death. I think it makes more sense for it to be just
        # another parameter.
        # Now works if the death rate is selected to be constant or time-variant
        self.vars['rate_death'] = 0.
        for label in self.labels:
            if self.inputs.time_variants['demo_life_expectancy'][u'time_variant'] == u'no':
                val = self.compartments[label] / self.params['demo_life_expectancy']
            elif self.inputs.time_variants['demo_life_expectancy'][u'time_variant'] == u'yes':
                val = self.compartments[label] / self.vars['demo_life_expectancy']
            self.flows[label] -= val
            self.vars['rate_death'] += val

        # extra death flows
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

        # This function collects some other functions that
        # previously led to a bug because not all of them
        # were called in the diagnostics round.
        # (Written by James, not Bosco)

        # Before clearing vars, we need to save the ones that are population sizes as its needed for the economics
        saved_vars = {}
        if self.eco_drives_epi:
            for key in self.vars.keys():
                if 'popsize' in key:
                    saved_vars[key] = self.vars[key]

        self.vars.clear() # clear all the vars
        self.vars = saved_vars # re-populate the saved vars
        self.calculate_vars_of_scaleup_fns()
        self.calculate_vars()
        self.calculate_flows()

    def make_derivate_fn(self):

        def derivative_fn(y, t):
            self.time = t
            self.compartments = self.convert_list_to_compartments(y)
            self.prepare_vars_flows()
            flow_vector = self.convert_compartments_to_list(self.flows)
            self.checks()
            return flow_vector

        return derivative_fn

    def init_run(self):

        if self.scenario == None:
            start_time = self.inputs.model_constants['start_time']
        else:
            start_time = self.inputs.model_constants['scenario_start_time']
        self.make_times(start_time,
                        self.inputs.model_constants['scenario_end_time'],
                        self.inputs.model_constants['time_step'])
        self.initialise_compartments()
        self.set_flows()
        self.var_labels = None
        self.soln_array = None
        self.var_array = None
        self.flow_array = None
        self.fraction_array = None
        assert not self.times is None, 'Times have not been set yet'

    def integrate_scipy(self, dt_max=0.05):
        """ Uses Adams method coded in the LSODA Fortran package. This method is programmed to "slow down" when a tricky
        point is encountered. Then we need to allow for a high maximal number of iterations (mxstep)so that the
        algorithm does not get stuck.
        Input:
            min_dt: represents the time step for calculation points. The attribute self.times will also be used to make sure
            that a solution is affected to the time points known by the model
        """
        self.init_run()
        init_y = self.get_init_list()
        derivative = self.make_derivate_fn()

        tt = [] # all the calculation times
        tt_record = [] # store the indices of tt corresponding to the calculation times to be stored

        time = self.times[0]
        tt.append(time)
        tt_record.append(0)
        i_tt = 0
        for i_time, new_time in enumerate(self.times):
            while time < new_time:
                time = time + dt_max
                if time > new_time:
                    time = new_time
                i_tt += 1
                tt.append(time)
                if time == new_time:
                    tt_record.append(i_tt)

        sol = odeint(derivative, init_y, tt, mxstep=5000000)
        self.soln_array = sol[tt_record, :]

        self.calculate_diagnostics()
        if self.run_costing:
            self.calculate_economics_diagnostics(self.end_period_costing)

    def integrate_explicit(self, dt_max=0.05):

        """ Uses Euler Explicit method.
            Input:
            min_dt: represents the time step for calculation points. The attribute self.times will also be used to make sure
            that a solution is affected to the time points known by the model
        """
        self.init_run()
        y = self.get_init_list()
        y_candidate = numpy.zeros((len(y)))
        n_compartment = len(y)
        n_time = len(self.times)
        self.soln_array = numpy.zeros((n_time, n_compartment))

        derivative = self.make_derivate_fn()
        old_time = self.times[0]
        time = old_time
        self.soln_array[0, :] = y
        dt_is_ok = True
        for i_time, new_time in enumerate(self.times):
            while time < new_time:
                if not dt_is_ok:
                    adaptive_dt_max = dt / 2.0
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

        self.calculate_diagnostics()
        if self.run_costing:
            self.calculate_economics_diagnostics(self.end_period_costing)

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

        derivative = self.make_derivate_fn()
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
                #old_time = time
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

                for i in range(n_compartment):
                    y[i] = y[i] + (dt/6.0) * (k1[i] + 2.0*k2[i] + 2.0*k3[i] + k4[i])


            if i_time < n_time - 1:
                self.soln_array[i_time + 1, :] = y

        self.calculate_diagnostics()
        if self.run_costing:
            self.calculate_economics_diagnostics()

    def calculate_output_vars(self):
        """
        Calculate diagnostic vars that can depend on self.flows as
        well as self.vars calculated in calculate_vars
        """
        pass

    def calculate_diagnostics(self):

        self.compartment_soln = {}
        for label in self.labels:
            if label in self.compartment_soln:
                continue
            self.compartment_soln[label] = self.get_compartment_soln(label)

        n_time = len(self.times)
        for i in range(n_time):

            self.time = self.times[i]

            for label in self.labels:
                self.compartments[label] = self.compartment_soln[label][i]

            self.prepare_vars_flows()
            self.calculate_output_vars()

            # only set after self.calculate_diagnostic_vars is
            # run so that we have all var_labels, including
            # the ones in calculate_diagnostic_vars
            if self.var_labels is None:
                self.var_labels = self.vars.keys()
                self.var_array = numpy.zeros((n_time, len(self.var_labels)))
                self.flow_array = numpy.zeros((n_time, len(self.labels)))

            for i_label, label in enumerate(self.var_labels):
                self.var_array[i, i_label] = self.vars[label]
            for i_label, label in enumerate(self.labels):
                self.flow_array[i, i_label] = self.flows[label]

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

    def calculate_additional_diagnostics(self):
        pass

    def coverage_over_time(self, param_key):
        """
        Define a function which returns the coverage over time associated with an intervention
        Args:
            model: model object, after integration
            param_key: the key of the parameter associated with the intervention

        Returns:
            a function which takes a time for argument an will return a coverage
        """
        coverage_function = self.scaleup_fns[param_key]
        return coverage_function

    def calculate_economics_diagnostics(self):
        """
        Run the economics diagnostics associated with a model run. Integration is supposed to have been run at this point
        Args:
            self the model object
        Returns:
            nothing
        """

        def get_cost_from_coverage(coverage, c_inflection_cost, saturation, unit_cost, pop_size, alpha=1.0):
            """
            Estimate the global uninflated cost associated with a given coverage
            Args:
                coverage: the coverage (as a proportion, then lives in 0-1)
                c_inflection_cost: cost at which inflection occurs on the curve. It's also the configuration leading to the
                                    best efficiency.
                saturation: maximal acceptable coverage, ie upper asymptote
                unit_cost: unit cost of the intervention
                pop_size: size of the population targeted by the intervention
                alpha: steepness parameter

            Returns:
                uninflated cost

            """
            if pop_size * unit_cost == 0:  # if unit cost or pop_size is null, return 0
                return 0
            assert 0 <= coverage < saturation, 'coverage must verify 0 <= coverage < saturation'

            a = saturation / (1.0 - 2 ** alpha)
            b = ((2.0 ** (alpha + 1.0)) / (alpha * (saturation - a) * unit_cost * pop_size))
            cost_uninflated = c_inflection_cost - 1.0 / b * scipy.log(
                (((saturation - a) / (coverage - a)) ** (1.0 / alpha)) - 1.0)
            return cost_uninflated

        def inflate_cost(cost_uninflated, current_cpi, cpi_time_variant):
            """
            Calculate the inflated cost associated with cost_uninflated and considering the current cpi and the cpi correponding
            to the date considered (cpi_time_variant)

            Returns:
                the inflated cost
            """
            return cost_uninflated * current_cpi / cpi_time_variant

        def discount_cost(cost_uninflated, discount_rate, t_into_future):
            """
            Calculate the discounted cost associated with cost_uninflated at time (t + t_into_future)
            Args:
                cost_uninflated: cost without accounting for discounting
                discount_rate: discount rate (/year)
                t_into_future: number of years into future at which we want to calculate the discounted cost

            Returns:
                the discounted cost
            """
            assert t_into_future >= 0, 't_into_future must be >= 0'
            return (cost_uninflated / ((1 + discount_rate) ** t_into_future))

        start_time = self.inputs.model_constants['recent_time']  # start time for cost calculations
        start_index = indices(self.times, lambda x: x >= start_time)[0]

        end_time_integration = self.inputs.model_constants['scenario_end_time']
        assert self.end_period_costing <= end_time_integration, 'period_end must be <= end_time_integration'
        end_index = indices(self.times, lambda x: x >= self.end_period_costing)[0]

        # prepare the references to fetch the data and model outputs
        param_key_base = 'econ_program_prop_'
        c_inflection_cost_base = 'econ_program_inflectioncost_'
        unitcost_base = 'econ_program_unitcost_'
        popsize_label_base = 'popsize_'

        discount_rate = 0.03  # not ideal... perhaps best to get it from a spreadsheet
        cpi_function = self.scaleup_fns['econ_cpi']
        year_current = self.inputs.model_constants['current_time']
        current_cpi = cpi_function(year_current)

        # prepare the storage. 'costs' will store all the costs and will be returned
        costs = {'cost_times': []}

        count_intervention = 0  # to count the interventions
        for intervention in self.interventions_to_cost:  # for each intervention
            count_intervention += 1
            costs[intervention] = {'uninflated_cost': [], 'inflated_cost': [], 'discounted_cost': []}

            param_key = param_key_base + intervention  # name of the corresponding parameter
            coverage_function = self.coverage_over_time(param_key)  # create a function to calculate coverage over time

            c_inflection_cost_label = c_inflection_cost_base + intervention  # key of the scale up function for inflection cost
            c_inflection_cost_function = self.scaleup_fns[c_inflection_cost_label]

            unitcost_label = unitcost_base + intervention  # key of the scale up function for unit cost
            unit_cost_function = self.scaleup_fns[unitcost_label]

            popsize_label = popsize_label_base + intervention
            pop_size_index = self.var_labels.index(
                popsize_label)  # column index in model.var_array that corresponds to the intervention

            saturation = 1.001  # provisional

            for i in range(start_index,
                           end_index + 1):  # for each step time. We may want to change this bit. No need for all time steps
                t = self.times[i]
                if count_intervention == 1:
                    costs['cost_times'].append(t)  # storage of the time
                # calculate the time variants that feed into the logistic function
                coverage = coverage_function(t)
                c_inflection_cost = c_inflection_cost_function(t)
                unit_cost = unit_cost_function(t)
                pop_size = self.var_array[i, pop_size_index]

                # calculate uninflated cost
                cost = get_cost_from_coverage(coverage, c_inflection_cost, saturation, unit_cost, pop_size)
                costs[intervention]['uninflated_cost'].append(cost)  # storage

                # calculate inflated cost
                cpi_time_variant = cpi_function(t)
                inflated_cost = inflate_cost(cost, current_cpi, cpi_time_variant)
                costs[intervention]['inflated_cost'].append(inflated_cost)  # storage

                # calculate discounted cost
                t_into_future = max(0, (t - year_current))
                discounted_cost = discount_cost(cost, discount_rate, t_into_future)
                costs[intervention]['discounted_cost'].append(discounted_cost)  # storage

        self.costs = costs

    def update_vars_from_cost(self):
        """
        update parameter values according to the funding allocated to each interventions. This process is done during
        integration
        Returns:
        Nothing
        """
        def get_coverage_from_cost(cost, c_inflection_cost, saturation, unit_cost, pop_size, alpha=1.0):
            """
            Estimate the coverage associated with a spending in a programme
            Args:
               cost: the amount of money allocated to a programme (absolute number, not a proportion of global funding)
               c_inflection_cost: cost at which inflection occurs on the curve. It's also the configuration leading to the
                                   best efficiency.
               saturation: maximal acceptable coverage, ie upper asymptote
               unit_cost: unit cost of the intervention
               pop_size: size of the population targeted by the intervention
               alpha: steepness parameter

            Returns:
               coverage (as a proportion, then lives in 0-1)
           """
            assert cost >= 0, 'cost must be positive or null'
            if cost <= c_inflection_cost:  # if cost is smaller thar c_inflection_cost, then the starting cost necessary to get coverage has not been reached
                return 0

            if pop_size * unit_cost == 0:  # if unit cost or pop_size is null, return 0
                return 0

            a = saturation / (1.0 - 2 ** alpha)
            b = ((2.0 ** (alpha + 1.0)) / (alpha * (saturation - a) * unit_cost * pop_size))
            coverage_estimated = a + (saturation - a) / (
                (1 + numpy.exp((-b) * (cost - c_inflection_cost))) ** alpha)
            return coverage_estimated

        interventions = ['vaccination', 'treatment_support']  # provisional
        interventions = self.interventions_to_cost  # provisional

        vars_key_base = 'program_prop_'
        popsize_label_base = 'popsize_'
        c_inflection_cost_base = 'econ_program_inflectioncost_'
        unitcost_base = 'econ_program_unitcost_'
        cost_base = 'econ_program_totalcost_'

        for int in interventions:
            vars_key = vars_key_base + int

            cost = self.vars[cost_base + int]  # dummy   . Should be obtained from scale_up functions
            unit_cost = self.vars[unitcost_base + int]
            c_inflection_cost = self.vars[c_inflection_cost_base + int]
            saturation = 0.9  # dummy   provisional

            popsize_key = popsize_label_base + int
            if popsize_key in self.vars.keys():
                pop_size = self.vars[popsize_key]
            else:
                pop_size = 0

            coverage = get_coverage_from_cost(cost, c_inflection_cost, saturation, unit_cost, pop_size, alpha=1.0)
            self.vars[vars_key] = coverage

    def get_compartment_soln(self, label):
        assert self.soln_array is not None, 'calculate_diagnostics has not been run'
        i_label = self.labels.index(label)
        return self.soln_array[:, i_label]

    def get_var_soln(self, label):
        assert self.var_array is not None, 'calculate_diagnostics has not been run'
        i_label = self.var_labels.index(label)
        return self.var_array[:, i_label]

    def get_flow_soln(self, label):
        assert self.flow_array is not None, 'calculate_diagnostics has not been run'
        i_label = self.labels.index(label)
        return self.flow_array[:, i_label]

    def load_state(self, i_time):

        self.time = self.times[i_time]
        for i_label, label in enumerate(self.labels):
            self.compartments[label] = \
                self.soln_array[i_time, i_label]
        self.calculate_vars()

        return self.compartments

    def checks(self, error_margin=0.1):

        """
        Assertion run during the simulation, should be overridden
        for each model.

        Args:
            error_margin: acceptable difference between target invariants

        Returns:

        """

        # Check all compartments are positive
        for label in self.labels:
            assert self.compartments[label] >= 0.
        # Check population is conserved across compartments
        # population_change = \
        #       self.vars['births_vac'] \
        #     - self.vars['births_unvac'] \
        #     - self.vars['rate_death'] \
        #     - self.vars['rate_infection_death']
        # assert abs(sum(self.flows.values()) - population_change ) < error_margin

    def make_graph(self, png):
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


class SimpleModel(BaseModel):

    """
    Initial Autumn model designed by James
    """

    def __init__(self):

        BaseModel.__init__(self)

        self.set_compartment('susceptible', 1e6)
        self.set_compartment('susceptible_vac', 1e6)
        self.set_compartment('latent_early', 0.)
        self.set_compartment('latent_late', 0.)
        self.set_compartment('active', 1.)
        self.set_compartment('treatment_infect', 0.)
        self.set_compartment('treatment_noninfect', 0.)

        self.set_parameter('total_population', 1E6)
        self.set_parameter('demo_rate_birth', 20. / 1e3)
        self.set_parameter('demo_rate_death', 1. / 65)

        self.set_parameter('tb_n_contact', 20.)
        self.set_parameter('tb_rate_earlyprogress', .2)
        self.set_parameter('tb_rate_lateprogress', .0001)
        self.set_parameter('tb_rate_stabilise', 2.3)
        self.set_parameter('tb_rate_recover', .3)
        self.set_parameter('tb_rate_death', .07)

        self.set_parameter('tb_bcg_multiplier', .5)

        self.set_parameter('program_prop_vac', 0.4)
        self.set_parameter('program_prop_unvac',
                           1 - self.params['program_prop_vac'])
        self.set_parameter('program_rate_detect', 1.)
        self.set_parameter('program_time_treatment', .5)

    def process_parameters(self):
        prop = self.params['program_prop_vac']
        self.set_compartment(
            'susceptible_vac',
            prop * self.params['total_population'])
        self.set_compartment(
            'susceptible',
            (1 - prop) * self.params['total_population'])
        time_treatment = self.params['program_time_treatment']
        self.set_parameter('program_rate_completion_infect', .9 / time_treatment)
        self.set_parameter('program_rate_default_infect', .05 / time_treatment)
        self.set_parameter('program_rate_death_infect', .05 / time_treatment)
        self.set_parameter('program_rate_completion_noninfect', .9 / time_treatment)
        self.set_parameter('program_rate_default_noninfect', .05 / time_treatment)
        self.set_parameter('program_rate_death_noninfect', .05 / time_treatment)

        y = 4
        self.set_scaleup_fn(
            'program_rate_detect',
            make_two_step_curve(0, 0.5 * y, y, 1950, 1995, 2015))


    def set_flows(self):
        self.set_var_entry_rate_flow('susceptible', 'births_unvac')
        self.set_var_entry_rate_flow('susceptible_vac', 'births_vac')

        self.set_var_transfer_rate_flow(
            'susceptible', 'latent_early', 'rate_force')

        self.set_var_transfer_rate_flow(
            'susceptible_vac', 'latent_early', 'rate_force_weak')

        self.set_fixed_transfer_rate_flow(
            'latent_early', 'active', 'tb_rate_earlyprogress')
        self.set_fixed_transfer_rate_flow(
            'latent_early', 'latent_late', 'tb_rate_stabilise')

        self.set_fixed_transfer_rate_flow(
            'latent_late', 'active', 'tb_rate_lateprogress')
        self.set_var_transfer_rate_flow(
            'latent_late', 'latent_early', 'rate_force_weak')

        self.set_fixed_transfer_rate_flow(
            'active', 'latent_late', 'tb_rate_recover')

        y = self.params['program_rate_detect']
        self.set_var_transfer_rate_flow(
            'active', 'treatment_infect', 'program_rate_detect')

        self.set_fixed_transfer_rate_flow(
            'treatment_infect', 'treatment_noninfect', 'program_rate_completion_infect')
        self.set_fixed_transfer_rate_flow(
            'treatment_infect', 'active', 'program_rate_default_infect')

        self.set_fixed_transfer_rate_flow(
            'treatment_noninfect', 'susceptible_vac', 'program_rate_completion_noninfect')
        self.set_fixed_transfer_rate_flow(
            'treatment_noninfect', 'active', 'program_rate_default_noninfect')

        self.set_population_death_rate('demo_rate_death')
        self.set_infection_death_rate_flow(
            'active', 'tb_rate_death')
        self.set_infection_death_rate_flow(
            'treatment_infect', 'program_rate_death_infect')
        self.set_infection_death_rate_flow(
            'treatment_noninfect', 'program_rate_death_noninfect')

    def calculate_vars(self):

        self.vars['population'] = sum(self.compartments.values())

        self.vars['rate_birth'] = \
            self.params['demo_rate_birth'] * self.vars['population']
        self.vars['births_unvac'] = \
            self.params['program_prop_unvac'] * self.vars['rate_birth']
        self.vars['births_vac'] = \
            self.params['program_prop_vac'] * self.vars['rate_birth']

        self.vars['infectious_population'] = 0.0
        for label in self.labels:
            if label in ['active', 'treatment_infect']:
                self.vars['infectious_population'] += \
                    self.compartments[label]
        self.vars['rate_force'] = \
            self.params['tb_n_contact'] \
              * self.vars['infectious_population'] \
              / self.vars['population']
        self.vars['rate_force_weak'] = \
            self.params['tb_bcg_multiplier'] \
              * self.vars['rate_force']

    def calculate_output_vars(self):

        rate_incidence = 0.
        rate_mortality = 0.
        rate_notifications = 0.
        for from_label, to_label, rate in self.fixed_transfer_rate_flows:
            if 'latent' in from_label and 'active' in to_label:
                rate_incidence += self.compartments[from_label] * rate
        self.vars['incidence'] = \
            rate_incidence \
            / self.vars['population'] * 1E5
        for from_label, to_label, rate in self.var_transfer_rate_flows:
            if 'active' in from_label and\
                    ('detect' in to_label or 'treatment_infect' in to_label):
                rate_notifications += self.compartments[from_label] * self.vars[rate]
        self.vars['notifications'] = \
            rate_notifications / self.vars['population'] * 1E5
        for from_label, rate in self.infection_death_rate_flows:
            rate_mortality \
                += self.compartments[from_label] * rate
        self.vars['mortality'] = \
            rate_mortality \
            / self.vars['population'] * 1E5

        self.vars['prevalence'] = 0.0
        for label in self.labels:
            if 'susceptible' not in label and 'latent' not in label:
                self.vars['prevalence'] += (
                    self.compartments[label]
                     / self.vars['population'] * 1E5)