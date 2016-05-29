import os

import numpy
from scipy.integrate import odeint

from autumn.curve import make_two_step_curve


class BaseModel():

    def __init__(self, age_breakpoints=[]):
        self.age_breakpoints = age_breakpoints

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

        self.flows = {}
        self.fixed_transfer_rate_flows = []
        self.fixed_infection_death_rate_flows = []
        self.var_transfer_rate_flows = []
        self.var_flows = []
        self.var_infection_death_rate_flows = []

    def define_age_structure(self):

        # Work out age-groups from list of breakpoints
        self.agegroups = []

        # If age-group breakpoints are supplied
        if len(self.age_breakpoints) > 0:
            for i in range(len(self.age_breakpoints)):
                if i == 0:

                    # The first age-group
                    self.agegroups +=\
                        ["_age0to" + str(self.age_breakpoints[i])]
                else:

                    # Middle age-groups
                    self.agegroups +=\
                        ["_age" + str(self.age_breakpoints[i - 1]) +
                         "to" + str(self.age_breakpoints[i])]

            # Last age-group
            self.agegroups += ["_age" + str(self.age_breakpoints[len(self.age_breakpoints) - 1]) + "up"]

        # Otherwise
        else:
            # List consisting of one empty string required
            # for the methods that iterate over strains
            self.agegroups += [""]

    def find_ageing_rates(self):

        # Calculate ageing rates as the reciprocal of the width of the age bracket
        for agegroup in self.agegroups:
            if "up" not in agegroup:
                self.set_parameter("ageing_rate" + agegroup, 1. / (
                    float(agegroup[agegroup.find("to") + 2:]) -
                    float(agegroup[agegroup.find("age") + 3: agegroup.find("to")])))

    def set_ageing_flows(self):

        # Set simple ageing flows for any number of strata
        for label in self.labels:
            for number_agegroup in range(len(self.agegroups)):
                if self.agegroups[number_agegroup] in label and number_agegroup < len(self.agegroups) - 1:
                    self.set_fixed_transfer_rate_flow(
                        label, label[0: label.find("_age")] + self.agegroups[number_agegroup + 1],
                               "ageing_rate" + self.agegroups[number_agegroup])

    def make_times(self, start, end, delta):
        "Return steps between start and end every delta"
        self.times = []
        step = start
        while step <= end:
            self.times.append(step)
            step += delta

    def make_times_with_n_step(self, start, end, n):
        "Return steps between start and in n increments"
        self.times = []
        step = start
        delta = (end - start) / float(n)
        while step <= end:
            self.times.append(step)
            step += delta

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

    def process_parameters(self):
        """
        To be overridden: calculates derived parameters from an initial set
        of raw parameters.
        """
        pass

    def get_init_list(self):
        return self.convert_compartments_to_list(self.init_compartments)

    def set_population_death_rate(self, death_label):
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

        # normal death flows
        self.vars["rate_death"] = 0.0
        for label in self.labels:
            val = self.compartments[label] * self.death_rate
            self.flows[label] -= val
            self.vars['rate_death'] += val

        # extra death flows
        self.vars["rate_infection_death"] = 0.0
        for label, rate in self.fixed_infection_death_rate_flows:
            val = self.compartments[label] * rate
            self.flows[label] -= val
            self.vars["rate_infection_death"] += val
        for label, rate in self.var_infection_death_rate_flows:
            val = self.compartments[label] * self.vars[vars_label]
            self.flows[label] -= val
            self.vars["rate_infection_death"] += val

    def make_derivate_fn(self):

        def derivative_fn(y, t):
            self.time = t
            self.compartments = self.convert_list_to_compartments(y)
            self.vars.clear()
            self.calculate_vars_of_scaleup_fns()
            self.calculate_vars()
            self.populate_output_arrays(t)
            self.calculate_flows()
            flow_vector = self.convert_compartments_to_list(self.flows)
            self.checks()
            return flow_vector

        return derivative_fn

    def populate_output_arrays(self, t):

        # This is James's dodgy code, that might not be in the right place.
        # However, it seems to function better than what was present before.
        if self.var_array_jt is None:
            self.var_array_jt = numpy.zeros((len(self.times), len(self.vars)))
            self.var_labels_jt = []
        for i_var, var in enumerate(self.vars):
            if var not in self.var_labels_jt:
                self.var_labels_jt += [var]
            for i in range(len(self.times)):
                if self.times[i] == t:
                    self.var_array_jt[i, i_var] = self.vars[var]

    def init_run(self):
        self.process_parameters()
        self.set_flows()
        self.var_labels = None
        self.soln_array = None
        self.var_array = None
        self.var_array_jt = None
        self.scaleup_array = None
        self.flow_array = None
        self.fraction_array = None
        assert not self.times is None, "Haven't set times yet"

    def integrate_scipy(self):

        self.init_run()
        init_y = self.get_init_list()
        derivative = self.make_derivate_fn()
        self.soln_array = odeint(derivative, init_y, self.times)
        self.calculate_diagnostics()

    def integrate_explicit(self, min_dt=0.05):

        self.init_run()
        y = self.get_init_list()
        n_compartment = len(y)
        n_time = len(self.times)
        self.soln_array = numpy.zeros((n_time, n_compartment))
        derivative = self.make_derivate_fn()
        time = self.times[0]
        self.soln_array[0, :] = y
        for i_time, new_time in enumerate(self.times):
            while time < new_time:
                f = derivative(y, time)
                old_time = time
                time = time + min_dt
                dt = min_dt
                if time > new_time:
                    dt = new_time - old_time
                    time = new_time
                for i in range(n_compartment):
                    y[i] = y[i] + dt * f[i]
            if i_time < n_time - 1:
                self.soln_array[i_time+1,:] = y

        self.calculate_diagnostics()

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

            self.calculate_vars()
            self.calculate_flows()
            self.calculate_output_vars()

            # only set after self.calculate_diagnostic_vars is
            # run so that we have all var_labels, including
            # the ones in calculate_diagnostic_vars
            if self.var_labels is None:
                self.var_labels = self.vars.keys()
                self.var_array = numpy.zeros((n_time, len(self.var_labels)))
                self.scaleup_array = numpy.zeros((n_time, len(self.scaleup_fns)))
                self.flow_array = numpy.zeros((n_time, len(self.labels)))

            for i_label, label in enumerate(self.var_labels):
                self.var_array[i, i_label] = self.vars[label]
            for i_label, label in enumerate(self.labels):
                self.flow_array[i, i_label] = self.flows[label]

            self.scaleup_list = []
            for i_label, label in enumerate(self.scaleup_fns):
                self.scaleup_array[i, i_label] = self.scaleup_fns[label](self.time)
                self.scaleup_list += [label]

        self.fraction_array = numpy.zeros((n_time, len(self.labels)))
        self.fraction_soln = {}
        for i_label, label in enumerate(self.labels):
            self.fraction_soln[label] = [
                v / t
                for v, t
                in zip(
                    self.compartment_soln[label],
                    self.get_var_soln("population"))]
            self.fraction_array[:, i_label] = self.fraction_soln[label]

        self.calculate_additional_diagnostics()

    def calculate_additional_diagnostics(self):
        pass

    def get_compartment_soln(self, label):
        assert self.soln_array is not None, "calculate_diagnostics has not been run"
        i_label = self.labels.index(label)
        return self.soln_array[:, i_label]

    def get_var_soln(self, label):
        assert self.var_array is not None, "calculate_diagnostics has not been run"
        i_label = self.var_labels.index(label)
        return self.var_array[:, i_label]

    def get_flow_soln(self, label):
        assert self.flow_array is not None, "calculate_diagnostics has not been run"
        i_label = self.labels.index(label)
        return self.flow_array[:, i_label]

    def load_state(self, i_time):
        self.time = self.times[i_time]
        for i_label, label in enumerate(self.labels):
            self.compartments[label] = \
                self.soln_array[i_time, i_label]
        self.calculate_vars()

    def checks(self, error_margin=0.1):
        """
        Assertion run during the simulation, should be overriden
        for each model.

        Args:
            error_margin: acceptable difference between target invariants

        Returns:

        """
        # Check all compartments are positive
        for label in self.labels:
            assert self.compartments[label] >= 0.0
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
                return "%.1fB" % (f/1E9)
            if abs_f > 1E6:
                return "%.1fM" % (f/1E6)
            if abs_f > 1E3:
                return "%.1fK" % (f/1E3)
            if abs_f > 100:
                return "%.0f" % f
            if abs_f > 0.5:
                return "%.1f" % f
            if abs_f > 0.05:
                return "%.2f" % f
            if abs_f > 0.0005:
                return "%.4f" % f
            if abs_f > 0.000005:
                return "%.6f" % f
            return str(f)

        self.graph = Digraph(format='png')
        for label in self.labels:
            self.graph.node(label)
        self.graph.node("tb_death")
        for from_label, to_label, var_label in self.var_transfer_rate_flows:
            self.graph.edge(from_label, to_label, label=var_label[:4])
        for from_label, to_label, rate in self.fixed_transfer_rate_flows:
            self.graph.edge(from_label, to_label, label=num_str(rate))
        for label, rate in self.fixed_infection_death_rate_flows:
            self.graph.edge(label, "tb_death", label=num_str(rate))
        for label, rate in self.var_infection_death_rate_flows:
            self.graph.edge(label, "tb_death", label=var_label[:4])
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

        self.set_compartment("susceptible", 1e6)
        self.set_compartment("susceptible_vac", 1e6)
        self.set_compartment("latent_early", 0.)
        self.set_compartment("latent_late", 0.)
        self.set_compartment("active", 1.)
        self.set_compartment("treatment_infect", 0.)
        self.set_compartment("treatment_noninfect", 0.)

        self.set_parameter("total_population", 1E6)
        self.set_parameter("demo_rate_birth", 20. / 1e3)
        self.set_parameter("demo_rate_death", 1. / 65)

        self.set_parameter("tb_n_contact", 20.)
        self.set_parameter("tb_rate_earlyprogress", .2)
        self.set_parameter("tb_rate_lateprogress", .0001)
        self.set_parameter("tb_rate_stabilise", 2.3)
        self.set_parameter("tb_rate_recover", .3)
        self.set_parameter("tb_rate_death", .07)

        self.set_parameter("tb_bcg_multiplier", .5)

        self.set_parameter("program_prop_vac", 0.4)
        self.set_parameter("program_prop_unvac",
                           1 - self.params["program_prop_vac"])
        self.set_parameter("program_rate_detect", 1.)
        self.set_parameter("program_time_treatment", .5)

    def process_parameters(self):
        prop = self.params["program_prop_vac"]
        self.set_compartment(
            "susceptible_vac",
            prop * self.params["total_population"])
        self.set_compartment(
            "susceptible",
            (1 - prop) * self.params["total_population"])
        time_treatment = self.params["program_time_treatment"]
        self.set_parameter("program_rate_completion_infect", .9 / time_treatment)
        self.set_parameter("program_rate_default_infect", .05 / time_treatment)
        self.set_parameter("program_rate_death_infect", .05 / time_treatment)
        self.set_parameter("program_rate_completion_noninfect", .9 / time_treatment)
        self.set_parameter("program_rate_default_noninfect", .05 / time_treatment)
        self.set_parameter("program_rate_death_noninfect", .05 / time_treatment)

        y = 4
        self.set_scaleup_fn(
            "program_rate_detect",
            make_two_step_curve(0, 0.5 * y, y, 1950, 1995, 2015))


    def set_flows(self):
        self.set_var_entry_rate_flow("susceptible", "births_unvac")
        self.set_var_entry_rate_flow("susceptible_vac", "births_vac")

        self.set_var_transfer_rate_flow(
            "susceptible", "latent_early", "rate_force")

        self.set_var_transfer_rate_flow(
            "susceptible_vac", "latent_early", "rate_force_weak")

        self.set_fixed_transfer_rate_flow(
            "latent_early", "active", "tb_rate_earlyprogress")
        self.set_fixed_transfer_rate_flow(
            "latent_early", "latent_late", "tb_rate_stabilise")

        self.set_fixed_transfer_rate_flow(
            "latent_late", "active", "tb_rate_lateprogress")
        self.set_var_transfer_rate_flow(
            "latent_late", "latent_early", "rate_force_weak")

        self.set_fixed_transfer_rate_flow(
            "active", "latent_late", "tb_rate_recover")

        y = self.params["program_rate_detect"]
        self.set_var_transfer_rate_flow(
            "active", "treatment_infect", "program_rate_detect")

        self.set_fixed_transfer_rate_flow(
            "treatment_infect", "treatment_noninfect", "program_rate_completion_infect")
        self.set_fixed_transfer_rate_flow(
            "treatment_infect", "active", "program_rate_default_infect")

        self.set_fixed_transfer_rate_flow(
            "treatment_noninfect", "susceptible_vac", "program_rate_completion_noninfect")
        self.set_fixed_transfer_rate_flow(
            "treatment_noninfect", "active", "program_rate_default_noninfect")

        self.set_population_death_rate("demo_rate_death")
        self.set_infection_death_rate_flow(
            "active", "tb_rate_death")
        self.set_infection_death_rate_flow(
            "treatment_infect", "program_rate_death_infect")
        self.set_infection_death_rate_flow(
            "treatment_noninfect", "program_rate_death_noninfect")

    def calculate_vars(self):

        self.vars["population"] = sum(self.compartments.values())

        self.vars["rate_birth"] = \
            self.params["demo_rate_birth"] * self.vars["population"]
        self.vars["births_unvac"] = \
            self.params["program_prop_unvac"] * self.vars["rate_birth"]
        self.vars["births_vac"] = \
            self.params["program_prop_vac"] * self.vars["rate_birth"]

        self.vars["infectious_population"] = 0.0
        for label in self.labels:
            if label in ["active", "treatment_infect"]:
                self.vars["infectious_population"] += \
                    self.compartments[label]
        self.vars["rate_force"] = \
            self.params["tb_n_contact"] \
              * self.vars["infectious_population"] \
              / self.vars["population"]
        self.vars["rate_force_weak"] = \
            self.params["tb_bcg_multiplier"] \
              * self.vars["rate_force"]

    def calculate_output_vars(self):

        rate_incidence = 0.
        rate_mortality = 0.
        rate_notifications = 0.
        for from_label, to_label, rate in self.fixed_transfer_rate_flows:
            if 'latent' in from_label and 'active' in to_label:
                rate_incidence += self.compartments[from_label] * rate
        self.vars["incidence"] = \
            rate_incidence \
            / self.vars["population"] * 1E5
        for from_label, to_label, rate in self.var_transfer_rate_flows:
            if 'active' in from_label and\
                    ('detect' in to_label or 'treatment_infect' in to_label):
                rate_notifications += self.compartments[from_label] * self.vars[rate]
        self.vars["notifications"] = \
            rate_notifications / self.vars["population"] * 1E5
        for from_label, rate in self.infection_death_rate_flows:
            rate_mortality \
                += self.compartments[from_label] * rate
        self.vars["mortality"] = \
            rate_mortality \
            / self.vars["population"] * 1E5

        self.vars["prevalence"] = 0.0
        for label in self.labels:
            if "susceptible" not in label and "latent" not in label:
                self.vars["prevalence"] += (
                    self.compartments[label]
                     / self.vars["population"] * 1E5)