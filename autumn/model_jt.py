# -*- coding: utf-8 -*-


"""

Base Population Model to handle different type of models.

Implicit time unit: years

"""

import os
from scipy.integrate import odeint
from scipy import exp, log
import numpy


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


def find_element_in_list(element, list_element):
    try:
        index_element = list_element.index(element)
        return index_element
    except ValueError:
        return -1


class BasePopulationSystem():

    def __init__(self):
        self.labels = []
        self.init_compartments = {}
        self.flows = {}
        self.vars = {}
        self.vars_array = None
        self.flows_array = None
        self.params = {}
        self.fixed_transfer_rate_flows = []
        self.infection_death_rate_flows = []
        self.var_transfer_rate_flows = []
        self.var_flows = []
        self.times = None

    def make_steps(self, start, end, delta):
        "Return steps with n or delta"
        self.steps = []
        step = start
        while step <= end:
            self.steps.append(step)
            step += delta
        self.times = self.steps

    def make_n_steps(self, start, end, n):
        "Return steps with n or delta"
        self.steps = []
        step = start
        delta = (end - start) / float(n)
        while step <= end:
            self.steps.append(step)
            step += delta
        self.times = self.steps

    def set_compartment(self, label, init_val=0.0):
        if label not in self.labels:
            self.labels.append(label)
        self.init_compartments[label] = init_val
        assert init_val >= 0, 'Start with negative compartment not permitted'

    def set_param(self, label, val):
        self.params[label] = val

    def convert_list_to_compartments(self, vec):
        return {l: vec[i] for i, l in enumerate(self.labels)}

    def convert_compartments_to_list(self, compartments):
        return [compartments[l] for l in self.labels]

    def get_init_list(self):
        return self.convert_compartments_to_list(self.init_compartments)

    def set_population_death_rate(self, death_label):
        self.death_rate = self.params[death_label]

    def set_infection_death_rate_flow(self, label, param_label):
        add_unique_tuple_to_list(
            self.infection_death_rate_flows,
            (label, self.params[param_label]))

    def set_fixed_transfer_rate_flow(self, from_label, to_label, param_label):
        add_unique_tuple_to_list(
            self.fixed_transfer_rate_flows,
            (from_label, to_label, self.params[param_label]))

    def set_var_transfer_rate_flow(self, from_label, to_label, vars_label):
        add_unique_tuple_to_list(
            self.var_transfer_rate_flows,
            (from_label, to_label, vars_label))

    def set_var_entry_rate_flow(self, label, vars_label):
        add_unique_tuple_to_list(
            self.var_flows,
            (label, vars_label))

    def calculate_flows(self):
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
        self.vars["rate_incidence"] = 0.0
        for from_label, to_label, rate in self.fixed_transfer_rate_flows:
            val = self.compartments[from_label] * rate
            self.flows[from_label] -= val
            self.flows[to_label] += val
            if 'latent' in from_label and 'active' in to_label:
                self.vars["rate_incidence"] += val

        # normal death flows
        self.vars["rate_death"] = 0.0

        for label in self.labels:
            val = self.compartments[label] * self.death_rate
            self.flows[label] -= val
            self.vars['rate_death'] += val

        # extra death flows
        self.vars["rate_infection_death"] = 0.0
        for label, rate in self.infection_death_rate_flows:
            val = self.compartments[label] * rate
            self.flows[label] -= val
            self.vars["rate_infection_death"] += val

    def calculate_vars(self):
        pass

    def make_derivate_fn(self):

        def derivative_fn(y, t):
            self.time = t
            self.compartments = self.convert_list_to_compartments(y)
            self.vars.clear()
            self.calculate_vars()
            self.calculate_flows()
            flow_vector = self.convert_compartments_to_list(self.flows)
            self.checks()
            return flow_vector

        return derivative_fn

    def integrate_scipy(self):
        self.set_flows()
        assert not self.times is None
        init_y = self.get_init_list()
        derivative = self.make_derivate_fn()
        self.soln_array = odeint(derivative, init_y, self.times)
        self.calculate_fractions()
        self.calculate_fractions_jt()

    def integrate_explicit(self, min_dt=0.05):
        self.set_flows()
        assert not self.times is None
        y = self.get_init_list()
        n_component = len(y)
        n_time = len(self.times)
        self.soln_array = numpy.zeros((n_time, n_component))

        derivative = self.make_derivate_fn()
        time = self.times[0]
        self.soln_array[0,:] = y
        for i_time, new_time in enumerate(self.times):
            while time < new_time:
                f = derivative(y, time)
                old_time = time
                time = time + min_dt
                dt = min_dt
                if time > new_time:
                    dt = new_time - old_time
                    time = new_time
                for i in range(n_component):
                    y[i] = y[i] + dt * f[i]
            if i_time < n_time - 1:
                self.soln_array[i_time+1,:] = y
        self.calculate_fractions()
        self.restrict_labels()

    def restrict_labels(self):
        self.inclusions = {
            "ever_infected": ["suscptible_treated", "latent", "active", "missed", "detect", "treatment"],
            "infected": ["latent", "active", "missed", "detect", "treatment"],
            "active": ["active", "missed", "detect", "treatment"],
            "infectious": ["active", "missed", "detect", "treatment_infect"],
            "identified": ["detect", "treatment"],
            "treatment": ["treatment"]
        }

        for i in self.inclusions:
            compartments_included = []
            compartments_excluded = []
            for label in self.labels:
                for working_inclusion in self.inclusions[i]:
                    if working_inclusion in label:
                        compartments_included.append(label)
            setattr(self, "labels_in_" + i, compartments_included)
            for label in self.labels:
                if label not in getattr(self, "labels_in_" + i):
                    compartments_excluded.append(label)
            setattr(self, "labels_not_in_" + i, compartments_excluded)

        self.calculate_fractions_jt()

    def calculate_fractions_jt(self):
        for restriction in self.inclusions:
            working_population = []
            n = len(self.times)
            for i in range(n):
                t = 0.0
                for label in getattr(self, "labels_in_" + restriction):
                    t += self.populations[label][i]
                working_population.append(t)
            setattr(self, "population_in_" + restriction, working_population)
            fractions = {}
            for label in getattr(self, "labels_in_" + restriction):
                fractions[label] = [
                    compartment / population
                    for compartment, population in
                    zip(
                        self.populations[label],
                        working_population
                    )
                ]
            setattr(self, "fractions_in_" + restriction, fractions)

    def calculate_fractions(self):
        self.populations = {}
        for label in self.labels:
            if label not in self.populations:
                self.populations[label] = self.get_compartment_soln(label)

        self.total_population = []
        n = len(self.times)
        for i in range(n):
            t = 0.0
            for label in self.labels:
                t += self.populations[label][i]
            self.total_population.append(t)

        self.fractions = {}
        for label in self.labels:
            self.fractions[label] = [
                v/t
                for v, t in
                zip(
                    self.populations[label],
                    self.total_population
                )
            ]

    def get_compartment_soln(self, label):
        i_label = self.labels.index(label)
        return self.soln_array[:, i_label]

    def get_var_soln(self, label):
        i_label = self.var_labels.index(label)
        return self.vars_array[:, i_label]

    def load_state(self, i_time):
        self.time = self.times[i_time]
        for i_label, label in enumerate(self.labels):
            self.compartments[label] = \
                self.soln_array[i_time, i_label]
        self.calculate_vars()
    
    def checks(self, error_margin=0.1):
        # Check all compartments are positive
        for label in self.labels:  
            assert self.compartments[label] >= 0.0
        # Check population is conserved across compartments
        population_change = \
              self.vars['rate_birth'] \
            - self.vars['rate_death'] \
            - self.vars['rate_infection_death']
        assert abs(sum(self.flows.values()) - population_change ) < error_margin

    def make_graph(self, png):
        from graphviz import Digraph

        styles = {
            'graph': {
                # 'label': 'A Fancy Graph',
                'fontsize': '16',
                'fontcolor': 'white',
                'bgcolor': '#333333',
                'rankdir': 'BT',
            },
            'nodes': {
                'fontname': 'Helvetica',
                'shape': 'box',
                'fontcolor': 'white',
                'color': 'white',
                'style': 'filled',
                'fillcolor': '#006699',
            },
            'edges': {
                'style': 'dashed',
                'color': 'white',
                'arrowhead': 'open',
                'fontname': 'Courier',
                'fontsize': '12',
                'fontcolor': 'white',
            }
        }

        styles = {
            'graph': {
                'label': 'Stage 5',
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

        self.graph = Digraph(format='png')
        for label in self.labels:
            self.graph.node(label)
        self.graph.node("tb_death")
        for from_label, to_label, var_label in self.var_transfer_rate_flows:
            self.graph.edge(from_label, to_label, label=var_label)
        for from_label, to_label, rate in self.fixed_transfer_rate_flows:
            self.graph.edge(from_label, to_label, label=str(rate))
        for label, rate in self.infection_death_rate_flows:
            self.graph.edge(label, "tb_death", label=str(rate))
        base, ext = os.path.splitext(png)
        if ext.lower() != '.png':
            base = png

        self.graph = apply_styles(self.graph, styles)

        self.graph.render(base)

    def check_converged_compartment_fraction(
            self, label, equil_time, test_fraction_diff):
        is_converged = True
        labels = self.labels
        self.calculate_fractions()
        times = self.times
        fraction = self.fractions[label]
        i = -2
        max_fraction_diff = 0
        time_diff = 0
        while time_diff < equil_time:
            i -= 1
            if -i >= len(times):
                is_converged = False
                break
            time_diff = abs(times[-1] - times[i])
            frac_diff = (fraction[-1] - fraction[i])
            if abs(frac_diff) > max_fraction_diff:
                max_fraction_diff = frac_diff
            if abs(frac_diff) > test_fraction_diff:
                return False
        return True



class NaivePopulation(BasePopulationSystem):

    """
    Adapted from the OptimaTB prototype.
    """

    def __init__(self):

        BasePopulationSystem.__init__(self)

        self.set_compartment("susceptible", 1e6)
        self.set_compartment("latent", 0.)
        self.set_compartment("active", 1e3)
        self.set_compartment("detected", 1.)
        self.set_compartment("treated", 0.)

        self.set_param("rate_birth", 0.02)

        self.set_param("rate_death", 1/70.)
        self.set_param("tbdeath", 0.2)
        self.set_param("treatdeath", 0.1)

        self.set_param("ncontacts", 5.)

        self.set_param("progress", 0.005)
        self.set_param("treat", 0.3)
        self.set_param("recov", 2.)
        self.set_param("test", 4.)

    def calculate_vars(self):
        self.vars["population"] = sum(self.compartments.values())
        self.vars["rate_birth"] = self.params["rate_birth"] * self.vars["population"]
        self.vars["force"] = \
              (self.compartments["active"] + self.compartments["detected"]) \
            / self.vars["population"] \
            * self.params['ncontacts']

    def set_flows(self):
        self.set_var_entry_rate_flow("susceptible", "rate_birth")

        self.set_var_transfer_rate_flow("susceptible", "latent", "force")

        self.set_fixed_transfer_rate_flow("latent", "active", "progress")
        self.set_fixed_transfer_rate_flow("active", "detected", "test")
        self.set_fixed_transfer_rate_flow("detected", "treated", "treat")
        self.set_fixed_transfer_rate_flow("treated", "susceptible", "recov")

        self.set_infection_death_rate_flow("active", "tbdeath")
        self.set_infection_death_rate_flow("detected", "tbdeath")
        self.set_infection_death_rate_flow("treated", "treatdeath")

        self.set_population_death_rate("rate_death")



class SingleComponentPopluationSystem(BasePopulationSystem):

    """
    Initial test Autumn model designed by James
    """

    def __init__(self):

        BasePopulationSystem.__init__(self)

        self.set_compartment("susceptible", 1e6)
        self.set_compartment("latent_early", 0.)
        self.set_compartment("latent_late", 0.)
        self.set_compartment("active", 1.)
        self.set_compartment("undertreatment", 0.)

        self.set_param("rate_birth", 20. / 1e3)
        self.set_param("rate_death", 1. / 65)

        self.set_param("n_tb_contact", 40.)
        self.set_param("rate_tb_earlyprog", .1 / .5)
        self.set_param("rate_tb_lateprog", .1 / 100.)
        self.set_param("rate_tb_stabilise", .9 / .5)
        self.set_param("rate_tb_recover", .6 / 3.)
        self.set_param("rate_tb_death", .4 / 3.)

        time_treatment = .5
        self.set_param("rate_program_detect", 1.)
        self.set_param("time_treatment", time_treatment)
        self.set_param("rate_program_completion", .9 / time_treatment)
        self.set_param("rate_program_default", .05 / time_treatment)
        self.set_param("rate_program_death", .05 / time_treatment)

    def calculate_vars(self):
        self.vars["pop_total"] = sum(self.compartments.values())

        self.vars["rate_birth"] = \
            self.params["rate_birth"] * self.vars["pop_total"]

        self.vars["rate_force"] = \
            self.params["n_tb_contact"] \
              * self.compartments["active"] \
              / self.vars["pop_total"]

    def set_flows(self):
        self.set_var_entry_rate_flow("susceptible", "rate_birth")

        self.set_var_transfer_rate_flow(
            "susceptible", "latent_early", "rate_force")

        self.set_fixed_transfer_rate_flow(
            "latent_early", "active", "rate_tb_earlyprog")
        self.set_fixed_transfer_rate_flow(
            "latent_early", "latent_late", "rate_tb_stabilise")

        self.set_fixed_transfer_rate_flow(
            "latent_late", "active", "rate_tb_lateprog")

        self.set_fixed_transfer_rate_flow(
            "active", "undertreatment", "rate_program_detect")
        self.set_fixed_transfer_rate_flow(
            "active", "latent_late", "rate_tb_recover")

        self.set_fixed_transfer_rate_flow(
            "undertreatment", "active", "rate_program_default")
        self.set_fixed_transfer_rate_flow(
            "undertreatment", "susceptible", "rate_program_completion")

        self.set_population_death_rate("rate_death")
        self.set_infection_death_rate_flow("active", "rate_tb_death")
        self.set_infection_death_rate_flow("undertreatment", "rate_program_death")



class Stage2PopulationSystem(BasePopulationSystem):

    """
    Initial test Autumn model designed by James
    """

    def __init__(self):

        BasePopulationSystem.__init__(self)

        self.set_compartment("susceptible", 1e6)
        self.set_compartment("latent_early", 0.)
        self.set_compartment("latent_late", 0.)
        self.set_compartment("active", 1.)
        self.set_compartment("treatment_infect", 0.)
        self.set_compartment("treatment_noninfect", 0.)

        self.set_param("rate_birth", 20. / 1e3)
        self.set_param("rate_death", 1. / 65)

        self.set_param("n_tb_contact", 40.)
        self.set_param("rate_tb_earlyprog", .1 / .5)
        self.set_param("rate_tb_lateprog", .1 / 100.)
        self.set_param("rate_tb_stabilise", .9 / .5)
        self.set_param("rate_tb_recover", .6 / 3.)
        self.set_param("rate_tb_death", .4 / 3.)

        time_treatment = .5
        self.set_param("rate_program_detect", 1.)
        self.set_param("time_treatment", time_treatment)
        self.set_param("rate_program_completion_infect", .9 / time_treatment)
        self.set_param("rate_program_default_infect", .05 / time_treatment)
        self.set_param("rate_program_death_infect", .05 / time_treatment)
        self.set_param("rate_program_completion_noninfect", .9 / time_treatment)
        self.set_param("rate_program_default_noninfect", .05 / time_treatment)
        self.set_param("rate_program_death_noninfect", .05 / time_treatment)

    def calculate_vars(self):
        self.vars["population"] = sum(self.compartments.values())
        self.vars["rate_birth"] = \
            self.params["rate_birth"] * self.vars["population"]
        self.vars["rate_force"] = \
                self.params["n_tb_contact"] \
              * self.compartments["active"] \
              / self.vars["population"]

    def set_flows(self):
        self.set_var_entry_rate_flow("susceptible", "rate_birth")

        self.set_var_transfer_rate_flow(
            "susceptible", "latent_early", "rate_force")

        self.set_fixed_transfer_rate_flow(
            "latent_early", "active", "rate_tb_earlyprog")
        self.set_fixed_transfer_rate_flow(
            "latent_early", "latent_late", "rate_tb_stabilise")

        self.set_fixed_transfer_rate_flow(
            "latent_late", "active", "rate_tb_lateprog")

        self.set_fixed_transfer_rate_flow(
            "active", "latent_late", "rate_tb_recover")
        self.set_fixed_transfer_rate_flow(
            "active", "treatment_infect", "rate_program_detect")

        self.set_fixed_transfer_rate_flow(
            "treatment_infect", "treatment_noninfect", "rate_program_completion_infect")
        self.set_fixed_transfer_rate_flow(
            "treatment_infect", "active", "rate_program_default_infect")

        self.set_fixed_transfer_rate_flow(
            "treatment_noninfect", "susceptible", "rate_program_completion_noninfect")
        self.set_fixed_transfer_rate_flow(
            "treatment_noninfect", "active", "rate_program_default_noninfect")

        self.set_population_death_rate("rate_death")
        self.set_infection_death_rate_flow(
            "active", "rate_tb_death")
        self.set_infection_death_rate_flow(
            "treatment_infect", "rate_program_death_infect")
        self.set_infection_death_rate_flow(
            "treatment_noninfect", "rate_program_death_noninfect")


class Stage6PopulationSystem(BasePopulationSystem):

    """
    This model based on James' thesis
    """

    def __init__(self, input_parameters, input_compartments):

        BasePopulationSystem.__init__(self)

        compartment_list = [
            "susceptible_fully", 
            "susceptible_vac", 
            "susceptible_treated",
            "latent_early", 
            "latent_late", 
            "active", 
            "detect", 
            "missed", 
            "treatment_infect", 
            "treatment_noninfect"
        ]

        self.pulmonary_status = [
            "_smearpos",
            "_smearneg",
            "_extrapul"]

        for compartment in compartment_list:
            if compartment in input_compartments:
                if "susceptible" in compartment or "latent" in compartment:
                    self.set_compartment(compartment, input_compartments[compartment])
                else:
                    for status in self.pulmonary_status:
                        self.set_compartment(compartment + status,
                                             input_compartments[compartment] / 3.)
            else:
                if "susceptible" in compartment or "latent" in compartment:
                    self.set_compartment(compartment, 0.)
                else:
                    for status in self.pulmonary_status:
                        self.set_compartment(compartment + status, 0.)

        for parameter in input_parameters:
            self.set_param(parameter, input_parameters[parameter])

        self.set_param("tb_rate_stabilise",
                       (1 - self.params["tb_proportion_early_progression"]) /
                       self.params["tb_timeperiod_early_latent"])

        if "tb_proportion_casefatality_untreated_extrapul" not in input_parameters:
            self.set_param("tb_proportion_casefatality_untreated_extrapul",
                            input_parameters["tb_proportion_casefatality_untreated_smearneg"])

        self.set_param("program_rate_detect",
                       1. / self.params["tb_timeperiod_activeuntreated"] /
                       (1. - self.params["program_proportion_detect"]))
        # Formula derived from CDR = (detection rate) / (detection rate and spontaneous resolution rates)

        self.set_param("program_rate_missed",
                       self.params["program_rate_detect"] *
                       (1. - self.params["program_algorithm_sensitivity"]) /
                       self.params["program_algorithm_sensitivity"])
        # Formula derived from (algorithm sensitivity) = (detection rate) / (detection rate and miss rate)

        # Code to determines the treatment flow rates from the input parameters
        self.outcomes = ["_success", "_death", "_default"]
        self.nonsuccess_outcomes = self.outcomes[1:3]
        self.treatment_stages = ["_infect", "_noninfect"]
        self.set_param("tb_timeperiod_noninfect_ontreatment",  # Find the non-infectious period
                       self.params["tb_timeperiod_treatment"] -
                       self.params["tb_timeperiod_infect_ontreatment"])
        for outcome in self.nonsuccess_outcomes:  # Find the proportion of deaths/defaults during infectious stage
            self.set_param("program_proportion" + outcome + "_infect",
                           1. - exp(log(1. - self.params["program_proportion" + outcome]) *
                                    self.params["tb_timeperiod_infect_ontreatment"] /
                                    self.params["tb_timeperiod_treatment"]))
        for outcome in self.nonsuccess_outcomes:  # Find the proportion of deaths/defaults during non-infectious stage
            self.set_param("program_proportion" + outcome + "_noninfect",
                           self.params["program_proportion" + outcome] -
                           self.params["program_proportion" + outcome + "_infect"])
        for treatment_stage in self.treatment_stages:  # Find the success proportions
            self.set_param("program_proportion_success" + treatment_stage,
                           1. - self.params["program_proportion_default" + treatment_stage] -
                           self.params["program_proportion_death" + treatment_stage])
            for outcome in self.outcomes:  # Find the corresponding rates from the proportions
                self.set_param("program_rate" + outcome + treatment_stage,
                               1. / self.params["tb_timeperiod" + treatment_stage + "_ontreatment"] *
                               self.params["program_proportion" + outcome + treatment_stage])

        for status in self.pulmonary_status:
            self.set_param("tb_rate_earlyprogress" + status,
                           self.params["tb_proportion_early_progression"]
                           / self.params["tb_timeperiod_early_latent"]
                           * self.params["epi_proportion_cases" + status])
            self.set_param("tb_rate_lateprogress" + status,
                           self.params["tb_rate_late_progression"]
                           * self.params["epi_proportion_cases" + status])
            self.set_param("tb_rate_recover" + status,
                           (1 - self.params["tb_proportion_casefatality_untreated" + status])
                           / self.params["tb_timeperiod_activeuntreated"])
            self.set_param("tb_demo_rate_death" + status,
                           self.params["tb_proportion_casefatality_untreated" + status]
                           / self.params["tb_timeperiod_activeuntreated"])

    def calculate_vars(self):
        self.vars["population"] = sum(self.compartments.values())

        self.vars["rate_birth"] = \
            self.params["demo_rate_birth"] * self.vars["population"]
        self.vars["births_unvac"] = \
            self.params["program_prop_unvac"] * self.vars["rate_birth"]
        self.vars["births_vac"] = \
            self.params["program_prop_vac"] * self.vars["rate_birth"]

        self.vars["infected_populaton"] = 0.
        for status in self.pulmonary_status:
            for label in self.labels:
                if status in label and "_noninfect" not in label:
                    self.vars["infected_populaton"] += \
                        self.params["tb_multiplier_force" + status] \
                           * self.compartments[label]

        self.vars["rate_force"] = \
              self.params["tb_n_contact"] \
            * self.vars["infected_populaton"] \
            / self.vars["population"]

        self.vars["rate_force_weak"] = \
            self.params["tb_multiplier_bcg_protection"] \
            * self.vars["rate_force"]

    def set_flows(self):
        self.set_var_entry_rate_flow(
            "susceptible_fully", "births_unvac")
        self.set_var_entry_rate_flow(
            "susceptible_vac", "births_vac")

        self.set_var_transfer_rate_flow(
            "susceptible_fully", "latent_early", "rate_force")
        self.set_var_transfer_rate_flow(
            "susceptible_vac", "latent_early", "rate_force_weak")
        self.set_var_transfer_rate_flow(
            "susceptible_treated", "latent_early", "rate_force_weak")
        self.set_var_transfer_rate_flow(
            "latent_late", "latent_early", "rate_force_weak")

        self.set_fixed_transfer_rate_flow(
            "latent_early", "latent_late", "tb_rate_stabilise")

        for status in self.pulmonary_status:
            self.set_fixed_transfer_rate_flow(
                "latent_early",
                "active" + status,
                "tb_rate_earlyprogress" + status)
            self.set_fixed_transfer_rate_flow(
                "latent_late",
                "active" + status,
                "tb_rate_lateprogress" + status)
            self.set_fixed_transfer_rate_flow(
                "active" + status,
                "latent_late",
                "tb_rate_recover" + status)
            self.set_fixed_transfer_rate_flow(
                "active" + status,
                "detect" + status,
                "program_rate_detect")
            self.set_fixed_transfer_rate_flow(
                "active" + status,
                "missed" + status,
                "program_rate_missed")
            self.set_fixed_transfer_rate_flow(
                "detect" + status,
                "treatment_infect" + status,
                "program_rate_start_treatment")
            self.set_fixed_transfer_rate_flow(
                "missed" + status,
                "active" + status,
                "program_rate_restart_presenting")
            self.set_fixed_transfer_rate_flow(
                "missed" + status,
                "latent_late",
                "tb_rate_recover" + status)
            self.set_fixed_transfer_rate_flow(
                "treatment_infect" + status,
                "treatment_noninfect" + status,
                "program_rate_success_infect")
            self.set_fixed_transfer_rate_flow(
                "treatment_infect" + status,
                "active" + status,
                "program_rate_default_infect")
            self.set_fixed_transfer_rate_flow(
                "treatment_noninfect" + status,
                "active" + status,
                "program_rate_default_noninfect")
            self.set_fixed_transfer_rate_flow(
                "treatment_noninfect" + status,
                "susceptible_treated",
                "program_rate_success_noninfect")

        # death flows
        self.set_population_death_rate("demo_rate_death")

        for status in self.pulmonary_status:
            self.set_infection_death_rate_flow(
                "active" + status,
                "tb_demo_rate_death" + status)
            self.set_infection_death_rate_flow(
                "detect" + status,
                "tb_demo_rate_death" + status)
            self.set_infection_death_rate_flow(
                "treatment_infect" + status,
                "program_rate_death_infect")
            self.set_infection_death_rate_flow(
                "treatment_noninfect" + status,
                "program_rate_death_noninfect")




