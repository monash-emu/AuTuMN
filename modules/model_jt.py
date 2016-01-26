# -*- coding: utf-8 -*-


"""

Base Population Model to handle different type of models.

Implicit time unit: years

"""

import os
from scipy.integrate import odeint
from scipy import exp, log
import numpy


class BasePopulationSystem():

    def __init__(self):
        self.labels = []
        self.init_compartments = {}
        self.flows = {}
        self.vars = {}
        self.params = {}
        self.fixed_transfer_rate_flows = []
        self.infection_death_rate_flows = []
        self.var_transfer_rate_flows = []
        self.var_flows = []

    def make_steps(self, start, end, delta):
        self.steps = []
        step = start
        while step <= end:
            self.steps.append(step)
            if type(delta) is float:
                step += delta
            elif type(delta) is int:
                step += (end - start) / float(delta)

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

    def set_death_rate_flow(self, label, param_label):
        self.infection_death_rate_flows.append((label, self.params[param_label]))

    def set_fixed_transfer_rate_flow(self, from_label, to_label, param_label):
        self.fixed_transfer_rate_flows.append((from_label, to_label, self.params[param_label]))

    def set_var_transfer_rate_flow(self, from_label, to_label, vars_label):
        self.var_transfer_rate_flows.append((from_label, to_label, vars_label))

    def set_var_entry_rate_flow(self, label, vars_label):
        self.var_flows.append((label, vars_label))

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
        for from_label, to_label, rate in self.fixed_transfer_rate_flows:
            val = self.compartments[from_label] * rate
            self.flows[from_label] -= val
            self.flows[to_label] += val

        # population-wide death flows
        self.vars["deaths"] = 0.0

        for label in self.labels:
            val = self.compartments[label] * self.death_rate
            self.flows[label] -= val
            self.vars['deaths'] += val

        # extra death flows
        for label, rate in self.infection_death_rate_flows:
            val = self.compartments[label] * rate
            self.flows[label] -= val
            self.vars['deaths'] += val

    def make_derivate_fn(self):

        def derivative_fn(y, t):
            self.time = t
            self.compartments = self.convert_list_to_compartments(y)
            self.vars.clear()
            self.calculate_vars()
            if len(self.fixed_transfer_rate_flows) == 0:
                self.set_flows()
            self.calculate_flows()
            flow_vector = self.convert_compartments_to_list(self.flows)
            self.checks()
            return flow_vector

        return derivative_fn

    def integrate_scipy(self):
        derivative = self.make_derivate_fn()
        init_y = self.get_init_list()
        self.soln_array = odeint(derivative, init_y, self.steps)
        self.calculate_fractions()
        
    def integrate_explicit(self, min_dt=0.05):
        y = self.get_init_list()

        n_component = len(y)
        n_time = len(self.steps)
        self.soln_array = numpy.zeros((n_time, n_component))

        derivative = self.make_derivate_fn()
        time = self.steps[0]
        self.soln_array[0, :] = y

        for i_time, new_time in enumerate(self.steps):
            while time < new_time:
                f = derivative(y, time)
                old_time = time
                time = time + min_dt
                dt = min_dt
                if time > new_time:
                    dt = new_time - old_time
                    time = new_time
                for i in range(n_component):
                    y[i] = y[i] + dt*f[i]
            if i_time < n_time - 1:
                self.soln_array[i_time+1, :] = y
        self.calculate_fractions()

    def calculate_fractions(self):
        self.populations = {}
        for label in self.labels:
            if label not in self.populations:
                self.populations[label] = self.get_soln(label)

        self.total_population = []
        n = len(self.steps)
        for i in range(n):
            t = 0.0
            for label in self.labels:
                t += self.populations[label][i]
            self.total_population.append(t)

        self.fractions = {}
        for label in self.labels:
            self.fractions[label] = [
                v/t for v, t in 
                    zip(
                        self.populations[label], 
                        self.total_population
                    )
            ]

    def get_soln(self, label):
        i_label = self.labels.index(label)
        return self.soln_array[:, i_label]

    def load_state(self, i_time):
        self.time = self.steps[i_time]
        for i_label, label in enumerate(self.labels):
            self.compartments[label] = \
                self.soln_array[i_time, i_label]
        self.calculate_vars()
    
    def checks(self, error_margin=0.1):
        # Check all compartments are positive
        for label in self.labels:  
            assert self.compartments[label] >= 0.0
        # Check population is conserved across compartments
        population_change = self.vars['births'] - self.vars['deaths']
        assert abs(sum(self.flows.values()) - population_change) < error_margin

    def make_graph(self, png):
        from graphviz import Digraph

        styles = {
            'graph': {
                # 'label': 'Flow chart of inter-compartmental flows',
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
                'label': 'Stage 6',
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
        self.graph.node("infection_death")
        for from_label, to_label, var_label in self.var_transfer_rate_flows:
            self.graph.edge(from_label, to_label, label=var_label)
        for from_label, to_label, rate in self.fixed_transfer_rate_flows:
            self.graph.edge(from_label, to_label, label=str(rate))
        for label, rate in self.infection_death_rate_flows:
            self.graph.edge(label, "infection_death", label=str(rate))
        base, ext = os.path.splitext(png)
        if ext.lower() != '.png':
            base = png

        self.graph = apply_styles(self.graph, styles)

        self.graph.render(base)

class Stage6PopulationSystem(BasePopulationSystem):

    """
    This model based on James' thesis
    """

    def __init__(self, input_parameters, input_compartments):

        BasePopulationSystem.__init__(self)

        compartment_list = ["susceptible_fully", "susceptible_vac", "susceptible_treated",
         "latent_early", "latent_late", "active", "detect", "missed", "treatment_infect",
         "treatment_noninfect"]

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

        """ The following code block takes the programmatic parameters and determines the flow rates for the
        treatment compartments. """
        self.outcomes = ["_default", "_death", "_success"]
        self.nonsuccess_outcomes = self.outcomes[0:2]
        self.treatment_stages = ["_infect", "_noninfect"]
        self.set_param("program_timeperiod_noninfect_ontreatment",  # Find the non-infectious period
                       self.params["program_timeperiod_treatment"] -
                       self.params["program_timeperiod_infect_ontreatment"])
        for outcome in self.nonsuccess_outcomes:  # Find the proportion of deaths/defaults during infectious stage
            self.set_param("program_proportion" + outcome + "_infect",
                           1. - exp(log(1. - self.params["program_proportion" + outcome]) *
                                    self.params["program_timeperiod_infect_ontreatment"] /
                                    self.params["program_timeperiod_treatment"]))
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
                               1. / self.params["program_timeperiod" + treatment_stage + "_ontreatment"] *
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

        self.vars["births"] = \
            self.params["demo_rate_birth"] * self.vars["population"]
        self.vars["births_unvac"] = \
            self.params["program_prop_unvac"] * self.vars["births"]
        self.vars["births_vac"] = \
            self.params["program_prop_vac"] * self.vars["births"]

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
                "program_rate_completion_noninfect")

        # death flows
        self.set_population_death_rate("demo_rate_death")

        for status in self.pulmonary_status:
            self.set_death_rate_flow(
                "active" + status,
                "tb_demo_rate_death" + status)
            self.set_death_rate_flow(
                "detect" + status,
                "tb_demo_rate_death" + status)
            self.set_death_rate_flow(
                "treatment_infect" + status,
                "program_rate_death_infect")
            self.set_death_rate_flow(
                "treatment_noninfect" + status,
                "program_rate_death_noninfect")
