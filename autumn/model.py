# -*- coding: utf-8 -*-


"""

Base Population Model to handle different type of models.

Implicit time unit: years

"""

import os
from scipy.integrate import odeint
import numpy

def make_steps(start, end, delta):
    steps = []
    step = start
    while step <= end:
        steps.append(step)
        step += delta
    return steps


class BasePopulationSystem():

    def __init__(self):
        self.labels = []
        self.init_compartments = {}
        self.flows = {}
        self.vars = {}
        self.params = {}
        self.fixed_transfer_rate_flows = []
        self.extra_death_rate_flows = []
        self.var_transfer_rate_flows = []
        self.var_flows = []

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

    def set_normal_death_rate(self, death_label):
        self.death_rate = self.params[death_label]

    def set_death_rate_flow(self, label, param_label):
        self.extra_death_rate_flows.append((label, self.params[param_label]))

    def set_fixed_transfer_rate_flow(self, from_label, to_label, param_label):
        self.fixed_transfer_rate_flows.append((from_label, to_label, self.params[param_label]))

    def set_var_transfer_rate_flow(self, from_label, to_label, vars_label):
        self.var_transfer_rate_flows.append((from_label, to_label, vars_label))

    def set_var_flow(self, label, vars_label):
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

        # normal death flows
        self.vars["deaths"] = 0.0

        for label in self.labels:
            val = self.compartments[label] * self.death_rate
            self.flows[label] -= val
            self.vars['deaths'] += val

        # extra death flows
        for label, rate in self.extra_death_rate_flows:
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

    def integrate_scipy(self, times):
        derivative = self.make_derivate_fn()
        self.times = times
        init_y = self.get_init_list()
        self.soln_array = odeint(derivative, init_y, times)
        self.calculate_fractions()
        
    def integrate_explicit(self, times):
        self.times = times
        y = self.get_init_list()

        n_component = len(y)
        n_time = len(self.times)
        self.soln_array = numpy.zeros((n_time, n_component))

        derivative = self.make_derivate_fn()
        time = self.times[0]
        self.soln_array[0,:] = y
        min_dt = 0.05
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
                    y[i] = y[i] + dt*f[i]
            if i_time < n_time - 1:
                self.soln_array[i_time+1,:] = y
        self.calculate_fractions()

    def calculate_fractions(self):
        self.populations = {}
        for label in self.labels:
            if label not in self.populations:
                self.populations[label] = self.get_soln(label)

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
        self.time = self.times[i_time]
        for i_label, label in enumerate(self.labels):
            self.compartments[label] = \
                self.soln_array[i_time,i_label]
        self.calculate_vars()
    
    def checks(self):
        # Check all compartments are positive
        for label in self.labels:  
            assert self.compartments[label] >= 0.0
        # Check population is conserved across compartments
        population_change = self.vars['births'] - self.vars['deaths']
        assert abs(sum(self.flows.values()) - population_change )  < 0.1

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
        for label, rate in self.extra_death_rate_flows:
            self.graph.edge(label, "tb_death", label=str(rate))
        base, ext = os.path.splitext(png)
        if ext.lower() != '.png':
            base = png

        self.graph = apply_styles(self.graph, styles)

        self.graph.render(base)



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
        self.vars["births"] = self.params["rate_birth"] * self.vars["population"]
        self.vars["force"] = \
              (self.compartments["active"] + self.compartments["detected"]) \
            / self.vars["population"] \
            * self.params['ncontacts']

    def set_flows(self):
        self.set_var_flow("susceptible", "births")

        self.set_var_transfer_rate_flow("susceptible", "latent", "force")

        self.set_fixed_transfer_rate_flow("latent", "active", "progress")
        self.set_fixed_transfer_rate_flow("active", "detected", "test")
        self.set_fixed_transfer_rate_flow("detected", "treated", "treat")
        self.set_fixed_transfer_rate_flow("treated", "susceptible", "recov")

        self.set_death_rate_flow("active", "tbdeath")
        self.set_death_rate_flow("detected", "tbdeath")
        self.set_death_rate_flow("treated", "treatdeath")

        self.set_normal_death_rate("rate_death")

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

        self.vars["births"] = \
            self.params["rate_birth"] * self.vars["pop_total"]

        self.vars["rate_force"] = \
            self.params["n_tb_contact"] \
              * self.compartments["active"] \
              / self.vars["pop_total"]

    def set_flows(self):
        self.set_var_flow("susceptible", "births")

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

        self.set_normal_death_rate("rate_death")
        self.set_death_rate_flow("active", "rate_tb_death")
        self.set_death_rate_flow("undertreatment", "rate_program_death")



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
        self.vars["births"] = \
            self.params["rate_birth"] * self.vars["population"]
        self.vars["rate_force"] = \
                self.params["n_tb_contact"] \
              * self.compartments["active"] \
              / self.vars["population"]

    def set_flows(self):
        self.set_var_flow("susceptible", "births")

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

        self.set_normal_death_rate("rate_death")
        self.set_death_rate_flow(
            "active", "rate_tb_death")
        self.set_death_rate_flow(
            "treatment_infect", "rate_program_death_infect")
        self.set_death_rate_flow(
            "treatment_noninfect", "rate_program_death_noninfect")


