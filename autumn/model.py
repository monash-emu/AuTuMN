# -*- coding: utf-8 -*-


"""

Base Population Model to handle different type of models.

Implicit time unit: years

"""

import os
from scipy.integrate import odeint
from scipy import exp, log
import numpy

from settings import default
from settings import philippines 

from curve import make_sigmoidal_curve

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


def label_intersects_tags(label, tags):
    for tag in tags:
        if tag in label:
            return True
    return False


class BaseModel():

    def __init__(self):
        self.labels = []
        self.init_compartments = {}
        self.params = {}
        self.times = None

        self.vars = {}

        self.soln_array = None
        self.var_labels = None
        self.var_array = None
        self.flow_array = None

        self.flows = {}
        self.fixed_transfer_rate_flows = []
        self.infection_death_rate_flows = []
        self.var_transfer_rate_flows = []
        self.var_flows = []

    def make_times(self, start, end, delta):
        "Return steps with n or delta"
        self.times = []
        step = start
        while step <= end:
            self.times.append(step)
            step += delta

    def make_times_with_n_step(self, start, end, n):
        "Return steps with n or delta"
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

    def calculate_vars(self):
        """
        Calculate self.vars that only depend on compartment values
        """
        pass

    def calculate_flows(self):
        """
        Calculate flows, which should only depend on compartment values
        and self.vars calculated in calculate_vars.
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
        for label, rate in self.infection_death_rate_flows:
            val = self.compartments[label] * rate
            self.flows[label] -= val
            self.vars["rate_infection_death"] += val

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

    def init_run(self):
        self.set_flows()
        self.var_labels = None
        self.soln_array = None
        self.var_array = None
        self.flow_array = None

    def integrate_scipy(self):
        self.init_run()
        assert not self.times is None, "Haven't set times yet"
        init_y = self.get_init_list()
        derivative = self.make_derivate_fn()
        self.soln_array = odeint(derivative, init_y, self.times)

        self.calculate_diagnostics()
        
    def integrate_explicit(self, min_dt=0.05):
        self.init_run()
        assert not self.times is None, "Haven't set times yet"
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
                    # hack to avoid errors due to time-step
                    if y[i] < 0.0:
                        y[i] = 0.0
            if i_time < n_time - 1:
                self.soln_array[i_time+1,:] = y

        self.calculate_diagnostics()

    def calculate_diagnostic_vars(self):
        """
        Calculate diagnostic vars that can depend on self.flows as
        well as self.vars calculated in calculate_vars
        """
        pass

    def calculate_diagnostics(self):
        self.population_soln = {}
        for label in self.labels:
            if label in self.population_soln:
                continue
            self.population_soln[label] = self.get_compartment_soln(label)

        n_time = len(self.times)
        for i in range(n_time):

            self.time = self.times[i]

            for label in self.labels:
                self.compartments[label] = self.population_soln[label][i]

            self.calculate_vars()
            self.calculate_flows()
            self.calculate_diagnostic_vars()

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

        self.fraction_soln = {}
        for label in self.labels:
            self.fraction_soln[label] = [
                v / t
                for v, t 
                in zip(
                    self.population_soln[label],
                    self.get_var_soln("population")
                )
            ]

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
        # # Check all compartments are positive
        # for label in self.labels:
        #     assert self.compartments[label] >= 0.0
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
            self.graph.edge(from_label, to_label, label=var_label)
        for from_label, to_label, rate in self.fixed_transfer_rate_flows:
            self.graph.edge(from_label, to_label, label=num_str(rate))
        for label, rate in self.infection_death_rate_flows:
            self.graph.edge(label, "tb_death", label=num_str(rate))
        base, ext = os.path.splitext(png)
        if ext.lower() != '.png':
            base = png

        self.graph = apply_styles(self.graph, styles)

        self.graph.render(base)

    def check_converged_compartment_fraction(
            self, label, equil_time, test_fraction_diff):
        labels = self.labels
        self.calculate_diagnostics()
        times = self.times
        fraction = self.fraction_soln[label]
        i = -2
        max_fraction_diff = 0
        time_diff = 0
        while time_diff < equil_time:
            i -= 1
            if -i >= len(times):
                break
            time_diff = abs(times[-1] - times[i])
            frac_diff = (fraction[-1] - fraction[i])
            if abs(frac_diff) > max_fraction_diff:
                max_fraction_diff = frac_diff
            if abs(frac_diff) > test_fraction_diff:
                return False
        return True


class BaseTbModel(BaseModel):
    def __init__(self):
        BaseModel.__init__(self)

    def initialise_compartments(self, input_compartments):

        # Initialise to zero
        for compartment in self.compartment_list:
            for comorbidity in self.comorbidities:
                if "susceptible" in compartment:  # Replicate for comorbidities only
                    self.set_compartment(compartment + comorbidity, 0.)
                elif "latent" in compartment:  # Replicate for comorbidities and strains
                    for strain in self.strains:
                        self.set_compartment(compartment + strain + comorbidity, 0.)
                else:
                    for strain in self.strains:
                        for organ in self.organ_status:
                            self.set_compartment(compartment + organ + strain + comorbidity, 0.)
            for compartment in self.compartment_list:
                if compartment in input_compartments:
                    if "susceptible" in compartment:
                        for comorbidity in self.comorbidities:
                            self.set_compartment(compartment + comorbidity,
                                                 input_compartments[compartment]
                                                 / len(self.comorbidities))
                    elif "latent" in compartment:
                        for comorbidity in self.comorbidities:
                            for strain in self.strains:
                                self.set_compartment(compartment + strain + comorbidity,
                                                     input_compartments[compartment]
                                                     / len(self.comorbidities)
                                                     / len(self.strains))
                    else:
                        for comorbidity in self.comorbidities:
                            for strain in self.strains:
                                for organ in self.organ_status:
                                    self.set_compartment(compartment + organ + strain + comorbidity,
                                                         input_compartments[compartment]
                                                         / len(self.comorbidities)
                                                         / len(self.strains)
                                                         / len(self.organ_status))

    def calculate_force_infection(self):
        # Force of infection calculation
        for strain in self.strains:
            self.vars["infectious_population" + strain] = 0.0
            for organ in self.organ_status:
                for label in self.labels:
                    if strain not in label:
                        continue
                    if organ not in label:
                        continue
                    if not label_intersects_tags(label, self.infectious_tags):
                        continue
                    self.vars["infectious_population" + strain] += \
                        self.params["tb_multiplier_force" + organ] \
                        * self.compartments[label]
            self.vars["rate_force" + strain] = \
                self.params["tb_n_contact"] \
                  * self.vars["infectious_population" + strain] \
                  / self.vars["population"]
            self.vars["rate_force_weak" + strain] = \
                self.params["tb_multiplier_bcg_protection"] \
                  * self.vars["rate_force" + strain]

    def strata_iterator(self):  # Inactive
        for strain in self.strains:
            for organ in self.organs:
                yield strain, organ
                # for morbidity in self.morbidities:
                #     yield strain, organ, morbidity

    def make_strata_label(self, base, strata):  # Inactive
        return base + "".join(strata)

    def get_organ(self, label):  # Inactive
        for organ in self.organ_status:
            if organ in label:
                return organ
        return None

    def find_flow_proportions_in_early_period(
            self, proportion, early_period, total_period):
        early_proportion = 1. - exp( log(1. - proportion) * early_period / total_period)
        late_proportion = proportion - early_proportion
        return early_proportion, late_proportion

    def set_treatment_flow_rates(self):
        outcomes = ["_success", "_death", "_default"]
        non_success_outcomes = outcomes[1:3]
        treatment_stages = ["_infect", "_noninfect"]

        # Find the non-infectious period
        self.set_param(
            "tb_timeperiod_noninfect_ontreatment",
            self.params["tb_timeperiod_treatment"]
              - self.params["tb_timeperiod_infect_ontreatment"])

        # Find the proportion of deaths/defaults during the infectious and non-infectious stages
        for outcome in non_success_outcomes:
            early_proportion, late_proportion = self.find_flow_proportions_in_early_period(
                self.params["program_proportion" + outcome],
                self.params["tb_timeperiod_infect_ontreatment"],
                self.params["tb_timeperiod_treatment"])
            self.set_param(
                "program_proportion" + outcome + "_infect",
                early_proportion)
            self.set_param(
                "program_proportion" + outcome + "_noninfect",
                late_proportion)

        # Find the success proportions
        for treatment_stage in treatment_stages:
            self.set_param(
                "program_proportion_success" + treatment_stage,
                1. - self.params["program_proportion_default" + treatment_stage]
                  - self.params["program_proportion_death" + treatment_stage])
            # Find the corresponding rates from the proportions
            for outcome in outcomes:
                self.set_param(
                    "program_rate" + outcome + treatment_stage,
                    1. / self.params["tb_timeperiod" + treatment_stage + "_ontreatment"]
                      * self.params["program_proportion" + outcome + treatment_stage])

        # Temporary code assigning non-DS strains the same outcomes *****
        for strain in self.strains:
            for treatment_stage in treatment_stages:
                for outcome in outcomes:
                    self.set_param(
                        "program_rate" + outcome + treatment_stage + strain,
                        self.params["program_rate" + outcome + treatment_stage])

    def calculate_diagnostic_vars(self):

        rate_incidence = 0.0
        rate_notification = 0.0
        rate_missed = 0.0
        rate_death_ontreatment = 0.0
        rate_default = 0.0
        rate_success = 0.0
        for from_label, to_label, rate in self.fixed_transfer_rate_flows:
            val = self.compartments[from_label] * rate
            if 'latent' in from_label and 'active' in to_label:
                rate_incidence += val
            elif 'active' in from_label and 'detect' in to_label:
                rate_notification += val
            elif 'active' in from_label and 'missed' in to_label:
                rate_missed += val
            elif 'treatment' in from_label and 'death' in to_label:
                rate_death_ontreatment += val
            elif 'treatment' in from_label and 'active' in to_label:
                rate_default += val
            elif 'treatment' in from_label and 'susceptible_treated' in to_label:
                rate_success += val

        # Main epidemiological indicators - note that denominator is not individuals
        self.vars["incidence"] = \
              rate_incidence \
            / self.vars["population"] * 1E5

        self.vars["notification"] = \
              rate_notification \
            / self.vars["population"] * 1E5

        self.vars["mortality"] = \
              self.vars["rate_infection_death"] \
            / self.vars["population"] * 1E5

        # Better term may be failed diagnosis, but using missed for
        # consistency with the compartment name for now
        self.vars["missed"] = \
              rate_missed \
            / self.vars["population"]

        self.vars["success"] = \
              rate_success \
            / self.vars["population"]

        self.vars["death_ontreatment"] = \
              rate_death_ontreatment \
            / self.vars["population"]

        self.vars["default"] = \
              rate_default \
            / self.vars["population"]


class SimplifiedModel(BaseTbModel):

    """
    Initial Autumn model designed by James
    """

    def __init__(self):

        BaseModel.__init__(self)

        self.set_compartment("susceptible", 1e6)
        self.set_compartment("latent_early", 0.)
        self.set_compartment("latent_late", 0.)
        self.set_compartment("active", 1.)
        self.set_compartment("treatment_infect", 0.)
        self.set_compartment("treatment_noninfect", 0.)

        self.set_param("demo_rate_birth", 20. / 1e3)
        self.set_param("demo_rate_death", 1. / 65)

        self.set_param("tb_n_contact", 40.)
        self.set_param("tb_rate_earlyprogress", .1 / .5)
        self.set_param("tb_rate_lateprogress", .1 / 100.)
        self.set_param("tb_rate_stabilise", .9 / .5)
        self.set_param("tb_rate_recover", .6 / 3.)
        self.set_param("tb_rate_death", .4 / 3.)

        self.set_param("program_rate_detect", 1.)
        time_treatment = .5
        self.set_param("program_time_treatment", time_treatment)
        self.set_param("program_rate_completion_infect", .9 / time_treatment)
        self.set_param("program_rate_default_infect", .05 / time_treatment)
        self.set_param("program_rate_death_infect", .05 / time_treatment)
        self.set_param("program_rate_completion_noninfect", .9 / time_treatment)
        self.set_param("program_rate_default_noninfect", .05 / time_treatment)
        self.set_param("program_rate_death_noninfect", .05 / time_treatment)

        curve1 = make_sigmoidal_curve(y_high=2, y_low=0, x_start=1950, x_inflect=1970, multiplier=4)
        curve2 = make_sigmoidal_curve(y_high=4, y_low=2, x_start=1995, x_inflect=2003, multiplier=3)
        self.test_curve = lambda x: curve1(x) if x < 1990 else curve2(x)

    def calculate_vars(self):
        self.vars["population"] = sum(self.compartments.values())
        self.vars["rate_birth"] = \
            self.params["demo_rate_birth"] * self.vars["population"]

        self.vars["infectious_population"] = 0.0
        for label in self.labels:
            if 'active' in label or '_infect' in label:
                self.vars["infectious_population"] += \
                    self.compartments[label]

        self.vars["rate_force"] = \
                self.params["tb_n_contact"] \
              * self.vars["infectious_population"] \
              / self.vars["population"]

        self.vars["program_rate_detect"] = self.test_curve(self.time)

    def set_flows(self):
        self.set_var_entry_rate_flow("susceptible", "rate_birth")

        self.set_var_transfer_rate_flow(
            "susceptible", "latent_early", "rate_force")

        self.set_fixed_transfer_rate_flow(
            "latent_early", "active", "tb_rate_earlyprogress")
        self.set_fixed_transfer_rate_flow(
            "latent_early", "latent_late", "tb_rate_stabilise")

        self.set_fixed_transfer_rate_flow(
            "latent_late", "active", "tb_rate_lateprogress")

        self.set_fixed_transfer_rate_flow(
            "active", "latent_late", "tb_rate_recover")

        self.set_var_transfer_rate_flow(
            "active", "treatment_infect", "program_rate_detect")

        self.set_fixed_transfer_rate_flow(
            "treatment_infect", "treatment_noninfect", "program_rate_completion_infect")
        self.set_fixed_transfer_rate_flow(
            "treatment_infect", "active", "program_rate_default_infect")

        self.set_fixed_transfer_rate_flow(
            "treatment_noninfect", "susceptible", "program_rate_completion_noninfect")
        self.set_fixed_transfer_rate_flow(
            "treatment_noninfect", "active", "program_rate_default_noninfect")

        self.set_population_death_rate("demo_rate_death")
        self.set_infection_death_rate_flow(
            "active", "tb_rate_death")
        self.set_infection_death_rate_flow(
            "treatment_infect", "program_rate_death_infect")
        self.set_infection_death_rate_flow(
            "treatment_noninfect", "program_rate_death_noninfect")

    def calculate_diagnostic_vars(self):

        rate_incidence = 0.0
        for from_label, to_label, rate in self.fixed_transfer_rate_flows:
            val = self.compartments[from_label] * rate
            if 'latent' in from_label and 'active' in to_label:
                rate_incidence += val

        # Main epidemiological indicators - note that denominator is not individuals
        self.vars["prevalence"] = \
              self.vars["infectious_population"] \
            / self.vars["population"] * 1E5

        self.vars["incidence"] = \
              rate_incidence \
            / self.vars["population"] * 1E5

        self.vars["mortality"] = \
              self.vars["rate_infection_death"] \
            / self.vars["population"] * 1E5

        self.vars["latent"] = 0.0
        for label in self.labels:
            if "latent" in label:
                self.vars["latent"] += (
                    self.compartments[label] 
                     / self.vars["population"] * 1E5)


class SingleStrainModel(BaseTbModel):

    """
    Includes organ status
    """

    def __init__(self):

        BaseTbModel.__init__(self)

        self.set_compartment("susceptible_fully", 1e6)
        self.set_compartment("susceptible_vac", 0.)
        self.set_compartment("susceptible_treated", 0.)
        self.set_compartment("latent_early", 0.)
        self.set_compartment("latent_late", 0.)

        self.organ_status = [
            "_smearpos",
            "_smearneg",
            "_extrapul"]

        self.set_param("proportion_cases_smearpos", 0.6)
        self.set_param("proportion_cases_smearneg", 0.2)
        self.set_param("proportion_cases_extrapul", 0.2)

        self.set_param("tb_multiplier_force_smearpos", 1.)
        self.set_param("tb_multiplier_force_smearneg", .25)
        self.set_param("tb_multiplier_force_extrapul", 0.0)

        self.set_param("tb_rate_earlyprogress", 0.2)

        for organ in self.organ_status:
            self.set_compartment("active" + organ, 1.)
            self.set_compartment("detect" + organ, 0.)
            self.set_compartment("missed" + organ, 0.)
            self.set_compartment("treatment_infect" + organ, 0.)
            self.set_compartment("treatment_noninfect" + organ, 0.)
            self.set_param(
                "tb_rate_earlyprogress" + organ,
                self.params["tb_rate_earlyprogress"]
                  * self.params["proportion_cases" + organ])
            self.set_param(
                "tb_rate_lateprogress" + organ,
                .0005 * self.params["proportion_cases" + organ])

        self.set_param("demo_rate_birth", 20. / 1e3)
        self.set_param("demo_rate_death", 1. / 65)

        self.set_param("tb_n_contact", 15.)

        self.set_param("tb_rate_lateprogress", .0005)
        self.set_param("tb_rate_stabilise", .8)
        self.set_param("tb_rate_recover", .5 * .3)
        self.set_param("tb_rate_death", .5 * .3)

        self.set_param("program_prop_vac", .9)
        self.set_param("program_prop_unvac", .1)

        self.set_param("program_rate_detect", 0.8)
        self.set_param("program_rate_missed", 0.2)

        self.set_param("program_rate_start_treatment", 26.)
        self.set_param("program_rate_giveup_waiting", 4.)

        self.set_param("program_rate_completion_infect", 26 * 0.9)
        self.set_param("program_rate_default_infect", 26 * 0.05)
        self.set_param("program_rate_death_infect", 26 * 0.05)

        self.set_param("program_rate_completion_noninfect", 2 * 0.7)
        self.set_param("program_rate_default_noninfect", 2 * 0.1)
        self.set_param("program_rate_death_noninfect", 2 * 0.1)

    def get_organ(self, label):
        for organ in self.organ_status:
            if organ in label:
                return organ
        return None

    def calculate_vars(self):
        self.vars["population"] = sum(self.compartments.values())

        self.vars["rate_birth"] = \
            self.params["demo_rate_birth"] * self.vars["population"]
        self.vars["births_unvac"] = \
            self.params['program_prop_unvac'] * self.vars["rate_birth"]
        self.vars["births_vac"] = \
            self.params['program_prop_vac'] * self.vars["rate_birth"]

        self.vars["infectious_population"] = 0.0
        self.vars["rate_force"] = 0.0
        for label in self.labels:
            if 'noninfect' in label:
                continue
            for organ in self.organ_status:
                if organ not in label:
                    continue
                self.vars["infectious_population"] += self.compartments[label]
                self.vars["rate_force"] += (
                    self.compartments[label] 
                      / self.vars["population"]
                      * self.params["tb_multiplier_force" + organ]
                      * self.params["tb_n_contact"])
  
        self.vars["rate_force_weak"] = 0.5 * self.vars["rate_force"]

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

        for organ in self.organ_status:
            self.set_fixed_transfer_rate_flow(
                "latent_early",
                "active" + organ,
                "tb_rate_earlyprogress" + organ)
            self.set_fixed_transfer_rate_flow(
                "latent_late",
                "active" + organ,
                "tb_rate_lateprogress" + organ)
            self.set_fixed_transfer_rate_flow(
                "active" + organ,
                "latent_late",
                "tb_rate_recover")
            self.set_fixed_transfer_rate_flow(
                "active" + organ,
                "detect" + organ,
                "program_rate_detect")
            self.set_fixed_transfer_rate_flow(
                "active" + organ,
                "missed" + organ,
                "program_rate_missed")
            self.set_fixed_transfer_rate_flow(
                "detect" + organ,
                "treatment_infect" + organ,
                "program_rate_start_treatment")
            self.set_fixed_transfer_rate_flow(
                "missed" + organ,
                "active" + organ,
                "program_rate_giveup_waiting")
            self.set_fixed_transfer_rate_flow(
                "treatment_infect" + organ,
                "treatment_noninfect" + organ,
                "program_rate_completion_infect")
            self.set_fixed_transfer_rate_flow(
                "treatment_infect" + organ,
                "active" + organ,
                "program_rate_default_infect")
            self.set_fixed_transfer_rate_flow(
                "treatment_noninfect" + organ,
                "active" + organ,
                "program_rate_default_noninfect")
            self.set_fixed_transfer_rate_flow(
                "treatment_noninfect" + organ,
                "susceptible_treated",
                "program_rate_completion_noninfect")

        # death flows
        self.set_population_death_rate("demo_rate_death")

        for organ in self.organ_status:
            self.set_infection_death_rate_flow(
                "active" + organ,
                "tb_rate_death")
            self.set_infection_death_rate_flow(
                "detect" + organ,
                "tb_rate_death")
            self.set_infection_death_rate_flow(
                "treatment_infect" + organ,
                "program_rate_death_infect")
            self.set_infection_death_rate_flow(
                "treatment_noninfect" + organ,
                "program_rate_death_noninfect")


    def calculate_diagnostic_vars(self):
        rate_incidence = 0.0
        rate_notification = 0.0
        for from_label, to_label, rate in self.fixed_transfer_rate_flows:
            val = self.compartments[from_label] * rate
            if 'latent' in from_label and 'active' in to_label:
                rate_incidence += val
            elif 'active' in from_label and 'detect' in to_label:
                rate_notification += val

        # Main epidemiological indicators - note that denominator is not individuals
        self.vars["incidence"] = \
              rate_incidence \
            / self.vars["population"] * 1E5

        self.vars["notification"] = \
              rate_notification \
            / self.vars["population"] * 1E5

        self.vars["mortality"] = \
              self.vars["rate_infection_death"] \
            / self.vars["population"] * 1E5


class NewFullModel(BaseTbModel):

    """
    Trying to create a harmonised model that can run any number of strains
    and organ statuses
    """

    def __init__(self, input_parameters=None, input_compartments=None):

        BaseTbModel.__init__(self)

        self.compartment_list = [
            "susceptible_fully",
            "susceptible_vac",
            "susceptible_treated",
            "latent_early",
            "latent_late",
            "active",
            "detect",
            "missed",
            "treatment_infect",
            "treatment_noninfect"]

        if input_compartments is None:
            input_compartments = {
                "susceptible_fully": 1e6,
                "active": 3.
            }

        # To remove comorbidities, set self.comorbidities to [""]
        self.comorbidities = [
            ""]

        # To remove strains, set self.strains to [""]
        self.strains = [
            ""]

        # To remove organ status, set self.organ_status to [""]
        self.organ_status = [
            "_smearpos",
            "_smearneg",
            "_extrapul"]

        self.initialise_compartments(input_compartments)

        self.infectious_tags = ["active", "missed", "detect", "treatment_infect"]

        if input_parameters is None:

            def get(param_set_name, param_name, prob=0.5):
                param_set = globals()[param_set_name]
                param = getattr(param_set, param_name)
                ppf = getattr(param, "ppf")
                return ppf(prob)

            input_parameters = {
                "demo_rate_birth": 20. / 1e3,
                "demo_rate_death": 1. / 65,
                "epi_proportion_cases_smearpos": 0.6,
                "epi_proportion_cases_smearneg": 0.2,
                "epi_proportion_cases_extrapul": 0.2,
                "tb_multiplier_force_smearpos": 1.,
                "tb_multiplier_force_smearneg":
                    get("default", "multiplier_force_smearneg"),
                "tb_multiplier_force_extrapul": 0.,
                "tb_n_contact":
                    get("default", "tb_n_contact"),
                "tb_proportion_early_progression":
                    get("default", "proportion_early_progression"),
                "tb_timeperiod_early_latent":
                    get("default", "timeperiod_early_latent"),
                "tb_rate_late_progression":
                    get("default", "rate_late_progression"),
                "tb_proportion_casefatality_untreated_smearpos":
                    get("default", "proportion_casefatality_active_untreated_smearpos"),
                "tb_proportion_casefatality_untreated_smearneg":
                    get("default", "proportion_casefatality_active_untreated_smearneg"),
                "tb_timeperiod_activeuntreated":
                    get("default", "timeperiod_activeuntreated"),
                "tb_multiplier_bcg_protection":
                    get("default", "multiplier_bcg_protection"),
                "program_prop_vac":
                    get("philippines", "bcg_coverage"),
                "program_prop_unvac":
                    1. - get("philippines", "bcg_coverage"),
                "program_proportion_detect":
                    get("philippines", "bcg_coverage"),
                "program_algorithm_sensitivity":
                    get("philippines", "algorithm_sensitivity"),
                "program_rate_start_treatment":
                    1. / get("philippines", "program_timeperiod_delayto_treatment"),
                "tb_timeperiod_treatment":
                    get("default", "timeperiod_treatment_ds"),
                "tb_timeperiod_infect_ontreatment":
                    get("default", "timeperiod_infect_ontreatment"),
                "program_proportion_default":
                    get("philippines", "proportion_default"),
                "program_proportion_death":
                    get("philippines", "proportion_death"),
                "program_rate_restart_presenting":
                    1. / get("philippines", "timeperiod_norepresentation")
            }

        # Now actually set the imported parameters
        for parameter in input_parameters:
            self.set_param(parameter, input_parameters[parameter])

        # If extrapulmonary case-fatality not stated
        if "tb_proportion_casefatality_untreated_extrapul" not in input_parameters:
            self.set_param(
                "tb_proportion_casefatality_untreated_extrapul",
                input_parameters["tb_proportion_casefatality_untreated_smearneg"])

        # Progression and stabilisation rates
        self.set_param("tb_rate_early_progression",  # Overall
                       self.params["tb_proportion_early_progression"]
                       / self.params["tb_timeperiod_early_latent"])
        self.set_param("tb_rate_stabilise", # Stabilisation rate
                       (1 - self.params["tb_proportion_early_progression"])
                       / self.params["tb_timeperiod_early_latent"])
        for organ in self.organ_status:
            self.set_param(
                "tb_rate_early_progression" + organ,
                self.params["tb_proportion_early_progression"]
                  / self.params["tb_timeperiod_early_latent"]
                  * self.params["epi_proportion_cases" + organ])
            self.set_param(
                "tb_rate_late_progression" + organ,
                self.params["tb_rate_late_progression"]
                * self.params["epi_proportion_cases" + organ])
            self.set_param(
                "tb_rate_death" + organ,
                self.params["tb_proportion_casefatality_untreated" + organ]
                / self.params["tb_timeperiod_activeuntreated"])
            self.set_param(
                "tb_rate_recover" + organ,
                (1 - self.params["tb_proportion_casefatality_untreated" + organ])
                / self.params["tb_timeperiod_activeuntreated"])

        # Rates of detection and failure of detection
        self.set_param(
            "program_rate_detect",
            1. / self.params["tb_timeperiod_activeuntreated"]
            / (1. - self.params["program_proportion_detect"]))
        # ( formula derived from CDR = (detection rate) / (detection rate and spontaneous resolution rates) )
        self.set_param(
            "program_rate_missed",
            self.params["program_rate_detect"]
            * (1. - self.params["program_algorithm_sensitivity"])
            / self.params["program_algorithm_sensitivity"])
        # ( formula derived from (algorithm sensitivity) = (detection rate) / (detection rate and miss rate) )

        # Temporarily set programmatic rates equal for all strains
        for strain in self.strains:
            self.set_param(
                "program_rate_detect" + strain,
                self.params["program_rate_detect"])
            self.set_param(
                "program_rate_missed" + strain,
                self.params["program_rate_missed"])
            self.set_param(
                "program_rate_start_treatment" + strain,
                self.params["program_rate_start_treatment"])
            self.set_param(
                "program_rate_restart_presenting" + strain,
                self.params["program_rate_restart_presenting"])

        self.set_treatment_flow_rates()

    def calculate_vars(self):
        self.vars["population"] = sum(self.compartments.values())

        self.vars["rate_birth"] = \
            self.params["demo_rate_birth"] * self.vars["population"]
        self.vars["births_unvac"] = \
            self.params["program_prop_unvac"] * self.vars["rate_birth"]
        self.vars["births_vac"] = \
            self.params["program_prop_vac"] * self.vars["rate_birth"]

        self.calculate_force_infection()

    def set_flows(self):
        for comorbidity in self.comorbidities:
            self.set_var_entry_rate_flow(
                "susceptible_fully" + comorbidity, "births_unvac")
            self.set_var_entry_rate_flow(
                "susceptible_vac" + comorbidity, "births_vac")

            for strain in self.strains:
                self.set_var_transfer_rate_flow(
                    "susceptible_fully" + comorbidity,
                    "latent_early" + strain + comorbidity,
                    "rate_force" + strain)
                self.set_var_transfer_rate_flow(
                    "susceptible_vac" + comorbidity,
                    "latent_early" + strain + comorbidity,
                    "rate_force_weak" + strain)
                self.set_var_transfer_rate_flow(
                    "susceptible_treated" + comorbidity,
                    "latent_early" + strain + comorbidity,
                    "rate_force_weak" + strain)
                self.set_var_transfer_rate_flow(
                    "latent_late" + comorbidity,
                    "latent_early" + strain + comorbidity,
                    "rate_force_weak" + strain)

                self.set_fixed_transfer_rate_flow(
                    "latent_early" + strain + comorbidity,
                    "latent_late" + strain + comorbidity,
                    "tb_rate_stabilise")

                for organ in self.organ_status:
                    self.set_fixed_transfer_rate_flow(
                        "latent_early" + strain + comorbidity,
                        "active" + organ + strain + comorbidity,
                        "tb_rate_early_progression" + organ)
                    self.set_fixed_transfer_rate_flow(
                        "latent_late" + strain + comorbidity,
                        "active" + organ + strain + comorbidity,
                        "tb_rate_late_progression" + organ)
                    self.set_fixed_transfer_rate_flow(
                        "active" + organ + strain + comorbidity,
                        "latent_late" + strain + comorbidity,
                        "tb_rate_recover" + organ)
                    self.set_fixed_transfer_rate_flow(
                        "active" + organ + strain + comorbidity,
                        "detect" + organ + strain + comorbidity,
                        "program_rate_detect")
                    self.set_fixed_transfer_rate_flow(
                        "active" + organ + strain + comorbidity,
                        "missed" + organ + strain + comorbidity,
                        "program_rate_missed")
                    self.set_fixed_transfer_rate_flow(
                        "detect" + organ + strain + comorbidity,
                        "treatment_infect" + organ + strain + comorbidity,
                        "program_rate_start_treatment")
                    self.set_fixed_transfer_rate_flow(
                        "missed" + organ + strain + comorbidity,
                        "active" + organ + strain + comorbidity,
                        "program_rate_restart_presenting")
                    self.set_fixed_transfer_rate_flow(
                        "treatment_infect" + organ + strain + comorbidity,
                        "treatment_noninfect" + organ + strain + comorbidity,
                        "program_rate_success_infect")
                    self.set_fixed_transfer_rate_flow(
                        "treatment_infect" + organ + strain + comorbidity,
                        "active" + organ + strain + comorbidity,
                        "program_rate_default_infect")
                    self.set_fixed_transfer_rate_flow(
                        "treatment_noninfect" + organ + strain + comorbidity,
                        "active" + organ + strain + comorbidity,
                        "program_rate_default_noninfect")
                    self.set_fixed_transfer_rate_flow(
                        "treatment_noninfect" + organ + strain + comorbidity,
                        "susceptible_treated" + comorbidity,
                        "program_rate_success_noninfect")

                    self.set_infection_death_rate_flow(
                        "active" + organ + strain + comorbidity,
                        "tb_rate_death" + organ)
                    self.set_infection_death_rate_flow(
                        "detect" + organ + strain + comorbidity,
                        "tb_rate_death" + organ)
                    self.set_infection_death_rate_flow(
                        "treatment_infect" + organ + strain + comorbidity,
                        "program_rate_death_infect")
                    self.set_infection_death_rate_flow(
                        "treatment_noninfect" + organ + strain + comorbidity,
                        "program_rate_death_noninfect")

        # death flows
        self.set_population_death_rate("demo_rate_death")




class FullModel(BaseTbModel):
    """
    Current model
    """

    def __init__(self, input_parameters=None, input_compartments=None):

        BaseModel.__init__(self)

        if input_parameters is None:

            def get(param_set_name, param_name, prob=0.5):
                param_set = globals()[param_set_name]
                param = getattr(param_set, param_name)
                ppf = getattr(param, "ppf")
                return ppf(prob)

            input_parameters = {
                "demo_rate_birth": 20. / 1e3,
                "demo_rate_death": 1. / 65,
                "epi_proportion_cases_smearpos": 0.6,
                "epi_proportion_cases_smearneg": 0.2,
                "epi_proportion_cases_extrapul": 0.2,
                "tb_multiplier_force_smearpos": 1.,
                "tb_multiplier_force_smearneg":
                    get("default", "multiplier_force_smearneg"),
                "tb_multiplier_force_extrapul": 0.,
                "tb_n_contact":
                    get("default", "tb_n_contact"),
                "tb_proportion_early_progression":
                    get("default", "proportion_early_progression"),
                "tb_timeperiod_early_latent":
                    get("default", "timeperiod_early_latent"),
                "tb_rate_late_progression":
                    get("default", "rate_late_progression"),
                "tb_proportion_casefatality_untreated_smearpos":
                    get("default", "proportion_casefatality_active_untreated_smearpos"),
                "tb_proportion_casefatality_untreated_smearneg":
                    get("default", "proportion_casefatality_active_untreated_smearneg"),
                "tb_timeperiod_activeuntreated":
                    get("default", "timeperiod_activeuntreated"),
                "tb_multiplier_bcg_protection":
                    get("default", "multiplier_bcg_protection"),
                "program_prop_vac":
                    get("philippines", "bcg_coverage"),
                "program_prop_unvac":
                    1. - get("philippines", "bcg_coverage"),
                "program_proportion_detect":
                    get("philippines", "bcg_coverage"),
                "program_algorithm_sensitivity":
                    get("philippines", "algorithm_sensitivity"),
                "program_rate_start_treatment":
                    1. / get("philippines", "program_timeperiod_delayto_treatment"),
                "tb_timeperiod_treatment":
                    get("default", "timeperiod_treatment_ds"),
                "tb_timeperiod_infect_ontreatment":
                    get("default", "timeperiod_infect_ontreatment"),
                "program_proportion_default":
                    get("philippines", "proportion_default"),
                "program_proportion_death":
                    get("philippines", "proportion_death"),
                "program_rate_restart_presenting":
                    1. / get("philippines", "timeperiod_norepresentation")
            }

        if input_compartments is None:
            input_compartments = {
                "susceptible_fully": 1e6,
                "active": 3.
            }

        self.set_input(input_parameters, input_compartments)

    def set_input(self, input_parameters, input_compartments):

        self.compartment_list = [
            "susceptible_fully",
            "susceptible_vac",
            "susceptible_treated",
            "latent_early",
            "latent_late",
            "active",
            "detect",
            "missed",
            "treatment_infect",
            "treatment_noninfect"]

        # WARNING: make sure names aren't subset of each other
        self.organs = [
            "_smearpos",
            "_smearneg",
            "_extrapul"]

        self.strains = [
            "_ds",
            "_mdr"]

        self.comorbidities = [
            "_hiv",
            "_diabetes",
            "_nocomorb"]

        self.infectious_tags = ["active", "missed", "detect", "treatment_infect"]

        self.initialise_compartments(input_compartments)

        for parameter in input_parameters:  # Set parameters from parameter module
            self.set_param(parameter, input_parameters[parameter])

        self.set_treatment_flow_rates()

        self.set_param(
            "tb_rate_stabilise",  # Calculate stabilisation rate
            (1 - self.params["tb_proportion_early_progression"])
            / self.params["tb_timeperiod_early_latent"])

        if "tb_proportion_casefatality_untreated_extrapul" not in input_parameters:
            self.set_param(
                "tb_proportion_casefatality_untreated_extrapul",
                input_parameters["tb_proportion_casefatality_untreated_smearneg"])

        self.set_param(
            "program_rate_detect",
            1. / self.params["tb_timeperiod_activeuntreated"]
            / (1. - self.params["program_proportion_detect"]))
        # Formula derived from CDR = (detection rate) / (detection rate and spontaneous resolution rates)

        self.set_param(
            "program_rate_missed",
            self.params["program_rate_detect"]
            * (1. - self.params["program_algorithm_sensitivity"])
            / self.params["program_algorithm_sensitivity"])
        # Formula derived from (algorithm sensitivity) = (detection rate) / (detection rate and miss rate)

        for strain in self.strains:  # Temporary, has to be changed
            self.set_param(
                "program_rate_detect" + strain,
                self.params["program_rate_detect"])
            self.set_param(
                "program_rate_missed" + strain,
                self.params["program_rate_missed"])
            self.set_param(
                "program_rate_start_treatment" + strain,
                self.params["program_rate_start_treatment"])
            self.set_param(
                "program_rate_restart_presenting" + strain,
                self.params["program_rate_restart_presenting"])

        for organ in self.organs:
            self.set_param(
                "tb_rate_earlyprogress" + organ,
                self.params["tb_proportion_early_progression"]
                  / self.params["tb_timeperiod_early_latent"]
                  * self.params["epi_proportion_cases" + organ])
            self.set_param(
                "tb_rate_lateprogress" + organ,
                self.params["tb_rate_late_progression"]
                * self.params["epi_proportion_cases" + organ])
            self.set_param(
                "tb_rate_recover" + organ,
                (1 - self.params["tb_proportion_casefatality_untreated" + organ])
                  / self.params["tb_timeperiod_activeuntreated"])
            self.set_param(
                "tb_demo_rate_death" + organ,
                self.params["tb_proportion_casefatality_untreated" + organ]
                  / self.params["tb_timeperiod_activeuntreated"])

    def initialise_compartments(self, input_compartments):

        # initialize to 0
        for compartment in self.compartment_list:
            if "susceptible" in compartment:  # Don't replicate
                self.set_compartment(compartment, 0.)
            elif "latent" in compartment:  # Replicate only for strains
                for strain in self.strains:
                    self.set_compartment(compartment + strain, 0.)
            else:  # Replicate for strains and organ status
                for strata in self.strata_iterator():
                    self.set_compartment(
                        self.make_strata_label(compartment, strata),
                        0.)

        for compartment in self.compartment_list:
            if compartment not in input_compartments:
                continue
            if "susceptible" in compartment:  # Don't replicate
                self.set_compartment(
                    compartment,
                    input_compartments[compartment])
            elif "latent" in compartment:  # Replicate only for strains
                for strain in self.strains:
                    self.set_compartment(
                        compartment + strain,
                        input_compartments[compartment])
            else:
                for strata in self.strata_iterator():
                    strain = strata[0]
                    if strain == "_ds":
                        self.set_compartment(
                            self.make_strata_label(compartment, strata),
                            input_compartments[compartment] / 3.)
                    else:
                        self.set_compartment(
                            self.make_strata_label(compartment, strata),
                            0.)

    def calculate_vars(self):
        self.vars["population"] = sum(self.compartments.values())

        self.vars["rate_birth"] = \
            self.params["demo_rate_birth"] * self.vars["population"]
        self.vars["births_unvac"] = \
            self.params["program_prop_unvac"] * self.vars["rate_birth"]
        self.vars["births_vac"] = \
            self.params["program_prop_vac"] * self.vars["rate_birth"]

        # Force of infection calculation
        for strain in self.strains:
            self.vars["infectious_population" + strain] = 0.0
            for organ in self.organs:
                for label in self.labels:
                    if strain not in label:
                        continue
                    if organ not in label:
                        continue
                    if not label_intersects_tags(label, self.infectious_tags):
                        continue
                    self.vars["infectious_population" + strain] += \
                        self.params["tb_multiplier_force" + organ] \
                        * self.compartments[label]
            self.vars["rate_force" + strain] = \
                self.params["tb_n_contact"] \
                  * self.vars["infectious_population" + strain] \
                  / self.vars["population"]
            self.vars["rate_force_weak" + strain] = \
                self.params["tb_multiplier_bcg_protection"] \
                  * self.vars["rate_force" + strain]

    def set_flows(self):
        self.set_var_entry_rate_flow(
            "susceptible_fully", "births_unvac")
        self.set_var_entry_rate_flow(
            "susceptible_vac", "births_vac")

        for strain in self.strains:
            self.set_var_transfer_rate_flow(
                "susceptible_fully",
                "latent_early" + strain,
                "rate_force" + strain)
            self.set_var_transfer_rate_flow(
                "susceptible_vac",
                "latent_early" + strain,
                "rate_force_weak" + strain)
            self.set_var_transfer_rate_flow(
                "susceptible_treated",
                "latent_early" + strain,
                "rate_force_weak" + strain)
            self.set_var_transfer_rate_flow(
                "latent_late" + strain,
                "latent_early" + strain,
                "rate_force_weak" + strain)

            self.set_fixed_transfer_rate_flow(
                "latent_early" + strain,
                "latent_late" + strain,
                "tb_rate_stabilise")

        for strata in self.strata_iterator():
            strain, organ = strata[0:2]
            self.set_fixed_transfer_rate_flow(
                "latent_early" + strain,
                self.make_strata_label("active", strata),
                "tb_rate_earlyprogress" + organ)
            self.set_fixed_transfer_rate_flow(
                "latent_late" + strain,
                self.make_strata_label("active", strata),
                "tb_rate_lateprogress" + organ)
            self.set_fixed_transfer_rate_flow(
                self.make_strata_label("active", strata),
                "latent_late" + strain,
                "tb_rate_recover" + organ)

            self.set_fixed_transfer_rate_flow(
                self.make_strata_label("active", strata),
                self.make_strata_label("detect", strata),
                "program_rate_detect" + strain)
            self.set_fixed_transfer_rate_flow(
                self.make_strata_label("active", strata),
                self.make_strata_label("missed", strata),
                "program_rate_missed" + strain)
            self.set_fixed_transfer_rate_flow(
                self.make_strata_label("detect", strata),
                self.make_strata_label("treatment_infect", strata),
                "program_rate_start_treatment" + strain)
            self.set_fixed_transfer_rate_flow(
                self.make_strata_label("missed", strata),
                self.make_strata_label("active", strata),
                "program_rate_restart_presenting" + strain)

            self.set_fixed_transfer_rate_flow(
                self.make_strata_label("missed", strata),
                "latent_late" + strain,
                "tb_rate_recover" + organ)
            self.set_fixed_transfer_rate_flow(
                self.make_strata_label("treatment_infect", strata),
                self.make_strata_label("treatment_noninfect", strata),
                "program_rate_success_infect" + strain)
            self.set_fixed_transfer_rate_flow(
                self.make_strata_label("treatment_infect", strata),
                self.make_strata_label("active", strata),
                "program_rate_default_infect" + strain)
            self.set_fixed_transfer_rate_flow(
                self.make_strata_label("treatment_noninfect", strata),
                self.make_strata_label("active", strata),
                "program_rate_default_noninfect" + strain)
            self.set_fixed_transfer_rate_flow(
                self.make_strata_label("treatment_noninfect", strata),
                "susceptible_treated",
                "program_rate_success_noninfect" + strain)

        # death flows
        self.set_population_death_rate("demo_rate_death")

        # Also will need changing
        for strata in self.strata_iterator():
            strain, organ = strata[0:2]
            self.set_infection_death_rate_flow(
                self.make_strata_label("active", strata),
                "tb_demo_rate_death" + organ)
            self.set_infection_death_rate_flow(
                self.make_strata_label("detect", strata),
                "tb_demo_rate_death" + organ)
            self.set_infection_death_rate_flow(
                self.make_strata_label("treatment_infect", strata),
                "program_rate_death_infect" + strain)
            self.set_infection_death_rate_flow(
                self.make_strata_label("treatment_noninfect", strata),
                "program_rate_death_noninfect" + strain)

