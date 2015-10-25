# -*- coding: utf-8 -*-


"""

Building up an object oriented disease spread model

time: years

"""


from scipy.integrate import odeint



def make_steps(start, end, delta_step):
    steps = []
    step = start
    while step < end:
        steps.append(step)
        step += delta_step
    return steps


class PopulationSystem():

    def __init__(self):
        self.labels = []
        self.init_compartments = {}
        self.flows = {}
        self.vars = {}
        self.params = {}

    def set_compartment(self, label, init_val = 0.0):
        self.labels.append(label)
        self.init_compartments[label] = init_val

    def set_param(self, label, val):
        self.params[label] = val
        assert val >= 0  # Ensure each individual parameter is positive

    def convert_list_to_compartments(self, vec):
        return {l: vec[i] for i, l in enumerate(self.labels)}

    def convert_compartments_to_list(self, compartments):
        return [compartments[l] for l in self.labels]

    def get_init_list(self):
        return self.convert_compartments_to_list(self.init_compartments)

    def calculate_vars(self):
        self.vars["pop_total"] = sum(self.compartments.values())

        self.vars["rate_forceinfection"] = \
            self.params["n_tbfixed_contact"] \
              * self.compartments["active"] \
              / self.vars["pop_total"]

        self.vars["births"] = \
            self.params["rate_pop_birth"] * self.vars["pop_total"]

        self.vars["deaths"] = \
            self.params["rate_tbfixed_death"] \
                * self.compartments["active"] \
            + self.params["rate_tbprog_death"] \
                * self.compartments["undertreatment"] \
            + self.params["rate_pop_death"] \
                * self.vars["pop_total"]

    def calculate_susceptible_flows(self):
        self.flows["susceptible"] = \
            self.vars["births"] \
            + self.compartments["undertreatment"] \
                * self.params["rate_tbprog_completion"] \
            - self.compartments["susceptible"] \
                * ( self.vars["rate_forceinfection"] \
                    + self.params["rate_pop_death"])

    def calculate_latent_flows(self):
        self.flows["latent_early"] = \
            self.compartments["susceptible"] * self.vars["rate_forceinfection"] \
            - self.compartments["latent_early"] \
                * (self.params["rate_tbfixed_earlyprog"] \
                    + self.params["rate_tbfixed_stabilise"] \
                    + self.params["rate_pop_death"])

        self.flows["latent_late"] = \
            self.compartments["latent_early"] * self.params["rate_tbfixed_stabilise"] \
            + self.compartments["active"] * self.params["rate_tbfixed_recover"] \
            - self.compartments["latent_late"] \
                * (self.params["rate_tbfixed_lateprog"] 
                    + self.params["rate_pop_death"]) 

    def calculate_active_flows(self):
        self.flows["active"] = \
            self.compartments["latent_early"] \
                * self.params["rate_tbfixed_earlyprog"] \
            + self.compartments["latent_late"] \
                * self.params["rate_tbfixed_lateprog"] \
            + self.compartments["undertreatment"] \
                * self.params["rate_tbprog_default"] \
            - self.compartments["active"] \
                * (   self.params["rate_tbprog_detect"] \
                    + self.params["rate_tbfixed_recover"] \
                    + self.params["rate_tbfixed_death"] \
                    + self.params["rate_pop_death"])

    def calculate_undertreatment_flows(self):
        self.flows["undertreatment"] = \
            self.compartments["active"] * self.params["rate_tbprog_detect"] \
            - self.compartments["undertreatment"] \
                * (   self.params["rate_tbprog_default"] \
                    + self.params["rate_tbprog_death"] \
                    + self.params["rate_pop_death"] \
                    + self.params["rate_tbprog_completion"])

    def calculate_flows(self):
        self.calculate_susceptible_flows()
        self.calculate_latent_flows()
        self.calculate_active_flows()
        self.calculate_undertreatment_flows()

    def checks(self):
        for label in self.labels:  # Check all compartments are positive
            assert self.compartments[label] >= 0.0
        # Check total flows sum (approximately) to zero
        assert abs(sum(self.flows.values()) + self.vars['deaths'] - self.vars['births'])  < 0.1
        
    def integrate(self, times):

        def derivative_fn(y, t):
            self.time = t
            self.compartments = self.convert_list_to_compartments(y)
            self.vars = {}
            self.calculate_vars()
            self.flows = {}
            self.calculate_flows()
            flow_vector = self.convert_compartments_to_list(self.flows)
            self.checks()
            return flow_vector

        self.times = times
        init_y = self.get_init_list()
        self.soln = odeint(derivative_fn, init_y, times)

    def get_soln(self, label):
        i_label = self.labels.index(label)
        return self.soln[:, i_label]

        
    



