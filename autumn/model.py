# -*- coding: utf-8 -*-


"""

Base Population Model to handle different type of models.

Implicit time unit: years

"""


from scipy.integrate import odeint
import numpy
import copy


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
        pass

    def calculate_flows(self):
        pass

    def checks(self):
        pass

    def make_derivate_fn(self):
        
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

        return derivative_fn

    def integrate_scipy(self, times):
        derivative = self.make_derivate_fn()
        self.times = times
        init_y = self.get_init_list()
        self.soln = odeint(derivative, init_y, times)

    def integrate_explicit(self, times):
        self.times = times
        y = self.get_init_list()

        n_component = len(y)
        n_time = len(self.times)
        self.soln = numpy.zeros((n_time, n_component))

        derivative = self.make_derivate_fn()
        time = self.times[0]
        self.soln[0,:] = y
        min_dt = 0.05
        for i_time, new_time in enumerate(self.times):
            while time < new_time:
                f = derivative(y, time)
                time = time + min_dt
                if time > new_time:
                    time = new_time
                    dt = new_time - time
                else:
                    dt = min_dt
                for i in range(n_component):
                    y[i] = y[i] + dt*f[i]
            if i_time < n_time - 1:
                self.soln[i_time+1,:] = y

    def calculate_fractions(self):
        solns = {}
        for label in self.labels:
            if label not in solns:
                solns[label] = self.get_soln(label)

        self.total = []
        n = len(self.times)
        for i in range(n):
            t = 0.0
            for label in self.labels:
                t += solns[label][i]
            self.total.append(t)

        self.fractions = {}
        for label in self.labels:
            fraction = [v/t for v, t in zip(solns[label], self.total)]
            self.fractions[label] = fraction

    def get_soln(self, label):
        i_label = self.labels.index(label)
        return self.soln[:, i_label]


        
class SimplePopluationSystem(BasePopulationSystem):

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

        self.set_param("birth", 0.02)
        self.set_param("death", 1/70.)
        self.set_param("tbdeath", 0.2)
        self.set_param("treatdeath", 0.1)
        self.set_param("ncontacts", 5.)
        self.set_param("progress", 0.005)
        self.set_param("treat", 0.3)
        self.set_param("recov", 2.)
        self.set_param("test", 4.)

    def calculate_vars(self):
        self.vars = {}

        self.vars["population"] = sum(self.compartments.values())

        self.vars["nbirths"] = self.vars["population"] * self.params['birth']

        self.vars["ninfections"] = \
              self.compartments["susceptible"] \
            * (self.compartments["active"] + self.compartments["detected"]) \
            / self.vars["population"] \
            * self.params['ncontacts']

        self.vars["nrecovered"] = self.compartments["treated"] * self.params['recov']

        self.vars["nprogress"] = self.compartments["latent"] * self.params['progress']

        self.vars["ndetected"] = self.compartments["active"] * self.params['test']
        self.vars["activedeaths"] = self.compartments["active"] * self.params['tbdeath']
        self.vars["ntbdeaths"] = self.vars["activedeaths"]

        self.vars["ntreated"] = self.compartments["detected"] * self.params['treat']
        self.vars["detecteddeaths"] = self.compartments["detected"] * self.params['tbdeath']
        self.vars["ntbdeaths"] += self.vars["detecteddeaths"]

    def calculate_flows(self):
        self.flows['susceptible'] = \
            - self.compartments["susceptible"] * self.params['death'] \
            + self.vars["nbirths"] \
            - self.vars["ninfections"] \
            + self.vars["nrecovered"] \
        
        self.flows['latent'] = \
            - self.compartments["latent"] * self.params['death'] \
            + self.vars["ninfections"] \
            - self.vars["nprogress"]
        
        self.flows['active'] = \
              self.vars["nprogress"] \
            - self.vars["activedeaths"] \
            - self.vars["ndetected"]
        
        self.flows['detected'] = \
              self.vars["ndetected"] \
            - self.vars["detecteddeaths"] \
            - self.vars["ntreated"]
        
        self.flows['treated'] = \
            - self.compartments["treated"] * self.params['treatdeath'] \
            + self.vars["ntreated"] \
            - self.vars["nrecovered"]

    def checks(self):
        # Check all compartments are positive
        for label in self.labels:  
            assert self.compartments[label] >= 0.0



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

        self.set_param("rate_pop_birth", 20. / 1e3)
        self.set_param("rate_pop_death", 1. / 65)

        self.set_param("n_tbfixed_contact", 10.)
        self.set_param("rate_tbfixed_earlyprog", .1 / .5)
        self.set_param("rate_tbfixed_lateprog", .1 / 100.)
        self.set_param("rate_tbfixed_stabilise", .9 / .5)
        self.set_param("rate_tbfixed_recover", .6 / 3.)
        self.set_param("rate_tbfixed_death", .4 / 3.)

        time_treatment = .5
        self.set_param("rate_tbprog_detect", 1.)
        self.set_param("time_treatment", time_treatment)
        self.set_param("rate_tbprog_completion", .9 / time_treatment)
        self.set_param("rate_tbprog_default", .05 / time_treatment)
        self.set_param("rate_tbprog_death", .05 / time_treatment)

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

    def calculate_flows(self):
        self.flows["susceptible"] = \
            + self.vars["births"] \
            + self.compartments["undertreatment"] \
                * self.params["rate_tbprog_completion"] \
            - self.compartments["susceptible"] \
                * (   self.vars["rate_forceinfection"] \
                    + self.params["rate_pop_death"])

        self.flows["latent_early"] = \
            + self.compartments["susceptible"] * self.vars["rate_forceinfection"] \
            - self.compartments["latent_early"] \
                * (   self.params["rate_tbfixed_earlyprog"] \
                    + self.params["rate_tbfixed_stabilise"] \
                    + self.params["rate_pop_death"])

        self.flows["latent_late"] = \
            + self.compartments["latent_early"] * self.params["rate_tbfixed_stabilise"] \
            + self.compartments["active"] * self.params["rate_tbfixed_recover"] \
            - self.compartments["latent_late"] \
                * (self.params["rate_tbfixed_lateprog"] 
                    + self.params["rate_pop_death"]) 

        self.flows["active"] = \
            + self.compartments["latent_early"] \
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

        self.flows["undertreatment"] = \
            + self.compartments["active"] * self.params["rate_tbprog_detect"] \
            - self.compartments["undertreatment"] \
                * (   self.params["rate_tbprog_default"] \
                    + self.params["rate_tbprog_death"] \
                    + self.params["rate_pop_death"] \
                    + self.params["rate_tbprog_completion"])

    def checks(self):
        for label in self.labels:  # Check all compartments are positive
            assert self.compartments[label] >= 0.0

        # Check total flows sum (approximately) to zero
        assert abs(sum(self.flows.values()) + self.vars['deaths'] - self.vars['births'])  < 0.1






