# -*- coding: utf-8 -*-


"""

Building up a library for disease spread model

time: years

"""

import math
from scipy.integrate import odeint
import matplotlib.pylab as pylab


class PopulationSystem():

    def __init__(self):
        self.labels = []
        self.init_compartments = {}
        self.flows = {}
        self.temp_vars = {}
        self.params = {}

    def set_compartment(self, label, init_val=0.0):
        self.labels.append(label)
        self.init_compartments[label] = init_val

    def set_param(self, label, val):
        self.params[label] = val

    def convert_list_to_compartments(self, vec):
        return {l:vec[i] for i, l in enumerate(self.labels)}

#        result = {}
#        n = len(self.labels)
#        for i in range(n):
#            key = self.labels[i]
#            val = vec[i]
#            result[key] = val
#        
#        return result


    def convert_compartments_to_list(self, compartments):
        return [compartments[l] for l in self.labels]

#        result = []
#        for l in self.labels:
#            c = compartments[l]
#            result.append(c)
#            
#        return result
        
    def get_init_list(self):
        return self.convert_compartments_to_list(self.init_compartments)

    def make_derivative_fn(self):

        def derivative_fn(y, t):
            self.calculate_flows(y)
            return self.convert_compartments_to_list(self.flows)

        return derivative_fn

    def integrate(self, times):
        self.times = times
        derivative_fn = self.make_derivative_fn()
        init_y = self.get_init_list()
        self.soln = odeint(derivative_fn, init_y, times)

    def make_time_plots(self, plot_labels, png=None):
        n_row = int(math.ceil(len(plot_labels) / 2.))

        for i_plot, plot_label in enumerate(plot_labels):
            i_label = self.labels.index(plot_label)
            vals = self.soln[:,i_label]
            pylab.subplot(n_row, 2, i_plot+1)
            pylab.plot(self.times, vals)
            pylab.ylabel(self.labels[i_label])

        pylab.xlabel('time')
        pylab.tight_layout()

        if png is None:
            pylab.show()
        else:
            pylab.savefig(png)

    def get_optima_compatible_soln(self):
        return self.soln[0,:]

    def calculate_temp_vars(self):
        self.temp_vars = {}
    
        self.temp_vars["pop_total"] = sum(self.compartments.values())
    
        self.temp_vars["rate_infection_multiplier"] = \
            self.params["n_infection_contact"] * self.compartments["infectious"] \
            / self.temp_vars["pop_total"]
        
    def calculate_latent_flows(self):
        self.flows["early_latents"] = \
            self.compartments["susceptibles"] * self.temp_vars["rate_infection_multiplier"]   \
            - self.compartments["early_latents"] \
                * (   self.params["rate_infection_early_progress"]  \
                    + self.params["rate_infection_stabilise"] \
                    + self.params["rate_pop_death"])
              
        self.flows["late_latents"] = \
            self.compartments["early_latents"] * self.params["rate_infection_stabilise"]  \
            + self.compartments["infectious"] * self.params["rate_infection_spont_recover"] \
            - self.compartments["late_latents"] \
                * (   self.params["rate_infection_late_progress"] 
                    + self.params["rate_pop_death"]) 

        
    def calculate_flows(self, y):
        self.compartments = self.convert_list_to_compartments(y)

        self.calculate_temp_vars()
    
        self.flows = {}
    
        self.flows["susceptibles"] = \
            self.params["rate_pop_birth"] * self.temp_vars["pop_total"] \
            + self.compartments["late_latents"] \
                * self.params["rate_treatment_completion"]  \
            - self.compartments["susceptibles"] \
                * (   self.temp_vars["rate_infection_multiplier"] \
                    + self.params["rate_pop_death"]) 
            
        self.calculate_latent_flows()
        
        self.flows["infectious"] = \
            self.compartments["early_latents"] \
                * self.params["rate_infection_early_progress"]   \
            + self.compartments["late_latents"] \
                * self.params["rate_infection_late_progress"]   \
            - self.compartments["infectious"] \
                * (   self.params["rate_treatment_detect"] \
                    + self.params["rate_infection_spont_recover"]  \
                    + self.params["rate_infection_death"]  \
                    + self.params["rate_pop_death"]) 
                
        self.flows["under_treatment"] = \
            self.compartments["infectious"] * self.params["rate_treatment_detect"]   \
            - self.compartments["under_treatment"] \
                * (   self.params["rate_treatment_default"] \
                    + self.params["rate_treatment_death"]  \
                    + self.params["rate_pop_death"] \
                    + self.params["rate_treatment_completion"])
           
        total_flow = 0.0
        for label in self.labels:
            total_flow = total_flow + self.flows[label]
            
        assert total_flow == 0.0

def make_times(start, end, step):
    times = []
    time = start
    while time < end:
        times.append(time)
        time += step
    return times




population = PopulationSystem()

population.set_compartment("susceptibles", 1e6)
population.set_compartment("early_latents", 0.)
population.set_compartment("late_latents", 0.)
population.set_compartment("infectious", 1.)
population.set_compartment("under_treatment", 0.)

population.set_param("rate_pop_birth", 20. / 1e3)
population.set_param("rate_pop_death", 1. / 65)
    
population.set_param("n_infection_contact", 10.)
population.set_param("rate_infection_early_progress", .1 / .5)
population.set_param("rate_infection_late_progress", .1 / 100.)
population.set_param("rate_infection_stabilise", .9 / .5)
population.set_param("rate_infection_spont_recover", .6 / 3.)
population.set_param("rate_infection_death", .4 / 3.)
    
time_treatment = .5
population.set_param("rate_treatment_detect", 1.)
population.set_param("time_treatment", time_treatment)
population.set_param("rate_treatment_completion", .9 / time_treatment)
population.set_param("rate_treatment_default", .05 / time_treatment)
population.set_param("rate_treatment_death", .05 / time_treatment)

times = make_times(0, 50, 5)
population.integrate(times)

labels = population.labels
population.make_time_plots(labels)



population2 = PopulationSystem()

population2.set_compartment("susceptibles", 1e6)
population2.set_compartment("early_latents", 0.)
population2.set_compartment("late_latents", 0.)
population2.set_compartment("infectious", 1.)
population2.set_compartment("under_treatment", 0.)

population2.set_param("rate_pop_birth", 20. / 1e3)
population2.set_param("rate_pop_death", 1. / 65)
    
population2.set_param("n_infection_contact", 10.)
population2.set_param("rate_infection_early_progress", .1 / .5)
population2.set_param("rate_infection_late_progress", .1 / 100.)
population2.set_param("rate_infection_stabilise", .9 / .5)
population2.set_param("rate_infection_spont_recover", .6 / 3.)
population2.set_param("rate_infection_death", .4 / 3.)
    
time_treatment = .23
population2.set_param("rate_treatment_detect", 1.)
population2.set_param("time_treatment", time_treatment)
population2.set_param("rate_treatment_completion", .9 / time_treatment)
population2.set_param("rate_treatment_default", .05 / time_treatment)
population2.set_param("rate_treatment_death", .05 / time_treatment)

times = make_times(0, 50, 2)
population2.integrate(times)

labels = population2.labels
population2.make_time_plots(labels, 'output.png')


print population.get_optima_compatible_soln()
print population2.get_optima_compatible_soln()