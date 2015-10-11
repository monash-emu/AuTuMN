# -*- coding: utf-8 -*-


"""

Building up an object oriented disease spread model

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
        self.tracked_vars = {}
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
        n_col=2

        for i_plot, plot_label in enumerate(plot_labels):
            i_label = self.labels.index(plot_label)
            vals = self.soln[:, i_label]
            pylab.subplot(n_row, n_col, i_plot+1)
            pylab.plot(self.times, vals, linewidth=2)
            pylab.ylabel(self.labels[i_label])
            pylab.xlabel('time')
            pylab.tight_layout()
        
        if png is None:
            pylab.show()
        else:
            pylab.savefig(png)
            
    def make_time_plots_color(self, plot_labels, png=None):
        n_row = int(math.ceil(len(plot_labels) / 2.))
        n_col=2

        for i_plot, plot_label in enumerate(plot_labels):
            i_label = self.labels.index(plot_label)
            vals = self.soln[:, i_label]
            colors = ('r', 'b', 'm', 'g', 'k') 
            for row in range(n_row):
                for col in range(n_col):
                    pylab.subplot(n_row, n_col, i_plot+1)
                    color = colors[i_plot % len(colors)]
                    pylab.plot(self.times, vals, linewidth=2, color=color)
        
        pylab.xlabel('time')
        pylab.tight_layout()

        if png is None:
            pylab.show()
        else:
            pylab.savefig(png)
            
#Plot all compartments in one panel     
    def make_time_plots_one_panel(self, plot_labels, png=None):
        for i_plot, plot_label in enumerate(plot_labels):
            i_label = self.labels.index(plot_label)
            print (i_label)
        pylab.subplot(211)
        pylab.plot(self.times,self.soln[:,0], '-r', label = 'Susceptible', linewidth=2)
        pylab.plot(self.times,self.soln[:,1], '-b', label = 'Latent_early', linewidth=2) 
        pylab.plot(self.times,self.soln[:,2], '-m', label = 'Latent_late', linewidth=2) 
        pylab.plot(self.times,self.soln[:,3], '-g', label = 'Active', linewidth=2) 
        pylab.plot(self.times,self.soln[:,4], '-k', label = 'Under treatment', linewidth=2) 
        pylab.xlabel('Time')
        pylab.ylabel('Number of patients')
        pylab.legend(loc=0)
        pylab.show()
        
        pylab.subplot(212)
        pylab.plot(self.times,self.soln[:,1], '-b', label = 'Latent_early', linewidth=2) 
        pylab.plot(self.times,self.soln[:,2], '-m', label = 'Latent_late', linewidth=2) 
        pylab.plot(self.times,self.soln[:,3], '-g', label = 'Active', linewidth=2) 
        pylab.plot(self.times,self.soln[:,4], '-k', label = 'Under treatment', linewidth=2) 
        pylab.xlabel('Time')
        pylab.ylabel('Number of patients')
        pylab.legend(loc=0)
        pylab.show()
        
    def get_optima_compatible_soln(self):
        return self.soln[0, :]

    def calculate_tracked_vars(self):
        self.tracked_vars = {}

        self.tracked_vars["pop_total"] = sum(self.compartments.values())

        self.tracked_vars["rate_forceinfection"] = \
            self.params["n_tbfixed_contact"] * self.compartments["active"] \
            / self.tracked_vars["pop_total"]

    def calculate_births_flows(self):
        self.flows["births"] = \
            self.params["rate_pop_birth"] * self.tracked_vars["pop_total"]

    def calculate_deaths_flows(self):
        self.flows["deaths"] = \
            self.params["rate_tbfixed_death"] * self.compartments["active"] \
            + self.params["rate_tbprog_death"] * self.compartments["undertreatment"] \
            + self.params["rate_pop_death"] * ( self.compartments["susceptible"] \
                    + self.compartments["latent_early"]
                    + self.compartments["latent_late"])

    def calculate_susceptible_flows(self):
        self.flows["susceptible"] = \
            self.flows["births"] \
            + self.compartments["undertreatment"] \
            * self.params["rate_tbprog_completion"] \
            - self.compartments["susceptible"] \
                * ( self.tracked_vars["rate_forceinfection"] \
                    + self.params["rate_pop_death"])

    def calculate_latent_flows(self):
        self.flows["latent_early"] = \
            self.compartments["susceptible"] * self.tracked_vars["rate_forceinfection"] \
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
                * (self.params["rate_tbprog_detect"] \
                    + self.params["rate_tbfixed_recover"] \
                    + self.params["rate_tbfixed_death"] \
                    + self.params["rate_pop_death"])

    def calculate_undertreatment_flows(self):
        self.flows["undertreatment"] = \
            self.compartments["active"] * self.params["rate_tbprog_detect"] \
            - self.compartments["undertreatment"] \
                * (self.params["rate_tbprog_default"] \
                    + self.params["rate_tbprog_death"] \
                    + self.params["rate_pop_death"] \
                    + self.params["rate_tbprog_completion"])

    def calculate_flows(self, y):
        self.compartments = self.convert_list_to_compartments(y)

        self.calculate_tracked_vars()
        self.flows = {}
        self.calculate_births_flows()
        self.calculate_deaths_flows()
        self.calculate_susceptible_flows()
        self.calculate_latent_flows()
        self.calculate_active_flows()
        self.calculate_undertreatment_flows()

        self.checks()

    def checks(self):
        for label in self.labels:  # Check all compartments are positive
            assert self.compartments[label] >= 0.0
            #assert sum(self.flows.values())==0.0 
        # Check total flows sum (approximately) to zero
        assert abs((sum(self.flows.values()) - 2 * self.flows["births"])) < 0.1
        
        
def make_times(start, end, step):
    times = []
    time = start
    while time < end:
        times.append(time)
        time += step
    return times


population = PopulationSystem()

population.set_compartment("susceptible", 1e6)
population.set_compartment("latent_early", 0.)
population.set_compartment("latent_late", 0.)
population.set_compartment("active", 1.)
population.set_compartment("undertreatment", 0.)

population.set_param("rate_pop_birth", 20. / 1e3)
population.set_param("rate_pop_death", 1. / 65)

population.set_param("n_tbfixed_contact", 10.)
population.set_param("rate_tbfixed_earlyprog", .1 / .5)
population.set_param("rate_tbfixed_lateprog", .1 / 100.)
population.set_param("rate_tbfixed_stabilise", .9 / .5)
population.set_param("rate_tbfixed_recover", .6 / 3.)
population.set_param("rate_tbfixed_death", .4 / 3.)

time_treatment = .5
population.set_param("rate_tbprog_detect", 1.)
population.set_param("time_treatment", time_treatment)
population.set_param("rate_tbprog_completion", .9 / time_treatment)
population.set_param("rate_tbprog_default", .05 / time_treatment)
population.set_param("rate_tbprog_death", .05 / time_treatment)

times = make_times(0, 50, 1)
population.integrate(times)

labels = population.labels
population.make_time_plots(labels)
population.make_time_plots_color(labels)
population.make_time_plots_one_panel(labels)

print (population.get_optima_compatible_soln())
