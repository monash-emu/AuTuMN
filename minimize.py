# -*- coding: utf-8 -*-

"""
Tests out the Parameter object in parameter_estimation.py (which is currently
just a prior setting object) and the Evidence object (which isn't yet used
within the Parameter object).
@author: James
"""

import os

import numpy
import pylab
import emcee
import scipy.optimize as optimize
from numpy import isfinite
from scipy.stats import beta, gamma, norm, truncnorm

from stage4 import Stage4PopulationSystem
from autumn.plotting import plot_fractions, plot_populations


def make_gamma_dist(mean, std):
    loc = 0
    upper_limit = 'No upper limit'
    shape = mean ** 2 / std ** 2  
    scale = std ** 2 / mean
    return gamma(shape, loc, scale)


class Sampler():

    def __init__(self):
        self.population = Stage4PopulationSystem()
        self.population.make_steps(0, 100, 1.)
        self.param_dict = [
            { 
                'init': 40,
                'scale': 10.,
                'title': 'n_tb_contact',
            },
            { 
                'init': 80E6,
                'scale': 100E6,
                'title': 'init_population',
            },
            { 
                'init': 0.2,
                'scale': .1,
                'title': 'tb_rate_early',
            },
        ]
        self.n_param = len(self.param_dict)
        self.n_walker = 2 * self.n_param

    def run_with_params(self, params):

        n_tb_contact, init_pop, tb_rate_early = params
        if init_pop < 0 or not isfinite(init_pop):
            self.sucess = False
            return

        self.population.set_param(
            "tb_n_contact", n_tb_contact)
        self.population.set_compartment(
            "susceptible_fully", init_pop)
        self.population.set_param(
            "tb_rate_earlyprogress", tb_rate_early)
        for status in self.population.pulmonary_status:
            self.population.set_param(
                "tb_rate_earlyprogress" + status,
                tb_rate_early * self.population.params["proportion_cases" + status])

        self.success = True
        try:
            self.population.integrate_explicit()
        except:
            self.sucess = False


    def ln_overall(self, params):
        self.run_with_params(params)
        if not self.success:
            return -numpy.inf

        n_tb_contact, init_pop, tb_rate_early = params
        if init_pop < 0 or not isfinite(init_pop):
            print "start_pop=error"
            return -numpy.inf
        final_pop = self.population.vars["population"]
        prevalence = self.population.vars["infected_populaton"]/final_pop*1E5
        mortality = self.population.vars["rate_infection_death"]/final_pop*1E5
        incidence = self.population.vars["rate_incidence"]/final_pop*1E5

        ln_prior = 0.0
        ln_prior += make_gamma_dist(40, 20).logpdf(n_tb_contact)
        ln_prior += norm(0.2, 0.5).logpdf(tb_rate_early)

        ln_posterior = 0.0
        ln_posterior += 5*norm(99E6, 40E6).logpdf(final_pop)
        ln_posterior += norm(417, 100).logpdf(prevalence)
        ln_posterior += norm(288, 50).logpdf(incidence)
        ln_posterior += norm(10, 10).logpdf(mortality)

        ln_overall = ln_prior + ln_posterior

        prints = [
           ("n={:.0f}", n_tb_contact),
           ("start_pop={:0.0f}", init_pop),
           ("final_pop={:0.0f}", final_pop),
           ("tb_rate_early={:0.4f}", tb_rate_early),
           ("prev={:0.0f}", prevalence),
           ("inci={:0.0f}", incidence),
           ("mort={:0.0f}", mortality),
           ("-lnprob={:0.2f}", -ln_overall),
        ]
        s = " ".join(p[0] for p in prints)
        print s.format(*[p[1] for p in prints])

        return ln_overall

    def scale_params(self, params):
        return [
            param / d['scale'] 
            for param, d in 
            zip(params, self.param_dict)]

    def revert_scaled_params(self, scaled_params):
        return [
            scaled_param * d['scale'] 
            for scaled_param, d 
            in zip(scaled_params, self.param_dict)]

    def minimize(self):

        def scaled_min_fn(scaled_params):
            params = self.revert_scaled_params(scaled_params)
            return -self.ln_overall(params)
        
        n_param = len(self.param_dict)
        init_params = [d['init'] for d in self.param_dict]
        scaled_init_params = self.scale_params(init_params)

        # select the TNC method which is a basic method that
        # allows bouns in the search
        bnds = [(0, None) for i in range(n_param)]
        self.minimum = optimize.minimize(
            scaled_min_fn, scaled_init_params, 
            method='TNC', bounds=bnds)

        self.minimum.best = self.revert_scaled_params(self.minimum.x)
        return self.minimum


sam = Sampler()
population = sam.population
minimum = sam.minimize()
print minimum.best
sam.run_with_params(minimum.best)
population.make_graph('stage4.graph.png')
plot_fractions(population, population.labels[:])
pylab.savefig('stage4.fraction.png', dpi=300)
plot_populations(population, population.labels[:])
pylab.savefig('stage4.pop.png', dpi=300)

# sampler = emcee.EnsembleSampler(n_walker, n_param, ln_overall)


# init_x_of_walkers = numpy.zeros((n_walker, n_param))
# for i_walker in range(n_walker):
#      for i_param, init_x in enumerate(init_params):
#           init_x_of_walkers[i_walker, i_param] = init_x + 1e-1 * init_x * numpy.random.uniform()

# n_mcmc_step = 40
# sampler.run_mcmc(init_x_of_walkers, n_mcmc_step)
# for i_param in range(n_param):
#     pylab.clf()
#     pylab.plot(range(n_mcmc_step), sampler.chain[0,:,i_param])
#     pylab.ylim([0, 1.2*sampler.chain[:,:,i_param].max()])

# pylab.clf()
# final_pops = []
# for i_mcmc_step in range(n_mcmc_step):
#     params = sampler.chain[0, i_mcmc_step, :]
#     self.run_with_params(params)
#     final_pops.append(population.vars["population"])
# pylab.plot(range(n_mcmc_step), final_pops)
# pylab.title("final_pops")
# pylab.savefig('sampler%d.png' % n_param)

# pylab.clf()
# n_mcmc_burn_step = 10
# samples = sampler.chain[:, n_mcmc_burn_step:, :].reshape((-1, n_param))
# n_sample = samples.shape[0]
# for i_sample in numpy.random.randint(n_sample, size=20):
#     params = samples[i_sample, :]
#     self.run_with_params(params)
#     population.calculate_fractions()
#     pylab.plot(times, population.total_population, color="k", alpha=0.1)
# self.run_with_params(params)
# population.calculate_fractions()
# pylab.plot(times, population.get_var_soln("population"), color="r", alpha=0.8)
# pylab.plot(times, population.get_var_soln("infected_populaton"), color="b", alpha=0.8)
# pylab.xlabel('year')
# pylab.ylabel('population')
# pylab.savefig('population.png')

# # os.system('open sampler.png stage2_mcmc_latentearly.png')


