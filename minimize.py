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

from stage4 import ThreeStrainSystem
from autumn.plotting import plot_fractions, plot_populations



def make_gamma_dist(mean, std):
    loc = 0
    shape = mean ** 2 / std ** 2  
    scale = std ** 2 / mean
    return gamma(shape, loc, scale)



class ModelRunner():

    def __init__(self):
        self.population = ThreeStrainSystem()
        self.population.make_times(0, 100, 1.)
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
        incidence = self.population.vars["incidence"]

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

    def mcmc(self):
        self.n_walker = 2 * self.n_param
        self.n_mcmc_step = 40
        self.sampler = emcee.EnsembleSampler(
            self.n_walker, 
            self.n_param, 
            lambda p: self.ln_overall(p))
        init_walkers = numpy.zeros((self.n_walker, self.n_param))
        init_params = [d['init'] for d in self.param_dict]
        for i_walker in range(self.n_walker):
             for i_param, init_x in enumerate(init_params):
                  init_walkers[i_walker, i_param] = init_x + 1e-1 * init_x * numpy.random.uniform()
        self.sampler.run_mcmc(init_walkers, self.n_mcmc_step)


def plot_params(sampler, base):
    n_walker, n_mcmc_step, n_param = sampler.chain.shape
    n_burn_step = 0
    samples = sampler.chain[:, n_burn_step:, :].reshape((-1, n_param))
    i_walker = 0
    for i_param in range(n_param):
        pylab.clf()
        vals = samples[:,i_param]
        pylab.plot(range(len(vals)), vals)
        pylab.ylim([0, 1.2 * vals.max()])
        pylab.savefig("%s.param%d.png" % (base, i_param))


model_runner = ModelRunner()

# minimum = model_runner.minimize()
# print minimum.best

# population = model_runner.population
# model_runner.run_with_params(minimum.best)
# population.make_graph('stage4.graph.png')
# plot_fractions(population, population.labels[:])
# pylab.savefig('stage4.fraction.png', dpi=300)
# plot_populations(population, population.labels[:])
# pylab.savefig('stage4.pop.png', dpi=300)

model_runner.mcmc()
plot_params(model_runner.sampler, 'minimize')

pylab.clf()
n_mcmc_burn_step = 0
chain = model_runner.sampler.chain[:, n_mcmc_burn_step:, :]
n_param = chain.shape[-1]
samples = chain.reshape((-1, n_param))
n_sample = samples.shape[0]
population = model_runner.population
times = population.times
for i_sample in numpy.random.randint(n_sample, size=20):
    params = samples[i_sample, :]
    model_runner.run_with_params(params)
    population.calculate_diagnostics()
    pylab.plot(times, population.total_population_soln, color="k", alpha=0.1)
model_runner.run_with_params(params)
population.calculate_diagnostics()
pylab.plot(times, population.get_var_soln("population"), color="r", alpha=0.8)
pylab.plot(times, population.get_var_soln("infected_populaton"), color="b", alpha=0.8)
pylab.xlabel('year')
pylab.ylabel('population')
pylab.savefig('population.png')



