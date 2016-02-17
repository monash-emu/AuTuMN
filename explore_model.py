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

from autumn.model import SingleStrainTreatmentModel
from autumn.plotting import plot_fractions, plot_populations



def make_gamma_dist(mean, std):
    loc = 0
    shape = mean ** 2 / std ** 2  
    scale = std ** 2 / mean
    return gamma(shape, loc, scale)



class ModelRunner():

    def __init__(self):
        self.model = SingleStrainTreatmentModel()
        self.model.make_times(0, 100, 1.)
        self.is_last_run_sucess = False
        self.param_dict = [
            { 
                'init': 40,
                'scale': 10.,
                'key': 'n_tb_contact',
            },
            { 
                'init': 80E6,
                'scale': 10E6,
                'key': 'init_population',
            },
        ]

    def convert_param_list_to_dict(self, param_list):
        params = {}
        for val, param_dict in zip(param_list, self.param_dict):
            key = param_dict['key']
            params[key] = val
        return params

    def run_with_params(self, param_list):
        params = self.convert_param_list_to_dict(param_list)

        if params['init_population'] < 0 or \
                not isfinite(params['init_population']):
            self.is_last_run_sucess = False
            return

        self.model.set_param(
            "tb_n_contact",
            params["n_tb_contact"])

        self.model.set_compartment(
            "susceptible_fully",
            params['init_population'])

        self.is_last_run_sucess = True
        try:
            self.model.integrate_explicit()
        except:
            self.is_last_run_sucess = False

    def ln_overall(self, param_list):
        self.run_with_params(param_list)
        if not self.is_last_run_sucess:
            return -numpy.inf

        params = self.convert_param_list_to_dict(param_list)

        final_pop = self.model.vars["population"]
        prevalence = self.model.vars["infectious_population"] / final_pop * 1E5
        mortality = self.model.vars["rate_infection_death"] / final_pop * 1E5
        incidence = self.model.vars["incidence"]

        ln_prior = 0.0
        ln_prior += make_gamma_dist(40, 20).logpdf(params['n_tb_contact'])

        ln_posterior = 0.0
        ln_posterior += 5*norm(99E6, 10E6).logpdf(final_pop)
        ln_posterior += norm(417, 100).logpdf(prevalence)
        # ln_posterior += norm(288, 50).logpdf(incidence)
        # ln_posterior += norm(10, 10).logpdf(mortality)

        ln_overall = ln_prior + ln_posterior

        prints = [
           ("n={:.0f}", params['n_tb_contact']),
           ("start_pop={:0.0f}", params['init_population']),
           ("final_pop={:0.0f}", final_pop),
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

    def mcmc(self, n_mcmc_step=40, n_walker_per_param=2):
        n_param = len(self.param_dict)
        self.n_walker = n_walker_per_param * n_param
        self.n_mcmc_step = n_mcmc_step
        self.sampler = emcee.EnsembleSampler(
            self.n_walker, 
            n_param,
            lambda p: self.ln_overall(p))
        init_walkers = numpy.zeros((self.n_walker, n_param))
        init_params = [d['init'] for d in self.param_dict]
        for i_walker in range(self.n_walker):
             for i_param, init_x in enumerate(init_params):
                  init_walkers[i_walker, i_param] = init_x + 1e-1 * init_x * numpy.random.uniform()
        self.sampler.run_mcmc(init_walkers, self.n_mcmc_step)
        # Emma cautions here that if the step size is proportional to the parameter value,
        # then detailed balance will not be present.


def plot_mcmc_params(model_runner, base, n_burn_step=0):
    sampler = model_runner.sampler
    n_walker, n_mcmc_step, n_param = sampler.chain.shape
    samples = sampler.chain[:, n_burn_step:, :].reshape((-1, n_param))
    for i_param in range(n_param):
        pylab.clf()
        vals = samples[:,i_param]
        pylab.plot(range(len(vals)), vals)
        pylab.ylim([0, 1.2 * vals.max()])
        key = model_runner.param_dict[i_param]['key']
        pylab.title("parameter: " + key)
        pylab.xlabel("MCMC with %d walkers over %d steps each" % (n_walker, n_mcmc_step))
        pylab.savefig("%s.param.%s.png" % (base, key))


def plot_mcmc_var(model_runner, var, base, n_burn_step=0, n_model_show=40):
    model = model_runner.model
    sampler = model_runner.sampler
    times = model.times

    pylab.clf()

    chain = sampler.chain[:, n_burn_step:, :]
    n_param = chain.shape[-1]
    samples = chain.reshape((-1, n_param))
    n_sample = samples.shape[0]

    for i_sample in numpy.random.randint(n_sample, size=n_model_show):
        params = samples[i_sample, :]
        model_runner.run_with_params(params)
        model.calculate_diagnostics()
        pylab.plot(times, model.get_var_soln(var), color="k", alpha=0.1)

    init_params = [d['init'] for d in model_runner.param_dict]
    model_runner.run_with_params(init_params)
    model.calculate_diagnostics()
    pylab.plot(times, model.get_var_soln(var), color="r", alpha=0.8)

    pylab.xlabel('year')
    pylab.ylabel(var)
    pylab.title('Modelled %s for selection of MCMC parameters' % var)
    pylab.savefig(base + '.' + var + '.png')


model_runner = ModelRunner()

# minimum = model_runner.minimize()
# print minimum.best

# population = model_runner.population
# model_runner.run_with_params(minimum.best)
# population.make_graph('stage4.graph.png')
# plot_fractions(population, population.labels[:])
# pylab.savefig('stage4.fraction.png', dpi=300)
# plot_populations(population, population.labels[:]180)
# pylab.savefig('stage4.pop.png', dpi=300)

out_dir = "explore_model"
if not os.path.isdir(out_dir):
    os.makedirs(out_dir)

model_runner.mcmc(n_mcmc_step=100)

base = os.path.join(out_dir, 'mcmc')
plot_mcmc_params(model_runner, base, n_burn_step=0)
plot_mcmc_var(model_runner, 'population', base, n_burn_step=50)
