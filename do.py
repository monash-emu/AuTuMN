# -*- coding: utf-8 -*-



__doc__ = """
Created on Wed Nov 25 16:00:29 2015

Parameter search over Stage 4 model as in stage4.py

@author: Bosco
"""

from collections import OrderedDict

import numpy
from scipy.stats import beta, gamma, norm, truncnorm
import pylab
import emcee

from modules.parameter_estimation import Parameter, Evidence
from modules.parameter_setting import rate_late_progression
from modules.model import make_steps

from stage4 import Stage4PopulationSystem


def make_neg_fn(fn):
    return lambda *args: -fn(*args)


def flat(p):
    return 1.0


def population_pdf(x):
    return norm.pdf((x-99E6)/100E6)


def contact_pdf(x):
    return norm.pdf((x-40.)/100.)


class SamplerWrapper():

    def __init__(self):
        self.n_mcmc_step = 80
        self.param_dicts = OrderedDict()
        self.times = make_steps(0, 30, 1)
        self.population = Stage4PopulationSystem()

        rate_late_progression.calculate_prior()
        self.param_dicts['n_tb_contact'] = {
            'init': 40,
            'pdf': contact_pdf,
        }
        self.param_dicts['init_population'] = {
            'init': 1E6,
            'pdf': flat,
        }

        self.n_dim = len(self.param_dicts)
        self.n_walker = self.n_dim * 2
        self.sampler = emcee.EnsembleSampler(
            self.n_walker,
            self.n_dim,
            lambda params: self.ln_posterior(params))

    def ln_posterior(self, params):
        prior = 1.0
        prior *= self.param_dicts['n_tb_contact']['pdf'](params[0])
        prior *= self.param_dicts['init_population']['pdf'](params[1])

        self.population = Stage4PopulationSystem()
        self.population.set_param('n_tb_contact', params[0])
        self.population.set_compartment('susceptible_unvac', params[1])
        self.population.set_flows()
        print "init", self.population.compartments["susceptible_unvac"]

        self.population.integrate_explicit(self.times)

        likelihood = 1.0
        pop = self.population.vars['population']
        likelihood *= population_pdf(pop)

        posterior = prior * likelihood
        print params, 1E6, self.population.compartments["susceptible_unvac"]

        if posterior <= 0.0:
            return -numpy.inf
        else:
            return numpy.log(posterior)

    def make_init_walkers(self):
        init_x_of_walkers = numpy.zeros((self.n_walker, self.n_dim))
        for i_walker in range(self.n_walker):
            for i_param, param_dict in enumerate(self.param_dicts.values()):
                init_x_of_walkers[i_walker, i_param] = \
                    param_dict['init'] + 1e-4 * numpy.random.randn(1)
        return init_x_of_walkers

    def sample(self):
        init_x_of_walkers = self.make_init_walkers()
        
        self.sampler.run_mcmc(init_x_of_walkers, self.n_mcmc_step)

        pylab.clf()
        pylab.plot(range(self.n_mcmc_step), self.sampler.chain[0, :, 0])
        pylab.ylim([0, 1.2 * self.sampler.chain[0].max()])
        pylab.show()

        pylab.clf()
        n_mcmc_burn_step = 20
        samples = self.sampler.chain[:, n_mcmc_burn_step:, :].reshape((-1, self.n_dim))
        n_sample = samples.shape[0]
        for i_sample in range(n_sample):
            val = samples[i_sample, 0]
            print "%d rate_tb_earlyprog=%.3f" % (i_sample, val)
            vals = self.population.get_soln("latent_early")
            pylab.plot(self.times, vals, color="k", alpha=0.1)
        vals = self.population.get_soln("latent_early")
        pylab.plot(self.times, vals, color="r", alpha=0.8)
        pylab.xlabel('year')
        pylab.ylabel('latent_early')
        pylab.savefig('do_latent_early.png')


sampler = SamplerWrapper()
sampler.sample()


