# -*- coding: utf-8 -*-

__doc__ = """
Created on Wed Nov 25 16:00:29 2015
Tests out the Parameter object in parameter_estimation.py (which is currently
just a prior setting object) and the Evidence object (which isn't yet used
within the Parameter object).
@author: James
"""

import os
import numpy
from modules.parameter_estimation import Parameter, Evidence
import pylab
import emcee
import stage4
import scipy.optimize as optimize
from modules.model import make_steps, Stage2PopulationSystem
from modules.parameter_setting import rate_late_progression


def make_neg_fn(fn):
    return lambda *args: -fn(*args)


class SamplerWrapper():
    def __init__(self):
        self.n_walker = 10
        self.n_dim = 1
        self.param_dicts = [
            {
                'param': rate_late_progression.pdf,
                'key': 'rate_late_progression'
            }
        ]
        self.sampler = emcee.EnsembleSampler(
            self.n_walker,
            self.n_dim,
            lambda params: self.ln_prior(params))

    def ln_prior(self, params):
        population = Stage2PopulationSystem()
        prob = 1.0
        for param, param_dict in zip(params, self.param_dicts):
            pdf = param_dict['pdf'](param)
            population.set_param(param_dict['key'], param)
            prob *= pdf
        population.set_flows()
        population.integrate_scipy(times)
        population.calculate_fractions()
        # return population.fractions[compartment]
        if prob <= 0.0:
            return -numpy.inf
        else:
            return numpy.log(prob)

    def get_pop(self, times, param, val, compartment):
        population = Stage2PopulationSystem()
        population.set_param(param, val)
        population.set_flows()
        population.integrate_scipy(times)
        population.calculate_fractions()
        return population.fractions[compartment]


rate_late_progression.graph_prior()

n_walker = 10
n_dim = 1
sampler = emcee.EnsembleSampler(n_walker, n_dim, ln_prior)

# minimimum = optimize.minimize(
#     make_neg_fn(ln_prior), [early_progression.prior_estimate])
init_x_of_walkers = numpy.zeros((n_walker, n_dim))
for i_walker in range(n_walker):
    init_x_of_walkers[i_walker, :] = \
        rate_late_progression.prior_estimate \
           + 1e-4 * numpy.random.randn(n_dim)

n_mcmc_step = 500
sampler.run_mcmc(init_x_of_walkers, n_mcmc_step)
pylab.clf()
pylab.plot(range(n_mcmc_step), sampler.chain[0, :, 0])
pylab.ylim([0, 1.2 * sampler.chain.max()])
pylab.show()

times = make_steps(0, 1000, 1)

pylab.clf()
n_mcmc_burn_step = 300
samples = sampler.chain[:, n_mcmc_burn_step:, :].reshape((-1, n_dim))
n_sample = samples.shape[0]
for i_sample in numpy.random.randint(n_sample, size=100):
    val = samples[i_sample, 0]
    print "%d rate_tb_earlyprog=%.3f" % (i_sample, val)
    vals = get_pop(times, "rate_tb_earlyprog", val, "active")
    pylab.plot(times, vals, color="k", alpha=0.1)
vals = get_pop(times, "rate_tb_earlyprog", rate_late_progression.prior_estimate, "active")
pylab.plot(times, vals, color="r", alpha=0.8)
pylab.xlabel('year')
pylab.ylabel('active')
pylab.savefig('stage2_mcmc_latentearly.png')

os.system('open pdf.png stage2_mcmc_latentearly.png')


# population = Stage4PopulationSystem()
# population.set_flows()
# population.make_graph('stage4.graph.png')
# population.integrate_scipy(make_steps(0, 50, 1))
# plot_fractions(population, population.labels[:])
# pylab.savefig('stage4.fraction.png', dpi=300)
#
# os.system('open -a "Google Chrome" stage4.graph.png stage4.fraction.png')
