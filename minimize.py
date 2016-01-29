# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 16:00:29 2015
Tests out the Parameter object in parameter_estimation.py (which is currently
just a prior setting object) and the Evidence object (which isn't yet used
within the Parameter object).
@author: James
"""

import os
import numpy
from autumn.parameter_estimation import Parameter, Evidence
import pylab
import emcee
import stage5
import scipy.optimize as optimize
from autumn.model import make_steps
from stage4 import Stage4PopulationSystem
from scipy.stats import beta, gamma, norm, truncnorm



def make_neg_fn(fn):
     return lambda *args: -fn(*args)


def safe_log(p):
    if p <= 0.0:
        return -numpy.inf
    else:
        return numpy.log(p)


def get_pop(params):
     n_tb_contact, init_pop = params
     if init_pop < 0:
          return 0.0
     population.set_param("n_tb_contact", n_tb_contact)
     population.set_compartment("susceptible_unvac", init_pop)
     population.make_steps(1, 50, 1)
     population.integrate_explicit()
     population.calculate_fractions()


def probability(params):
    get_pop(params)

    n_tb_contact, init_pop = params
    prior = 1.0
    prior *= norm.pdf((n_tb_contact-40)/20.)
    final_pop = population.vars["population"]
    mortality = population.vars["rate_infection_death"]/final_pop*1E5
    incidence = population.vars["incidence"]/final_pop*1E5
    prevalence = population.vars["rate_prevalence"]/final_pop*1E5
    likelihood = 0.0
    likelihood += 10E6*norm(99E6, 10E6).pdf(final_pop)
    likelihood += 10E6*norm(417, 30).pdf(prevalence)
    prob = prior * likelihood

    prints = [
       ("n_tb_contact={:.0f}", n_tb_contact),
       ("init_pop={:0.0f}", init_pop),
       ("final_pop={:0.0f}", final_pop),
       ("delta={:0.0f}", (final_pop/init_pop-1.)*100.),
       ("mortality={:0.0f}", mortality),
       ("incidence={:0.0f}", incidence),
       ("prevalence={:0.0f}", prevalence),
       ("prior={:0.3f}", prior),
       ("likelihood={:0.3f}", likelihood),
       ("prob={:0.3f}", prob),
       ("log_prob={:0.3f}", safe_log(prob)),
    ]
    s = " ".join(p[0] for p in prints)
    print s.format(*[p[1] for p in prints])

    return prob


def ln_prob(params):
    return safe_log(probability(params))


n_walker = 4
n_param = 2
population = Stage4PopulationSystem()
init_params = [20, 40E6]
scales = [100., 100E6]
titles_param = ["n_tb_contact", "init_population"]
times = make_steps(1, 50, 1)

scaled_init_params = [param/scale for param, scale in zip(init_params, scales)]
def neg_scaled_fn(scaled_params):
    params = [scale*scaled_param for scale, scaled_param in zip(scales, scaled_params)]
    return -probability(params)

minimimum = optimize.minimize( neg_scaled_fn, scaled_init_params)
print " ".join(["%.f" % (scale * x) for scale, x in zip(scales, minimimum.x)])

# sampler = emcee.EnsembleSampler(n_walker, n_param, ln_prob)


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
#     get_pop(params)
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
#     get_pop(params)
#     population.calculate_fractions()
#     pylab.plot(times, population.total_population, color="k", alpha=0.1)
# get_pop(params)
# population.calculate_fractions()
# pylab.plot(times, population.get_var_soln("population"), color="r", alpha=0.8)
# pylab.plot(times, population.get_var_soln("infected_populaton"), color="b", alpha=0.8)
# pylab.xlabel('year')
# pylab.ylabel('population')
# pylab.savefig('population.png')

# # os.system('open sampler.png stage2_mcmc_latentearly.png')


