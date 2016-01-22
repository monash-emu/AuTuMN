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
from modules.parameter_estimation import Parameter, Evidence
import pylab
import emcee
import stage5
import scipy.optimize as optimize
from modules.model import make_steps
from stage4 import Stage4PopulationSystem
from scipy.stats import beta, gamma, norm, truncnorm



def make_neg_fn(fn):
     return lambda *args: -fn(*args)


def get_pop(params):
     n_tb_contact, init_pop = params
     population.set_param("n_tb_contact", n_tb_contact)
     population.set_compartment("susceptible_unvac", init_pop)
     population.integrate_explicit(times)
     population.calculate_fractions()

     # print population.vars["population"], population.compartments["susceptible_unvac"]
     # prevalence = population.vars["extra_deaths"]/1E6
     # print population.vars["population"], prevalence
     # return population.fractions[compartment] 


def ln_prob(params):
     n_tb_contact, init_pop = params

     if init_pop < 0:
          return 0.0

     population.set_param("n_tb_contact", n_tb_contact)
     population.set_compartment("susceptible_unvac", init_pop)
     population.integrate_explicit(times)

     prob = 1.0
     prob *= norm.pdf((n_tb_contact-20)/2.)
     final_pop = population.vars["population"]
     likelihood = norm.pdf((final_pop-99E6)/20E6)
     prob *= likelihood

     print "n_tb_contact=%.1f, init_pop=%.f, final_pop=%.f, delta=%.f%% likelihood=%f" % (n_tb_contact, init_pop, final_pop, (final_pop/init_pop-1.)*100., likelihood)

     if prob <= 0.0:
          return -numpy.inf
     else:
          return numpy.log(prob)


n_walker = 4
n_param = 2
population = Stage4PopulationSystem()
init_params = [40, 40E6]
titles_param = ["n_tb_contact", "init_population"]
times = make_steps(1, 50, 1)

sampler = emcee.EnsembleSampler(n_walker, n_param, ln_prob)

# minimimum = optimize.minimize(
#      make_neg_fn(ln_prob), [early_progression.prior_estimate])

init_x_of_walkers = numpy.zeros((n_walker, n_param))
for i_walker in range(n_walker):
     for i_param, init_x in enumerate(init_params):
          init_x_of_walkers[i_walker, i_param] = init_x + 1e-2 * init_x * numpy.random.uniform()
print init_x_of_walkers
n_mcmc_step = 40
sampler.run_mcmc(init_x_of_walkers, n_mcmc_step)
for i_param in range(n_param):
     pylab.clf()
     pylab.plot(range(n_mcmc_step), sampler.chain[0,:,i_param])
     pylab.ylim([0, 1.2*sampler.chain[:,:,i_param].max()])

pylab.clf()
final_pops = []
for i_mcmc_step in range(n_mcmc_step):
     params = sampler.chain[0, i_mcmc_step, :]
     get_pop(params)
     final_pops.append(population.vars["population"])
pylab.plot(range(n_mcmc_step), final_pops)
pylab.title("final_pops")
pylab.savefig('sampler%d.png' % n_param)

pylab.clf()
n_mcmc_burn_step = 10
samples = sampler.chain[:, n_mcmc_burn_step:, :].reshape((-1, n_param))
n_sample = samples.shape[0]
for i_sample in numpy.random.randint(n_sample, size=20):
     params = samples[i_sample, :]
     get_pop(params)
     population.calculate_fractions()
     pylab.plot(times, population.total_population, color="k", alpha=0.1)
get_pop(params)
population.calculate_fractions()
pylab.plot(times, population.total_population, color="r", alpha=0.8)
pylab.xlabel('year')
pylab.ylabel('population')
pylab.savefig('population.png')

# os.system('open sampler.png stage2_mcmc_latentearly.png')


