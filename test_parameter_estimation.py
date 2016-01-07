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
from autumn.model import make_steps, Stage2PopulationSystem


def make_neg_fn(fn):
     return lambda *args: -fn(*args)

early_progression = Parameter(
     'Early progression proportion',
     'beta_symmetric_params2',
     numpy.mean([5, 10]) / 100 / 2, 
     0.02,
     [0.01, 0.05])

early_progression.graph_prior()
pylab.savefig('pdf.png')

def ln_prior(params):
     p = params[0]
     prob = early_progression.pdf(p)
     if prob <= 0.0:
          return -numpy.inf
     else:
          return numpy.log(prob)

def get_pop(times, param, val, compartment):
     population = Stage2PopulationSystem()
     population.set_param(param, val)
     population.set_flows()
     population.integrate_scipy(times)
     population.calculate_fractions()
     return population.fractions[compartment] 

n_walker = 10
n_dim = 1
sampler = emcee.EnsembleSampler(n_walker, n_dim, ln_prior)

# minimimum = optimize.minimize(
#      make_neg_fn(ln_prior), [early_progression.prior_estimate])
init_x_of_walkers = numpy.zeros((n_walker, n_dim))
for i_walker in range(n_walker):
     init_x_of_walkers[i_walker, :] = \
          early_progression.prior_estimate \
           + 1e-4*numpy.random.randn(n_dim)

n_mcmc_step = 500
sampler.run_mcmc(init_x_of_walkers, n_mcmc_step)
# pylab.clf()
# pylab.plot(range(n_mcmc_step), sampler.chain[0,:,0])
# pylab.ylim([0, 1.2*sampler.chain.max()])
# pylab.show()

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
vals = get_pop(times, "rate_tb_earlyprog", early_progression.prior_estimate,  "active")
pylab.plot(times, vals, color="r", alpha=0.8)
pylab.xlabel('year')
pylab.ylabel('active')
pylab.savefig('stage2_mcmc_latentearly.png')

os.system('open pdf.png stage2_mcmc_latentearly.png')


# early_progression_sloot2014 \
#     = Evidence('Early progression proportion', 0.075,
#                'Early progression proportion, Sloot et al. 2014',
#                'early_progression_sloot2014.pdf',
#                'From Figure 2, approximately 7 to 8% of the 739 contacts ' +
#                'with evidence of infection developed active TB in the early ' +
#                'high risk period, which lasted for around 6 months.',
#                'Sloot R, Schim van der Loeff MF, Kouw PM, Borgdorff MW. ' +
#                'Risk of Tuberculosis after Recent Exinit_x_of_walkersure. A 10-Year ' +
#                'Follow-up Study of Contacts in Amsterdam. Am J Respir ' +
#                'Crit Care Med. 2014;190(9):1044-1052.')
#early_progression_sloot2014.open_pdf()
