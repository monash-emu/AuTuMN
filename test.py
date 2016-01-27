
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


n_walker = 4
n_param = 2
population = Stage4PopulationSystem()
init_params = [20, 40E6]
scale = [100, 100E6]
titles_param = ["n_tb_contact", "init_population"]
times = make_steps(1, 50, 1)


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
     population.integrate_explicit(times)
     population.calculate_fractions()


def probability(params):
    get_pop(params)

    n_tb_contact, init_pop = params
    prior = 1.0
    prior *= norm.pdf((n_tb_contact-20)/2.)
    final_pop = population.vars["population"]
    mortality = population.vars["rate_disease_death"]/final_pop*1E5
    incidence = population.vars["incidence"]/final_pop*1E5
    prevalence = population.vars["rate_prevalence"]/final_pop*1E5
    likelihood = 0.0
    likelihood += 99E6*norm(99E6, 10E6).pdf(final_pop)
    likelihood += 417*norm(417, 200).pdf(prevalence)
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


mean = 417
std = mean/3.
xvals = make_steps(0, 2*mean, mean/100.)
fn = lambda x: mean*norm(mean, std).pdf(x)
# fn = lambda x: norm().pdf((x-mean)/std)
pylab.plot(xvals, map(fn, xvals))
pylab.savefig('test.png')

probability([20, 40E6])
probability([17, 45E6])