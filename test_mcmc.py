# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 10:42:36 2015
Testing mcmc and likelihood together. Starts with a populations (not integrated), 
some actual data points and the parameters (and initial estimates) to be tested.
Outputs are in the mcmc folder.
@author: Nick
"""

from autumn.model import SingleComponentPopluationSystem
from autumn.mcmc import mcmc

population = SingleComponentPopluationSystem()

# Need to write code to import data, this is a placeholding example
npops = 1
data = [dict() for p in range(npops)]
data[0]['prev'] = dict()
data[0]['prev']['years'] = [10, 15, 17]
data[0]['prev']['vals'] = [0.003, 0.003, 0.003]

# Need to write code to extract actual parameters used for the model.
# meta_prefit is an initial guess (or literature estimate)
meta_prefit = dict()
meta_prefit['n_tbfixed_contact'] = 40
meta_prefit['rate_pop_birth'] = 0.02
meta_prefit['rate_pop_death'] = 0.015384615384615385
meta_prefit['rate_tbfixed_earlyprog'] = 0.2
meta_prefit['rate_tbprog_completion'] = 1.8
for i in range(npops):
    meta_prefit['bias '+str(i)] = 0  # append bias for each population.


M = mcmc(population, meta_prefit, npops, data, nwalkers=50, nsteps=500, nburn=100)
