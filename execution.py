# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 15:44:37 2015

@author: ntdoan
"""

from autumn.read_spreadsheet import get_input 
from autumn.model import SingleComponentPopluationSystem, make_steps
from autumn.plotting import make_time_plots_color, make_time_plots_one_panel, plot_fractions
import pylab
from pprint import pprint
import math




def cc2eqn(x, p):
    '''
    2-parameter equation defining cc curves.
    x is total cost, p is a list of parameters (of length 2):
        p[0] = saturation
        p[1] = growth rate
    Returns y which is coverage. '''
    y =  p[0] + (p[1] - p[0])/(1. + math.exp(-(x-p[2])/p[3]))
   # 2*p[0] / (1 + exp(-p[1]*x)) - p[0]
    return y
   

p = [0.2, 0.5, 0.3, 0.1]


def new_cost_fx (x): 
    return cc2eqn (x, p)  # closure 
    


import_params = get_input('autumn/input.xlsx')
pprint(import_params)


population = SingleComponentPopluationSystem()


initials = import_params['const']['initials_for_compartments']
for key, value in initials.items():
    population.set_compartment(key, value[0])
    
parameters = import_params['const']['model_parameters']
for key, value in parameters.items():
    population.set_param(key, value[0])
    print(key, value, population.params[key])
      

budget = 100
key = 'rate_tbprog_detect'
print('default %s=%f' % (key, population.params[key]))
population.set_param(key, new_cost_fx(budget))
print('value after $%f %s=%f' % (budget, key, population.params[key]))

budget_for_tbprog_detect = 100
population.set_param(
    'rate_tbprog_detect',
    new_cost_fx(budget_for_tbprog_detect))


population.integrate_scipy(make_steps(0, 50, 1))
labels = population.labels
plot_fractions(population, population.labels)
pylab.show()
make_time_plots_color(population, labels)
make_time_plots_one_panel(population, labels, labels[1:])
