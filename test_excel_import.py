# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 15:44:37 2015

@author: ntdoan
"""

from autumn.read_spreadsheet import get_input #, read_values_from_xls
from autumn.model import SingleComponentPopluationSystem, make_steps
from autumn.plotting import make_time_plots_color, make_time_plots_one_panel, plot_fractions
import pylab
from pprint import pprint

import_params = get_input('data/model_input.xlsx')
pprint(import_params)

population = SingleComponentPopluationSystem()

initials = import_params['const']['initials_for_compartments']
for key, value in initials.items():
    population.set_compartment(key, value[0])
    
parameters = import_params['const']['model_parameters']
for key, value in parameters.items():
    population.set_param(key, value[0])
    print(key,value,population.params[key])
      
#read_values_from_xls(population, 'tests/model_input.xlsx')
population.integrate_scipy(make_steps(0, 50, 1))
labels = population.labels
plot_fractions(population, population.labels)
pylab.show()
make_time_plots_color(population, labels)
make_time_plots_one_panel(population, labels, labels[1:])
