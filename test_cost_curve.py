# -*- coding: utf-8 -*-
"""

Test
- implement a simple 4-params cost-coverage curve 
- apply to transmission model

"""

import math
import numpy
from autumn.model import SingleComponentPopluationSystem, make_steps
from autumn.plotting import plot_fractions
import pylab
import autumn.ccocs as ccocs
import numpy
import pylab



def cost_coverage_with_params(x, params):
    """
    Returns cost-coverage for x (total cost). 
    Params:
    - p[0]: lower
    - p[1]: upper
    - p[2]: x-inflection point
    - p[3]: grwoth rate
    """
    return  p[0] + ( p[1] - p[0] ) / ( 1. + math.exp( -( x - p[2] ) / p[3] ) )


p = [0.2, 0.5, 0.3, 0.1]
def cost_coverage(x): 
    return cost_coverage_with_params (x, p)  

y_values = []

for x in x_values: 
    y_values.append (new_cost_fx(x))
print(y_values)    

import pylab 
pylab.plot(x_values, y_values)
pylab.ylim ([0, 1])
pylab.show()



codata = [0, 0, 2, 2]
c =  ccocs.co_cofun(codata)
x = numpy.linspace(0,1,100)
y = c.evaluate(x) 
pylab.plot(x,y)
pylab.show()


codata = [1, 0]
c =  ccocs.co_linear(codata)
x = numpy.linspace(0,1,100)
y = c.evaluate(x) 
pylab.plot(x,y)
pylab.show()


population = SingleComponentPopluationSystem()

budget_for_tbprog_detect = 100
population.set_param(
    'rate_tbprog_detect',
    cost_coverage(budget_for_tbprog_detect))

times = numpy.linspace(0, 50, 51)
population.integrate_scipy(times)

plot_fractions(population, population.labels)
pylab.show()
