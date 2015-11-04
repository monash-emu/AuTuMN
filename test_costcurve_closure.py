# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 11:31:20 2015

@author: ntdoan
"""
def cc2eqn(x, p):
    '''
    2-parameter equation defining cc curves.
    x is total cost, p is a list of parameters (of length 2):
        p[0] = saturation
        p[1] = growth rate
    Returns y which is coverage. '''
    y =  p[0] + (p[1] - p[0])/(1. + exp(-(x-p[2])/p[3]))
   # 2*p[0] / (1 + exp(-p[1]*x)) - p[0]
    return y
   
from autumn.model import make_steps
x_values = make_steps (0., 1., 0.01)
print(steps)
p = [0.2, 0.5, 0.3, 0.1]


def new_cost_fx (x): 
    return cc2eqn (x, p)  # closure 
    
y_values = []

for x in x_values: 
    y_values.append (new_cost_fx(x))
print(y_values)    

import pylab 
pylab.plot(x_values, y_values)
pylab.ylim ([0, 1])
pylab.show()

